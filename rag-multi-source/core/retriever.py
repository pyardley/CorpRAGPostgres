"""
Retriever — turns ``(query, filter_dict)`` into a list of :class:`RetrievedChunk`.

Multi-tenancy is enforced at two layers:

1. **Application-layer filter (primary).** ``filter_dict`` (built via
   :func:`core.vector_store.build_query_filter`) becomes the SQL
   ``WHERE`` clause that constrains the search to the user's
   accessible resources. There is no ``user_id`` column on the vector
   table — the filter is the original security boundary.

2. **PostgreSQL Row-Level Security (defence-in-depth).** When
   ``settings.ENABLE_RLS`` is True, the retriever also binds the
   active user_id to the per-session GUC ``app.current_user_id`` via
   :func:`app.utils.set_current_user_for_rls` so the RLS policies on
   ``vector_chunks`` / ``user_accessible_resources`` are checked by
   the database itself. If the application-layer ``WHERE`` clause is
   ever bypassed by a regression, the policies still refuse to
   surface chunks the current user wasn't granted.

Retrieval modes
---------------
``settings.RETRIEVAL_MODE`` selects between two strategies:

* ``"vector"`` — pure cosine similarity over the HNSW index. Original
  behaviour. Best for conceptual / semantic queries.
* ``"hybrid"`` — runs the vector search and a Postgres FTS keyword
  search in parallel, then fuses the two ranked lists with Reciprocal
  Rank Fusion (RRF). The keyword side is essential for rare-token
  queries (error codes, ticket numbers, identifiers like ``XYZ999``)
  where the embedding is dominated by surrounding semantic context
  rather than the literal token.

Both modes share the per-source quota, the RLS bind, and the
``hnsw.ef_search`` tuning. Hybrid mode requires the
``vector_chunks.text_search`` generated column + GIN index that
``models.migrations`` provisions automatically on first run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from sqlalchemy import func, literal_column, select, text

from app.config import settings
from app.utils import get_db, set_current_user_for_rls
from core.llm import get_embeddings
from core.vector_store import build_query_filter, filter_to_where
from models.vector_chunk import VectorChunk


@dataclass
class RetrievedChunk:
    """One similarity-search hit returned to the chat layer."""

    resource_id: str
    source: str
    chunk_index: int
    score: float                # cosine similarity in [-1, 1]; we expect 0..1
    text: str
    title: str
    url: str
    metadata: dict[str, Any]

    @property
    def citation_label(self) -> str:
        return self.title or self.resource_id


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _select_columns(similarity_expr):
    """
    Column projection shared by the vector and keyword searches so per-source
    quota, RRF fusion, and row-to-RetrievedChunk conversion can treat both
    result sets interchangeably.
    """
    return (
        VectorChunk.resource_id,
        VectorChunk.source,
        VectorChunk.chunk_index,
        VectorChunk.text,
        VectorChunk.title,
        VectorChunk.url,
        VectorChunk.project_key,
        VectorChunk.space_key,
        VectorChunk.db_name,
        VectorChunk.git_scope,
        VectorChunk.email_provider,
        VectorChunk.object_name,
        VectorChunk.last_updated,
        VectorChunk.extra,
        similarity_expr,
    )


def _run_vector_search(db, embedding, where_clause, score_threshold, limit):
    """Pure cosine-similarity search ordered by HNSW distance."""
    similarity_expr = (
        1 - VectorChunk.embedding.cosine_distance(embedding)
    ).label("similarity")
    distance_expr = VectorChunk.embedding.cosine_distance(embedding)

    stmt = (
        select(*_select_columns(similarity_expr))
        .where(where_clause)
        .where(similarity_expr >= score_threshold)
        .order_by(distance_expr)
        .limit(limit)
    )
    return db.execute(stmt).all()


def _run_keyword_search(db, query_text, embedding, where_clause, limit):
    """
    Postgres FTS keyword search.

    Uses the ``text_search`` generated column (managed by
    ``models.migrations``) plus ``websearch_to_tsquery`` for permissive
    user-input parsing — quoted phrases, OR-of-terms, and a leading
    ``-`` for negation all work without raising on weird syntax.

    The ``score`` field on returned rows is still cosine similarity,
    computed alongside the FTS rank, so downstream code that displays
    ``RetrievedChunk.score`` doesn't have to special-case keyword-only
    hits. Note ``score_threshold`` is *not* applied here — the whole
    point of the hybrid path is to surface lexical matches the
    embedding missed (low cosine, high lexical relevance).

    ``literal_column("text_search")`` references the generated column
    by name without requiring it to be declared on the SQLAlchemy
    model (the model stays focused on application-managed columns;
    the generated column is the migration's responsibility).
    """
    similarity_expr = (
        1 - VectorChunk.embedding.cosine_distance(embedding)
    ).label("similarity")
    text_search_col = literal_column("text_search")

    # Postgres has two ``websearch_to_tsquery`` overloads — ``(text)`` and
    # ``(regconfig, text)`` — and *no* ``(text, text)`` form. SQLAlchemy
    # binds plain Python strings as text, so we have to disambiguate the
    # first argument explicitly. We do that by inlining the language as a
    # ``::regconfig``-cast literal via ``literal_column``. The value comes
    # from settings (not user input) and is restricted to a string by
    # pydantic, so injection-by-config would require an attacker who can
    # already write to .env — a non-threat. ``query_text`` *is* user
    # input and stays a parameter binding, which is where the safety
    # actually matters.
    fts_regconfig = literal_column(f"'{settings.FTS_LANGUAGE}'::regconfig")
    ts_query = func.websearch_to_tsquery(fts_regconfig, query_text)
    rank_expr = func.ts_rank_cd(text_search_col, ts_query).label("rank")

    stmt = (
        select(*_select_columns(similarity_expr), rank_expr)
        .where(where_clause)
        .where(text_search_col.op("@@")(ts_query))
        .order_by(rank_expr.desc())
        .limit(limit)
    )
    return db.execute(stmt).all()


def _rrf_fuse(vector_rows, keyword_rows, k):
    """
    Reciprocal Rank Fusion across two ranked lists.

    For each chunk that appears in either list, accumulate
    ``1 / (k + rank)`` (rank is 1-based). The fused output is sorted
    by that score descending. ``k`` smooths the contribution of
    lower-ranked items; 60 is the value from the original RRF paper
    and is rarely worth tuning.

    Fusion key is ``(resource_id, chunk_index)`` so a chunk found in
    both lists collapses to a single row whose RRF score is the sum
    of its two reciprocal ranks — exactly the chunks RRF promotes
    most aggressively, since they're endorsed by both retrievers.

    Note we keep the *original* row from whichever list saw it first,
    so the row's ``similarity`` column (cosine) carries through
    unchanged. The RRF score is used only for ordering.
    """
    scores: dict[tuple[str, int], float] = {}
    rows_by_key: dict[tuple[str, int], Any] = {}

    for rank, row in enumerate(vector_rows, start=1):
        key = (row._mapping["resource_id"], int(row._mapping["chunk_index"]))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        rows_by_key.setdefault(key, row)
    for rank, row in enumerate(keyword_rows, start=1):
        key = (row._mapping["resource_id"], int(row._mapping["chunk_index"]))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        rows_by_key.setdefault(key, row)

    ordered_keys = sorted(scores.keys(), key=lambda kk: scores[kk], reverse=True)
    return [rows_by_key[kk] for kk in ordered_keys]


def _apply_source_quota(rows, top_k, max_per_source):
    """
    Soft cap on how many chunks each source can contribute to top_k.

    Walk score-ordered ``rows`` once: take a chunk if its source still
    has room under the cap, otherwise stash it in ``overflow``.
    Backfill from overflow if fewer than ``top_k`` chunks made it
    through (single-source filters, near-empty corpora). Disabled when
    ``max_per_source >= top_k``.
    """
    if max_per_source >= top_k:
        return rows[:top_k]

    per_source: dict[str, int] = {}
    primary: list[Any] = []
    overflow: list[Any] = []
    for row in rows:
        src = row._mapping["source"]
        if per_source.get(src, 0) < max_per_source:
            primary.append(row)
            per_source[src] = per_source.get(src, 0) + 1
            if len(primary) >= top_k:
                break
        else:
            overflow.append(row)
    if len(primary) < top_k:
        for row in overflow:
            primary.append(row)
            if len(primary) >= top_k:
                break
    return primary


def _rows_to_chunks(rows) -> list[RetrievedChunk]:
    """Materialise SQLAlchemy rows into ``RetrievedChunk`` objects."""
    hits: list[RetrievedChunk] = []
    for row in rows:
        m = row._mapping  # SQLAlchemy 2.x row mapping
        # Re-assemble metadata so the rest of the pipeline (citation
        # rendering) sees a single dict with promoted columns folded in.
        # NB the row-mapping key is the Python attribute name "extra", not
        # the underlying DB column name "metadata" — the column was
        # declared as `extra = Column("metadata", JSONB, …)`.
        metadata: dict[str, Any] = dict(m["extra"] or {})
        for col in (
            "project_key",
            "space_key",
            "db_name",
            "git_scope",
            "email_provider",
            "object_name",
            "last_updated",
        ):
            value = m[col]
            if value is not None:
                metadata.setdefault(col, value)

        hits.append(
            RetrievedChunk(
                resource_id=m["resource_id"],
                source=m["source"],
                chunk_index=int(m["chunk_index"]),
                score=float(m["similarity"]),
                text=m["text"] or "",
                title=m["title"] or m["object_name"] or "",
                url=m["url"] or "",
                metadata=metadata,
            )
        )
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    filter_dict: dict[str, Any],
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    *,
    user_id: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    Run a similarity search constrained by ``filter_dict``.

    Vector mode emits roughly::

        SELECT
            id, resource_id, source, chunk_index, text,
            title, url, project_key, space_key, db_name, git_scope,
            object_name, last_updated, metadata,
            1 - (embedding <=> :query_vec) AS similarity
        FROM vector_chunks
        WHERE <filter clauses derived from filter_dict>
          AND 1 - (embedding <=> :query_vec) >= :threshold
        ORDER BY embedding <=> :query_vec
        LIMIT :candidate_limit

    Hybrid mode runs the same SELECT plus a parallel FTS query::

        SELECT ..., ts_rank_cd(text_search, q) AS rank
        FROM vector_chunks
        WHERE <filter clauses>
          AND text_search @@ websearch_to_tsquery(:lang, :user_query)
        ORDER BY rank DESC
        LIMIT :candidate_limit

    and fuses the two ranked lists with Reciprocal Rank Fusion.

    `<=>` is pgvector's cosine-distance operator (1 - similarity); ordering
    by it directly uses the HNSW index for sub-millisecond top-K on tens of
    millions of rows.

    Parameters
    ----------
    user_id
        Active user's UUID. Bound to the session's
        ``app.current_user_id`` GUC before issuing the SELECT so the
        Row-Level Security policies on ``vector_chunks`` recognise the
        caller. Required when ``settings.ENABLE_RLS`` is True;
        without it the RLS policies will deny every row and the
        retriever will return an empty list. Optional otherwise.
    """
    if not query.strip():
        return []

    top_k = top_k or settings.TOP_K
    score_threshold = (
        settings.SCORE_THRESHOLD if score_threshold is None else score_threshold
    )

    where_clause = filter_to_where(filter_dict)
    if where_clause is None:
        # No accessible scopes for any selected source — nothing to search.
        logger.debug("Empty filter; returning no hits.")
        return []

    embedding = get_embeddings().embed_query(query)

    # Fetch a wider candidate pool than ``top_k`` so the per-source quota
    # (applied below) and RRF fusion have something to work with. Without
    # this, capping any source would just hand the top-K back short.
    fetch_multiplier = max(1, int(settings.RETRIEVAL_FETCH_MULTIPLIER))
    candidate_limit = top_k * fetch_multiplier

    mode = settings.RETRIEVAL_MODE

    with get_db() as db:
        # Defence-in-depth: bind the user_id so the RLS policies can
        # double-check the rows the WHERE clause is about to filter.
        # No-op when ENABLE_RLS=False or user_id is missing.
        set_current_user_for_rls(db, user_id)

        # Raise HNSW recall for this statement only. The pgvector default
        # ``hnsw.ef_search=40`` causes the graph traversal to converge in
        # whichever cluster it enters first — when one source has a dense
        # neighbourhood near the query embedding (e.g. many similar
        # marketing emails) it crowds out a higher-scoring chunk in
        # another source even though that chunk would rank above the
        # cluster globally. ``SET LOCAL`` scopes the change to the current
        # transaction so it never leaks to other sessions on the pool.
        # See ``settings.HNSW_EF_SEARCH`` for tuning rationale.
        #
        # NB: Postgres ``SET`` doesn't accept bind parameters, so we
        # inline the integer literal. ``int(...)`` neutralises any
        # injection risk from a pathological config override.
        ef_search = int(settings.HNSW_EF_SEARCH)
        db.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

        vector_rows = _run_vector_search(
            db, embedding, where_clause, score_threshold, candidate_limit
        )

        if mode == "hybrid":
            keyword_rows = _run_keyword_search(
                db, query, embedding, where_clause, candidate_limit
            )
            fused_rows = _rrf_fuse(
                vector_rows, keyword_rows, k=int(settings.RRF_K)
            )
            logger.debug(
                "[hybrid] vector={} keyword={} fused={}",
                len(vector_rows),
                len(keyword_rows),
                len(fused_rows),
            )
        else:
            fused_rows = vector_rows

    # ── Per-source diversity quota ──────────────────────────────────────
    # ``fused_rows`` is already in score order — vector mode by cosine
    # similarity, hybrid mode by RRF score. Capping per source preserves
    # diversity when a single source has a dense cluster near the query;
    # the backfill from overflow guarantees single-source queries still
    # return TOP_K results. Disabled when MAX_HITS_PER_SOURCE >= TOP_K.
    max_per_source = max(1, int(settings.MAX_HITS_PER_SOURCE))
    rows = _apply_source_quota(fused_rows, top_k, max_per_source)

    hits = _rows_to_chunks(rows)

    logger.info(
        "Retrieved {} chunks (mode={}, threshold {}, top_k {})",
        len(hits),
        mode,
        score_threshold,
        top_k,
    )
    return hits


def deduplicate_by_resource(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """At most one chunk per resource_id, keeping the highest-scoring one."""
    best: dict[str, RetrievedChunk] = {}
    for hit in hits:
        existing = best.get(hit.resource_id)
        if existing is None or hit.score > existing.score:
            best[hit.resource_id] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


# Re-exported so callers can do `from core.retriever import build_query_filter`
__all__ = [
    "RetrievedChunk",
    "retrieve",
    "deduplicate_by_resource",
    "build_query_filter",
]
