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

Per-source candidate fetching
-----------------------------
The retriever does **not** issue one giant SELECT spanning every
enabled source. Instead it runs the vector (and, in hybrid mode,
FTS) search **once per enabled source**, each with its own
``WHERE source = X AND <scope_col> IN (…)`` clause and its own
``LIMIT``. The per-source result sets are then pooled, RRF-fused,
and passed through the per-source quota.

Why: with a single global SELECT, a source whose vectors form a
dense neighbourhood near the query embedding (e.g. tens of
thousands of similar marketing emails) can completely fill the
HNSW candidate pool, leaving no room for a higher-scoring chunk
from another source. Bumping ``hnsw.ef_search`` and the candidate
multiplier helps but never *guarantees* fairness — at sufficient
imbalance the dense source still wins. Per-source fetching makes
the guarantee structural: every selected source gets its own
candidate budget, so SQL Server schema chunks (or any other
sparser source) can't be evicted by sheer email volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from sqlalchemy import func, literal_column, select, text

from app.config import settings
from app.utils import get_db, set_current_user_for_rls
from core.llm import get_embeddings
from core.runtime_config import get_fts_language
from core.vector_store import (
    build_query_filter,
    filter_to_where,
    filter_to_where_for_source,
)
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
    # from ``core.runtime_config`` (UI-toggleable, falls back to
    # ``settings.FTS_LANGUAGE``) and is whitelisted to ``english`` /
    # ``simple`` by ``set_fts_language`` before it ever reaches DDL, so
    # injection-by-config would require an attacker who can already write
    # to the runtime-config file — a non-threat. ``query_text`` *is* user
    # input and stays a parameter binding, which is where the safety
    # actually matters.
    fts_regconfig = literal_column(f"'{get_fts_language()}'::regconfig")
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

    For each enabled source in ``filter_dict["by_source"]``, vector
    mode emits roughly::

        SELECT
            id, resource_id, source, chunk_index, text,
            title, url, project_key, space_key, db_name, git_scope,
            object_name, last_updated, metadata,
            1 - (embedding <=> :query_vec) AS similarity
        FROM vector_chunks
        WHERE source = :this_source
          AND <scope_col> = ANY(:this_source_scopes)
          AND 1 - (embedding <=> :query_vec) >= :threshold
        ORDER BY embedding <=> :query_vec
        LIMIT :per_source_limit

    Hybrid mode also issues a parallel FTS query per source::

        SELECT ..., ts_rank_cd(text_search, q) AS rank
        FROM vector_chunks
        WHERE source = :this_source
          AND <scope_col> = ANY(:this_source_scopes)
          AND text_search @@ websearch_to_tsquery(:lang, :user_query)
        ORDER BY rank DESC
        LIMIT :per_source_limit

    The per-source result sets are pooled, re-sorted by their global
    scoring metric, RRF-fused, and finally passed through the
    per-source quota for the TOP_K cut.

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

    by_source: dict[str, list[str]] = (filter_dict or {}).get("by_source") or {}
    if not by_source:
        # No accessible scopes for any selected source — nothing to search.
        logger.debug("Empty filter; returning no hits.")
        return []

    embedding = get_embeddings().embed_query(query)

    fetch_multiplier = max(1, int(settings.RETRIEVAL_FETCH_MULTIPLIER))
    max_per_source = max(1, int(settings.MAX_HITS_PER_SOURCE))

    # Per-source candidate budget.
    #
    # Each enabled source gets its own SELECT with its own LIMIT, so the
    # budget here is per source — not the old ``top_k * multiplier``
    # global pool. We want enough headroom that the per-source quota
    # (and its overflow backfill, when other sources are sparse) has
    # plenty to draw from.
    #
    # ``max_per_source * fetch_multiplier`` covers the quota's primary
    # slots with multiplier-fold headroom for backfill. We also floor at
    # ``top_k`` so a single-source query (only one source enabled) still
    # gets enough rows to fill TOP_K from one source's overflow.
    #
    # Defaults (top_k=8, max_per_source=4, fetch_multiplier=4) give a
    # per-source limit of 16 — comfortably above the 4-cap and TOP_K=8
    # backfill ceiling.
    per_source_limit = max(top_k, max_per_source * fetch_multiplier)

    mode = settings.RETRIEVAL_MODE

    with get_db() as db:
        # Defence-in-depth: bind the user_id so the RLS policies can
        # double-check the rows the WHERE clauses below filter. No-op
        # when ENABLE_RLS=False or user_id is missing.
        set_current_user_for_rls(db, user_id)

        # Raise HNSW recall for every SELECT in this transaction. With
        # per-source fetching the recall pressure on any single SELECT is
        # already much lower (each query searches a single source's
        # subset, not the whole index) — but a higher ef_search is still
        # cheap insurance against intra-source dense clusters. ``SET
        # LOCAL`` scopes the change to this transaction so it never
        # leaks to other sessions on the pool.
        #
        # NB: Postgres ``SET`` doesn't accept bind parameters, so we
        # inline the integer literal. ``int(...)`` neutralises any
        # injection risk from a pathological config override.
        ef_search = int(settings.HNSW_EF_SEARCH)
        db.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

        # ── Per-source candidate fetching ────────────────────────────
        # One vector SELECT (and, in hybrid mode, one FTS SELECT) per
        # enabled source. Each runs in its own subset of the index, so
        # the result is N independent ranked lists — no source can
        # crowd another out of its candidate slots.
        all_vector_rows: list[Any] = []
        all_keyword_rows: list[Any] = []
        per_source_counts: dict[str, dict[str, int]] = {}

        for source in by_source.keys():
            src_where = filter_to_where_for_source(filter_dict, source)
            if src_where is None:
                # Unknown source key or empty scope list — skip cleanly.
                continue

            v_rows = _run_vector_search(
                db, embedding, src_where, score_threshold, per_source_limit
            )
            all_vector_rows.extend(v_rows)
            counts = per_source_counts.setdefault(source, {"vector": 0, "keyword": 0})
            counts["vector"] = len(v_rows)

            if mode == "hybrid":
                k_rows = _run_keyword_search(
                    db, query, embedding, src_where, per_source_limit
                )
                all_keyword_rows.extend(k_rows)
                counts["keyword"] = len(k_rows)

        # Pool the per-source results into a single global ranking.
        # Each per-source SELECT returned rows already ordered by its
        # local distance / FTS rank, but those orderings are local —
        # we re-sort the pooled lists by the absolute scoring metric
        # (cosine similarity for vector, ts_rank_cd for FTS) so the
        # ranks fed into RRF reflect global standing across sources.
        all_vector_rows.sort(
            key=lambda r: float(r._mapping["similarity"]), reverse=True
        )

        if mode == "hybrid":
            all_keyword_rows.sort(
                key=lambda r: float(r._mapping["rank"]), reverse=True
            )
            fused_rows = _rrf_fuse(
                all_vector_rows, all_keyword_rows, k=int(settings.RRF_K)
            )
            logger.debug(
                "[hybrid per-source] sources={} vector={} keyword={} fused={} per_source={}",
                list(by_source.keys()),
                len(all_vector_rows),
                len(all_keyword_rows),
                len(fused_rows),
                per_source_counts,
            )
        else:
            fused_rows = all_vector_rows
            logger.debug(
                "[vector per-source] sources={} vector={} per_source={}",
                list(by_source.keys()),
                len(all_vector_rows),
                per_source_counts,
            )

    # ── Per-source diversity quota ──────────────────────────────────────
    # ``fused_rows`` is already in score order — vector mode by cosine
    # similarity, hybrid mode by RRF score. Per-source fetching means
    # every selected source has had a fair shot at the candidate pool;
    # the quota's job here is the final TOP_K selection with its
    # MAX_HITS_PER_SOURCE soft cap and overflow backfill.
    rows = _apply_source_quota(fused_rows, top_k, max_per_source)

    hits = _rows_to_chunks(rows)

    logger.info(
        "Retrieved {} chunks (mode={}, threshold {}, top_k {}, sources {})",
        len(hits),
        mode,
        score_threshold,
        top_k,
        list(by_source.keys()),
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
