"""
Retriever — turns ``(query, filter_dict)`` into a list of :class:`RetrievedChunk`.

Multi-tenancy is enforced here by translating ``filter_dict`` (built via
:func:`core.vector_store.build_query_filter`) into a SQL ``WHERE`` clause that
constrains the search to the user's accessible resources. There is **no
``user_id`` column** on the vector table — the filter is the security
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from sqlalchemy import select

from app.config import settings
from app.utils import get_db
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


def retrieve(
    query: str,
    filter_dict: dict[str, Any],
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Run a similarity search constrained by ``filter_dict``.

    The SQL we emit is roughly::

        SELECT
            id, resource_id, source, chunk_index, text,
            title, url, project_key, space_key, db_name, git_scope,
            object_name, last_updated, metadata,
            1 - (embedding <=> :query_vec) AS similarity
        FROM vector_chunks
        WHERE <filter clauses derived from filter_dict>
          AND 1 - (embedding <=> :query_vec) >= :threshold
        ORDER BY embedding <=> :query_vec
        LIMIT :top_k

    `<=>` is pgvector's cosine-distance operator (1 - similarity); ordering
    by it directly uses the HNSW index for sub-millisecond top-K on tens of
    millions of rows.
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

    # `1 - (embedding <=> :vec)` -> cosine similarity. We can't bind a Vector
    # parameter through SQLAlchemy 2.x with a generic operator, so we lean on
    # pgvector's SQLAlchemy adapter via the column expression.
    similarity_expr = (
        1 - VectorChunk.embedding.cosine_distance(embedding)
    ).label("similarity")
    distance_expr = VectorChunk.embedding.cosine_distance(embedding)

    stmt = (
        select(
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
            VectorChunk.object_name,
            VectorChunk.last_updated,
            VectorChunk.extra,
            similarity_expr,
        )
        .where(where_clause)
        .where(similarity_expr >= score_threshold)
        .order_by(distance_expr)
        .limit(top_k)
    )

    with get_db() as db:
        rows = db.execute(stmt).all()

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

    logger.info(
        "Retrieved {} chunks (threshold {}, top_k {})",
        len(hits),
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
