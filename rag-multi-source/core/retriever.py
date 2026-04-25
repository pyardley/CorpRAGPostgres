"""
Retriever — turns ``(query, filter_dict)`` into a list of :class:`RetrievedChunk`.

Multi-tenancy is **not** enforced here by adding a ``user_id`` clause; that's
the old behaviour. Instead, callers must build ``filter_dict`` via
:func:`core.vector_store.build_query_filter`, which translates the user's
selected sources + accessible-resources rows into a Pinecone metadata filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from app.config import settings
from core.llm import get_embeddings
from core.vector_store import build_query_filter, ensure_index, get_index


@dataclass
class RetrievedChunk:
    """One similarity-search hit returned to the chat layer."""

    resource_id: str
    source: str
    chunk_index: int
    score: float
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

    Parameters
    ----------
    query
        The user question.
    filter_dict
        Pinecone metadata filter — typically built via
        :func:`core.vector_store.build_query_filter`.
    top_k
        Number of chunks to return; defaults to ``settings.TOP_K``.
    score_threshold
        Minimum cosine similarity score; defaults to ``settings.SCORE_THRESHOLD``.
    """
    if not query.strip():
        return []

    top_k = top_k or settings.TOP_K
    score_threshold = (
        settings.SCORE_THRESHOLD if score_threshold is None else score_threshold
    )

    ensure_index()
    index = get_index()
    vector = get_embeddings().embed_query(query)

    logger.debug("Pinecone query top_k={} filter={}", top_k, filter_dict)
    response = index.query(
        vector=vector,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True,
    )

    matches = (
        response.get("matches")
        if isinstance(response, dict)
        else getattr(response, "matches", [])
    )

    hits: list[RetrievedChunk] = []
    for match in matches or []:
        score = match["score"] if isinstance(match, dict) else match.score
        if score is None or score < score_threshold:
            continue
        md = (match["metadata"] if isinstance(match, dict) else match.metadata) or {}
        hits.append(
            RetrievedChunk(
                resource_id=md.get("resource_id", ""),
                source=md.get("source", ""),
                chunk_index=int(md.get("chunk_index", 0)),
                score=float(score),
                text=md.get("text", ""),
                title=md.get("title") or md.get("object_name") or "",
                url=md.get("url", ""),
                metadata=md,
            )
        )

    logger.info(
        "Retrieved {}/{} chunks (threshold {})",
        len(hits),
        len(matches or []),
        score_threshold,
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
