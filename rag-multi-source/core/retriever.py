"""Build a Pinecone retriever with user-scoped metadata filters."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.documents import Document
from loguru import logger

from core.llm import get_embeddings
from core.vector_store import get_vector_store


def build_filter(
    user_id: str,
    sources: Optional[list[str]] = None,
    project_keys: Optional[list[str]] = None,
    space_keys: Optional[list[str]] = None,
    db_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Construct a Pinecone metadata filter dict.

    *user_id* is always required (multi-tenancy).
    All other params narrow the scope further.
    Omitting a param means "all" for that dimension.
    """
    filt: dict[str, Any] = {"user_id": {"$eq": user_id}}

    if sources:
        filt["source"] = {"$in": sources}

    sub_conditions: list[dict] = []

    if project_keys:
        sub_conditions.append({"project_key": {"$in": project_keys}})
    if space_keys:
        sub_conditions.append({"space_key": {"$in": space_keys}})
    if db_names:
        sub_conditions.append({"db_name": {"$in": db_names}})

    if sub_conditions:
        if len(sub_conditions) == 1:
            filt.update(sub_conditions[0])
        else:
            filt["$or"] = sub_conditions

    return filt


def retrieve(
    query: str,
    user_id: str,
    sources: Optional[list[str]] = None,
    project_keys: Optional[list[str]] = None,
    space_keys: Optional[list[str]] = None,
    db_names: Optional[list[str]] = None,
    top_k: int = 8,
) -> list[Document]:
    """
    Retrieve the top-k most relevant documents for *query*,
    filtered to this user's selected sources and scopes.
    """
    from app.config import settings

    embeddings = get_embeddings()
    vs = get_vector_store(embeddings)

    filt = build_filter(
        user_id=user_id,
        sources=sources,
        project_keys=project_keys,
        space_keys=space_keys,
        db_names=db_names,
    )

    logger.debug("Retrieval filter: {}", filt)

    docs_with_scores = vs.similarity_search_with_score(
        query=query, k=top_k, filter=filt
    )

    # Filter out low-confidence results
    threshold = settings.SCORE_THRESHOLD
    filtered = [
        doc for doc, score in docs_with_scores if score >= threshold
    ]

    logger.info(
        "Retrieved {}/{} docs above threshold={} for query={!r}",
        len(filtered),
        len(docs_with_scores),
        threshold,
        query[:80],
    )
    return filtered
