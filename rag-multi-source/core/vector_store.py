"""Pinecone serverless vector store: index lifecycle and upsert helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from loguru import logger
from pinecone import Pinecone, ServerlessSpec

from app.config import settings


def _pinecone_client() -> Pinecone:
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set. Check your .env file.")
    return Pinecone(api_key=settings.PINECONE_API_KEY)


def ensure_index(embedding_dim: int | None = None) -> str:
    """Create the Pinecone index if it does not exist. Returns index name."""
    dim = embedding_dim or settings.embedding_dim
    pc = _pinecone_client()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.PINECONE_INDEX_NAME not in existing:
        logger.info(
            "Creating Pinecone serverless index '{}' dim={}", settings.PINECONE_INDEX_NAME, dim
        )
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )
        logger.info("Index created.")
    else:
        logger.debug("Pinecone index '{}' already exists.", settings.PINECONE_INDEX_NAME)
    return settings.PINECONE_INDEX_NAME


def get_vector_store(embeddings: Embeddings) -> PineconeVectorStore:
    """Return a LangChain PineconeVectorStore backed by our shared index."""
    ensure_index()
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY,
    )


def delete_vectors(filter_dict: dict[str, Any]) -> None:
    """
    Delete all vectors matching *filter_dict* metadata.
    A 404 "Namespace not found" is silently ignored — it just means the index
    is empty for that scope, which is fine on the first full load.
    """
    pc = _pinecone_client()
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    try:
        index.delete(filter=filter_dict)
        logger.info("Deleted vectors matching filter: {}", filter_dict)
    except Exception as exc:
        # 404 = namespace/scope doesn't exist yet → nothing to delete
        if "404" in str(exc) or "Namespace not found" in str(exc) or "not found" in str(exc).lower():
            logger.info("No existing vectors to delete for filter: {} (index is empty for this scope)", filter_dict)
        else:
            raise
