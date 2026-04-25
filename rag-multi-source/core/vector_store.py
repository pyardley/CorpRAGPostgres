"""
Pinecone vector store — single shared org-wide index.

KEY DESIGN POINTS
-----------------
1. **One index** per organisation (`settings.PINECONE_INDEX_NAME`). The
   previous architecture stored a separate copy of every chunk per user.
   This module is the only place that talks to Pinecone, which makes the
   "store-once" promise enforceable.

2. **Stable, deterministic IDs.** Each ingested resource is identified by a
   `resource_id` like `jira:PROJ-123`, `confluence:page-987654321`, or
   `sql:server1.dbname.proc_GetCustomerData`. A resource that produces N
   chunks gets vector IDs of the form ``f"{resource_id}::chunk-{i}"`` so that:
     * upserting the same resource twice replaces the previous chunks,
     * we can wipe a resource cleanly by deleting the IDs we generated.

3. **No `user_id` metadata.** Multi-tenancy is enforced at *query* time by
   passing a Pinecone metadata filter built from the user's accessible
   resources (see :func:`build_query_filter` and `core.retriever`).

4. **Required metadata fields per chunk** (lower-cased keys, JSON-safe):
     resource_id, source, project_key | space_key | db_name,
     object_name, title, url, last_updated, chunk_index, text
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Optional

from loguru import logger

from app.config import settings
from core.llm import get_embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceChunk:
    """One chunk of one resource, ready to upsert."""

    resource_id: str          # e.g. "jira:PROJ-123"
    source: str               # "jira" | "confluence" | "sql"
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vector_id(self) -> str:
        # `::` is reserved (never appears in any of our resource_ids).
        return f"{self.resource_id}::chunk-{self.chunk_index}"


# ──────────────────────────────────────────────────────────────────────────────
# Pinecone client
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _pinecone_client():
    """Lazy-import the Pinecone SDK and ensure the shared index exists."""
    from pinecone import Pinecone, ServerlessSpec

    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set.")

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    name = settings.PINECONE_INDEX_NAME

    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        logger.info("Creating Pinecone index {!r}", name)
        pc.create_index(
            name=name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION
            ),
        )
        # Wait until the index is ready before we try to use it.
        while not pc.describe_index(name).status.get("ready", False):
            time.sleep(1)

    return pc


def ensure_index() -> str:
    """Make sure the shared index exists and return its name."""
    _pinecone_client()  # side-effect: create-if-missing
    return settings.PINECONE_INDEX_NAME


def get_index():
    """Return a handle to the shared Pinecone index."""
    return _pinecone_client().Index(settings.PINECONE_INDEX_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Existence / lookup
# ──────────────────────────────────────────────────────────────────────────────

def list_chunk_ids_for_resource(resource_id: str) -> list[str]:
    """
    Return the existing vector IDs that belong to ``resource_id``.

    Pinecone serverless does not offer a cheap "list IDs by metadata filter",
    so we scope by ID prefix — every chunk we upsert is keyed
    ``{resource_id}::chunk-{n}``, which makes the prefix exact.
    """
    index = get_index()
    ids: list[str] = []
    try:
        for page in index.list(prefix=f"{resource_id}::"):
            ids.extend(page)
    except AttributeError:
        page = index.list(prefix=f"{resource_id}::")
        if page:
            ids.extend(page)
    except Exception as exc:  # pragma: no cover - SDK shape can vary
        logger.warning("list() failed for prefix {}: {}", resource_id, exc)
    return ids


def resource_exists(resource_id: str) -> bool:
    """True if at least one chunk for this resource is already in Pinecone."""
    return bool(list_chunk_ids_for_resource(resource_id))


# ──────────────────────────────────────────────────────────────────────────────
# Upsert
# ──────────────────────────────────────────────────────────────────────────────

def _batched(seq: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def upsert_chunks(chunks: list[ResourceChunk]) -> int:
    """
    Embed and upsert chunks. Returns the number of vectors written.

    For idempotency the caller is expected to either:
      * always pass *every* chunk for a resource (so re-upserts overwrite), OR
      * call :func:`delete_resource` first.
    """
    if not chunks:
        return 0

    embeddings = get_embeddings()
    index = get_index()

    total = 0
    for batch in _batched(chunks, settings.PINECONE_BATCH_SIZE):
        texts = [c.text for c in batch]
        vectors = embeddings.embed_documents(texts)

        payload = []
        for chunk, vector in zip(batch, vectors):
            md = dict(chunk.metadata)
            # Always overwrite the canonical metadata fields so callers can't
            # accidentally drop them.
            md["resource_id"] = chunk.resource_id
            md["source"] = chunk.source
            md["chunk_index"] = chunk.chunk_index
            # Keep chunk text in metadata so retrieval can hand it to the LLM
            # without a second round-trip.
            md.setdefault("text", chunk.text)
            payload.append(
                {"id": chunk.vector_id, "values": vector, "metadata": md}
            )

        index.upsert(vectors=payload)
        total += len(payload)

    logger.info("Upserted {} chunks to {}", total, settings.PINECONE_INDEX_NAME)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Delete
# ──────────────────────────────────────────────────────────────────────────────

def delete_resource(resource_id: str) -> int:
    """Delete every chunk for a single resource. Returns count deleted."""
    index = get_index()
    ids = list_chunk_ids_for_resource(resource_id)
    if not ids:
        return 0
    for batch in _batched(ids, settings.PINECONE_BATCH_SIZE):
        index.delete(ids=batch)
    logger.info("Deleted {} chunks for resource {}", len(ids), resource_id)
    return len(ids)


def delete_by_filter(filter_dict: dict[str, Any]) -> None:
    """
    Bulk delete using Pinecone's metadata-filter delete.

    Used for ``--mode full`` ingestion: e.g. wipe all vectors for
    ``{"source": "jira", "project_key": {"$in": ["PROJ"]}}`` before the rebuild.
    A 404 / "not found" is treated as a no-op (the scope was empty already).
    """
    index = get_index()
    logger.info("Deleting vectors by filter: {}", filter_dict)
    try:
        index.delete(filter=filter_dict)
    except Exception as exc:
        msg = str(exc).lower()
        if "404" in msg or "not found" in msg:
            logger.info("Nothing to delete for filter {} (empty scope).", filter_dict)
        else:
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Stats (used by the sidebar)
# ──────────────────────────────────────────────────────────────────────────────

def index_stats() -> dict[str, Any]:
    try:
        return get_index().describe_index_stats()
    except Exception as exc:
        logger.warning("Could not fetch Pinecone stats: {}", exc)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Filter builder — the workhorse for query-time tenancy
# ──────────────────────────────────────────────────────────────────────────────

def build_query_filter(
    selected_sources: list[str],
    accessible_jira_projects: Optional[list[str]] = None,
    accessible_confluence_spaces: Optional[list[str]] = None,
    accessible_databases: Optional[list[str]] = None,
    accessible_git_scopes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Translate (selected sources × accessible-resources rows) into a Pinecone
    metadata filter.

    Rules
    -----
    * ``source`` must be in ``selected_sources``.
    * Within each source, the per-source key
      (``project_key`` / ``space_key`` / ``db_name`` / ``git_scope``) must be in
      the matching accessible-list — UNLESS that list is empty, in which case
      that source is excluded.
    * The per-source clauses are OR'd together so one query can span
      Jira + Confluence + SQL + Git.
    """
    selected_sources = [s for s in selected_sources if s]
    if not selected_sources:
        # Pinecone rejects empty filters; this clause never matches anything,
        # which is the correct behaviour when the user has selected no sources.
        return {"source": {"$in": ["__none__"]}}

    or_clauses: list[dict[str, Any]] = []

    if "jira" in selected_sources and accessible_jira_projects:
        or_clauses.append(
            {
                "source": "jira",
                "project_key": {"$in": list(accessible_jira_projects)},
            }
        )
    if "confluence" in selected_sources and accessible_confluence_spaces:
        or_clauses.append(
            {
                "source": "confluence",
                "space_key": {"$in": list(accessible_confluence_spaces)},
            }
        )
    if "sql" in selected_sources and accessible_databases:
        or_clauses.append(
            {
                "source": "sql",
                "db_name": {"$in": list(accessible_databases)},
            }
        )
    if "git" in selected_sources and accessible_git_scopes:
        or_clauses.append(
            {
                "source": "git",
                "git_scope": {"$in": list(accessible_git_scopes)},
            }
        )

    if not or_clauses:
        return {"source": {"$in": ["__none__"]}}
    if len(or_clauses) == 1:
        return or_clauses[0]
    return {"$or": or_clauses}
