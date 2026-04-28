"""
Vector store — PostgreSQL + pgvector.

The vector index lives in the same Postgres database as ``users``,
``user_credentials``, and ``user_accessible_resources``. Retrieval is a single
SQL statement that combines cosine-similarity ranking with the user's
accessible resources — no second hop to a managed vector DB, and no risk of
drift between the access table and the index.

Public surface
--------------
* :class:`ResourceChunk` — what each ingestor produces (one chunk of one
  resource, ready to embed + insert).
* :func:`upsert_chunks(chunks)` — embed + ``INSERT … ON CONFLICT DO UPDATE``
  by ``(resource_id, chunk_index)``. Re-running is a clean replace.
* :func:`delete_resource(resource_id)` — wipe one resource cleanly.
* :func:`delete_by_filter(filter_dict)` — bulk wipe (used by ``--mode full``).
* :func:`build_query_filter(...)` — helper for the chat layer; returns a dict
  consumed by :func:`core.retriever.retrieve`.

Multi-tenancy
-------------
Vectors are shared org-wide. Tenancy is enforced **at query time** by the
``WHERE`` clause built from ``build_query_filter`` + the user's
``user_accessible_resources`` rows. There is no ``user_id`` column on
``vector_chunks`` and no per-user duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from loguru import logger
from sqlalchemy import delete, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.utils import get_db
from core.llm import get_embeddings
from models.vector_chunk import VectorChunk


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceChunk:
    """One chunk of one resource, ready to upsert."""

    resource_id: str          # e.g. "jira:PROJ-123"
    source: str               # "jira" | "confluence" | "sql" | "git"
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────

# Columns we promote out of `chunk.metadata` onto their own DB column for
# fast indexed filtering. The ingestors put the right value into metadata
# under the matching key.
_PROMOTED_FIELDS = ("project_key", "space_key", "db_name", "git_scope")

# Per-source filter column. Used by both `build_query_filter` and the
# retriever to know which column to constrain.
_FILTER_COLUMNS_BY_SOURCE = {
    "jira": "project_key",
    "confluence": "space_key",
    "sql": "db_name",
    "git": "git_scope",
}


def _row_payload(chunk: ResourceChunk, embedding: list[float]) -> dict[str, Any]:
    md = dict(chunk.metadata)

    promoted = {key: md.pop(key, None) for key in _PROMOTED_FIELDS}

    return {
        "resource_id": chunk.resource_id,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "embedding": embedding,
        "title": md.pop("title", None),
        "url": md.pop("url", None),
        "object_name": md.pop("object_name", None),
        "last_updated": md.pop("last_updated", None),
        "extra": md,
        **promoted,
    }


def _batched(seq: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ──────────────────────────────────────────────────────────────────────────────
# Existence / lookup
# ──────────────────────────────────────────────────────────────────────────────

def resource_exists(resource_id: str) -> bool:
    """True if at least one chunk for this resource is already stored."""
    with get_db() as db:
        return db.execute(
            select(VectorChunk.id)
            .where(VectorChunk.resource_id == resource_id)
            .limit(1)
        ).first() is not None


# ──────────────────────────────────────────────────────────────────────────────
# Upsert
# ──────────────────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[ResourceChunk]) -> int:
    """
    Embed and ``INSERT … ON CONFLICT DO UPDATE`` the chunks. Returns the
    number of rows written.

    Idempotent — calling with the same chunks twice is a no-op (each row's
    ``updated_at`` advances). The conflict target is
    ``(resource_id, chunk_index)``, so re-ingesting a resource overwrites
    its chunks in place.
    """
    if not chunks:
        return 0

    embeddings = get_embeddings()
    total = 0
    with get_db() as db:
        for batch in _batched(chunks, settings.VECTOR_UPSERT_BATCH_SIZE):
            vectors = embeddings.embed_documents([c.text for c in batch])
            rows = [
                _row_payload(chunk, vec) for chunk, vec in zip(batch, vectors)
            ]
            stmt = pg_insert(VectorChunk).values(rows)
            update_cols = {
                col.name: stmt.excluded[col.name]
                for col in VectorChunk.__table__.columns
                if col.name not in {"id", "created_at"}
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_vector_chunks_resource_chunk",
                set_=update_cols,
            )
            db.execute(stmt)
            total += len(rows)

    logger.info("Upserted {} chunks to vector_chunks.", total)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Delete
# ──────────────────────────────────────────────────────────────────────────────

def delete_resource(resource_id: str) -> int:
    """Delete every chunk for a single resource. Returns count deleted."""
    with get_db() as db:
        result = db.execute(
            delete(VectorChunk).where(VectorChunk.resource_id == resource_id)
        )
        n = int(result.rowcount or 0)
    if n:
        logger.info("Deleted {} chunks for resource {}", n, resource_id)
    return n


def delete_by_filter(filter_dict: dict[str, Any]) -> int:
    """
    Bulk-delete chunks that match a simple filter dict.

    Supported keys:

      * ``source`` — string
      * ``project_key`` / ``space_key`` / ``db_name`` / ``git_scope`` —
        a string, a list, or ``{"$in": [...]}``

    Used by ``--mode full`` to wipe a scope before re-ingesting it.
    """
    stmt = delete(VectorChunk)
    clauses = []
    for key, value in filter_dict.items():
        column = getattr(VectorChunk, key, None)
        if column is None:
            continue
        if isinstance(value, dict) and "$in" in value:
            clauses.append(column.in_(list(value["$in"])))
        elif isinstance(value, (list, tuple, set)):
            clauses.append(column.in_(list(value)))
        else:
            clauses.append(column == value)
    if clauses:
        stmt = stmt.where(*clauses)

    with get_db() as db:
        result = db.execute(stmt)
        n = int(result.rowcount or 0)
    logger.info("Deleted {} chunks matching {}", n, filter_dict)
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────

def index_stats() -> dict[str, Any]:
    """Return per-source row counts. Cheap — `GROUP BY source`."""
    with get_db() as db:
        rows = db.execute(
            text(
                "SELECT source, COUNT(*) AS n "
                "FROM vector_chunks GROUP BY source ORDER BY source"
            )
        ).all()
    counts: dict[str, Any] = {r[0]: int(r[1]) for r in rows}
    counts["__total__"] = sum(int(v) for v in counts.values())
    return counts


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
    Translate (selected sources × accessible-resources rows) into a structured
    filter dict consumed by :func:`core.retriever.retrieve`.

    Per-source clauses are OR'd together so one query can span Jira +
    Confluence + SQL + Git.

    Example output::

        {
            "by_source": {
                "jira":       ["PROJ", "OPS"],
                "confluence": ["DOCS"],
                "git":        ["acme/widgets@main"],
            }
        }
    """
    by_source: dict[str, list[str]] = {}

    if "jira" in selected_sources and accessible_jira_projects:
        by_source["jira"] = list(accessible_jira_projects)
    if "confluence" in selected_sources and accessible_confluence_spaces:
        by_source["confluence"] = list(accessible_confluence_spaces)
    if "sql" in selected_sources and accessible_databases:
        by_source["sql"] = list(accessible_databases)
    if "git" in selected_sources and accessible_git_scopes:
        by_source["git"] = list(accessible_git_scopes)

    return {"by_source": by_source}


def filter_to_where(filter_dict: dict[str, Any]):
    """
    Turn a :func:`build_query_filter` result into a SQLAlchemy WHERE
    expression. Returns ``None`` when the filter would match nothing
    (caller should short-circuit with an empty result).
    """
    by_source: dict[str, list[str]] = (filter_dict or {}).get("by_source") or {}
    if not by_source:
        return None

    clauses = []
    for source, scope_values in by_source.items():
        col_name = _FILTER_COLUMNS_BY_SOURCE.get(source)
        if not col_name or not scope_values:
            continue
        column = getattr(VectorChunk, col_name)
        clauses.append(
            (VectorChunk.source == source) & column.in_(scope_values)
        )

    if not clauses:
        return None
    return or_(*clauses) if len(clauses) > 1 else clauses[0]
