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
  by ``(resource_id, chunk_index)``. Re-running is a clean replace, and now
  short-circuits when an existing row's ``content_fingerprint`` already
  matches the incoming chunk (skips the embed call entirely).
* :func:`compute_content_fingerprint(text)` — SHA-256 over the
  whitespace-normalised, lower-cased chunk body. Stable across re-ingests
  and across sources, so the same paragraph copy-pasted into Confluence
  AND Jira hashes to the same digest.
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

When ``settings.ENABLE_RLS`` is on, every read/write path additionally
binds the active user_id to the ``app.current_user_id`` GUC before
issuing the statement so the database-level Row-Level Security policies
on ``vector_chunks`` / ``user_accessible_resources`` recognise the
caller. See :func:`app.utils.set_current_user_for_rls` for the helper.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from loguru import logger
from sqlalchemy import delete, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.utils import get_db, set_current_user_for_rls
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
    # SHA-256 hex digest of the *normalised* chunk text. Optional on the
    # dataclass so legacy callers continue to work; the upsert path will
    # compute it lazily if missing. Populated by BaseIngestor._chunk()
    # for every new ingest so the dedup short-circuit fires on
    # subsequent runs without recomputing.
    content_fingerprint: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Fingerprint helper
# ──────────────────────────────────────────────────────────────────────────────

# Whitespace runs collapse to a single space; combined with lower-casing
# this means "Hello   World\n\n" and "hello world" hash to the same
# digest — the dedup is robust to formatting churn (trailing newlines,
# CRLF↔LF, double-spacing) that doesn't change semantics.
_WHITESPACE_RE = re.compile(r"\s+")


def _normalise_for_fingerprint(text: str) -> str:
    """Stable canonical form for hashing — lower-cased, whitespace-collapsed."""
    if not text:
        return ""
    # NUL bytes are stripped at the row level (see _strip_nul) but
    # filter here too so they can't poison the hash space.
    cleaned = text.replace("\x00", "").strip().lower()
    return _WHITESPACE_RE.sub(" ", cleaned)


def compute_content_fingerprint(text: str) -> str:
    """
    SHA-256 hex digest of the normalised chunk body.

    OB1-inspired: lets the upsert path skip re-embedding rows whose
    body hasn't changed (saves the embedding API hit, which dominates
    the cost of an incremental re-ingest) and lets us notice when the
    same content has been ingested twice through different sources
    (e.g. a README that lives both in Git and in a Confluence page).

    The 64-character hex digest fits exactly in the CHAR(64) column on
    ``vector_chunks.content_fingerprint``. Returns ``""`` for empty
    input — callers should treat empty fingerprints as "skip the
    short-circuit" rather than "match every empty row".
    """
    norm = _normalise_for_fingerprint(text)
    if not norm:
        return ""
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────

# Columns we promote out of `chunk.metadata` onto their own DB column for
# fast indexed filtering. The ingestors put the right value into metadata
# under the matching key.
_PROMOTED_FIELDS = (
    "project_key",
    "space_key",
    "db_name",
    "git_scope",
    "email_provider",
)

# Per-source filter column. Used by both `build_query_filter` and the
# retriever to know which column to constrain.
_FILTER_COLUMNS_BY_SOURCE = {
    "jira": "project_key",
    "confluence": "space_key",
    "sql": "db_name",
    "git": "git_scope",
    "email": "email_provider",
}


def _strip_nul(value: Any) -> Any:
    """
    Recursively remove NUL (``\\x00``) bytes from string values.

    PostgreSQL's TEXT/VARCHAR types reject NUL bytes outright
    (``psycopg.DataError: PostgreSQL text fields cannot contain NUL
    (0x00) bytes``) — they have no semantic meaning in a text column
    and can't be escaped. Marketing emails, Outlook-generated MSO
    conditional comments, and any payload that's been transcoded
    UTF-16 → UTF-8 with errors='replace' frequently contain stray NULs
    inside otherwise-valid text/HTML/CSS, which then bombs the entire
    INSERT batch (32 chunks at a time) on the first offending row.

    Strip rather than reject: the user's intent is "store this email",
    not "fail because of an invisible artefact in a tracking-pixel
    style block". JSONB columns *do* allow ``\\u0000`` so we still
    sanitise the metadata recursively to keep behaviour consistent and
    avoid future schema-level surprises.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, list):
        return [_strip_nul(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_nul(v) for v in value)
    if isinstance(value, dict):
        return {k: _strip_nul(v) for k, v in value.items()}
    return value


def _row_payload(chunk: ResourceChunk, embedding: list[float]) -> dict[str, Any]:
    md = _strip_nul(dict(chunk.metadata))

    promoted = {key: md.pop(key, None) for key in _PROMOTED_FIELDS}

    # Lazy fingerprint computation — the BaseIngestor pre-fills this
    # but legacy / direct callers may pass a ResourceChunk without one.
    fingerprint = chunk.content_fingerprint
    if not fingerprint:
        fingerprint = compute_content_fingerprint(chunk.text)

    return {
        "resource_id": _strip_nul(chunk.resource_id),
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "text": _strip_nul(chunk.text),
        "embedding": embedding,
        "title": _strip_nul(md.pop("title", None)),
        "url": _strip_nul(md.pop("url", None)),
        "object_name": _strip_nul(md.pop("object_name", None)),
        "last_updated": _strip_nul(md.pop("last_updated", None)),
        "extra": md,
        # Empty-string fingerprint -> store NULL so the partial index
        # excludes the row. We only ever hash non-empty bodies; the
        # ingestor already drops empty chunks earlier in the pipeline.
        "content_fingerprint": fingerprint or None,
        **promoted,
    }


def _batched(seq: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ──────────────────────────────────────────────────────────────────────────────
# Existence / lookup
# ──────────────────────────────────────────────────────────────────────────────

def resource_exists(
    resource_id: str, *, user_id: Optional[str] = None
) -> bool:
    """True if at least one chunk for this resource is already stored.

    When RLS is enabled, the lookup honours the per-user policy — pass
    ``user_id`` for an authoritative answer in tenancy-sensitive code
    paths. Admin tooling can omit it to mean "any chunk visible to a
    BYPASSRLS role".
    """
    with get_db() as db:
        set_current_user_for_rls(db, user_id)
        return db.execute(
            select(VectorChunk.id)
            .where(VectorChunk.resource_id == resource_id)
            .limit(1)
        ).first() is not None


# ──────────────────────────────────────────────────────────────────────────────
# Upsert
# ──────────────────────────────────────────────────────────────────────────────

def _upsert_stmt(rows: list[dict[str, Any]]):
    """Build the ``INSERT … ON CONFLICT DO UPDATE`` statement once."""
    stmt = pg_insert(VectorChunk).values(rows)
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VectorChunk.__table__.columns
        if col.name not in {"id", "created_at"}
    }
    return stmt.on_conflict_do_update(
        constraint="uq_vector_chunks_resource_chunk",
        set_=update_cols,
    )


def _existing_fingerprints(
    db, candidates: list[tuple[str, int]]
) -> dict[tuple[str, int], Optional[str]]:
    """
    For the given (resource_id, chunk_index) pairs, return the currently
    stored ``content_fingerprint`` keyed by the same tuple. Missing rows
    map to ``None`` (they need to be inserted regardless of dedup).
    Used to short-circuit re-embeds when the body hasn't changed.
    """
    if not candidates:
        return {}

    # Group by resource_id so we can issue one IN-clause per resource —
    # the (resource_id, chunk_index) unique index makes this very fast.
    by_resource: dict[str, list[int]] = {}
    for rid, idx in candidates:
        by_resource.setdefault(rid, []).append(idx)

    found: dict[tuple[str, int], Optional[str]] = {}
    for rid, indices in by_resource.items():
        rows = db.execute(
            select(
                VectorChunk.resource_id,
                VectorChunk.chunk_index,
                VectorChunk.content_fingerprint,
            ).where(
                VectorChunk.resource_id == rid,
                VectorChunk.chunk_index.in_(indices),
            )
        ).all()
        for r in rows:
            found[(r[0], int(r[1]))] = r[2]
    return found


def upsert_chunks(
    chunks: list[ResourceChunk],
    *,
    user_id: Optional[str] = None,
) -> int:
    """
    Embed and ``INSERT … ON CONFLICT DO UPDATE`` the chunks. Returns the
    number of rows actually written.

    Idempotent — calling with the same chunks twice is a no-op (each row's
    ``updated_at`` advances). The conflict target is
    ``(resource_id, chunk_index)``, so re-ingesting a resource overwrites
    its chunks in place.

    Content-fingerprint dedup (OB1-inspired)
    ----------------------------------------
    Before embedding any batch we look up the currently-stored
    ``content_fingerprint`` for each ``(resource_id, chunk_index)`` and
    drop any incoming chunk whose hash matches what's already there.
    This is the hot path during incremental re-ingests of unchanged
    documents and saves the OpenAI embedding API call entirely. The
    full check is per-row equality on a SHA-256 hex digest; a
    collision would require an attacker who can already write into
    ``vector_chunks``, at which point the dedup is the least of your
    worries.

    Robustness: each batch is wrapped in a SAVEPOINT (``begin_nested``)
    so a single bad row (e.g. a previously-undetected control-character
    or Postgres-incompatible payload) doesn't poison the whole outer
    transaction. On batch failure we rollback the savepoint, log the
    error, and retry the rows **one at a time** with their own
    savepoints — successful rows still land, failures are logged with
    their ``resource_id`` + ``chunk_index`` so the next "weird payload"
    is one ``grep`` away from being identifiable.

    Parameters
    ----------
    chunks
        The chunks to insert / update. Pre-populated fingerprints on
        each ``ResourceChunk`` are honoured; missing ones are computed
        on the fly.
    user_id
        Optional user_id bound to the session for RLS. Required when
        ``settings.ENABLE_RLS`` is True — the policies will reject
        writes from a session with no GUC set.
    """
    if not chunks:
        return 0

    embeddings = get_embeddings()
    total = 0
    skipped = 0
    deduped = 0
    with get_db() as db:
        # RLS: bind user_id to the session before any read or write.
        # No-op when ENABLE_RLS is False (see app.utils).
        set_current_user_for_rls(db, user_id)

        for batch in _batched(chunks, settings.VECTOR_UPSERT_BATCH_SIZE):
            # ── Pre-compute fingerprints for every chunk in the batch
            #    so we can short-circuit identical rows before embedding.
            for c in batch:
                if not c.content_fingerprint:
                    c.content_fingerprint = compute_content_fingerprint(c.text)

            # ── Dedup: ask the DB what fingerprints we already have
            #    for these (resource_id, chunk_index) pairs. Skip any
            #    chunk whose stored fingerprint matches — body
            #    unchanged means we don't need to spend an embedding
            #    API call OR a DB write on it.
            to_process: list[ResourceChunk] = []
            if settings.ENABLE_CONTENT_FINGERPRINT_DEDUP:
                existing = _existing_fingerprints(
                    db, [(c.resource_id, c.chunk_index) for c in batch]
                )
                for c in batch:
                    stored = existing.get((c.resource_id, c.chunk_index))
                    if stored and c.content_fingerprint and stored == c.content_fingerprint:
                        deduped += 1
                        continue
                    to_process.append(c)
            else:
                to_process = list(batch)

            if not to_process:
                continue

            vectors = embeddings.embed_documents([c.text for c in to_process])
            rows = [
                _row_payload(chunk, vec)
                for chunk, vec in zip(to_process, vectors)
            ]

            # Fast path: single bulk INSERT inside a savepoint.
            try:
                with db.begin_nested():
                    db.execute(_upsert_stmt(rows))
                total += len(rows)
                continue
            except SQLAlchemyError as exc:
                logger.warning(
                    "Bulk upsert of {} rows failed ({}); falling back "
                    "to per-row inserts to isolate the bad payload.",
                    len(rows),
                    exc.__class__.__name__,
                )

            # Slow path: one row per savepoint so one rotten payload
            # only kills itself.
            for row in rows:
                try:
                    with db.begin_nested():
                        db.execute(_upsert_stmt([row]))
                    total += 1
                except SQLAlchemyError as exc:
                    skipped += 1
                    text_preview = (row.get("text") or "")[:160].replace(
                        "\n", " "
                    )
                    logger.error(
                        "[upsert_chunks] DROP row resource_id={} "
                        "chunk_index={} source={} reason={}: {} | "
                        "text[0:160]={!r}",
                        row.get("resource_id"),
                        row.get("chunk_index"),
                        row.get("source"),
                        exc.__class__.__name__,
                        str(exc.orig)[:300] if hasattr(exc, "orig") else str(exc)[:300],
                        text_preview,
                    )

    if skipped:
        logger.warning(
            "Upserted {} chunks to vector_chunks ({} deduped by fingerprint, "
            "{} dropped due to DB-level rejection — see ERROR lines above).",
            total,
            deduped,
            skipped,
        )
    else:
        logger.info(
            "Upserted {} chunks to vector_chunks ({} deduped by fingerprint).",
            total,
            deduped,
        )
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Delete
# ──────────────────────────────────────────────────────────────────────────────

def delete_resource(
    resource_id: str, *, user_id: Optional[str] = None
) -> int:
    """Delete every chunk for a single resource. Returns count deleted.

    Honours RLS when ``user_id`` is supplied — required for the
    ingestor's per-resource pre-delete in incremental mode.
    """
    with get_db() as db:
        set_current_user_for_rls(db, user_id)
        result = db.execute(
            delete(VectorChunk).where(VectorChunk.resource_id == resource_id)
        )
        n = int(result.rowcount or 0)
    if n:
        logger.info("Deleted {} chunks for resource {}", n, resource_id)
    return n


def delete_by_filter(
    filter_dict: dict[str, Any], *, user_id: Optional[str] = None
) -> int:
    """
    Bulk-delete chunks that match a simple filter dict.

    Supported keys:

      * ``source`` — string
      * ``project_key`` / ``space_key`` / ``db_name`` / ``git_scope`` /
        ``email_provider`` — a string, a list, or ``{"$in": [...]}``

    Used by ``--mode full`` to wipe a scope before re-ingesting it.

    When RLS is enabled, ``user_id`` must be supplied for the policy
    check; otherwise the DELETE will silently affect zero rows.
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
        set_current_user_for_rls(db, user_id)
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
    accessible_email_providers: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Translate (selected sources × accessible-resources rows) into a structured
    filter dict consumed by :func:`core.retriever.retrieve`.

    Per-source clauses are OR'd together so one query can span Jira +
    Confluence + SQL + Git + Email.

    Example output::

        {
            "by_source": {
                "jira":       ["PROJ", "OPS"],
                "confluence": ["DOCS"],
                "git":        ["acme/widgets@main"],
                "email":      ["outlook", "gmail"],
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
    if "email" in selected_sources and accessible_email_providers:
        by_source["email"] = list(accessible_email_providers)

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
