"""
Base ingestor — the contract every source-specific ingestor must satisfy.

There is one shared ``vector_chunks`` table in Postgres for the whole
organisation. The base class is the only place that talks to the vector
store during ingestion, which makes the "store-once + dedup" guarantee
enforceable.

Subclass responsibilities
-------------------------
1. Set the ``source`` class attribute (e.g. ``"jira"``).
2. Implement :meth:`fetch_resources` — yield :class:`SourceResource` objects.
3. Implement :meth:`scope_filter` — filter dict consumed by
   ``core.vector_store.delete_by_filter`` to wipe the user's selected scope
   when ``mode == "full"``.
4. Implement :meth:`resource_identifier_for` — the value to store in
   ``user_accessible_resources.resource_identifier`` (the project_key /
   space_key / db_name / git_scope).

The base class handles:
  * chunking with the shared text splitter,
  * deterministic chunk identity via ``(resource_id, chunk_index)`` ON
    CONFLICT DO UPDATE so re-ingest is a clean overwrite,
  * full-mode wipe-then-rebuild via metadata filter (no per-user data),
  * ``IngestionLog`` rows for full / incremental baselines,
  * ``grant_access()`` into ``user_accessible_resources``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sqlalchemy.orm import Session


# Type alias for the optional progress callback. Receives a snapshot dict like
# {"source": "...", "items_processed": 12, "vectors_upserted": 87,
#  "current_item": "PROJ-123 — fix login"}. The UI can render whatever it
# wants from this; the CLI uses loguru and ignores the callback.
ProgressCallback = Callable[[dict[str, Any]], None]

from app.config import settings
from app.utils import grant_access
from core.vector_store import (
    ResourceChunk,
    compute_content_fingerprint,
    delete_by_filter,
    delete_resource,
    upsert_chunks,
)
from models.ingestion_log import IngestionLog


# ──────────────────────────────────────────────────────────────────────────────
# Shared dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceResource:
    """A logical document fetched from a source, before chunking."""

    resource_id: str
    title: str
    text: str
    url: str
    last_updated: str  # ISO-8601 string
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    items_processed: int = 0
    vectors_upserted: int = 0
    skipped_unchanged: int = 0
    last_item_updated_at: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Text splitter (shared singleton)
# ──────────────────────────────────────────────────────────────────────────────

_splitter: Optional[RecursiveCharacterTextSplitter] = None


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _splitter


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseIngestor(ABC):
    """Subclass and implement the abstract methods to add a new source."""

    source: str = ""  # MUST be overridden by subclasses

    def __init__(
        self,
        db: Session,
        user_id: str,
        credentials: dict[str, str],
        scope: str,
        mode: str = "incremental",
    ):
        if mode not in {"full", "incremental"}:
            raise ValueError(f"mode must be 'full' or 'incremental', got {mode!r}")
        if not self.source:
            raise NotImplementedError("Subclasses must set `source`.")

        self.db = db
        self.user_id = user_id
        self.credentials = credentials
        self.scope = scope          # "all" | project_key | space_key | db_name
        self.mode = mode

    # ── Abstract API ──────────────────────────────────────────────────────────

    @abstractmethod
    def fetch_resources(
        self, since: Optional[datetime] = None
    ) -> Iterable[SourceResource]:
        """Yield resources to ingest. ``since`` is honoured in incremental mode."""

    @abstractmethod
    def scope_filter(self) -> dict[str, Any]:
        """Filter dict used to wipe the scope in ``--mode full``."""

    @abstractmethod
    def resource_identifier_for(self, resource: SourceResource) -> str:
        """The project_key / space_key / db_name for this resource."""

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self, on_progress: Optional[ProgressCallback] = None
    ) -> IngestionResult:
        """
        Execute the ingestion. If ``on_progress`` is supplied it is called once
        before fetching starts (so the UI can flip into "running" state) and
        after every successfully-processed resource.
        """
        log = IngestionLog(
            user_id=self.user_id,
            source=self.source,
            scope=self.scope,
            mode=self.mode,
            status="running",
        )
        self.db.add(log)
        # Commit the "running" row immediately so it survives a hard
        # interruption (browser close, Streamlit watcher rerun, Ctrl+C).
        # Without this, the outer get_db() context manager would
        # roll back this row along with every grant_access() insert
        # and the user would see vectors in the DB but no scope in the
        # picker — exactly the orphaned-state bug we're hardening against.
        self.db.commit()

        result = IngestionResult()
        if on_progress:
            on_progress(
                {
                    "source": self.source,
                    "scope": self.scope,
                    "stage": "starting",
                    "items_processed": 0,
                    "vectors_upserted": 0,
                    "current_item": "",
                }
            )

        try:
            if self.mode == "full":
                logger.info(
                    "[{}] FULL ingest scope={} — wiping existing vectors.",
                    self.source,
                    self.scope,
                )
                # user_id is passed for RLS; the policy reads it from
                # the per-session GUC on the connection used by
                # delete_by_filter's own get_db() block.
                delete_by_filter(self.scope_filter(), user_id=self.user_id)
                since = None
            else:
                since = self._incremental_since()
                logger.info(
                    "[{}] INCREMENTAL ingest scope={} since={}",
                    self.source,
                    self.scope,
                    since,
                )

            for resource in self.fetch_resources(since=since):
                self._process_resource(resource, result)
                # Update the running log row with cumulative counters so
                # the "Recent runs" sidebar panel shows real progress and
                # so a hard kill leaves a sane partial-status row instead
                # of one that's frozen at items=0/vectors=0.
                # _process_resource() already committed after
                # grant_access(); this flush makes the updated counters
                # part of the *next* commit (whether that's the next
                # resource's commit or the final one in the finally block).
                log.items_processed = result.items_processed
                log.vectors_upserted = result.vectors_upserted
                log.last_item_updated_at = result.last_item_updated_at
                self.db.flush()
                if on_progress:
                    on_progress(
                        {
                            "source": self.source,
                            "scope": self.scope,
                            "stage": "processing",
                            "items_processed": result.items_processed,
                            "vectors_upserted": result.vectors_upserted,
                            "current_item": resource.title,
                        }
                    )

            log.status = "success"
        except Exception as exc:
            logger.exception("[{}] Ingestion failed: {}", self.source, exc)
            log.status = "error"
            log.error_message = str(exc)[:1000]
            raise
        finally:
            log.items_processed = result.items_processed
            log.vectors_upserted = result.vectors_upserted
            log.last_item_updated_at = result.last_item_updated_at
            log.completed_at = datetime.utcnow()
            # Commit the terminal status (success/error) so it survives
            # the wrapping get_db() block being torn down by an
            # exception. ``except Exception`` in get_db() rolls back any
            # *uncommitted* work, but a committed row is durable.
            try:
                self.db.commit()
            except Exception:  # noqa: BLE001 - best-effort bookkeeping
                logger.exception(
                    "[{}] Could not commit final ingestion log row.",
                    self.source,
                )
                self.db.rollback()

        if on_progress:
            on_progress(
                {
                    "source": self.source,
                    "scope": self.scope,
                    "stage": "done",
                    "items_processed": result.items_processed,
                    "vectors_upserted": result.vectors_upserted,
                    "current_item": "",
                }
            )

        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _incremental_since(self) -> Optional[datetime]:
        last = (
            self.db.query(IngestionLog)
            .filter_by(
                user_id=self.user_id,
                source=self.source,
                scope=self.scope,
                status="success",
            )
            .order_by(IngestionLog.completed_at.desc())
            .first()
        )
        if not last or not last.last_item_updated_at:
            return None
        try:
            return datetime.fromisoformat(last.last_item_updated_at.rstrip("Z"))
        except ValueError:
            return None

    def _process_resource(
        self, resource: SourceResource, result: IngestionResult
    ) -> None:
        chunks = self._chunk(resource)
        if not chunks:
            logger.debug("Resource {} produced 0 chunks; skipping.", resource.resource_id)
            return

        # Wipe stale chunks for this resource so re-ingest with fewer chunks
        # leaves no orphan tail entries (full mode already wiped the whole scope).
        # Note: when fingerprint-dedup short-circuits a re-embed, the
        # per-resource pre-delete here would still wipe the row before
        # the dedup check sees it — so we skip the pre-delete entirely
        # when ENABLE_CONTENT_FINGERPRINT_DEDUP is on. The ON CONFLICT
        # DO UPDATE inside upsert_chunks() still gives us in-place
        # overwrite semantics for changed rows; the only loss is
        # "orphan tail" chunks (chunk_index N+1..) when the new content
        # is shorter than the old. Those are explicitly cleaned up
        # below by deleting any stored chunk_index >= len(chunks).
        if self.mode == "incremental":
            if settings.ENABLE_CONTENT_FINGERPRINT_DEDUP:
                self._prune_orphan_tail_chunks(resource.resource_id, len(chunks))
            else:
                delete_resource(resource.resource_id, user_id=self.user_id)

        upserted = upsert_chunks(chunks, user_id=self.user_id)
        result.items_processed += 1
        result.vectors_upserted += upserted

        if (
            result.last_item_updated_at is None
            or resource.last_updated > result.last_item_updated_at
        ):
            result.last_item_updated_at = resource.last_updated

        # Make this scope queryable by the user (idempotent).
        grant_access(
            self.db,
            self.user_id,
            self.source,
            self.resource_identifier_for(resource),
        )
        # Persist the grant + any pending log-row updates immediately.
        # Vector upserts in core.vector_store.upsert_chunks() commit on
        # their own engine session per batch, so without this commit the
        # bookkeeping would lag behind real data and a mid-run interrupt
        # would leave orphaned vectors with no user_accessible_resources
        # row to surface them in the sidebar picker.
        self.db.commit()

    def _chunk(self, resource: SourceResource) -> list[ResourceChunk]:
        splitter = _get_splitter()
        pieces = splitter.split_text(resource.text or "")
        chunks: list[ResourceChunk] = []
        for i, piece in enumerate(pieces):
            md = {
                "title": resource.title,
                "url": resource.url,
                "last_updated": resource.last_updated,
                **resource.metadata,
            }
            # Compute the SHA-256 fingerprint up-front so the upsert
            # path can short-circuit re-embeds for unchanged chunks
            # without a second pass over the text. Same canonical form
            # (lower-cased, whitespace-collapsed) used in vector_store.
            fingerprint = compute_content_fingerprint(piece)
            chunks.append(
                ResourceChunk(
                    resource_id=resource.resource_id,
                    source=self.source,
                    chunk_index=i,
                    text=piece,
                    metadata=md,
                    content_fingerprint=fingerprint,
                )
            )
        return chunks

    def _prune_orphan_tail_chunks(
        self, resource_id: str, kept_chunks: int
    ) -> None:
        """
        Remove any stored chunks for ``resource_id`` whose ``chunk_index``
        is >= ``kept_chunks``. Called when fingerprint dedup is enabled
        so we keep the in-place overwrite behaviour without nuking the
        rows we'd otherwise short-circuit.

        Without this, a resource that shrinks from 10 chunks to 3
        would leave chunks 3..9 as stale residue.
        """
        from sqlalchemy import delete as _delete

        from app.utils import get_db
        from models.vector_chunk import VectorChunk

        with get_db() as db:
            from app.utils import set_current_user_for_rls

            set_current_user_for_rls(db, self.user_id)
            result = db.execute(
                _delete(VectorChunk)
                .where(VectorChunk.resource_id == resource_id)
                .where(VectorChunk.chunk_index >= kept_chunks)
            )
            n = int(result.rowcount or 0)
        if n:
            logger.debug(
                "Pruned {} orphan-tail chunks for {} (kept {}).",
                n,
                resource_id,
                kept_chunks,
            )
