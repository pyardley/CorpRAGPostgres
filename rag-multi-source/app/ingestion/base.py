"""
Base ingestor — the contract every source-specific ingestor must satisfy.

In the new architecture there is **one** Pinecone index for the whole
organisation. The base class is the only place that talks to the vector store
during ingestion, which makes the "store-once + dedup" guarantee enforceable.

Subclass responsibilities
-------------------------
1. Set the ``source`` class attribute (e.g. ``"jira"``).
2. Implement :meth:`fetch_resources` — yield :class:`SourceResource` objects.
3. Implement :meth:`scope_filter` — Pinecone filter used to wipe the user's
   selected scope when ``mode == "full"``.
4. Implement :meth:`resource_identifier_for` — the value to store in
   ``user_accessible_resources.resource_identifier`` (the project_key /
   space_key / db_name).

The base class handles:
  * chunking with the shared text splitter,
  * deterministic vector IDs,
  * dedup against Pinecone (incremental skips re-fetching if you don't),
  * full-mode wipe-then-rebuild via metadata filter (no `user_id` involved),
  * `IngestionLog` rows for full / incremental baselines,
  * `grant_access()` into ``user_accessible_resources``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sqlalchemy.orm import Session

from app.config import settings
from app.utils import grant_access
from core.vector_store import (
    ResourceChunk,
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
        """Pinecone metadata filter used to wipe the scope in ``--mode full``."""

    @abstractmethod
    def resource_identifier_for(self, resource: SourceResource) -> str:
        """The project_key / space_key / db_name for this resource."""

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> IngestionResult:
        log = IngestionLog(
            user_id=self.user_id,
            source=self.source,
            scope=self.scope,
            mode=self.mode,
            status="running",
        )
        self.db.add(log)
        self.db.flush()

        result = IngestionResult()
        try:
            if self.mode == "full":
                logger.info(
                    "[{}] FULL ingest scope={} — wiping existing vectors.",
                    self.source,
                    self.scope,
                )
                delete_by_filter(self.scope_filter())
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
            self.db.flush()

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
        if self.mode == "incremental":
            delete_resource(resource.resource_id)

        upserted = upsert_chunks(chunks)
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
            chunks.append(
                ResourceChunk(
                    resource_id=resource.resource_id,
                    source=self.source,
                    chunk_index=i,
                    text=piece,
                    metadata=md,
                )
            )
        return chunks
