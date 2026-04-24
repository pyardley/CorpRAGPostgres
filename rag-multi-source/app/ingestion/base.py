"""Abstract base class for all source ingestors."""

from __future__ import annotations

import abc
import time
from datetime import datetime, timezone
from typing import Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from app.config import settings
from app.utils import get_db
from core.llm import get_embeddings
from core.vector_store import delete_vectors, get_vector_store
from models.ingestion_log import IngestionLog


class BaseIngestor(abc.ABC):
    """
    Subclasses implement :meth:`list_scopes`, :meth:`load_documents`.
    This base handles chunking, embedding, upsert, and log management.
    """

    SOURCE: str  # "jira" | "confluence" | "sql"
    BATCH_SIZE: int = 100  # vectors per Pinecone upsert call
    RATE_LIMIT_SLEEP: float = 0.25  # seconds between API calls

    def __init__(self, user_id: str, credentials: dict[str, str]) -> None:
        self.user_id = user_id
        self.credentials = credentials
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def list_scopes(self) -> list[dict[str, str]]:
        """
        Return available scopes (projects / spaces / databases).
        Each entry is a dict with at least {"key": ..., "name": ...}.
        """

    @abc.abstractmethod
    def load_documents(
        self, scope_key: str, since: Optional[str] = None
    ) -> list[Document]:
        """
        Fetch and return LangChain Documents for *scope_key*.
        If *since* is provided (ISO-8601 string), only return items
        modified after that timestamp (incremental mode).
        Each Document.metadata must include all relevant filter fields:
          user_id, source, title, url, last_updated,
          + source-specific: project_key | space_key | db_name, object_name…
        """

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        scope_key: str = "all",
        mode: str = "incremental",
    ) -> IngestionLog:
        """
        Execute a full or incremental ingestion for *scope_key*.
        Returns the completed IngestionLog row.
        """
        log = self._start_log(scope_key, mode)

        try:
            since: Optional[str] = None
            if mode == "incremental":
                since = self._last_success_timestamp(scope_key)
                logger.info(
                    "[{}] Incremental mode – fetching items since {}", self.SOURCE, since
                )
            else:
                logger.info("[{}] Full load – clearing scope '{}' first.", self.SOURCE, scope_key)
                self._delete_scope(scope_key)

            docs = self.load_documents(scope_key, since=since)
            logger.info("[{}] Loaded {} raw documents.", self.SOURCE, len(docs))

            chunks = self._chunk(docs)
            logger.info("[{}] Split into {} chunks.", self.SOURCE, len(chunks))

            vectors_upserted = self._embed_and_upsert(chunks)

            last_updated = self._max_last_updated(docs)
            self._complete_log(log, len(docs), vectors_upserted, last_updated)
            # Reflect the final state on the in-memory object so callers see
            # correct values (the SQLAlchemy row was updated in a separate session).
            log.status = "success"
            log.items_processed = len(docs)
            log.vectors_upserted = vectors_upserted
            log.last_item_updated_at = last_updated

        except Exception as exc:
            logger.exception("[{}] Ingestion failed for scope '{}'.", self.SOURCE, scope_key)
            self._fail_log(log, str(exc))
            log.status = "error"
            log.error_message = str(exc)[:2000]

        return log

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk(self, docs: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for i, doc in enumerate(docs):
            split = self._splitter.split_documents([doc])
            for j, chunk in enumerate(split):
                chunk.metadata["chunk_index"] = j
                chunks.append(chunk)
        return chunks

    # ── Embedding & upsert ────────────────────────────────────────────────────

    def _embed_and_upsert(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0

        embeddings = get_embeddings()
        vs = get_vector_store(embeddings)
        total = 0

        for start in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[start : start + self.BATCH_SIZE]
            vs.add_documents(batch)
            total += len(batch)
            logger.debug("[{}] Upserted {}/{} chunks.", self.SOURCE, total, len(chunks))
            time.sleep(self.RATE_LIMIT_SLEEP)

        return total

    # ── Scope deletion (full load) ────────────────────────────────────────────

    def _delete_scope(self, scope_key: str) -> None:
        filt = self._scope_filter(scope_key)
        filt["user_id"] = self.user_id
        delete_vectors(filt)

    def _scope_filter(self, scope_key: str) -> dict[str, Any]:
        """Override in subclasses to produce the right metadata filter."""
        return {"source": self.SOURCE}

    # ── Log management ────────────────────────────────────────────────────────

    def _start_log(self, scope_key: str, mode: str) -> IngestionLog:
        with get_db() as db:
            log = IngestionLog(
                user_id=self.user_id,
                source=self.SOURCE,
                scope=scope_key,
                mode=mode,
                status="running",
            )
            db.add(log)
            db.flush()
            db.expunge(log)
        return log

    def _complete_log(
        self,
        log: IngestionLog,
        items_processed: int,
        vectors_upserted: int,
        last_item_updated_at: Optional[str],
    ) -> None:
        with get_db() as db:
            row = db.query(IngestionLog).filter_by(id=log.id).first()
            if row:
                row.status = "success"
                row.items_processed = items_processed
                row.vectors_upserted = vectors_upserted
                row.last_item_updated_at = last_item_updated_at
                row.completed_at = datetime.now(timezone.utc)

    def _fail_log(self, log: IngestionLog, error: str) -> None:
        with get_db() as db:
            row = db.query(IngestionLog).filter_by(id=log.id).first()
            if row:
                row.status = "error"
                row.error_message = error[:2000]
                row.completed_at = datetime.now(timezone.utc)

    def _last_success_timestamp(self, scope_key: str) -> Optional[str]:
        """Return the last_item_updated_at from the most recent successful run."""
        with get_db() as db:
            row = (
                db.query(IngestionLog)
                .filter_by(user_id=self.user_id, source=self.SOURCE, scope=scope_key, status="success")
                .order_by(IngestionLog.completed_at.desc())
                .first()
            )
            return row.last_item_updated_at if row else None

    @staticmethod
    def _max_last_updated(docs: list[Document]) -> Optional[str]:
        dates = [d.metadata.get("last_updated") for d in docs if d.metadata.get("last_updated")]
        return max(dates) if dates else None
