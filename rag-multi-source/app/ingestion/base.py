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
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sqlalchemy.orm import Session

if TYPE_CHECKING:  # pragma: no cover - import-cycle dodge
    # Recipe lives in a sibling top-level package and is purely
    # *optional* for ingestion. We import lazily / under TYPE_CHECKING
    # so the existing hardcoded ingestors don't pay a circular-import
    # cost and so unit tests that stub out app.ingestion.base don't
    # need to also stub recipes.
    from recipes.recipe import Recipe


# Type alias for the optional progress callback. Receives a snapshot dict like
# {"source": "...", "items_processed": 12, "vectors_upserted": 87,
#  "current_item": "PROJ-123 — fix login"}. The UI can render whatever it
# wants from this; the CLI uses loguru and ignores the callback.
ProgressCallback = Callable[[dict[str, Any]], None]

from app.config import settings
from app.utils import grant_access
from core.entity_graph import upsert_edges
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
class ImageRef:
    """
    An image discovered on a resource (Confluence attachment, Jira
    attachment, …), not yet fetched.

    `fetch_bytes` is a closure supplied by the ingestor that discovered
    the image — it already has whatever client/session/instance context
    is needed to authenticate the download (e.g. which Confluence
    instance, for multi-instance setups), so `BaseIngestor` never needs
    source-specific fetch logic. Returns `None` on failure.
    """

    filename: str
    alt_text: str
    mime_type: str
    fetch_bytes: Callable[[], Optional[bytes]]


@dataclass
class SourceResource:
    """A logical document fetched from a source, before chunking."""

    resource_id: str
    title: str
    text: str
    url: str
    last_updated: str  # ISO-8601 string
    metadata: dict[str, Any] = field(default_factory=dict)
    image_refs: list[ImageRef] = field(default_factory=list)
    # Deterministic (subject, predicate, object) edges the ingestor
    # already knows from structured fields (e.g. Jira assignee/reporter,
    # Git commit author) — see BaseIngestor._persist_entity_edges.
    entity_edges: list[tuple[str, str, str]] = field(default_factory=list)


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
        *,
        recipe: "Optional[Recipe]" = None,
    ):
        if mode not in {"full", "incremental"}:
            raise ValueError(f"mode must be 'full' or 'incremental', got {mode!r}")

        # Recipe-driven mode: a Recipe can supply the ``source`` for a
        # subclass that doesn't hardcode one (e.g. the generic builtin
        # parser used by RecipeRunner). Subclasses that DO set ``source``
        # at the class level continue to take precedence — recipes can
        # decorate them, never override them.
        if not self.source and recipe is not None:
            # Mutating an instance attribute is fine; class-level
            # ``source`` stays empty for subclasses without one.
            self.source = recipe.source
        if not self.source:
            raise NotImplementedError(
                "Subclasses must set `source` (or be driven by a Recipe)."
            )

        self.db = db
        self.user_id = user_id
        self.credentials = credentials
        self.scope = scope          # "all" | project_key | space_key | db_name
        self.mode = mode
        # Available to subclasses that opt into recipe metadata. Always
        # safe to read — defaults to None when invoked the legacy way.
        self.recipe: "Optional[Recipe]" = recipe
        # Hard cap on vision-LLM calls across this ingestor instance's
        # whole run (see settings.MAX_IMAGES_PER_INGESTION_RUN) — a
        # per-resource cap alone doesn't bound a `--mode full` re-ingest
        # of an entire space/project.
        self._images_captioned = 0

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
        text_chunks = self._chunk(resource)
        image_chunks = self._caption_image_chunks(
            resource, next_index=len(text_chunks)
        )
        chunks = text_chunks + image_chunks
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

        self._persist_entity_edges(resource)

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
        # Recipe-driven chunk size / overlap takes precedence when set;
        # falling back to the shared splitter (settings.CHUNK_SIZE) is
        # still the default for every legacy ingestor.
        if self.recipe is not None and (
            self.recipe.chunk_size is not None
            or self.recipe.chunk_overlap is not None
        ):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.recipe.chunk_size or settings.CHUNK_SIZE,
                chunk_overlap=self.recipe.chunk_overlap
                if self.recipe.chunk_overlap is not None
                else settings.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
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

    def _caption_image_chunks(
        self, resource: SourceResource, next_index: int
    ) -> list[ResourceChunk]:
        """
        Caption each of `resource.image_refs` with a vision-capable LLM
        and turn it into one extra `ResourceChunk`, appended after the
        text chunks so it's searchable like any other chunk.

        No-ops (returns `[]`) unless `settings.ENABLE_MULTIMODAL_INGESTION`
        is set. Fails OPEN per image — a caption failure (bad model,
        oversized/corrupt image, network error) is logged and that image
        is skipped; it never fails the resource or the ingestion run.
        `MAX_IMAGES_PER_INGESTION_RUN` is a hard stop across the whole
        run so a `--mode full` re-ingest of a huge space/project can't
        run up an unbounded vision-LLM bill.
        """
        if not settings.ENABLE_MULTIMODAL_INGESTION or not resource.image_refs:
            return []

        from core.vision import caption_image

        chunks: list[ResourceChunk] = []
        for i, ref in enumerate(resource.image_refs[: settings.MAX_IMAGES_PER_RESOURCE]):
            if self._images_captioned >= settings.MAX_IMAGES_PER_INGESTION_RUN:
                logger.warning(
                    "[{}] MAX_IMAGES_PER_INGESTION_RUN reached — skipping "
                    "remaining images for {}",
                    self.source,
                    resource.resource_id,
                )
                break
            try:
                image_bytes = ref.fetch_bytes()
                if not image_bytes or len(image_bytes) > settings.MAX_IMAGE_BYTES:
                    continue
                caption = caption_image(
                    image_bytes, ref.mime_type, alt_text=ref.alt_text
                )
                if not caption.strip():
                    continue
            except Exception:
                logger.warning(
                    "[{}] image caption failed for {!r} on {}",
                    self.source,
                    ref.filename,
                    resource.resource_id,
                )
                continue

            self._images_captioned += 1
            text = f"[Image: {ref.filename}]\n{caption}"
            chunks.append(
                ResourceChunk(
                    resource_id=resource.resource_id,
                    source=self.source,
                    chunk_index=next_index + i,
                    text=text,
                    metadata={
                        "title": resource.title,
                        "url": resource.url,
                        "last_updated": resource.last_updated,
                        **resource.metadata,
                        "type": "image_caption",
                        "image_filename": ref.filename,
                    },
                    content_fingerprint=compute_content_fingerprint(text),
                )
            )
        return chunks

    def _persist_entity_edges(self, resource: SourceResource) -> int:
        """
        Persist `resource.entity_edges` (deterministic, set by the
        ingestor — e.g. Jira assignee/reporter, Git commit author) plus,
        when `settings.ENABLE_ENTITY_EXTRACTION_LLM` is set, edges
        extracted from `resource.text` by an LLM pass.

        No-ops (returns 0) unless `settings.ENABLE_ENTITY_GRAPH` is set,
        or this is a SQL ingestor with `settings.ENABLE_SQL_DEPENDENCY_GRAPH`
        set, or a Git ingestor with `settings.ENABLE_GIT_DEPENDENCY_GRAPH`
        set (see `app.ingestion.sql_ingestor` / `git_ingestor` — separate
        flags since both dependency graphs are deterministic, zero-cost
        parse passes, not the LLM-relationship feature `ENABLE_ENTITY_GRAPH`
        gates). The LLM extraction pass fails OPEN — an extraction error is
        logged and skipped; deterministic edges (and the resource's
        chunks) still persist normally.

        The LLM pass never runs for `source == "sql"`, even when
        `ENABLE_ENTITY_EXTRACTION_LLM` is on: `core.entity_extraction`'s
        prompt is framed around Jira/Git free text (ticket descriptions,
        commit messages) with no SQL-object scoping, and running it over
        stored-procedure/function definitions produces relationship
        noise (predicates like `has_column`, `defines`, `executes`,
        `related_to`) rather than code-structure signal — confirmed live
        against the RetailReportingDemo fixture, where it wrote 457 such
        edges alongside 67 clean deterministic ones from
        `core.sql_dependency_extraction.find_references`. SQL's
        dependency graph relies on the deterministic edges only.

        Same reasoning carves out Git *file* resources (source code is the
        same class of noise as a SQL object body) — but not Git *commit*
        resources, whose text is a natural-language commit message, the
        shape `core.entity_extraction`'s prompt is actually designed for.
        """
        sql_dependency_graph_active = (
            self.source == "sql" and settings.ENABLE_SQL_DEPENDENCY_GRAPH
        )
        git_dependency_graph_active = (
            self.source == "git" and settings.ENABLE_GIT_DEPENDENCY_GRAPH
        )
        if (
            not settings.ENABLE_ENTITY_GRAPH
            and not sql_dependency_graph_active
            and not git_dependency_graph_active
        ):
            return 0

        edges: list[tuple[str, str, str, str]] = [
            (subject, predicate, obj, "deterministic")
            for subject, predicate, obj in resource.entity_edges
        ]

        skip_llm_extraction = self.source == "sql" or (
            self.source == "git" and resource.metadata.get("git_type") == "file"
        )
        if settings.ENABLE_ENTITY_EXTRACTION_LLM and not skip_llm_extraction:
            from core.entity_extraction import extract_entities

            # extract_entities() already fails open (logs + returns []),
            # so no try/except needed here.
            edges += [
                (subject, predicate, obj, "llm")
                for subject, predicate, obj in extract_entities(resource.text)
            ]

        return upsert_edges(
            self.source,
            self.resource_identifier_for(resource),
            resource.resource_id,
            edges,
            user_id=self.user_id,
        )

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
