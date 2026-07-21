"""
EntityEdge — lightweight entity relationship graph (README "Possible
enhancements" — Lightweight entity graph, GraphRAG-inspired).

Each row is a (subject, predicate, object) triple derived at ingestion
time — either deterministically from structured API fields (Jira
assignee/reporter, Git commit author) or, when
``settings.ENABLE_ENTITY_EXTRACTION_LLM`` is set, extracted from free
text by an LLM pass (see ``core.entity_extraction``).

Tenancy mirrors ``vector_chunks``: ``source`` + ``resource_identifier``
match the same values stored in ``user_accessible_resources``, and RLS
policies (see ``models.migrations``) enforce the same access rule at
the database level.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String

from models.base import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class EntityEdge(Base):
    """One (subject, predicate, object) relationship triple."""

    __tablename__ = "entity_edges"

    id = Column(String(36), primary_key=True, default=_new_uuid)

    # "jira" | "git" — which source this edge was derived from.
    source = Column(String(50), nullable=False, index=True)

    # project_key (jira) or git_scope (git) — same value stored in
    # user_accessible_resources.resource_identifier for this source.
    resource_identifier = Column(String(255), nullable=False, index=True)

    # The resource this edge was derived from, e.g. "jira:PROJ-123".
    source_resource_id = Column(String(500), nullable=False, index=True)

    subject = Column(String(500), nullable=False)
    predicate = Column(String(100), nullable=False)
    object = Column(String(500), nullable=False)

    # "deterministic" | "llm"
    extraction_method = Column(String(20), nullable=False, default="deterministic")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<EntityEdge {self.subject!r} -{self.predicate}-> "
            f"{self.object!r} ({self.source})>"
        )
