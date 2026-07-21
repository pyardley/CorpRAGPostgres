"""
ResponseCacheEntry — semantic cache of full chat answers (README
"Possible enhancements" — Semantic response cache, in the same
Postgres database).

Keyed on (question embedding, `scope_fingerprint`) rather than
`user_id`: `scope_fingerprint` is a hash of the *resolved* accessible
scopes actually used for retrieval (see `core.response_cache.scope_fingerprint`),
so two users only ever share a cache hit when their accessible data is
identical — the fingerprint match, not `user_id`, is the access-control
boundary. `user_id` is kept purely as an audit trail (who generated
this entry).

RLS on this table is intentionally GUC-presence-only, not owner-scoped
— see `models.migrations` and `core.response_cache` for the full
reasoning.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.config import settings
from models.base import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class ResponseCacheEntry(Base):
    __tablename__ = "response_cache"

    id = Column(String(36), primary_key=True, default=_new_uuid)

    # Audit trail only — NOT part of the access-control key.
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    question = Column(Text, nullable=False)
    question_embedding = Column(Vector(settings.embedding_dim), nullable=False)

    # sha256 of the canonicalised (filter_dict["by_source"], fts_language)
    # that produced this answer. The real tenancy boundary — see module
    # docstring.
    scope_fingerprint = Column(String(64), nullable=False, index=True)

    answer = Column(Text, nullable=False)

    # Serialized list[RetrievedChunk] (dicts), reconstructed on a hit.
    citations = Column(JSONB, nullable=False, default=list)

    # "rag" — MCP/hybrid answers are never cached (see core.response_cache).
    source_type = Column(String(16), nullable=False, default="rag")

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"<ResponseCacheEntry {self.question[:60]!r} fp={self.scope_fingerprint[:8]}>"
