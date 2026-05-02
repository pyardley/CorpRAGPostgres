"""
QueryAuditLog — one row per user prompt answered by the chat layer.

This is the **high-level** audit table. It captures *what* happened (who
asked what, which provider/model answered, how many tokens were spent,
how long it took, whether it succeeded). The detailed per-step
breakdown — vector retrieval, MCP tool calls, LLM invocation, post-
processing — lives on a child table (``query_step_timings``) keyed on
``audit_id``.

We always log a row, even on errors, so the audit trail is complete and
the UI's "Audit Log" view can show failures next to successes. The
model is intentionally narrow (no embeddings, no full responses) so the
table stays small even for chatty users; the prompt text is stored but
truncated by the helper at write time to a configurable cap.

Design notes
------------
* ``id`` is a UUID string for symmetry with the other tables in the
  project (``users``, ``ingestion_logs``, …) — easier to grep in logs
  than a bigint.
* ``timestamp`` is UTC and indexed so the sidebar's "last 100 queries"
  ORDER BY ts DESC LIMIT 100 stays fast.
* ``source_type`` records which answer path served the prompt:
    - ``"rag"``      — pure RAG answer (``core.rag_chain``).
    - ``"hybrid"``   — RAG context + bound MCP tools
                       (``core.mcp_chain``), at least one LLM call.
    - ``"mcp_only"`` — direct-SELECT fast path that bypassed the LLM
                       entirely (still counts as MCP for accounting).
* ``estimated_cost_usd`` uses the same rate table as the chat status
  bar so the two views never disagree.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from models.base import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class QueryAuditLog(Base):
    """High-level audit row for one user prompt + answer cycle."""

    __tablename__ = "query_audit_logs"

    id = Column(String(36), primary_key=True, default=_new_uuid)

    # NULL allowed for system-level / health-check style queries that
    # aren't owned by a user, but in practice every chat prompt has one.
    user_id = Column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )

    timestamp = Column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    # Truncated prompt text (see app.utils.log_query_audit cap). We keep
    # the prompt so the audit page can show what the user asked, but a
    # very long pasted SQL block doesn't bloat the table.
    prompt_text = Column(Text, nullable=True)

    # Wall-clock from the chat layer's perspective — covers retrieval,
    # reasoning, tool calls, post-processing.
    total_duration_ms = Column(Integer, nullable=False, default=0)

    # Token + cost accounting. May all be zero on the direct-SELECT
    # MCP fast path (no LLM call).
    tokens_prompt = Column(Integer, nullable=False, default=0)
    tokens_completion = Column(Integer, nullable=False, default=0)
    estimated_cost_usd = Column(Float, nullable=False, default=0.0)

    # Provider + model that served this prompt. Stored as plain strings
    # rather than enums so adding a new provider doesn't need a
    # migration.
    llm_provider = Column(String(32), nullable=True)
    llm_model = Column(String(128), nullable=True)

    success = Column(Boolean, nullable=False, default=True, index=True)
    error_message = Column(Text, nullable=True)

    # "rag" | "hybrid" | "mcp_only" — see module docstring.
    source_type = Column(String(16), nullable=False, default="rag")

    user = relationship("User")
    step_timings = relationship(
        "QueryStepTiming",
        back_populates="audit",
        cascade="all, delete-orphan",
        order_by="QueryStepTiming.id",
    )

    __table_args__ = (
        # Sidebar query: "last 100 queries by user, newest first".
        Index("ix_query_audit_logs_user_ts", "user_id", "timestamp"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<QueryAuditLog id={self.id} user={self.user_id} "
            f"ts={self.timestamp} source={self.source_type} "
            f"success={self.success}>"
        )
