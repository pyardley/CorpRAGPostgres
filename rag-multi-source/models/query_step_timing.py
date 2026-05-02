"""
QueryStepTiming — one row per measured step inside a single audit log.

This is the **detailed** companion to ``query_audit_logs``. Every major
phase of the answer pipeline (filter build, vector retrieval, each MCP
tool call, the LLM invocation, post-processing) is timed with
``time.perf_counter()`` and persisted here, keyed on ``audit_id``.

Why a separate table
--------------------
Keeping the per-step breakdown out of the audit row means:

* The high-level audit view ("last 100 prompts") doesn't have to drag
  along a JSON blob of timings — the sidebar table stays light.
* Steps are queryable in their own right (``SELECT step_name,
  AVG(duration_ms) FROM query_step_timings GROUP BY step_name`` is the
  canonical "where are we slow?" query).
* A single prompt can produce a variable number of MCP tool calls
  without us having to reshape the audit row each time.

Step-name conventions
---------------------
The chains use a short, namespaced ``step_name`` so a flat list still
reads cleanly:

* ``"build_filter"``               — translate sidebar selection into
                                      the vector-store metadata filter.
* ``"vector_retrieval"``           — pgvector top-K cosine + de-dup.
* ``"llm_invocation"``             — single LLM call (the RAG path
                                      emits exactly one of these; the
                                      hybrid path emits one per hop and
                                      indexes them via metadata).
* ``"mcp_tool_call:sql_table_query"``     — one MCP tool round-trip.
* ``"mcp_tool_call:sql_list_databases"``  — same, for the listing tool.
* ``"post_processing"``            — citation rendering / answer
                                      formatting after the LLM returns.
* ``"total"``                      — duplicate of the audit row's
                                      ``total_duration_ms`` so a single
                                      query against this table can
                                      still surface end-to-end timing
                                      without a JOIN.

``metadata`` is JSONB so each step can attach whatever extra info is
useful: row counts for MCP calls, hop index for hybrid LLM calls, the
number of chunks retrieved for vector_retrieval, and so on.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from models.base import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class QueryStepTiming(Base):
    """One measured step inside a ``QueryAuditLog`` row."""

    __tablename__ = "query_step_timings"

    id = Column(String(36), primary_key=True, default=_new_uuid)

    audit_id = Column(
        String(36),
        ForeignKey("query_audit_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # See module docstring for the conventional names.
    step_name = Column(String(128), nullable=False, index=True)

    duration_ms = Column(Integer, nullable=False, default=0)

    # Free-form bag for step-specific extras: row_count, tool_name,
    # hop, retrieved_count, error, etc. Defaulted to {} so callers don't
    # have to think about NULL handling.
    extra = Column("metadata", JSONB, nullable=False, default=dict)

    audit = relationship("QueryAuditLog", back_populates="step_timings")

    __table_args__ = (
        # Common analytics query: "what's the average duration of step X?".
        Index(
            "ix_query_step_timings_step_duration",
            "step_name",
            "duration_ms",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<QueryStepTiming audit={self.audit_id} step={self.step_name} "
            f"duration_ms={self.duration_ms}>"
        )
