"""
VectorChunk — one row per (resource, chunk_index).

This is the only place embeddings live. Vectors and tenancy metadata sit in
the same Postgres database as ``users``, ``user_credentials``, and
``user_accessible_resources``, so ranking and filtering happen in a single
SQL query — no second hop to a managed vector DB.

Key design points
-----------------
* `(resource_id, chunk_index)` is unique. Re-ingesting a resource overwrites
  its chunks via PostgreSQL's `ON CONFLICT DO UPDATE`, leaving no orphans.
* Per-source identifier columns (`project_key`, `space_key`, `db_name`,
  `git_scope`) are nullable and individually indexed. Each row populates
  exactly the one column that matches its source. The query layer ANDs
  `source = X` with `<source's column> = ANY(:user_scopes)` so hits stay
  inside the user's accessible resources list.
* Source-specific extras (e.g. Jira issue_type / status, Git sha / file_path)
  go into the JSONB ``metadata`` column so the schema stays narrow and
  ingest changes don't require migrations.
* Embeddings are stored using ``pgvector.sqlalchemy.Vector``. The HNSW
  ANN index is created in ``models.migrations`` because column-level index
  declarations don't accept the ``vector_cosine_ops`` operator class on
  every SQLAlchemy version.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.config import settings
from models.base import Base


class VectorChunk(Base):
    __tablename__ = "vector_chunks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Stable identity — re-ingesting a resource overwrites these rows.
    resource_id = Column(String(512), nullable=False, index=True)
    source = Column(String(50), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)

    # Per-source filter columns. Exactly one is populated per row, matched
    # to `source`. The query layer indexes lookups on these directly.
    project_key = Column(String(255), nullable=True)
    space_key = Column(String(255), nullable=True)
    db_name = Column(String(255), nullable=True)
    git_scope = Column(String(512), nullable=True)
    # Email source: "outlook" | "gmail". Tracked separately so the access
    # table and the query-time filter can target each mailbox individually.
    email_provider = Column(String(32), nullable=True)

    # Common displayable metadata (cheaper to read than a JSONB extract).
    title = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    object_name = Column(String(512), nullable=True)
    last_updated = Column(String(64), nullable=True)

    # Free-form bag for source-specific fields (Jira issue_type, Git sha, …).
    extra = Column("metadata", JSONB, nullable=False, default=dict)

    # The chunk text + its embedding.
    text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dim), nullable=False)

    created_at = Column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        # Re-ingesting a chunk overwrites in place via ON CONFLICT.
        UniqueConstraint(
            "resource_id", "chunk_index", name="uq_vector_chunks_resource_chunk"
        ),
        # Filter helpers. Partial indexes keep them small — only rows for the
        # matching source ever populate each column.
        Index(
            "ix_vector_chunks_project_key",
            "project_key",
            postgresql_where=Column("project_key").isnot(None),
        ),
        Index(
            "ix_vector_chunks_space_key",
            "space_key",
            postgresql_where=Column("space_key").isnot(None),
        ),
        Index(
            "ix_vector_chunks_db_name",
            "db_name",
            postgresql_where=Column("db_name").isnot(None),
        ),
        Index(
            "ix_vector_chunks_git_scope",
            "git_scope",
            postgresql_where=Column("git_scope").isnot(None),
        ),
        Index(
            "ix_vector_chunks_email_provider",
            "email_provider",
            postgresql_where=Column("email_provider").isnot(None),
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<VectorChunk id={self.id} resource_id={self.resource_id} "
            f"chunk={self.chunk_index} source={self.source}>"
        )
