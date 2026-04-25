"""
UserAccessibleResource — replaces the old `user_id` filter on every Pinecone vector.

Each row says: "user X is allowed to query vectors whose `<source>` metadata field
equals `<resource_identifier>`". The vector store itself is shared across the whole
organisation; multi-tenancy is enforced *only* at query time by translating the
rows in this table into a Pinecone metadata filter.

The (user_id, source, resource_identifier) tuple is unique.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import relationship

from models.base import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class UserAccessibleResource(Base):
    """One row per (user, source, scope) the user is allowed to retrieve from."""

    __tablename__ = "user_accessible_resources"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    user_id = Column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )

    # "jira" | "confluence" | "sql"
    source = Column(String(50), nullable=False, index=True)

    # project_key (jira), space_key (confluence), or db_name (sql)
    resource_identifier = Column(String(255), nullable=False, index=True)

    last_synced = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="accessible_resources")

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "source",
            "resource_identifier",
            name="uq_user_source_resource",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<UserAccessibleResource user={self.user_id} "
            f"source={self.source} id={self.resource_identifier}>"
        )
