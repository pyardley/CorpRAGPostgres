from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from models.base import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class IngestionLog(Base):
    """Tracks the state of each ingestion run so incremental updates work."""

    __tablename__ = "ingestion_logs"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # source: "jira" | "confluence" | "sql"
    source = Column(String(50), nullable=False, index=True)

    # Scope within the source (project_key / space_key / db_name / "all")
    scope = Column(String(255), nullable=False, index=True)

    # "full" | "incremental"
    mode = Column(String(20), nullable=False)

    # "running" | "success" | "error"
    status = Column(String(20), nullable=False, default="running")

    items_processed = Column(Integer, default=0)
    vectors_upserted = Column(Integer, default=0)

    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # ISO-8601 timestamp of the most recently ingested item (for incremental baseline)
    last_item_updated_at = Column(String(50), nullable=True)

    error_message = Column(Text, nullable=True)

    user = relationship("User", back_populates="ingestion_logs")

    def __repr__(self) -> str:
        return (
            f"<IngestionLog user={self.user_id} source={self.source} "
            f"scope={self.scope} status={self.status}>"
        )
