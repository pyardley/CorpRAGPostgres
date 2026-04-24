from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from models.base import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    credentials = relationship(
        "UserCredential", back_populates="user", cascade="all, delete-orphan"
    )
    ingestion_logs = relationship(
        "IngestionLog", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


class UserCredential(Base):
    """Encrypted storage for per-user connector credentials."""

    __tablename__ = "user_credentials"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    # source: "jira" | "confluence" | "sql"
    source = Column(String(50), nullable=False)
    # credential_key: "url" | "email" | "api_token" | "conn_str" | "db_name"
    credential_key = Column(String(100), nullable=False)
    encrypted_value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    user = relationship("User", back_populates="credentials")

    __table_args__ = (
        UniqueConstraint("user_id", "source", "credential_key", name="uq_user_source_key"),
    )

    def __repr__(self) -> str:
        return f"<UserCredential user={self.user_id} source={self.source} key={self.credential_key}>"
