"""Database session management, encryption helpers, and shared utilities."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator, Optional

from cryptography.fernet import Fernet
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from models.base import Base

# ── Database engine ───────────────────────────────────────────────────────────

_connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables (idempotent – safe to call on every startup)."""
    # Import models so they register with Base metadata
    import models.user  # noqa: F401
    import models.ingestion_log  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised.")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Encryption ────────────────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    key = settings.ENCRYPTION_KEY
    if not key:
        # Auto-generate a key the first time and tell the operator to persist it
        key = Fernet.generate_key().decode()
        logger.warning(
            "ENCRYPTION_KEY not set. Generated a temporary key for this session. "
            "Set ENCRYPTION_KEY={} in your .env to persist encrypted credentials.",
            key,
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _get_fernet().decrypt(ciphertext.encode()).decode()


# ── Credential helpers ────────────────────────────────────────────────────────

def save_credential(db: Session, user_id: str, source: str, key: str, value: str) -> None:
    from models.user import UserCredential

    encrypted = encrypt(value)
    existing = (
        db.query(UserCredential)
        .filter_by(user_id=user_id, source=source, credential_key=key)
        .first()
    )
    if existing:
        existing.encrypted_value = encrypted
    else:
        db.add(UserCredential(user_id=user_id, source=source, credential_key=key, encrypted_value=encrypted))
    db.flush()


def load_credential(db: Session, user_id: str, source: str, key: str) -> Optional[str]:
    from models.user import UserCredential

    row = (
        db.query(UserCredential)
        .filter_by(user_id=user_id, source=source, credential_key=key)
        .first()
    )
    if row is None:
        return None
    try:
        return decrypt(row.encrypted_value)
    except Exception:
        logger.exception("Failed to decrypt credential {}/{}/{}", user_id, source, key)
        return None


def load_all_credentials(db: Session, user_id: str, source: str) -> dict[str, str]:
    from models.user import UserCredential

    rows = db.query(UserCredential).filter_by(user_id=user_id, source=source).all()
    result: dict[str, str] = {}
    for row in rows:
        try:
            result[row.credential_key] = decrypt(row.encrypted_value)
        except Exception:
            logger.warning("Could not decrypt {}/{}/{}", user_id, source, row.credential_key)
    return result
