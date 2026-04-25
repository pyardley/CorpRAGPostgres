"""
Tiny SQLite-friendly migration helper.

We are intentionally not pulling in Alembic for a single-table change. This module
exposes `run_migrations(engine)` which:

  * Creates any missing tables defined on `Base.metadata` (idempotent).
  * Adds any new columns introduced after the original schema using ALTER TABLE
    (best-effort — tested against SQLite).

Call from `app.utils.init_db()` so it runs on every Streamlit/CLI startup.
"""

from __future__ import annotations

from typing import Iterable

from loguru import logger
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from models.base import Base


# Add (table_name, column_name, column_ddl) tuples here as the schema evolves.
# Example: ("users", "is_admin", "BOOLEAN NOT NULL DEFAULT 0")
_ADDITIVE_COLUMNS: Iterable[tuple[str, str, str]] = ()


def _existing_columns(engine: Engine, table: str) -> set[str]:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table)}


def run_migrations(engine: Engine) -> None:
    """Ensure schema is up-to-date. Safe to call repeatedly."""
    # Import all model modules so they register with Base.metadata
    import models.user  # noqa: F401
    import models.ingestion_log  # noqa: F401
    import models.user_accessible_resource  # noqa: F401

    # 1. Create any missing tables (this is what handles the new
    #    user_accessible_resources table on existing DBs).
    Base.metadata.create_all(bind=engine)

    # 2. Apply additive column changes.
    with engine.begin() as conn:
        for table, column, ddl in _ADDITIVE_COLUMNS:
            if column in _existing_columns(engine, table):
                continue
            logger.info("Adding column {}.{}", table, column)
            conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN {column} {ddl}'))

    logger.info("Schema migrations complete.")
