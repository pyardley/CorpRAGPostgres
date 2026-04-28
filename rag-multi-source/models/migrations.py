"""
Tiny migration helper. Targets PostgreSQL with the pgvector extension.

Exposes `run_migrations(engine)` which:

  1. Ensures the ``vector`` extension is enabled.
  2. Creates any missing tables defined on ``Base.metadata`` (idempotent).
  3. Creates the HNSW ANN index on ``vector_chunks.embedding`` if it isn't
     already present. HNSW gives sub-millisecond top-K cosine queries up to
     several million rows on a tiny instance.
  4. Applies any additive ``ALTER TABLE`` column changes listed in
     ``_ADDITIVE_COLUMNS`` (still best-effort, no down-migration support).

Called from ``app.utils.init_db()`` on every Streamlit / CLI startup, so a
fresh database becomes usable just by setting ``DATABASE_URL`` and running
the app once.
"""

from __future__ import annotations

from typing import Iterable

from loguru import logger
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from models.base import Base


# (table_name, column_name, column_ddl). Example:
#   ("users", "is_admin", "BOOLEAN NOT NULL DEFAULT FALSE")
_ADDITIVE_COLUMNS: Iterable[tuple[str, str, str]] = ()


def _existing_columns(engine: Engine, table: str) -> set[str]:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table)}


def _index_exists(engine: Engine, table: str, index_name: str) -> bool:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return False
    return any(
        ix.get("name") == index_name for ix in inspector.get_indexes(table)
    )


def run_migrations(engine: Engine) -> None:
    """Ensure the schema (and pgvector) are up to date. Safe to re-run."""

    # Import all model modules so SQLAlchemy registers them on Base.metadata.
    import models.user  # noqa: F401
    import models.ingestion_log  # noqa: F401
    import models.user_accessible_resource  # noqa: F401
    import models.vector_chunk  # noqa: F401

    # 1. pgvector extension must exist before vector_chunks can be created.
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # 2. Tables.
    Base.metadata.create_all(bind=engine)

    # 3. HNSW vector index. Created post-table because SQLAlchemy doesn't
    #    have first-class support for `USING hnsw` + opclass at column-decl
    #    time. m=16 / ef_construction=64 are pgvector's defaults — fine for
    #    most corpora; tune higher if recall isn't satisfactory.
    with engine.begin() as conn:
        if not _index_exists(engine, "vector_chunks", "ix_vector_chunks_hnsw"):
            logger.info("Creating HNSW index on vector_chunks.embedding…")
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vector_chunks_hnsw "
                    "ON vector_chunks USING hnsw "
                    "(embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                )
            )

    # 4. Additive column changes.
    with engine.begin() as conn:
        for table, column, ddl in _ADDITIVE_COLUMNS:
            if column in _existing_columns(engine, table):
                continue
            logger.info("Adding column {}.{}", table, column)
            conn.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN {column} {ddl}')
            )

    logger.info("Schema migrations complete.")
