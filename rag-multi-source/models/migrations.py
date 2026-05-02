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
  5. Creates additive indexes (composite / partial) listed in
     ``_ADDITIVE_INDEXES`` — used for query-shape-specific tuning that
     SQLAlchemy's column-decl indexes don't cover.

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
_ADDITIVE_COLUMNS: Iterable[tuple[str, str, str]] = (
    # Email source — added after the original 4-source schema was deployed.
    # Nullable so existing rows (jira/confluence/sql/git) stay valid.
    ("vector_chunks", "email_provider", "VARCHAR(32) NULL"),
)

# (index_name, table_name, ddl). Idempotent — checked via _index_exists.
_ADDITIVE_INDEXES: Iterable[tuple[str, str, str]] = (
    (
        "ix_vector_chunks_email_provider",
        "vector_chunks",
        "CREATE INDEX IF NOT EXISTS ix_vector_chunks_email_provider "
        "ON vector_chunks (email_provider) "
        "WHERE email_provider IS NOT NULL",
    ),
    # ── Audit logging indexes ───────────────────────────────────────────
    # SQLAlchemy already declares these via Index(...) in the model
    # files, but listing them here as well keeps the migration robust
    # against models being imported in a different order or being
    # partially registered when a fresh DB is being bootstrapped.
    (
        "ix_query_audit_logs_user_ts",
        "query_audit_logs",
        "CREATE INDEX IF NOT EXISTS ix_query_audit_logs_user_ts "
        "ON query_audit_logs (user_id, timestamp DESC)",
    ),
    (
        "ix_query_audit_logs_timestamp",
        "query_audit_logs",
        "CREATE INDEX IF NOT EXISTS ix_query_audit_logs_timestamp "
        "ON query_audit_logs (timestamp DESC)",
    ),
    (
        "ix_query_audit_logs_success",
        "query_audit_logs",
        "CREATE INDEX IF NOT EXISTS ix_query_audit_logs_success "
        "ON query_audit_logs (success)",
    ),
    (
        "ix_query_step_timings_audit_id",
        "query_step_timings",
        "CREATE INDEX IF NOT EXISTS ix_query_step_timings_audit_id "
        "ON query_step_timings (audit_id)",
    ),
    (
        "ix_query_step_timings_step_duration",
        "query_step_timings",
        "CREATE INDEX IF NOT EXISTS ix_query_step_timings_step_duration "
        "ON query_step_timings (step_name, duration_ms)",
    ),
)


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
    import models.query_audit_log  # noqa: F401
    import models.query_step_timing  # noqa: F401

    # 1. pgvector extension must exist before vector_chunks can be created.
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # 2. Tables. ``create_all`` is idempotent — existing tables are skipped.
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

    # 5. Additive indexes (CREATE INDEX IF NOT EXISTS already idempotent,
    #    but we double-check the inspector first to avoid noisy logs).
    with engine.begin() as conn:
        for index_name, table, ddl in _ADDITIVE_INDEXES:
            if _index_exists(engine, table, index_name):
                continue
            logger.info("Creating index {} on {}", index_name, table)
            conn.execute(text(ddl))

    logger.info("Schema migrations complete.")
