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
  6. Optionally applies Row-Level Security policies on ``vector_chunks``
     and ``user_accessible_resources`` (gated on ``settings.ENABLE_RLS``).
     Policies key on the per-session GUC ``app.current_user_id`` set by
     :func:`app.utils.set_current_user_for_rls` — see the
     ``_apply_rls_policies`` block below for the full SQL.

Called from ``app.utils.init_db()`` on every Streamlit / CLI startup, so a
fresh database becomes usable just by setting ``DATABASE_URL`` and running
the app once.
"""

from __future__ import annotations

from typing import Iterable

from loguru import logger
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.config import settings
from models.base import Base


# (table_name, column_name, column_ddl). Example:
#   ("users", "is_admin", "BOOLEAN NOT NULL DEFAULT FALSE")
_ADDITIVE_COLUMNS: Iterable[tuple[str, str, str]] = (
    # Email source — added after the original 4-source schema was deployed.
    # Nullable so existing rows (jira/confluence/sql/git) stay valid.
    ("vector_chunks", "email_provider", "VARCHAR(32) NULL"),
    # Content fingerprint (SHA-256 hex) — OB1-inspired dedup. Nullable
    # so existing rows are valid; the next re-ingest backfills them.
    # CHAR(64) is exact-fit for a SHA-256 hex digest.
    ("vector_chunks", "content_fingerprint", "CHAR(64) NULL"),
    # Hybrid retrieval — Postgres full-text search vector. Generated
    # column kept in sync automatically by Postgres on every INSERT /
    # UPDATE, so the application write path is unchanged. ``setweight``
    # gives title matches a stronger weight ('A') than body matches
    # ('B'); ``ts_rank_cd`` honours those weights at query time.
    # ``coalesce`` keeps the expression total over NULL columns so the
    # generated column is never NULL.
    #
    # NB: the language config is hard-coded to ``english`` here because
    # generated-column expressions must be immutable, and ``to_tsvector``
    # is only treated as immutable when its first argument is a regclass
    # literal. If you change ``FTS_LANGUAGE`` in settings you must also
    # rebuild this column with the matching language — drop it, change
    # this DDL, and re-run migrations.
    (
        "vector_chunks",
        "text_search",
        "tsvector GENERATED ALWAYS AS ("
        "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(text, '')), 'B')"
        ") STORED",
    ),
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
    # ── Content fingerprint dedup (OB1-inspired) ────────────────────────
    # Equality lookup on the SHA-256 hex; partial so the index only
    # carries rows that have been processed by the new fingerprint code
    # path (fresh ingests + any row touched after the migration).
    (
        "ix_vector_chunks_content_fingerprint",
        "vector_chunks",
        "CREATE INDEX IF NOT EXISTS ix_vector_chunks_content_fingerprint "
        "ON vector_chunks (content_fingerprint) "
        "WHERE content_fingerprint IS NOT NULL",
    ),
    # ── Hybrid retrieval (Postgres FTS) ─────────────────────────────────
    # GIN index over the generated ``text_search`` tsvector column drives
    # the keyword side of hybrid retrieval (``core.retriever`` runs an
    # FTS query in parallel with the vector search and fuses the results
    # via Reciprocal Rank Fusion). GIN scales linearly with the corpus
    # and is the standard pgsql FTS index — no extension required.
    (
        "ix_vector_chunks_text_search",
        "vector_chunks",
        "CREATE