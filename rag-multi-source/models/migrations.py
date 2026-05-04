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
        "CREATE INDEX IF NOT EXISTS ix_vector_chunks_text_search "
        "ON vector_chunks USING GIN (text_search)",
    ),
)


# ── Row-Level Security (defence-in-depth) ─────────────────────────────────
#
# The application-layer ``WHERE`` clause built from
# ``user_accessible_resources`` is still the primary tenancy boundary.
# Enabling RLS lets the database itself reject any row the current
# session-bound user_id wasn't granted, so a regression in
# ``filter_to_where`` or a naked ``SELECT * FROM vector_chunks`` from a
# misbehaving script can't leak chunks across tenants.
#
# Each policy reads ``current_setting('app.current_user_id', true)`` —
# the trailing ``true`` makes the GUC missing-safe (returns NULL rather
# than raising) so admin tooling that hasn't called
# ``set_current_user_for_rls`` simply sees no rows. The app's
# ``get_db()`` wrapper sets this GUC per request via SET LOCAL so it's
# scoped to the active transaction and never leaks across pooled
# connections.
#
# ``BYPASSRLS`` is intentionally NOT granted — superuser-only by default.
# An operator who needs to run admin SQL ad-hoc should either bypass
# via psql as the DB owner (default behaviour) or set the GUC manually.

_RLS_STATEMENTS: tuple[tuple[str, str], ...] = (
    # 1. Enable RLS on the two tenancy-sensitive tables.
    (
        "vector_chunks_enable_rls",
        "ALTER TABLE vector_chunks ENABLE ROW LEVEL SECURITY",
    ),
    (
        "user_accessible_resources_enable_rls",
        "ALTER TABLE user_accessible_resources ENABLE ROW LEVEL SECURITY",
    ),
    # 2. SELECT policy on vector_chunks — only allow rows whose
    #    (source, per-source identifier) appears in the current user's
    #    user_accessible_resources rows. The COALESCE picks the one
    #    populated identifier column for the row (project_key /
    #    space_key / db_name / git_scope / email_provider).
    (
        "vector_chunks_select_policy",
        """
        CREATE POLICY vector_chunks_tenant_select ON vector_chunks
        FOR SELECT
        USING (
            current_setting('app.current_user_id', true) IS NOT NULL
            AND EXISTS (
                SELECT 1
                FROM user_accessible_resources uar
                WHERE uar.user_id = current_setting('app.current_user_id', true)
                  AND uar.source = vector_chunks.source
                  AND uar.resource_identifier = COALESCE(
                      vector_chunks.project_key,
                      vector_chunks.space_key,
                      vector_chunks.db_name,
                      vector_chunks.git_scope,
                      vector_chunks.email_provider
                  )
            )
        )
        """,
    ),
    # 3. Write policies on vector_chunks — ingestion runs under the
    #    same DB role as the chat layer, so we let any authenticated
    #    user (i.e. one with the GUC set) write. Tenancy of writes is
    #    enforced at app-level by BaseIngestor.run() / grant_access()
    #    which only writes scopes the user has credentials for. The
    #    USING / WITH CHECK clauses both succeed unconditionally; the
    #    presence of the GUC is the only requirement (so an
    #    unauthenticated session can't write at all).
    (
        "vector_chunks_insert_policy",
        """
        CREATE POLICY vector_chunks_tenant_insert ON vector_chunks
        FOR INSERT
        WITH CHECK (current_setting('app.current_user_id', true) IS NOT NULL)
        """,
    ),
    (
        "vector_chunks_update_policy",
        """
        CREATE POLICY vector_chunks_tenant_update ON vector_chunks
        FOR UPDATE
        USING (current_setting('app.current_user_id', true) IS NOT NULL)
        WITH CHECK (current_setting('app.current_user_id', true) IS NOT NULL)
        """,
    ),
    (
        "vector_chunks_delete_policy",
        """
        CREATE POLICY vector_chunks_tenant_delete ON vector_chunks
        FOR DELETE
        USING (current_setting('app.current_user_id', true) IS NOT NULL)
        """,
    ),
    # 4. user_accessible_resources — a user only sees / mutates their
    #    own grant rows. Same GUC-driven pattern.
    (
        "uar_select_policy",
        """
        CREATE POLICY uar_owner_select ON user_accessible_resources
        FOR SELECT
        USING (user_id = current_setting('app.current_user_id', true))
        """,
    ),
    (
        "uar_insert_policy",
        """
        CREATE POLICY uar_owner_insert ON user_accessible_resources
        FOR INSERT
        WITH CHECK (user_id = current_setting('app.current_user_id', true))
        """,
    ),
    (
        "uar_update_policy",
        """
        CREATE POLICY uar_owner_update ON user_accessible_resources
        FOR UPDATE
        USING (user_id = current_setting('app.current_user_id', true))
        WITH CHECK (user_id = current_setting('app.current_user_id', true))
        """,
    ),
    (
        "uar_delete_policy",
        """
        CREATE POLICY uar_owner_delete ON user_accessible_resources
        FOR DELETE
        USING (user_id = current_setting('app.current_user_id', true))
        """,
    ),
)


# Names of the policies created above — used so we can detect "already
# applied" and avoid the noisy ``DuplicateObject`` errors that would
# otherwise fire on every restart. Maps policy name -> table name.
_RLS_POLICY_NAMES: dict[str, str] = {
    "vector_chunks_tenant_select": "vector_chunks",
    "vector_chunks_tenant_insert": "vector_chunks",
    "vector_chunks_tenant_update": "vector_chunks",
    "vector_chunks_tenant_delete": "vector_chunks",
    "uar_owner_select": "user_accessible_resources",
    "uar_owner_insert": "user_accessible_resources",
    "uar_owner_update": "user_accessible_resources",
    "uar_owner_delete": "user_accessible_resources",
}


def _existing_columns(engine: Engine, table: str) -> set[str]:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table)}


def _index_exists(engine: Engine, table: str, index_name: str) -> bool:
    """
    True if a Postgres index with this name exists in the current schema.

    Uses ``pg_class`` / ``pg_namespace`` directly instead of SQLAlchemy's
    ``inspector.get_indexes()``. The inspector silently omits expression-
    based and some GIN indexes (notably GIN over a generated ``tsvector``
    column) on certain SQLAlchemy versions, which would let
    ``run_migrations`` fall through to ``CREATE INDEX IF NOT EXISTS``
    for an index that already exists. ``IF NOT EXISTS`` is *supposed*
    to be a no-op in that case, but in practice we've seen it surface
    a ``pg_class_relname_nsp_index`` unique violation under specific
    catalog states (e.g. previous migration partially completed,
    invalid-index edges), so we answer the question authoritatively
    up front against the same catalog Postgres itself enforces
    uniqueness on.

    Note: the ``table`` parameter is retained for signature stability
    but no longer used in the lookup — index names are unique per
    schema, not per table, which is exactly the constraint we're
    trying to honour.
    """
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT 1 FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relkind = 'i' "
                "  AND c.relname = :index_name "
                "  AND n.nspname = current_schema()"
            ),
            {"index_name": index_name},
        ).first()
    return row is not None


def _policy_exists(engine: Engine, table: str, policy_name: str) -> bool:
    """True if a Postgres RLS policy with this name exists on this table."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT 1 FROM pg_policies "
                "WHERE schemaname = 'public' "
                "  AND tablename = :table "
                "  AND policyname = :policy"
            ),
            {"table": table, "policy": policy_name},
        ).first()
    return row is not None


def _rls_enabled(engine: Engine, table: str) -> bool:
    """True if RLS is already enabled on this table (avoids noisy logs)."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT relrowsecurity "
                "FROM pg_class "
                "WHERE relname = :table "
                "  AND relnamespace = 'public'::regnamespace"
            ),
            {"table": table},
        ).first()
    return bool(row[0]) if row else False


def _apply_rls_policies(engine: Engine) -> None:
    """
    Enable RLS + create the tenancy policies on ``vector_chunks`` and
    ``user_accessible_resources``. Idempotent — skips ENABLE if already
    on, and skips CREATE POLICY for any policy already present in
    ``pg_policies``.

    Quietly returns if ``settings.ENABLE_RLS`` is False so the helper
    is opt-in and existing deployments continue to work unchanged.
    """
    if not settings.ENABLE_RLS:
        logger.info("RLS disabled (ENABLE_RLS=false); skipping policy setup.")
        return

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "vector_chunks" not in table_names or "user_accessible_resources" not in table_names:
        logger.warning(
            "RLS skipped — required tables not yet present "
            "(this is expected on the very first bootstrap call)."
        )
        return

    with engine.begin() as conn:
        for name, statement in _RLS_STATEMENTS:
            sql = " ".join(statement.split()).strip()

            # Distinguish ENABLE-RLS statements from CREATE POLICY statements
            # so we can check the right system catalogue before running.
            if sql.upper().startswith("ALTER TABLE"):
                # Pattern: ALTER TABLE <name> ENABLE ROW LEVEL SECURITY
                table = sql.split()[2]
                if _rls_enabled(engine, table):
                    continue
                logger.info("Enabling RLS on {}", table)
                conn.execute(text(sql))
            elif sql.upper().startswith("CREATE POLICY"):
                # Pattern: CREATE POLICY <policy_name> ON <table>...
                tokens = sql.split()
                policy_name = tokens[2]
                table = _RLS_POLICY_NAMES.get(policy_name)
                if table is None:
                    logger.warning(
                        "Unknown RLS policy {} — applying without idempotency check",
                        policy_name,
                    )
                elif _policy_exists(engine, table, policy_name):
                    continue
                logger.info(
                    "Creating RLS policy {} (key={})", policy_name, name
                )
                conn.execute(text(sql))


def drop_rls_policies(engine: Engine) -> None:
    """
    Tear down every policy this module installs and disable RLS on the
    affected tables. Useful for tests, for backing out the feature on a
    single deployment, or for an admin who needs ad-hoc bypass without
    setting the GUC. Not called by ``run_migrations`` — invoke manually
    from a one-off script when needed.
    """
    with engine.begin() as conn:
        for policy_name, table in _RLS_POLICY_NAMES.items():
            logger.info("Dropping RLS policy {} on {}", policy_name, table)
            conn.execute(
                text(f'DROP POLICY IF EXISTS "{policy_name}" ON "{table}"')
            )
        for table in {"vector_chunks", "user_accessible_resources"}:
            logger.info("Disabling RLS on {}", table)
            conn.execute(
                text(f'ALTER TABLE "{table}" DISABLE ROW LEVEL SECURITY')
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

    # 5. Additive indexes. Each index is created in its OWN transaction
    #    so a failure on one (e.g. a stale catalog entry from a partial
    #    earlier run) doesn't poison the rest of the migration. We
    #    skip when ``_index_exists`` already says the name is taken,
    #    and otherwise tolerate a UniqueViolation on relname — that
    #    means the catalog already has an index with this name, which
    #    is exactly the outcome we wanted anyway.
    from sqlalchemy.exc import IntegrityError

    for index_name, table, ddl in _ADDITIVE_INDEXES:
        if _index_exists(engine, table, index_name):
            continue
        logger.info("Creating index {} on {}", index_name, table)
        try:
            with engine.begin() as conn:
                conn.execute(text(ddl))
        except IntegrityError as exc:
            # ``CREATE INDEX IF NOT EXISTS`` should normally swallow this,
            # but specific catalog states (invalid-index leftovers from
            # an aborted earlier run, etc.) can still trip the
            # ``pg_class_relname_nsp_index`` unique constraint. The end
            # state we care about — "an index with this name exists" —
            # is now true regardless, so we log and move on.
            if "pg_class_relname_nsp_index" in str(exc.orig):
                logger.warning(
                    "Index {} already present in catalog — skipping create.",
                    index_name,
                )
            else:
                raise

    # 6. Row-Level Security (defence-in-depth). Gated on
    #    settings.ENABLE_RLS so existing deployments can opt out.
    _apply_rls_policies(engine)

    logger.info("Schema migrations complete.")
