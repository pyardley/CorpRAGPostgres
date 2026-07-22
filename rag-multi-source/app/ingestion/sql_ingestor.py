"""
SQL Server ingestor — extracts stored procedures, functions, views and table
schemas, one resource per object.

resource_id format:
    sql:{server}.{db_name}.{schema}.{object_name}

`server` is derived from the connection string (a sanitised SERVER=… value).
For full-mode wipes the scope is the database name.

resource_identifier (for the access table): the db_name.

Required credentials:
  conn_str   — a full ODBC connect string, e.g.
               "DRIVER={ODBC Driver 18 for SQL Server};SERVER=tcp:host,1433;
                DATABASE=master;UID=user;PWD=secret;TrustServerCertificate=yes"
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable, Optional

from loguru import logger

from app.config import settings
from app.ingestion.base import BaseIngestor, SourceResource
from core.sql_ddl import render_table_ddl
from core.sql_dependency_extraction import find_references


_SQL_ROUTINES = """
SELECT
    r.ROUTINE_SCHEMA     AS schema_name,
    r.ROUTINE_NAME       AS object_name,
    r.ROUTINE_TYPE       AS object_type,
    r.ROUTINE_DEFINITION AS definition,
    r.LAST_ALTERED       AS last_altered
FROM INFORMATION_SCHEMA.ROUTINES r
WHERE r.ROUTINE_TYPE IN ('PROCEDURE', 'FUNCTION')
  AND r.ROUTINE_DEFINITION IS NOT NULL
ORDER BY r.ROUTINE_SCHEMA, r.ROUTINE_NAME
"""

_SQL_VIEWS = """
SELECT
    v.TABLE_SCHEMA     AS schema_name,
    v.TABLE_NAME       AS object_name,
    'VIEW'             AS object_type,
    v.VIEW_DEFINITION  AS definition
FROM INFORMATION_SCHEMA.VIEWS v
WHERE v.VIEW_DEFINITION IS NOT NULL
ORDER BY v.TABLE_SCHEMA, v.TABLE_NAME
"""

_SQL_TRIGGERS = """
SELECT
    s.name             AS schema_name,
    tr.name            AS trigger_name,
    OBJECT_NAME(tr.parent_id) AS table_name,
    sm.definition      AS definition,
    tr.is_disabled     AS is_disabled,
    tr.modify_date     AS modify_date,
    (
        SELECT STRING_AGG(te.type_desc, ', ')
        FROM sys.trigger_events te
        WHERE te.object_id = tr.object_id
    )                  AS trigger_events
FROM sys.triggers tr
JOIN sys.sql_modules sm ON sm.object_id = tr.object_id
JOIN sys.tables t ON t.object_id = tr.parent_id
JOIN sys.schemas s ON s.schema_id = t.schema_id
WHERE tr.is_ms_shipped = 0
ORDER BY s.name, t.name, tr.name
"""

_SQL_TABLE_COLS = """
SELECT
    c.TABLE_SCHEMA, c.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE,
    c.CHARACTER_MAXIMUM_LENGTH, c.IS_NULLABLE, c.COLUMN_DEFAULT
FROM INFORMATION_SCHEMA.COLUMNS c
JOIN INFORMATION_SCHEMA.TABLES t
  ON t.TABLE_SCHEMA = c.TABLE_SCHEMA AND t.TABLE_NAME = c.TABLE_NAME
WHERE t.TABLE_TYPE = 'BASE TABLE'
ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
"""


class SQLIngestor(BaseIngestor):
    source = "sql"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._base_conn_str = (self.credentials.get("conn_str") or "").strip()
        if not self._base_conn_str:
            raise ValueError("SQL credentials incomplete: need conn_str.")
        self._server = _extract_server(self._base_conn_str)

    # ── Engine ───────────────────────────────────────────────────────────────

    def _make_engine(self, db_name: Optional[str] = None):
        """
        Build a SQLAlchemy engine for the given database.

        Why this is more involved than just ``DATABASE=...`` in the conn
        string: with some SQL Server setups (especially Windows
        authentication where the login has a server-level *default
        database* assignment), the ``DATABASE=`` clause in the ODBC
        connection string is silently ignored — the login lands in its
        default DB regardless. The reliable way to guarantee we're inside
        the target catalog is to run ``USE [db_name]`` immediately after
        every connect.

        We do that via a SQLAlchemy ``connect`` event listener so it
        fires for every pooled connection automatically.
        """
        from sqlalchemy import create_engine, event
        from sqlalchemy.engine import URL

        conn_str = self._base_conn_str
        if db_name and db_name != "all":
            # Still adjust the DATABASE= clause as a hint to the driver —
            # cheap, and helps when the server *does* respect it.
            if "DATABASE=" in conn_str.upper():
                conn_str = re.sub(
                    r"DATABASE=[^;]+",
                    f"DATABASE={db_name}",
                    conn_str,
                    flags=re.IGNORECASE,
                )
            else:
                conn_str += f";DATABASE={db_name}"

        url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
        engine = create_engine(url, fast_executemany=True)

        if db_name and db_name != "all":
            target = db_name  # capture before the listener closure

            @event.listens_for(engine, "connect")
            def _force_db(dbapi_conn, _conn_record):
                # Use the raw DB-API cursor; SQLAlchemy isn't ready yet at
                # connect-time. The bracket-quoted name handles dbs with
                # spaces / hyphens.
                cursor = dbapi_conn.cursor()
                try:
                    cursor.execute(f"USE [{target}]")
                finally:
                    cursor.close()

        return engine

    # ── Abstract API ─────────────────────────────────────────────────────────

    def scope_filter(self) -> dict[str, Any]:
        if self.scope == "all":
            return {"source": "sql"}
        return {"source": "sql", "db_name": self.scope}

    def resource_identifier_for(self, resource: SourceResource) -> str:
        return resource.metadata["db_name"]

    def fetch_resources(
        self, since: Optional[datetime] = None
    ) -> Iterable[SourceResource]:
        databases: list[str]
        if self.scope == "all":
            databases = self._list_databases()
        else:
            databases = [self.scope]

        since_iso = since.isoformat() if since else None
        for db_name in databases:
            yield from self._iter_database(db_name, since_iso)

    # ── Database discovery ───────────────────────────────────────────────────

    def _list_databases(self) -> list[str]:
        from sqlalchemy import text

        try:
            engine = self._make_engine()
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT name FROM sys.databases "
                        "WHERE state_desc = 'ONLINE' "
                        "AND name NOT IN ('master','tempdb','model','msdb') "
                        "ORDER BY name"
                    )
                ).fetchall()
            return [r[0] for r in rows]
        except Exception as exc:
            logger.warning("[sql] Could not list databases: {}", exc)
            return []

    # ── Per-database iteration ───────────────────────────────────────────────

    def _iter_database(
        self, db_name: str, since: Optional[str]
    ) -> Iterable[SourceResource]:
        try:
            engine = self._make_engine(db_name)
        except Exception as exc:
            logger.error("[sql] Cannot connect to {}: {}", db_name, exc)
            return

        try:
            # Fetch pass: every object's raw rows, before building any
            # SourceResource. This lets us assemble a whole-database
            # catalog of known object names up front (see
            # _build_known_objects) so each object's definition text can
            # be scanned for references to any OTHER object in the same
            # database — the static dependency graph (see
            # core.sql_dependency_extraction.find_references) needs the
            # full catalog to exist before it can classify a single match.
            routine_rows = self._fetch_routines(engine, db_name)
            view_rows = self._fetch_views(engine, db_name)
            table_groups = self._fetch_tables(engine, db_name)
            trigger_rows = self._fetch_triggers(engine, db_name)

            known_objects = self._build_known_objects(
                routine_rows, view_rows, table_groups, trigger_rows
            )

            # Build pass: rows -> SourceResource, now with entity_edges
            # populated from the catalog built above.
            yield from self._resources_from_routines(
                routine_rows, db_name, since, known_objects
            )
            yield from self._resources_from_views(view_rows, db_name, known_objects)
            yield from self._resources_from_tables(
                table_groups, db_name, known_objects
            )
            yield from self._resources_from_triggers(
                trigger_rows, db_name, since, known_objects
            )
        finally:
            engine.dispose()

    # ── Fetch pass (raw rows, no SourceResource construction) ────────────────

    def _fetch_routines(self, engine, db_name: str) -> list[Any]:
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                return conn.execute(text(_SQL_ROUTINES)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Routines failed for {}: {}", db_name, exc)
            return []

    def _fetch_views(self, engine, db_name: str) -> list[Any]:
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                return conn.execute(text(_SQL_VIEWS)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Views failed for {}: {}", db_name, exc)
            return []

    def _fetch_tables(self, engine, db_name: str) -> dict[str, list]:
        """Column rows grouped by `schema.table_name` — the natural
        "fetch" step for tables, since a table's DDL is reconstructed
        from its column set rather than a single definition column."""
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(_SQL_TABLE_COLS)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Tables failed for {}: {}", db_name, exc)
            return {}

        tables: dict[str, list] = {}
        for row in rows:
            full_name = f"{row[0]}.{row[1]}"
            tables.setdefault(full_name, []).append(row)
        return tables

    def _fetch_triggers(self, engine, db_name: str) -> list[Any]:
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                return conn.execute(text(_SQL_TRIGGERS)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Triggers failed for {}: {}", db_name, exc)
            return []

    # ── Whole-database object catalog (for static dependency extraction) ────

    def _build_known_objects(
        self,
        routine_rows: list[Any],
        view_rows: list[Any],
        table_groups: dict[str, list],
        trigger_rows: list[Any],
    ) -> dict[str, tuple[str, str]]:
        """
        Lowercased "schema.name" -> (canonical_name, object_type) for
        every table/view/procedure/function/trigger in this database.
        Temp tables and CTEs never appear here since they never appear
        in INFORMATION_SCHEMA/sys.triggers — exactly the set of names
        `core.sql_dependency_extraction.find_references` should be
        searching for.
        """
        known: dict[str, tuple[str, str]] = {}
        for schema_name, object_name, object_type, _definition, _last_altered in routine_rows:
            full_name = f"{schema_name}.{object_name}"
            known[full_name.lower()] = (full_name, object_type.lower())
        for schema_name, object_name, _object_type, _definition in view_rows:
            full_name = f"{schema_name}.{object_name}"
            known[full_name.lower()] = (full_name, "view")
        for full_name in table_groups:
            known[full_name.lower()] = (full_name, "table")
        for schema_name, trigger_name, _table_name, _definition, _is_disabled, _modify_date, _trigger_events in trigger_rows:
            full_name = f"{schema_name}.{trigger_name}"
            known[full_name.lower()] = (full_name, "trigger")
        return known

    # ── Build pass (rows -> SourceResource) ──────────────────────────────────

    def _resources_from_routines(
        self,
        rows: list[Any],
        db_name: str,
        since: Optional[str],
        known_objects: dict[str, tuple[str, str]],
    ) -> Iterable[SourceResource]:
        for schema_name, object_name, object_type, definition, last_altered in rows:
            last_altered_str = str(last_altered) if last_altered else ""
            if since and last_altered_str and last_altered_str < since:
                continue

            full_name = f"{schema_name}.{object_name}"
            text_body = (
                f"SQL Server {object_type}: {full_name}\n"
                f"Database: {db_name}\n\n"
                f"-- Definition:\n{definition or '(definition not available)'}"
            )
            yield SourceResource(
                resource_id=f"sql:{self._server}.{db_name}.{full_name}",
                title=f"{db_name}: {full_name} ({object_type})",
                text=text_body,
                url="",
                last_updated=last_altered_str or datetime.utcnow().isoformat(),
                metadata={
                    "db_name": db_name,
                    "schema_name": schema_name,
                    "object_name": full_name,
                    "object_type": object_type.lower(),
                    "server": self._server,
                },
                entity_edges=_qualified_edges(
                    definition or "", full_name, known_objects, db_name
                ),
            )

    def _resources_from_views(
        self,
        rows: list[Any],
        db_name: str,
        known_objects: dict[str, tuple[str, str]],
    ) -> Iterable[SourceResource]:
        for schema_name, object_name, _object_type, definition in rows:
            full_name = f"{schema_name}.{object_name}"
            text_body = (
                f"SQL Server VIEW: {full_name}\n"
                f"Database: {db_name}\n\n"
                f"-- Definition:\n{definition or '(definition not available)'}"
            )
            yield SourceResource(
                resource_id=f"sql:{self._server}.{db_name}.{full_name}",
                title=f"{db_name}: {full_name} (VIEW)",
                text=text_body,
                url="",
                last_updated=datetime.utcnow().isoformat(),
                metadata={
                    "db_name": db_name,
                    "schema_name": schema_name,
                    "object_name": full_name,
                    "object_type": "view",
                    "server": self._server,
                },
                entity_edges=_qualified_edges(
                    definition or "", full_name, known_objects, db_name
                ),
            )

    def _resources_from_tables(
        self,
        table_groups: dict[str, list],
        db_name: str,
        known_objects: dict[str, tuple[str, str]],
    ) -> Iterable[SourceResource]:
        for full_name, cols in table_groups.items():
            schema_name = cols[0][0]
            ddl = render_table_ddl(full_name, cols)
            text_body = (
                f"SQL Server TABLE: {full_name}\n"
                f"Database: {db_name}\n\n"
                f"-- Columns:\n{ddl}"
            )
            yield SourceResource(
                resource_id=f"sql:{self._server}.{db_name}.{full_name}",
                title=f"{db_name}: {full_name} (TABLE)",
                text=text_body,
                url="",
                last_updated=datetime.utcnow().isoformat(),
                metadata={
                    "db_name": db_name,
                    "schema_name": schema_name,
                    "object_name": full_name,
                    "object_type": "table",
                    "server": self._server,
                },
                # No FK/REFERENCES clause appears in the rendered DDL
                # above, so this always comes back empty today — kept
                # for consistency and in case the DDL rendering grows
                # FK info later. See core.sql_dependency_extraction's
                # module docstring.
                entity_edges=_qualified_edges(ddl, full_name, known_objects, db_name),
            )

    def _resources_from_triggers(
        self,
        rows: list[Any],
        db_name: str,
        since: Optional[str],
        known_objects: dict[str, tuple[str, str]],
    ) -> Iterable[SourceResource]:
        for (
            schema_name,
            trigger_name,
            table_name,
            definition,
            is_disabled,
            modify_date,
            trigger_events,
        ) in rows:
            modify_date_str = str(modify_date) if modify_date else ""
            if since and modify_date_str and modify_date_str < since:
                continue

            full_name = f"{schema_name}.{trigger_name}"
            parent_full_name = f"{schema_name}.{table_name}"
            status_str = "DISABLED" if is_disabled else "ENABLED"
            text_body = (
                f"SQL Server TRIGGER: {full_name}\n"
                f"Database: {db_name}\n"
                f"Table: {parent_full_name}\n"
                f"Events: {trigger_events or '(unknown)'}\n"
                f"Status: {status_str}\n\n"
                f"-- Definition:\n{definition or '(definition not available)'}"
            )
            yield SourceResource(
                resource_id=f"sql:{self._server}.{db_name}.{full_name}",
                title=f"{db_name}: {full_name} (TRIGGER on {parent_full_name})",
                text=text_body,
                url="",
                last_updated=modify_date_str or datetime.utcnow().isoformat(),
                metadata={
                    "db_name": db_name,
                    "schema_name": schema_name,
                    "object_name": full_name,
                    "object_type": "trigger",
                    "server": self._server,
                },
                entity_edges=_qualified_edges(
                    definition or "", full_name, known_objects, db_name
                ),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _qualified_edges(
    definition: str,
    full_name: str,
    known_objects: dict[str, tuple[str, str]],
    db_name: str,
) -> list[tuple[str, str, str]]:
    """
    ``find_references`` scoped to this ingest run: no-ops entirely when
    ``settings.ENABLE_SQL_DEPENDENCY_GRAPH`` is off, and prefixes both
    sides of each edge with ``db_name`` so subject/object match the
    ``{db_name}.{schema}.{object_name}`` form ``entity_edges`` expects
    (matching ``resource_identifier``'s tenancy scope for this source).
    """
    if not settings.ENABLE_SQL_DEPENDENCY_GRAPH:
        return []
    edges = find_references(definition, full_name.lower(), known_objects)
    return [(f"{db_name}.{s}", p, f"{db_name}.{o}") for s, p, o in edges]


def _extract_server(conn_str: str) -> str:
    """
    Pull a stable server identifier out of an ODBC connect string.

    Strips ``tcp:`` prefix and any port (``,1433``) so the resource_id is
    stable across deployments that use different transport hints.
    """
    match = re.search(r"SERVER\s*=\s*([^;]+)", conn_str, flags=re.IGNORECASE)
    if not match:
        return "server"
    raw = match.group(1).strip()
    raw = re.sub(r"^tcp:", "", raw, flags=re.IGNORECASE)
    raw = raw.split(",")[0]              # drop port
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", raw)
