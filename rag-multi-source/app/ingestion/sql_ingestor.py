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

from app.ingestion.base import BaseIngestor, SourceResource


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
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL

        conn_str = self._base_conn_str
        if db_name and db_name != "all":
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
        return create_engine(url, fast_executemany=True)

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
            yield from self._iter_routines(engine, db_name, since)
            yield from self._iter_views(engine, db_name)
            yield from self._iter_tables(engine, db_name)
        finally:
            engine.dispose()

    def _iter_routines(
        self, engine, db_name: str, since: Optional[str]
    ) -> Iterable[SourceResource]:
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(_SQL_ROUTINES)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Routines failed for {}: {}", db_name, exc)
            return

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
            )

    def _iter_views(self, engine, db_name: str) -> Iterable[SourceResource]:
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(_SQL_VIEWS)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Views failed for {}: {}", db_name, exc)
            return

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
            )

    def _iter_tables(self, engine, db_name: str) -> Iterable[SourceResource]:
        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(_SQL_TABLE_COLS)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Tables failed for {}: {}", db_name, exc)
            return

        tables: dict[str, list] = {}
        for row in rows:
            full_name = f"{row[0]}.{row[1]}"
            tables.setdefault(full_name, []).append(row)

        for full_name, cols in tables.items():
            schema_name = cols[0][0]
            ddl_lines: list[str] = []
            for col in cols:
                col_name = col[2]
                data_type = col[3]
                max_len = col[4]
                nullable = col[5]
                default = col[6]
                type_str = data_type + (f"({max_len})" if max_len else "")
                null_str = "NULL" if nullable == "YES" else "NOT NULL"
                default_str = f" DEFAULT {default}" if default else ""
                ddl_lines.append(f"  {col_name} {type_str} {null_str}{default_str}")

            text_body = (
                f"SQL Server TABLE: {full_name}\n"
                f"Database: {db_name}\n\n"
                f"-- Columns:\nCREATE TABLE {full_name} (\n"
                + ",\n".join(ddl_lines)
                + "\n)"
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
            )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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
