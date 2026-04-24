"""SQL Server ingestor – extracts stored procedures, functions, views, and table schemas."""

from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.documents import Document
from loguru import logger

from app.ingestion.base import BaseIngestor


# ── SQL to extract object definitions ─────────────────────────────────────────

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
    v.VIEW_DEFINITION  AS definition,
    NULL               AS last_altered
FROM INFORMATION_SCHEMA.VIEWS v
WHERE v.VIEW_DEFINITION IS NOT NULL
ORDER BY v.TABLE_SCHEMA, v.TABLE_NAME
"""

_SQL_TABLES = """
SELECT
    c.TABLE_SCHEMA                    AS schema_name,
    c.TABLE_NAME                      AS object_name,
    'TABLE'                           AS object_type,
    c.COLUMN_NAME                     AS column_name,
    c.DATA_TYPE                       AS data_type,
    c.CHARACTER_MAXIMUM_LENGTH        AS max_length,
    c.IS_NULLABLE                     AS is_nullable,
    c.COLUMN_DEFAULT                  AS column_default,
    t.TABLE_TYPE                      AS table_type
FROM INFORMATION_SCHEMA.COLUMNS c
JOIN INFORMATION_SCHEMA.TABLES t
  ON t.TABLE_SCHEMA = c.TABLE_SCHEMA AND t.TABLE_NAME = c.TABLE_NAME
WHERE t.TABLE_TYPE = 'BASE TABLE'
ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
"""


class SQLIngestor(BaseIngestor):
    SOURCE = "sql"

    def __init__(self, user_id: str, credentials: dict[str, str]) -> None:
        super().__init__(user_id, credentials)
        # conn_str may be set at the user level; db_name scopes it further
        self._base_conn_str = credentials.get("conn_str", "")
        if not self._base_conn_str:
            raise ValueError("SQL credentials incomplete: need conn_str.")

    def _make_engine(self, db_name: Optional[str] = None):
        import re
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL

        conn_str = self._base_conn_str
        if db_name and db_name != "all":
            if "DATABASE=" in conn_str.upper():
                conn_str = re.sub(r"DATABASE=[^;]+", f"DATABASE={db_name}", conn_str, flags=re.IGNORECASE)
            else:
                conn_str += f";DATABASE={db_name}"

        # Use URL.create so SQLAlchemy handles percent-encoding of the ODBC
        # connect string — passing it raw in an f-string breaks on the braces
        # and spaces in e.g. DRIVER={ODBC Driver 18 for SQL Server}.
        url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
        return create_engine(url, fast_executemany=True)

    # ── Abstract implementations ──────────────────────────────────────────────

    def list_scopes(self) -> list[dict[str, str]]:
        """Return databases accessible via the connection string."""
        try:
            engine = self._make_engine()
            from sqlalchemy import text
            with engine.connect() as conn:
                rows = conn.execute(text("SELECT name FROM sys.databases WHERE state_desc = 'ONLINE' ORDER BY name")).fetchall()
            return [{"key": r[0], "name": r[0]} for r in rows]
        except Exception as exc:
            logger.warning("[sql] Could not list databases: {}", exc)
            return []

    def load_documents(
        self, scope_key: str, since: Optional[str] = None
    ) -> list[Document]:
        if scope_key == "all":
            db_names = [s["key"] for s in self.list_scopes()]
        else:
            db_names = [scope_key]

        docs: list[Document] = []
        for db_name in db_names:
            docs.extend(self._load_database(db_name, since))
        return docs

    # ── Per-database loading ──────────────────────────────────────────────────

    def _load_database(self, db_name: str, since: Optional[str]) -> list[Document]:
        docs: list[Document] = []
        try:
            engine = self._make_engine(db_name)
            docs.extend(self._load_routines(engine, db_name, since))
            docs.extend(self._load_views(engine, db_name))
            docs.extend(self._load_tables(engine, db_name))
        except Exception as exc:
            logger.error("[sql] Failed to load database {}: {}", db_name, exc)
        return docs

    def _load_routines(self, engine, db_name: str, since: Optional[str]) -> list[Document]:
        from sqlalchemy import text
        docs: list[Document] = []

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(_SQL_ROUTINES)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Could not fetch routines from {}: {}", db_name, exc)
            return docs

        for row in rows:
            schema_name, object_name, object_type, definition, last_altered = row
            last_altered_str = str(last_altered) if last_altered else ""

            if since and last_altered_str and last_altered_str <= since:
                continue

            full_name = f"{schema_name}.{object_name}"
            text_content = (
                f"SQL Server {object_type}: {full_name}\n"
                f"Database: {db_name}\n\n"
                f"-- Definition:\n{definition or '(definition not available)'}"
            )

            metadata: dict[str, Any] = {
                "user_id": self.user_id,
                "source": "sql",
                "db_name": db_name,
                "schema_name": schema_name,
                "object_name": full_name,
                "object_type": object_type.lower(),
                "title": f"{db_name}: {full_name} ({object_type})",
                "url": "",
                "last_updated": last_altered_str,
            }
            docs.append(Document(page_content=text_content, metadata=metadata))

        return docs

    def _load_views(self, engine, db_name: str) -> list[Document]:
        from sqlalchemy import text
        docs: list[Document] = []

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(_SQL_VIEWS)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Could not fetch views from {}: {}", db_name, exc)
            return docs

        for row in rows:
            schema_name, object_name, object_type, definition, _ = row
            full_name = f"{schema_name}.{object_name}"
            text_content = (
                f"SQL Server VIEW: {full_name}\n"
                f"Database: {db_name}\n\n"
                f"-- Definition:\n{definition or '(definition not available)'}"
            )
            metadata: dict[str, Any] = {
                "user_id": self.user_id,
                "source": "sql",
                "db_name": db_name,
                "schema_name": schema_name,
                "object_name": full_name,
                "object_type": "view",
                "title": f"{db_name}: {full_name} (VIEW)",
                "url": "",
                "last_updated": "",
            }
            docs.append(Document(page_content=text_content, metadata=metadata))

        return docs

    def _load_tables(self, engine, db_name: str) -> list[Document]:
        from sqlalchemy import text
        docs: list[Document] = []

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(_SQL_TABLES)).fetchall()
        except Exception as exc:
            logger.warning("[sql] Could not fetch tables from {}: {}", db_name, exc)
            return docs

        # Group columns by table
        tables: dict[str, list] = {}
        for row in rows:
            schema_name, table_name = row[0], row[1]
            key = f"{schema_name}.{table_name}"
            tables.setdefault(key, []).append(row)

        for full_name, cols in tables.items():
            schema_name = cols[0][0]
            table_name = cols[0][1]
            col_lines = []
            for col in cols:
                col_name = col[3]
                data_type = col[4]
                max_len = col[5]
                nullable = col[6]
                default = col[7]
                type_str = data_type
                if max_len:
                    type_str += f"({max_len})"
                null_str = "NULL" if nullable == "YES" else "NOT NULL"
                default_str = f" DEFAULT {default}" if default else ""
                col_lines.append(f"  {col_name} {type_str} {null_str}{default_str}")

            text_content = (
                f"SQL Server TABLE: {full_name}\n"
                f"Database: {db_name}\n\n"
                f"-- Columns:\nCREATE TABLE {full_name} (\n"
                + ",\n".join(col_lines)
                + "\n)"
            )

            metadata: dict[str, Any] = {
                "user_id": self.user_id,
                "source": "sql",
                "db_name": db_name,
                "schema_name": schema_name,
                "object_name": full_name,
                "object_type": "table",
                "title": f"{db_name}: {full_name} (TABLE)",
                "url": "",
                "last_updated": "",
            }
            docs.append(Document(page_content=text_content, metadata=metadata))

        return docs

    def _scope_filter(self, scope_key: str) -> dict:
        filt: dict = {"source": "sql"}
        if scope_key != "all":
            filt["db_name"] = scope_key
        return filt
