"""
Shared SQLAlchemy engine cache for live SQL Server MCP tools.

Both `mcp_server.tools.sql_tools` (ad-hoc row queries) and
`mcp_server.tools.sql_schema_tools` (object definitions / dependency
DMV queries) need a live connection to a specific (user, database).
This is the one place that builds and caches those engines, so both
modules share the same pool instead of `sql_schema_tools` reaching into
`sql_tools`'s private state.
"""

from __future__ import annotations

import re

from loguru import logger
from sqlalchemy import event
from sqlalchemy.engine import URL, Engine, create_engine

from app.utils import get_db, load_credential
from mcp_server.config import mcp_settings

_engine_cache: dict[tuple[str, str], Engine] = {}


def get_engine(user_id: str, db_name: str) -> Engine:
    """
    Build (or reuse) a SQLAlchemy engine for the given user/database.

    Mirrors `app.ingestion.sql_ingestor.SQLIngestor._make_engine`: we
    rewrite/append `DATABASE=` and pin the catalog with a `USE [db]`
    connect-event so logins with a server-default DB still land in the
    right place.
    """
    cache_key = (user_id, db_name)
    if cache_key in _engine_cache:
        return _engine_cache[cache_key]

    with get_db() as db:
        conn_str = load_credential(db, user_id, "sql", "conn_str")
    if not conn_str:
        raise PermissionError(
            "User has no stored SQL Server connection string."
        )

    if "DATABASE=" in conn_str.upper():
        conn_str = re.sub(
            r"DATABASE=[^;]+",
            f"DATABASE={db_name}",
            conn_str,
            flags=re.IGNORECASE,
        )
    else:
        conn_str = conn_str.rstrip(";") + f";DATABASE={db_name}"

    url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
        # Per-statement timeout (seconds). The pyodbc layer respects this.
        connect_args={"timeout": mcp_settings.MCP_SQL_QUERY_TIMEOUT_SECONDS},
    )

    @event.listens_for(engine, "connect")
    def _force_db(dbapi_conn, _conn_record):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute(f"USE [{db_name}]")
        finally:
            cursor.close()

    _engine_cache[cache_key] = engine
    return engine


def shutdown() -> None:
    """Dispose every cached engine on server shutdown."""
    for key, engine in list(_engine_cache.items()):
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            logger.warning("Engine dispose failed for {}", key)
    _engine_cache.clear()
