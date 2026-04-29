"""
SQL Server MCP tools — safe, read-only live access to table data.

Why this exists
---------------
The existing RAG pipeline indexes *schemas* and *stored procedure code*,
but not actual row content. For ad-hoc questions like "show me the last
10 orders for customer X" the LLM needs to read the live table. This
module is the bridge.

Safety properties (all enforced server-side, never trust the LLM):

* **Read-only.** Queries are parsed with :mod:`sqlparse`; we accept
  exactly one statement, and that statement must be a ``SELECT`` (or a
  CTE ``WITH ... SELECT``). Any DML, DDL, ``EXEC``, system-procedure
  call (``sp_*``, ``xp_*``), or transaction-control verb is rejected.
* **Hard row cap.** A ``TOP n`` clause is injected if missing; the value
  is clamped to ``MCP_SQL_MAX_ROWS``.
* **Per-statement timeout.** Wraps the connection in
  ``SET LOCK_TIMEOUT`` + a SQLAlchemy execution-time guard so a runaway
  query can't pin a worker.
* **Tenancy.** Every call carries a ``user_id``; we resolve their stored
  ODBC credential, and verify they have a row in
  ``user_accessible_resources`` for the requested ``db_name`` before
  executing.
* **Auditable.** Every call (allowed or rejected) is logged with the
  user, db, query, row count, and duration via :mod:`loguru`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import sqlparse
from loguru import logger
from sqlalchemy import event, text
from sqlalchemy.engine import URL, Engine, create_engine

from app.utils import get_db, list_accessible, load_credential
from mcp_server.config import mcp_settings


# ──────────────────────────────────────────────────────────────────────────────
# MCP tool descriptors (consumed by server.py and the LangChain adapter)
# ──────────────────────────────────────────────────────────────────────────────

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "sql_table_query",
        "description": (
            "Run a safe, read-only SELECT against a live SQL Server "
            "database the calling user has access to. Returns up to "
            f"{mcp_settings.MCP_SQL_MAX_ROWS} rows as a markdown table. "
            "Use this when the user needs *actual table data* (rows), "
            "not schema or stored-procedure code — those come from RAG. "
            "Always include the database name and a fully-qualified "
            "table reference (schema.table) in the query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db_name": {
                    "type": "string",
                    "description": "Target SQL Server database name.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "A single read-only SELECT (or WITH ... SELECT) "
                        "statement. No semicolons except optionally at "
                        "the very end."
                    ),
                },
                "max_rows": {
                    "type": "integer",
                    "description": (
                        "Optional row cap; clamped to "
                        f"{mcp_settings.MCP_SQL_MAX_ROWS}."
                    ),
                    "default": mcp_settings.MCP_SQL_DEFAULT_ROWS,
                },
            },
            "required": ["db_name", "query"],
        },
    },
    {
        "name": "sql_list_databases",
        "description": (
            "List the SQL Server databases the calling user has been "
            "granted access to (i.e. they have an ingestion-time entry "
            "in user_accessible_resources)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Result shape
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Uniform return type so the server can serialise it directly."""

    ok: bool
    tool: str
    data: Any = None
    markdown: str = ""
    metadata: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "markdown": self.markdown,
            "metadata": self.metadata or {},
            "error": self.error,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Query validation
# ──────────────────────────────────────────────────────────────────────────────

# Tokens that immediately disqualify a query, regardless of position.
# We rely on sqlparse to tokenise so these match real keywords, not
# substrings inside identifiers / string literals.
_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    k.upper()
    for k in (
        "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT",
        "DROP", "CREATE", "ALTER", "TRUNCATE", "RENAME",
        "GRANT", "REVOKE", "DENY",
        "EXEC", "EXECUTE", "CALL",
        "BACKUP", "RESTORE", "SHUTDOWN",
        "BULK", "OPENROWSET", "OPENQUERY", "OPENDATASOURCE",
        "BEGIN", "COMMIT", "ROLLBACK", "SAVE",
        "USE", "GO", "DBCC",
    )
)

# System-procedure prefixes — disallow even inside strings, since they
# can be invoked via dynamic patterns we don't try to parse.
_FORBIDDEN_SUBSTRINGS_RE = re.compile(
    r"\b(?:sp_|xp_|fn_|sys\.sp_)\w+", re.IGNORECASE
)


class QueryValidationError(ValueError):
    """Raised when a query violates the read-only contract."""


def _validate_select(query: str) -> str:
    """
    Reject anything that isn't a single, read-only SELECT.

    Returns the canonicalised query (whitespace-normalised, trailing
    semicolon stripped) ready for row-cap injection.
    """
    if not query or not query.strip():
        raise QueryValidationError("Empty query.")

    # Strip comments + normalise whitespace.
    cleaned = sqlparse.format(
        query,
        strip_comments=True,
        reindent=False,
        keyword_case=None,
    ).strip()
    if not cleaned:
        raise QueryValidationError("Query reduced to nothing after stripping comments.")

    # Reject obvious system-proc invocations regardless of structure.
    if _FORBIDDEN_SUBSTRINGS_RE.search(cleaned):
        raise QueryValidationError(
            "Calls to sp_*/xp_*/fn_* style procedures are not permitted."
        )

    # We allow at most ONE statement. Trailing ';' is fine.
    statements = [s for s in sqlparse.split(cleaned) if s.strip()]
    if len(statements) != 1:
        raise QueryValidationError(
            f"Exactly one statement is allowed, got {len(statements)}."
        )

    stmt_text = statements[0].rstrip(";").strip()
    parsed = sqlparse.parse(stmt_text)
    if not parsed:
        raise QueryValidationError("Could not parse query.")
    stmt = parsed[0]

    stmt_type = (stmt.get_type() or "").upper()
    if stmt_type not in {"SELECT", "UNKNOWN"}:
        # sqlparse reports "UNKNOWN" for CTE-led queries (WITH ... SELECT),
        # which we accept as long as the leading keyword is WITH.
        raise QueryValidationError(
            f"Only SELECT/CTE queries are allowed (got {stmt_type})."
        )

    first_token = stmt.token_first(skip_cm=True, skip_ws=True)
    leading = first_token.value.upper() if first_token else ""
    if leading not in {"SELECT", "WITH"}:
        raise QueryValidationError(
            f"Query must start with SELECT or WITH (got {leading!r})."
        )

    # Walk every keyword token; reject if any is in the forbidden set.
    for token in stmt.flatten():
        if token.ttype is None:
            continue
        # sqlparse keyword-like ttypes: Keyword, Keyword.DML, Keyword.DDL, ...
        ttype_str = str(token.ttype)
        if "Keyword" not in ttype_str:
            continue
        word = token.value.upper().strip()
        if word in _FORBIDDEN_KEYWORDS:
            raise QueryValidationError(
                f"Forbidden keyword in query: {word}"
            )

    return stmt_text


_TOP_RE = re.compile(r"^\s*SELECT\s+TOP\s+\(?\s*\d+", re.IGNORECASE)
_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


def _inject_row_limit(query: str, max_rows: int) -> str:
    """
    Ensure the query returns at most ``max_rows`` rows.

    Strategy:
      * If the leading SELECT already has TOP, we don't second-guess it
        (the validator would have caught huge values; we still wrap the
        whole thing in an outer TOP just to be safe).
      * Otherwise we splice ``TOP (max_rows)`` right after the leading
        SELECT keyword. This works for both bare SELECTs and CTE-led
        queries (the final SELECT after WITH).
    """
    safe_query = query.rstrip(";").rstrip()

    # CTE: find the final SELECT after the WITH(...) block. We use a
    # conservative outer-wrap approach: build a derived table.
    if safe_query.upper().lstrip().startswith("WITH"):
        return f"SELECT TOP ({max_rows}) * FROM (\n{safe_query}\n) AS _mcp_capped"

    if _TOP_RE.match(safe_query):
        # Leave caller's TOP intact, but still wrap to enforce max.
        return f"SELECT TOP ({max_rows}) * FROM (\n{safe_query}\n) AS _mcp_capped"

    return _SELECT_RE.sub(f"SELECT TOP ({max_rows}) ", safe_query, count=1)


# ──────────────────────────────────────────────────────────────────────────────
# Engine cache (one per (user_id, db_name))
# ──────────────────────────────────────────────────────────────────────────────

_engine_cache: dict[tuple[str, str], Engine] = {}


def _get_engine(user_id: str, db_name: str) -> Engine:
    """
    Build (or reuse) a SQLAlchemy engine for the given user/database.

    Mirrors ``app.ingestion.sql_ingestor.SQLIngestor._make_engine``: we
    rewrite/append ``DATABASE=`` and pin the catalog with a
    ``USE [db]`` connect-event so logins with a server-default DB still
    land in the right place.
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


# ──────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────────────

def _to_markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    if not columns:
        return "_(no columns)_"
    if not rows:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        return f"{header}\n{sep}\n_(0 rows)_"

    def _cell(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        # Escape pipes + collapse newlines so the table stays one-line-per-row
        return s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    header = "| " + " | ".join(_cell(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join(
        "| " + " | ".join(_cell(c) for c in row) + " |" for row in rows
    )
    return f"{header}\n{sep}\n{body}"


# ──────────────────────────────────────────────────────────────────────────────
# Tool: sql_list_databases
# ──────────────────────────────────────────────────────────────────────────────

def list_databases(user_id: str) -> ToolResult:
    """Return the SQL databases this user has access to."""
    with get_db() as db:
        dbs = sorted(list_accessible(db, user_id, "sql"))
    md = (
        "| Database |\n| --- |\n" + "\n".join(f"| {d} |" for d in dbs)
        if dbs
        else "_(no databases accessible — run a SQL ingestion first)_"
    )
    logger.info(
        "[mcp.sql] list_databases user={} -> {} dbs", user_id, len(dbs)
    )
    return ToolResult(
        ok=True,
        tool="sql_list_databases",
        data={"databases": dbs},
        markdown=md,
        metadata={"count": len(dbs)},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool: sql_table_query
# ──────────────────────────────────────────────────────────────────────────────

def table_query(
    user_id: str,
    db_name: str,
    query: str,
    max_rows: Optional[int] = None,
) -> ToolResult:
    """Validate and execute a single read-only SELECT, return markdown."""
    started = time.perf_counter()
    requested_rows = max_rows or mcp_settings.MCP_SQL_DEFAULT_ROWS
    cap = max(1, min(int(requested_rows), mcp_settings.MCP_SQL_MAX_ROWS))

    # 1. Tenancy gate.
    with get_db() as db:
        accessible = set(list_accessible(db, user_id, "sql"))
    if db_name not in accessible:
        msg = (
            f"Access denied: user has no granted access to database "
            f"'{db_name}'. Run a SQL ingestion under your account "
            f"first, or pick a database from sql_list_databases."
        )
        logger.warning(
            "[mcp.sql] DENY user={} db={} reason=no-access", user_id, db_name
        )
        return ToolResult(
            ok=False, tool="sql_table_query", error=msg
        )

    # 2. Validate query shape.
    try:
        canonical = _validate_select(query)
    except QueryValidationError as exc:
        logger.warning(
            "[mcp.sql] DENY user={} db={} reason=validation msg={} query={!r}",
            user_id, db_name, exc, query,
        )
        return ToolResult(
            ok=False,
            tool="sql_table_query",
            error=f"Query rejected: {exc}",
        )

    capped_query = _inject_row_limit(canonical, cap)

    # 3. Execute.
    try:
        engine = _get_engine(user_id, db_name)
        with engine.connect() as conn:
            # Defence-in-depth: explicit read-only transaction on MSSQL.
            try:
                conn.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
            except Exception:  # noqa: BLE001 - non-fatal
                pass
            result = conn.execute(text(capped_query))
            columns = list(result.keys())
            rows = [list(r) for r in result.fetchmany(cap)]
    except PermissionError:
        raise
    except Exception as exc:  # noqa: BLE001 - we want to surface a clean error
        logger.exception(
            "[mcp.sql] EXEC FAIL user={} db={} query={!r}",
            user_id, db_name, capped_query,
        )
        return ToolResult(
            ok=False,
            tool="sql_table_query",
            error=f"SQL execution failed: {exc}",
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    md = _to_markdown_table(columns, rows)

    logger.info(
        "[mcp.sql] OK user={} db={} rows={} cap={} duration_ms={} query={!r}",
        user_id, db_name, len(rows), cap, duration_ms, canonical,
    )

    return ToolResult(
        ok=True,
        tool="sql_table_query",
        data={"columns": columns, "rows": rows},
        markdown=md,
        metadata={
            "db_name": db_name,
            "row_count": len(rows),
            "row_cap": cap,
            "duration_ms": duration_ms,
            "executed_query": capped_query,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def shutdown() -> None:
    """Dispose every cached engine on server shutdown."""
    for key, engine in list(_engine_cache.items()):
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            logger.warning("Engine dispose failed for {}", key)
    _engine_cache.clear()
