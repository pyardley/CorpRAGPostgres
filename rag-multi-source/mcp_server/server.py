"""
FastAPI MCP server.

Exposes the SQL tools defined in :mod:`mcp_server.tools.sql_tools` over
HTTP/JSON. Endpoints:

* ``GET  /healthz``                    — liveness probe (no auth)
* ``GET  /mcp/tools``                  — list available tool descriptors
* ``POST /mcp/tools/sql_table_query``  — run a read-only SELECT
* ``POST /mcp/tools/sql_list_databases`` — list user-accessible DBs

All ``/mcp/*`` endpoints require the ``X-MCP-Token`` header to match
:attr:`mcp_server.config.mcp_settings.MCP_SHARED_TOKEN`.

Run standalone with::

    uvicorn mcp_server.server:app --host 127.0.0.1 --port 8765

…or let the Streamlit app supervise it via :mod:`app.mcp_manager`.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from mcp_server.config import generate_token, mcp_settings
from mcp_server.tools import sql_tools


# Shared on-disk token location — kept in sync with app/mcp_manager.py so
# the Streamlit in-process client can always discover the live token,
# regardless of how the server was launched (manager-spawned, standalone
# uvicorn, systemd, etc.).
_TOKEN_FILE = Path(".streamlit") / "mcp" / "token"


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    level=mcp_settings.MCP_LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<7}</level> | "
        "<cyan>mcp</cyan> | <level>{message}</level>"
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

# If no token was set in the environment, generate one for the lifetime
# of this process. ``app.mcp_manager`` writes/reads it via a runtime file
# so the in-process client picks it up without env-var coordination.
_RUNTIME_TOKEN: str = mcp_settings.MCP_SHARED_TOKEN or generate_token()


def get_runtime_token() -> str:
    return _RUNTIME_TOKEN


async def require_token(
    x_mcp_token: Optional[str] = Header(default=None, alias="X-MCP-Token"),
) -> None:
    if not x_mcp_token or x_mcp_token != _RUNTIME_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing MCP token.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class SQLTableQueryRequest(BaseModel):
    user_id: str = Field(..., description="Calling user's UUID.")
    db_name: str = Field(..., description="Target SQL Server database name.")
    query: str = Field(..., description="A single read-only SELECT.")
    max_rows: Optional[int] = Field(
        default=None,
        description="Optional row cap; clamped server-side.",
    )


class SQLListDatabasesRequest(BaseModel):
    user_id: str = Field(..., description="Calling user's UUID.")


class ToolEnvelope(BaseModel):
    """Uniform response envelope returned by every tool endpoint."""

    ok: bool
    tool: str
    data: Any | None = None
    markdown: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

def _publish_token(token: str) -> None:
    """
    Write the live token to ``.streamlit/mcp/token`` so the Streamlit
    in-process client (and any peer process in the same project root)
    can pick it up without out-of-band coordination.

    We always overwrite, even when ``MCP_SHARED_TOKEN`` was set via env —
    a stable env value just means the file content matches anyway, and
    keeping a single source of truth simplifies the manager's fast-path
    logic ("server is up → read token file → connect").
    """
    try:
        _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_FILE.write_text(token, encoding="utf-8")
        logger.info("MCP token published to {}", _TOKEN_FILE)
    except Exception as exc:  # noqa: BLE001 - non-fatal
        logger.warning("Could not publish MCP token to {}: {}", _TOKEN_FILE, exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "MCP server starting on {}:{} (sql_tools enabled)",
        mcp_settings.MCP_HOST,
        mcp_settings.MCP_PORT,
    )
    _publish_token(_RUNTIME_TOKEN)
    try:
        yield
    finally:
        logger.info("MCP server shutting down — disposing SQL engines.")
        sql_tools.shutdown()


app = FastAPI(
    title="CorporateRAG MCP Server",
    version="0.1.0",
    description=(
        "Read-only Model-Context-Protocol-style tools for the "
        "CorporateRAG hybrid RAG+MCP chat agent."
    ),
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/mcp/tools", dependencies=[Depends(require_token)])
async def list_tools() -> dict[str, Any]:
    """Return all registered MCP tool descriptors."""
    return {"tools": sql_tools.TOOL_SPECS}


@app.post(
    "/mcp/tools/sql_list_databases",
    response_model=ToolEnvelope,
    dependencies=[Depends(require_token)],
)
async def call_sql_list_databases(req: SQLListDatabasesRequest) -> ToolEnvelope:
    result = sql_tools.list_databases(user_id=req.user_id)
    return ToolEnvelope(**result.to_dict())


@app.post(
    "/mcp/tools/sql_table_query",
    response_model=ToolEnvelope,
    dependencies=[Depends(require_token)],
)
async def call_sql_table_query(req: SQLTableQueryRequest) -> ToolEnvelope:
    result = sql_tools.table_query(
        user_id=req.user_id,
        db_name=req.db_name,
        query=req.query,
        max_rows=req.max_rows,
    )
    return ToolEnvelope(**result.to_dict())


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint helper
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the server with uvicorn (used by ``python -m mcp_server.server``)."""
    import uvicorn

    uvicorn.run(
        "mcp_server.server:app",
        host=mcp_settings.MCP_HOST,
        port=mcp_settings.MCP_PORT,
        log_level=mcp_settings.MCP_LOG_LEVEL.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
