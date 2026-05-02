"""
MCP server configuration.

Reads from the same ``.env`` as the Streamlit app so operators only have
one config surface. Every setting has a sensible default suitable for
local single-host deployment.
"""

from __future__ import annotations

import secrets
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    """Server-side configuration for the MCP service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Network ──────────────────────────────────────────────────────────────
    # Bind loopback by default; the MCP server is internal infrastructure,
    # not a public API. Override MCP_HOST=0.0.0.0 only if you intend to
    # expose it (and put it behind a real reverse proxy + auth).
    MCP_HOST: str = "127.0.0.1"
    MCP_PORT: int = 8765

    # ── Auth ─────────────────────────────────────────────────────────────────
    # Shared secret presented via the ``X-MCP-Token`` header. If unset, the
    # manager auto-generates one per process and writes it to a runtime
    # token file so the in-process client can read it back. Set
    # MCP_SHARED_TOKEN explicitly when you want a stable token across
    # restarts (e.g. when running uvicorn separately).
    MCP_SHARED_TOKEN: Optional[str] = None

    # ── Safety limits ────────────────────────────────────────────────────────
    # Hard ceiling on rows returned by any SQL tool call regardless of what
    # the caller asks for. The default of 100 is a deliberately tight
    # production value.
    MCP_SQL_MAX_ROWS: int = 100
    MCP_SQL_DEFAULT_ROWS: int = 50
    MCP_SQL_QUERY_TIMEOUT_SECONDS: int = 15

    # ── Logging ──────────────────────────────────────────────────────────────
    MCP_LOG_LEVEL: str = "INFO"


mcp_settings = MCPSettings()


def generate_token() -> str:
    """Cryptographically random shared secret for the MCP service."""
    return secrets.token_urlsafe(32)
