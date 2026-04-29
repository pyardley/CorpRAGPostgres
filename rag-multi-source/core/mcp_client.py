"""
Thin HTTP client + LangChain tool adapters for the MCP server.

The two halves are deliberately decoupled:

* :class:`MCPClient` is a plain ``httpx`` wrapper — call it from
  anywhere (tests, scripts, the chat layer) without any LangChain
  dependency.
* :func:`build_mcp_tools` returns LangChain ``StructuredTool`` instances
  bound to a specific ``user_id`` so the LLM can invoke them safely
  through ``llm.bind_tools(...)`` /agent loops.

The shape mirrors what ``langchain-mcp-adapters`` produces, so we can
swap the transport (HTTP → MCP stdio/SSE) later without touching the
agent code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

from mcp_server.config import mcp_settings


# Same path the manager + server write to; we re-read it on 401 to
# transparently recover from a token rotation that happened in another
# Streamlit process / a manual server restart.
_TOKEN_FILE = Path(".streamlit") / "mcp" / "token"


def _read_disk_token() -> Optional[str]:
    try:
        return _TOKEN_FILE.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class MCPToolResult:
    """Mirror of ``mcp_server.tools.sql_tools.ToolResult`` on the client side."""

    ok: bool
    tool: str
    data: Any
    markdown: str
    metadata: dict[str, Any]
    error: Optional[str]

    @classmethod
    def from_envelope(cls, payload: dict[str, Any]) -> "MCPToolResult":
        return cls(
            ok=bool(payload.get("ok")),
            tool=str(payload.get("tool", "")),
            data=payload.get("data"),
            markdown=str(payload.get("markdown", "")),
            metadata=dict(payload.get("metadata") or {}),
            error=payload.get("error"),
        )


class MCPClient:
    """HTTP client for the CorporateRAG MCP server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or f"http://{mcp_settings.MCP_HOST}:{mcp_settings.MCP_PORT}"
        ).rstrip("/")
        self.token = token  # may be None until the manager primes it
        self._client = httpx.Client(timeout=timeout)

    # ── Plumbing ─────────────────────────────────────────────────────────────

    def set_token(self, token: str) -> None:
        self.token = token

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError(
                "MCP client has no token. The MCP manager should set one "
                "via MCPClient.set_token() during app startup."
            )
        return {"X-MCP-Token": self.token, "Content-Type": "application/json"}

    def _refresh_token_from_disk(self) -> bool:
        """Reload the shared token from disk; True iff a new value was picked up."""
        disk_token = _read_disk_token()
        if disk_token and disk_token != self.token:
            logger.info(
                "MCP client refreshing token from {} (was stale)", _TOKEN_FILE
            )
            self.token = disk_token
            return True
        return False

    def _post(self, path: str, json: dict[str, Any]) -> MCPToolResult:
        url = f"{self.base_url}{path}"
        tool_name = path.rsplit("/", 1)[-1]

        try:
            r = self._client.post(url, json=json, headers=self._headers())
        except httpx.HTTPError as exc:
            logger.error("MCP request failed for {}: {}", path, exc)
            return MCPToolResult(
                ok=False,
                tool=tool_name,
                data=None,
                markdown="",
                metadata={},
                error=f"MCP server unreachable: {exc}",
            )

        # Self-heal on 401: the MCP server may have been restarted (and
        # rotated its token) in another process. Re-read the on-disk
        # shared token and retry exactly once.
        if r.status_code == 401 and self._refresh_token_from_disk():
            try:
                r = self._client.post(url, json=json, headers=self._headers())
            except httpx.HTTPError as exc:
                logger.error("MCP retry failed for {}: {}", path, exc)
                return MCPToolResult(
                    ok=False,
                    tool=tool_name,
                    data=None,
                    markdown="",
                    metadata={},
                    error=f"MCP server unreachable on retry: {exc}",
                )

        if r.status_code != 200:
            return MCPToolResult(
                ok=False,
                tool=tool_name,
                data=None,
                markdown="",
                metadata={"status_code": r.status_code},
                error=f"MCP server returned {r.status_code}: {r.text[:200]}",
            )
        return MCPToolResult.from_envelope(r.json())

    # ── Health / discovery ───────────────────────────────────────────────────

    def healthz(self) -> bool:
        """True iff the server responds 200 to /healthz."""
        try:
            r = self._client.get(f"{self.base_url}/healthz", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def list_tools(self) -> list[dict[str, Any]]:
        try:
            r = self._client.get(
                f"{self.base_url}/mcp/tools",
                headers=self._headers(),
                timeout=5.0,
            )
            r.raise_for_status()
            return r.json().get("tools", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list MCP tools: {}", exc)
            return []

    # ── Typed tool calls ─────────────────────────────────────────────────────

    def sql_list_databases(self, user_id: str) -> MCPToolResult:
        return self._post(
            "/mcp/tools/sql_list_databases", {"user_id": user_id}
        )

    def sql_table_query(
        self,
        user_id: str,
        db_name: str,
        query: str,
        max_rows: Optional[int] = None,
    ) -> MCPToolResult:
        return self._post(
            "/mcp/tools/sql_table_query",
            {
                "user_id": user_id,
                "db_name": db_name,
                "query": query,
                "max_rows": max_rows,
            },
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Singleton client
# ──────────────────────────────────────────────────────────────────────────────

_client_singleton: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Process-wide MCPClient. The token must have been set by the manager."""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = MCPClient()
    return _client_singleton


# ──────────────────────────────────────────────────────────────────────────────
# LangChain tool adapter
# ──────────────────────────────────────────────────────────────────────────────

def build_mcp_tools(user_id: str) -> list[Any]:
    """
    Return LangChain ``StructuredTool`` objects bound to ``user_id``.

    The LLM never sees the user_id — it's curried in here so the tool
    descriptions stay clean and the LLM can't accidentally (or
    intentionally) impersonate someone else.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    client = get_mcp_client()

    class SqlTableQueryArgs(BaseModel):
        db_name: str = Field(
            ..., description="Target SQL Server database name."
        )
        query: str = Field(
            ...,
            description=(
                "A single read-only SELECT (or WITH ... SELECT). "
                "Reference tables as schema.table_name. No DDL/DML."
            ),
        )
        max_rows: Optional[int] = Field(
            default=None,
            description=(
                "Optional row cap (default 50, max 100 — clamped server-side)."
            ),
        )

    class SqlListDatabasesArgs(BaseModel):
        pass

    def _run_sql_query(
        db_name: str, query: str, max_rows: Optional[int] = None
    ) -> str:
        result = client.sql_table_query(
            user_id=user_id,
            db_name=db_name,
            query=query,
            max_rows=max_rows,
        )
        if not result.ok:
            return f"ERROR: {result.error}"
        meta = result.metadata
        header = (
            f"_(Live data via MCP from `{meta.get('db_name', db_name)}` — "
            f"{meta.get('row_count', '?')} row(s), "
            f"{meta.get('duration_ms', '?')} ms)_\n\n"
        )
        return header + result.markdown

    def _run_list_dbs() -> str:
        result = client.sql_list_databases(user_id=user_id)
        return result.markdown if result.ok else f"ERROR: {result.error}"

    return [
        StructuredTool.from_function(
            func=_run_sql_query,
            name="sql_table_query",
            description=(
                "Run a safe, read-only SELECT against a live SQL Server "
                "database the user has access to. Returns up to 100 "
                "rows as a markdown table. Use this when the user "
                "wants ACTUAL TABLE DATA / ROW VALUES — not schema or "
                "stored-procedure code (those come from RAG context). "
                "Always use schema.table_name and add a sensible WHERE "
                "to keep results small."
            ),
            args_schema=SqlTableQueryArgs,
        ),
        StructuredTool.from_function(
            func=_run_list_dbs,
            name="sql_list_databases",
            description=(
                "List the SQL Server databases the user has been granted "
                "access to. Call this first if the user hasn't named a "
                "specific database."
            ),
            args_schema=SqlListDatabasesArgs,
        ),
    ]
