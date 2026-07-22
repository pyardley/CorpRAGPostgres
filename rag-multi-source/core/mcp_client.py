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

from app.config import settings
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

    def entity_graph_query(
        self,
        user_id: str,
        entity: str,
        max_results: Optional[int] = None,
    ) -> MCPToolResult:
        return self._post(
            "/mcp/tools/entity_graph_query",
            {
                "user_id": user_id,
                "entity": entity,
                "max_results": max_results,
            },
        )

    def sql_dependency_graph(
        self,
        user_id: str,
        object_name: str,
        direction: str = "both",
        max_hops: Optional[int] = None,
    ) -> MCPToolResult:
        return self._post(
            "/mcp/tools/sql_dependency_graph",
            {
                "user_id": user_id,
                "object_name": object_name,
                "direction": direction,
                "max_hops": max_hops,
            },
        )

    def sql_object_definition(
        self,
        user_id: str,
        db_name: str,
        object_name: str,
    ) -> MCPToolResult:
        return self._post(
            "/mcp/tools/sql_object_definition",
            {"user_id": user_id, "db_name": db_name, "object_name": object_name},
        )

    def sql_object_dependencies(
        self,
        user_id: str,
        db_name: str,
        object_name: str,
        direction: str = "both",
        max_hops: Optional[int] = None,
    ) -> MCPToolResult:
        return self._post(
            "/mcp/tools/sql_object_dependencies",
            {
                "user_id": user_id,
                "db_name": db_name,
                "object_name": object_name,
                "direction": direction,
                "max_hops": max_hops,
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

    tools = [
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

    # Only bound when settings.ENABLE_ENTITY_GRAPH is set — checked
    # internally here rather than requiring every caller to pass a flag,
    # matching how core.reranker.rerank() / core.query_rewrite.rewrite_query()
    # check their own settings.
    if settings.ENABLE_ENTITY_GRAPH:

        class EntityGraphQueryArgs(BaseModel):
            entity: str = Field(
                ...,
                description=(
                    "A resource id (e.g. 'jira:PROJ-123'), a person's "
                    "name/accountId/email, or a repo scope (e.g. "
                    "'owner/repo@main') to search for."
                ),
            )
            max_results: Optional[int] = Field(
                default=None, description="Max edges to return (default 20)."
            )

        def _run_entity_graph_query(entity: str, max_results: Optional[int] = None) -> str:
            result = client.entity_graph_query(
                user_id=user_id, entity=entity, max_results=max_results
            )
            return result.markdown if result.ok else f"ERROR: {result.error}"

        tools.append(
            StructuredTool.from_function(
                func=_run_entity_graph_query,
                name="entity_graph_query",
                description=(
                    "Search the entity relationship graph for edges "
                    "involving a person, ticket, or repository — "
                    "assigned_to, reported_by (Jira), modified_by (Git), "
                    "plus any LLM-extracted relationships. Use this for "
                    "relationship questions plain text search can't "
                    "answer, e.g. 'who else worked on tickets like this "
                    "one' or 'which repos does alice maintain'."
                ),
                args_schema=EntityGraphQueryArgs,
            )
        )

    # Bound whenever the static SQL dependency graph is populated at
    # ingestion time — checked internally, same precedent as
    # ENABLE_ENTITY_GRAPH above.
    if settings.ENABLE_SQL_DEPENDENCY_GRAPH:

        class SqlDependencyGraphArgs(BaseModel):
            object_name: str = Field(
                ...,
                description=(
                    "A table/view/procedure/function/trigger name to "
                    "start from, e.g. "
                    "'dbo.usp_BuildReport_CustomerChurnRisk' or just "
                    "'usp_BuildReport_CustomerChurnRisk'."
                ),
            )
            direction: str = Field(
                default="both",
                description=(
                    "'downstream' = what depends on this object (impact "
                    "analysis / blast radius). 'upstream' = what this "
                    "object depends on (lineage / trace to source). "
                    "'both' = both directions."
                ),
            )
            max_hops: Optional[int] = Field(
                default=None, description="Max traversal hops (default 3, max 5)."
            )

        def _run_sql_dependency_graph(
            object_name: str, direction: str = "both", max_hops: Optional[int] = None
        ) -> str:
            result = client.sql_dependency_graph(
                user_id=user_id,
                object_name=object_name,
                direction=direction,
                max_hops=max_hops,
            )
            return result.markdown if result.ok else f"ERROR: {result.error}"

        tools.append(
            StructuredTool.from_function(
                func=_run_sql_dependency_graph,
                name="sql_dependency_graph",
                description=(
                    "Traverse the static SQL object dependency graph "
                    "built at ingestion time from table/view/procedure/"
                    "function/trigger definitions. Use "
                    "direction='downstream' for 'what breaks if I "
                    "change/drop X' (blast radius), and "
                    "direction='upstream' for 'trace X back to its "
                    "source tables' (lineage). Returns hop-labeled "
                    "(subject, predicate, object) edges. This is a "
                    "STATIC, text-derived graph — an empty result is "
                    "inconclusive, not proof of no dependency; it can't "
                    "see dynamic SQL."
                ),
                args_schema=SqlDependencyGraphArgs,
            )
        )

    # Live-catalog counterpart to the static graph above — needs VIEW
    # DEFINITION / VIEW DATABASE STATE permission on the stored SQL
    # login, hence its own flag so an operator without those can
    # disable it cleanly rather than seeing repeated tool-call failures.
    if settings.ENABLE_SQL_DEPENDENCY_MCP_TOOLS:

        class SqlObjectDefinitionArgs(BaseModel):
            db_name: str = Field(..., description="Target SQL Server database name.")
            object_name: str = Field(
                ...,
                description=(
                    "A table/view/procedure/function/trigger name, e.g. "
                    "'dbo.fn_NetLineAmount' or just 'fn_NetLineAmount'."
                ),
            )

        class SqlObjectDependenciesArgs(BaseModel):
            db_name: str = Field(..., description="Target SQL Server database name.")
            object_name: str = Field(
                ...,
                description=(
                    "A table/view/procedure/function/trigger name to "
                    "start from, e.g. 'dbo.usp_BuildReport_X' or just "
                    "'usp_BuildReport_X'."
                ),
            )
            direction: str = Field(
                default="both",
                description=(
                    "'downstream' = what depends on this object (impact "
                    "analysis / blast radius). 'upstream' = what this "
                    "object depends on (lineage / trace to source). "
                    "'both' = both directions."
                ),
            )
            max_hops: Optional[int] = Field(
                default=None, description="Max traversal hops (default 3)."
            )

        def _run_sql_object_definition(db_name: str, object_name: str) -> str:
            result = client.sql_object_definition(
                user_id=user_id, db_name=db_name, object_name=object_name
            )
            return result.markdown if result.ok else f"ERROR: {result.error}"

        def _run_sql_object_dependencies(
            db_name: str,
            object_name: str,
            direction: str = "both",
            max_hops: Optional[int] = None,
        ) -> str:
            result = client.sql_object_dependencies(
                user_id=user_id,
                db_name=db_name,
                object_name=object_name,
                direction=direction,
                max_hops=max_hops,
            )
            return result.markdown if result.ok else f"ERROR: {result.error}"

        tools.append(
            StructuredTool.from_function(
                func=_run_sql_object_definition,
                name="sql_object_definition",
                description=(
                    "Fetch the COMPLETE, unchunked definition of a live "
                    "SQL Server table/view/procedure/function/trigger — "
                    "straight from the database, not the (possibly "
                    "fragmented or stale) ingested copy. Use this "
                    "whenever you need to open up a called function/"
                    "procedure's actual logic rather than naming it."
                ),
                args_schema=SqlObjectDefinitionArgs,
            )
        )
        tools.append(
            StructuredTool.from_function(
                func=_run_sql_object_dependencies,
                name="sql_object_dependencies",
                description=(
                    "Traverse LIVE SQL Server dependency metadata "
                    "(sys.dm_sql_referenced_entities / "
                    "sys.dm_sql_referencing_entities), multiple hops "
                    "deep. More authoritative than sql_dependency_graph "
                    "since it reflects the database's current state, "
                    "not the last ingestion. Use direction='downstream' "
                    "for 'what breaks if I change X', direction='upstream' "
                    "for 'trace X back to source'."
                ),
                args_schema=SqlObjectDependenciesArgs,
            )
        )

    return tools
