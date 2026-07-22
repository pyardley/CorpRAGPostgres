"""
Live SQL Server schema/dependency MCP tools.

Why this exists
----------------
`mcp_server.tools.entity_graph_tools.traverse_sql_dependencies` walks a
*static* dependency graph built once at ingestion time by regex-scanning
object definitions (`core.sql_dependency_extraction`). It proves an
edge *exists*, but never returns the referenced object's actual body —
so a lineage-tracing answer can confirm "this proc calls
fn_NetLineAmount" without ever seeing what that function computes.

This module closes that gap with two tools that talk to the *live*
database instead of the ingested/chunked copy:

* `sql_object_definition` — fetch a complete, unchunked object body on
  demand (`sys.sql_modules` for procs/functions/views/triggers, or a
  reconstructed `CREATE TABLE` for tables via `core.sql_ddl`). This is
  what lets the LLM actually open up a called function's formula
  instead of naming it.
* `sql_object_dependencies` — multi-hop traversal via SQL Server's own
  dependency DMVs (`sys.dm_sql_referenced_entities` /
  `sys.dm_sql_referencing_entities`), which are exhaustive over
  schema-bound references but — unlike the static graph — always
  reflect the database's *current* state, not what was true as of the
  last ingestion. Supplemented by the same
  `core.sql_dependency_extraction.find_references` text search the
  static graph uses, as a fallback for what those DMVs themselves miss
  (dynamic SQL, some non-schema-bound references) — results are tagged
  `"evidence": "dmv"` vs `"evidence": "text-fallback"` so the caller can
  tell which is authoritative.

Safety/tenancy properties mirror `mcp_server.tools.sql_tools`: every
call carries a `user_id`, resolved against `user_accessible_resources`
before touching the target database. Both tools are read-only by
construction (they only query system catalogs/DMVs, never user data),
but still require `VIEW DEFINITION` (and, for the dependency DMVs,
ideally `VIEW DATABASE STATE`) permission on the stored SQL login —
missing permissions fail open with a clear `ToolResult(ok=False, ...)`
rather than a raw driver exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from sqlalchemy import text as sql_text

from app.utils import get_db, list_accessible
from core.live_acl import revalidate
from core.sql_ddl import render_table_ddl
from core.sql_dependency_extraction import find_references
from mcp_server.config import mcp_settings
from mcp_server.tools._sql_engine import get_engine


# ──────────────────────────────────────────────────────────────────────────────
# MCP tool descriptors
# ──────────────────────────────────────────────────────────────────────────────

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "sql_object_definition",
        "description": (
            "Fetch the COMPLETE, unchunked definition of a live SQL "
            "Server table/view/procedure/function/trigger — straight "
            "from the database, not the (possibly fragmented or stale) "
            "ingested copy. Use this whenever you need to open up a "
            "called function/procedure's actual logic rather than "
            "naming it, or when a RAG-retrieved object body looks "
            "truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db_name": {
                    "type": "string",
                    "description": "Target SQL Server database name.",
                },
                "object_name": {
                    "type": "string",
                    "description": (
                        "A table/view/procedure/function/trigger name, "
                        "e.g. 'dbo.fn_NetLineAmount' or just "
                        "'fn_NetLineAmount' (schema defaults to dbo)."
                    ),
                },
            },
            "required": ["db_name", "object_name"],
        },
    },
    {
        "name": "sql_object_dependencies",
        "description": (
            "Traverse LIVE SQL Server dependency metadata "
            "(sys.dm_sql_referenced_entities / "
            "sys.dm_sql_referencing_entities) starting from an object, "
            "multiple hops deep. Use direction='downstream' for 'what "
            "breaks if I change/drop X' (blast radius), and "
            "direction='upstream' for 'trace X back to its source "
            "tables' (lineage). More authoritative than the static "
            "sql_dependency_graph tool since it reflects the database's "
            "current state, not the last ingestion — but still can't "
            "see dynamic SQL; a text-search fallback over the object's "
            "own definition supplements what the DMVs miss for the "
            "starting object (tagged 'evidence':'text-fallback' vs "
            "'evidence':'dmv' in the results)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db_name": {
                    "type": "string",
                    "description": "Target SQL Server database name.",
                },
                "object_name": {
                    "type": "string",
                    "description": (
                        "A table/view/procedure/function/trigger name to "
                        "start from, e.g. 'dbo.usp_BuildReport_X' or "
                        "just 'usp_BuildReport_X'."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["upstream", "downstream", "both"],
                    "description": (
                        "'downstream' = what depends on this object "
                        "(impact analysis). 'upstream' = what this "
                        "object depends on (lineage). 'both' = both "
                        "directions."
                    ),
                    "default": "both",
                },
                "max_hops": {
                    "type": "integer",
                    "description": (
                        "Max traversal hops (default 3, hard ceiling "
                        f"{mcp_settings.MCP_SQL_MAX_DEPENDENCY_HOPS})."
                    ),
                    "default": 3,
                },
            },
            "required": ["db_name", "object_name"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Result shape (mirrors mcp_server.tools.sql_tools.ToolResult)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
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
# Object resolution helpers
# ──────────────────────────────────────────────────────────────────────────────

_TYPE_DESC_MAP = {
    "SQL_STORED_PROCEDURE": "procedure",
    "SQL_SCALAR_FUNCTION": "function",
    "SQL_TABLE_VALUED_FUNCTION": "function",
    "SQL_INLINE_TABLE_VALUED_FUNCTION": "function",
    "VIEW": "view",
    "USER_TABLE": "table",
    "SQL_TRIGGER": "trigger",
}

_KNOWN_OBJECTS_QUERY = """
SELECT s.name, o.name, o.type_desc
FROM sys.objects o
JOIN sys.schemas s ON s.schema_id = o.schema_id
WHERE o.type IN ('P','FN','TF','IF','V','U','TR')
"""


def _split_object_name(object_name: str) -> tuple[str, str]:
    """
    Accept 'schema.object', bare 'object' (schema defaults to dbo), or
    'db.schema.object' (the db part is ignored — the caller already
    picked the target database via db_name) — takes the last two
    dot-separated parts.
    """
    parts = [p.strip().strip("[]") for p in object_name.strip().split(".")]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "dbo", parts[-1]


def _resolve_object(engine, schema: str, name: str) -> Optional[tuple[str, str, str]]:
    """Returns (canonical_schema, canonical_name, object_type) or None."""
    with engine.connect() as conn:
        row = conn.execute(
            sql_text(
                "SELECT s.name, o.name, o.type_desc "
                "FROM sys.objects o "
                "JOIN sys.schemas s ON s.schema_id = o.schema_id "
                "WHERE s.name = :schema AND o.name = :name "
                "AND o.type IN ('P','FN','TF','IF','V','U','TR')"
            ),
            {"schema": schema, "name": name},
        ).first()
    if not row:
        return None
    canonical_schema, canonical_name, type_desc = row
    return canonical_schema, canonical_name, _TYPE_DESC_MAP.get(type_desc, type_desc.lower())


def _fetch_known_objects(engine) -> dict[str, tuple[str, str]]:
    """Lowercased 'schema.name' -> (canonical_name, object_type) for the
    whole live database — used to build the text-fallback catalog for
    `find_references`, the same shape `sql_ingestor.py` builds at
    ingestion time, but via one lean sys.objects query."""
    with engine.connect() as conn:
        rows = conn.execute(sql_text(_KNOWN_OBJECTS_QUERY)).fetchall()
    known: dict[str, tuple[str, str]] = {}
    for schema_name, object_name, type_desc in rows:
        full_name = f"{schema_name}.{object_name}"
        known[full_name.lower()] = (full_name, _TYPE_DESC_MAP.get(type_desc, type_desc.lower()))
    return known


def _fetch_live_definition(engine, schema: str, name: str) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(
            sql_text("SELECT definition FROM sys.sql_modules WHERE object_id = OBJECT_ID(:qualified)"),
            {"qualified": f"[{schema}].[{name}]"},
        ).first()
    return row[0] if row and row[0] else None


def _check_access(user_id: str, db_name: str) -> Optional[str]:
    """Return an error message if the user lacks access to db_name, else None."""
    with get_db() as db:
        accessible = set(revalidate(db, user_id, "sql", list_accessible(db, user_id, "sql")))
    if db_name not in accessible:
        return (
            f"Access denied: user has no granted access to database "
            f"'{db_name}'. Run a SQL ingestion under your account first, "
            "or pick a database from sql_list_databases."
        )
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Tool: sql_object_definition
# ──────────────────────────────────────────────────────────────────────────────

def object_definition(user_id: str, db_name: str, object_name: str) -> ToolResult:
    """Fetch the complete, unchunked live definition of one SQL object."""
    access_error = _check_access(user_id, db_name)
    if access_error:
        logger.warning("[mcp.sql_schema] DENY user={} db={} reason=no-access", user_id, db_name)
        return ToolResult(ok=False, tool="sql_object_definition", error=access_error)

    schema, name = _split_object_name(object_name)

    try:
        engine = get_engine(user_id, db_name)
        resolved = _resolve_object(engine, schema, name)
        if resolved is None:
            return ToolResult(
                ok=False,
                tool="sql_object_definition",
                error=f"Object '{schema}.{name}' not found in database '{db_name}'.",
            )
        canonical_schema, canonical_name, object_type = resolved
        full_name = f"{canonical_schema}.{canonical_name}"

        if object_type == "table":
            with engine.connect() as conn:
                rows = conn.execute(
                    sql_text(
                        "SELECT c.TABLE_SCHEMA, c.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE, "
                        "c.CHARACTER_MAXIMUM_LENGTH, c.IS_NULLABLE, c.COLUMN_DEFAULT "
                        "FROM INFORMATION_SCHEMA.COLUMNS c "
                        "WHERE c.TABLE_SCHEMA = :schema AND c.TABLE_NAME = :name "
                        "ORDER BY c.ORDINAL_POSITION"
                    ),
                    {"schema": canonical_schema, "name": canonical_name},
                ).fetchall()
            if not rows:
                return ToolResult(
                    ok=False,
                    tool="sql_object_definition",
                    error=f"No columns found for table '{full_name}'.",
                )
            definition = render_table_ddl(full_name, rows)
        else:
            definition = _fetch_live_definition(engine, canonical_schema, canonical_name)
            if not definition:
                return ToolResult(
                    ok=False,
                    tool="sql_object_definition",
                    error=(
                        f"No definition available for '{full_name}' — it may be "
                        "encrypted (WITH ENCRYPTION), or the login lacks "
                        "VIEW DEFINITION permission."
                    ),
                )
    except PermissionError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[mcp.sql_schema] object_definition FAIL user={} db={} object={}",
            user_id, db_name, object_name,
        )
        return ToolResult(
            ok=False, tool="sql_object_definition", error=f"Failed to fetch definition: {exc}"
        )

    logger.info(
        "[mcp.sql_schema] object_definition user={} db={} object={} type={} chars={}",
        user_id, db_name, full_name, object_type, len(definition),
    )
    return ToolResult(
        ok=True,
        tool="sql_object_definition",
        data={"object_name": full_name, "object_type": object_type, "definition": definition},
        markdown=f"**{full_name}** ({object_type}):\n\n```sql\n{definition}\n```",
        metadata={"db_name": db_name, "object_name": full_name, "object_type": object_type},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool: sql_object_dependencies
# ──────────────────────────────────────────────────────────────────────────────

def _dmv_hop(engine, schema: str, name: str, forward: bool) -> list[tuple[str, str]]:
    """
    One hop via the appropriate dependency DMV.

    forward=True (upstream/lineage): `sys.dm_sql_referenced_entities` —
    what `schema.name` itself references.
    forward=False (downstream/impact): `sys.dm_sql_referencing_entities`
    — what references `schema.name`.
    """
    full_name = f"{schema}.{name}"
    if forward:
        query = sql_text(
            "SELECT DISTINCT referenced_schema_name, referenced_entity_name "
            "FROM sys.dm_sql_referenced_entities(:full_name, 'OBJECT') "
            "WHERE referenced_entity_name IS NOT NULL "
            "AND referenced_schema_name IS NOT NULL"
        )
    else:
        query = sql_text(
            "SELECT DISTINCT referencing_schema_name, referencing_entity_name "
            "FROM sys.dm_sql_referencing_entities(:full_name, 'OBJECT')"
        )
    with engine.connect() as conn:
        rows = conn.execute(query, {"full_name": full_name}).fetchall()
    return [(r[0], r[1]) for r in rows if r[0] and r[1]]


def _traverse_dmv(
    engine, start_schema: str, start_name: str, forward: bool, max_hops: int
) -> list[dict[str, Any]]:
    visited = {(start_schema.lower(), start_name.lower())}
    frontier = [(start_schema, start_name)]
    results: list[dict[str, Any]] = []

    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        next_frontier: list[tuple[str, str]] = []
        for schema, name in frontier:
            try:
                found = _dmv_hop(engine, schema, name, forward)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[mcp.sql_schema] DMV hop failed for {}.{} (forward={}): {}",
                    schema, name, forward, exc,
                )
                continue
            for f_schema, f_name in found:
                key = (f_schema.lower(), f_name.lower())
                subj = f"{schema}.{name}" if forward else f"{f_schema}.{f_name}"
                obj = f"{f_schema}.{f_name}" if forward else f"{schema}.{name}"
                results.append(
                    {"hop": hop, "subject": subj, "predicate": "references", "object": obj, "evidence": "dmv"}
                )
                if key not in visited:
                    visited.add(key)
                    next_frontier.append((f_schema, f_name))
        frontier = next_frontier

    return results


def _render_dependency_markdown(
    object_name: str, edge_sets: dict[str, list[dict[str, Any]]]
) -> str:
    parts: list[str] = []
    for label, rows in edge_sets.items():
        if not rows:
            parts.append(f"**{label}** — _(no edges found)_")
            continue
        header = (
            f"**{label}** (from {object_name}):\n\n"
            "| Hop | Subject | Predicate | Object | Evidence |\n"
            "| --- | --- | --- | --- | --- |\n"
        )
        body = "\n".join(
            f"| {r['hop']} | {r['subject']} | {r['predicate']} | {r['object']} | {r['evidence']} |"
            for r in rows
        )
        parts.append(header + body)
    return "\n\n".join(parts) if parts else f"_(no edges found for {object_name!r})_"


def object_dependencies(
    user_id: str,
    db_name: str,
    object_name: str,
    direction: str = "both",
    max_hops: Optional[int] = None,
) -> ToolResult:
    """Multi-hop live traversal of SQL Server's dependency DMVs, with a
    text-search fallback (over the starting object's own definition)
    for what those DMVs miss."""
    access_error = _check_access(user_id, db_name)
    if access_error:
        logger.warning("[mcp.sql_schema] DENY user={} db={} reason=no-access", user_id, db_name)
        return ToolResult(ok=False, tool="sql_object_dependencies", error=access_error)

    if direction not in ("upstream", "downstream", "both"):
        direction = "both"
    hops = max(1, min(int(max_hops or 3), mcp_settings.MCP_SQL_MAX_DEPENDENCY_HOPS))

    schema, name = _split_object_name(object_name)
    full_name = f"{schema}.{name}"

    try:
        engine = get_engine(user_id, db_name)
        resolved = _resolve_object(engine, schema, name)
        if resolved is None:
            return ToolResult(
                ok=False,
                tool="sql_object_dependencies",
                error=f"Object '{full_name}' not found in database '{db_name}'.",
            )
        canonical_schema, canonical_name, object_type = resolved
        full_name = f"{canonical_schema}.{canonical_name}"

        edge_sets: dict[str, list[dict[str, Any]]] = {}
        if direction in ("upstream", "both"):
            edge_sets["upstream"] = _traverse_dmv(
                engine, canonical_schema, canonical_name, forward=True, max_hops=hops
            )
        if direction in ("downstream", "both"):
            edge_sets["downstream"] = _traverse_dmv(
                engine, canonical_schema, canonical_name, forward=False, max_hops=hops
            )

        # Text-fallback supplement: only for the starting object's own
        # definition (not recursively at every hop) and only for
        # upstream, since that's what scanning ONE object's own text can
        # discover — a downstream text-fallback would mean scanning
        # every OTHER object's text, which the static sql_dependency_graph
        # tool already does at ingestion time.
        if "upstream" in edge_sets and object_type != "table":
            live_definition = _fetch_live_definition(engine, canonical_schema, canonical_name)
            if live_definition:
                known_objects = _fetch_known_objects(engine)
                self_key = full_name.lower()
                already = {e["object"].lower() for e in edge_sets["upstream"]}
                for _subj, predicate, obj in find_references(live_definition, self_key, known_objects):
                    if obj.lower() not in already:
                        edge_sets["upstream"].append(
                            {
                                "hop": 1,
                                "subject": full_name,
                                "predicate": predicate,
                                "object": obj,
                                "evidence": "text-fallback",
                            }
                        )
                        already.add(obj.lower())
    except PermissionError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[mcp.sql_schema] object_dependencies FAIL user={} db={} object={}",
            user_id, db_name, object_name,
        )
        return ToolResult(
            ok=False,
            tool="sql_object_dependencies",
            error=(
                f"Failed to traverse dependencies for '{full_name}': {exc}. "
                "This may mean the SQL login lacks VIEW DEFINITION / VIEW "
                "DATABASE STATE permission."
            ),
        )

    total = sum(len(rows) for rows in edge_sets.values())
    logger.info(
        "[mcp.sql_schema] object_dependencies user={} db={} object={} direction={} -> {} edges",
        user_id, db_name, full_name, direction, total,
    )
    return ToolResult(
        ok=True,
        tool="sql_object_dependencies",
        data={"object_name": full_name, "edges": edge_sets},
        markdown=_render_dependency_markdown(full_name, edge_sets),
        metadata={"db_name": db_name, "object_name": full_name, "count": total},
    )
