"""
Entity graph MCP tool — query relationship data pure text search can't
answer (README "Possible enhancements" — Lightweight entity graph,
GraphRAG-inspired).

Pure chunk similarity can't answer "who else has touched tickets
related to this outage?" or "which repos does this author maintain?".
`entity_edges` (see `models.entity_edge`) stores (subject, predicate,
object) triples populated at ingestion time — deterministically from
Jira assignee/reporter and Git commit author, optionally enriched by an
LLM pass over free text (see `core.entity_extraction`). This tool lets
the LLM query that graph directly.

Tenancy mirrors `sql_tools.table_query`: resolve the calling user's
accessible Jira project keys / Git scopes / SQL databases and restrict
the query to rows whose `(source, resource_identifier)` matches one of
them — enforced here at the application layer, reinforced by the
`entity_edges` RLS policies (`models.migrations`) as defence-in-depth.

Also exposes `traverse_sql_dependencies` (the `sql_dependency_graph`
tool) — a multi-hop BFS over `source="sql"` edges, which
`app.ingestion.sql_ingestor` populates via
`core.sql_dependency_extraction.find_references` when
`settings.ENABLE_SQL_DEPENDENCY_GRAPH` is on. `query_entities` itself
only does a single-hop lookup with no directionality, which isn't
enough to answer "what breaks if I change X" (needs to walk multiple
hops downstream) or "trace X back to source" (needs to walk upstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from sqlalchemy import and_, or_

from app.utils import get_db, list_accessible
from core.live_acl import revalidate
from mcp_server.tools._edge_bfs import bfs
from models.entity_edge import EntityEdge

# ──────────────────────────────────────────────────────────────────────────────
# MCP tool descriptor (consumed by server.py and the LangChain adapter)
# ──────────────────────────────────────────────────────────────────────────────

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "entity_graph_query",
        "description": (
            "Search the entity relationship graph for edges involving a "
            "person, ticket, or repository — assigned_to, reported_by "
            "(Jira), modified_by (Git), plus any LLM-extracted "
            "relationships. Use this for relationship questions that "
            "plain text search can't answer, e.g. 'who else worked on "
            "tickets like this one', 'who reported PROJ-123', or 'which "
            "repos does alice@example.com maintain'. Search `entity` as "
            "either the subject or the object of an edge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": (
                        "A resource id (e.g. 'jira:PROJ-123'), a "
                        "person's name/accountId/email, or a repo scope "
                        "(e.g. 'owner/repo@main') to search for."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max edges to return.",
                    "default": 20,
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "sql_dependency_graph",
        "description": (
            "Traverse the static SQL object dependency graph built at "
            "ingestion time from table/view/procedure/function/trigger "
            "definitions. Use direction='downstream' for 'what breaks if "
            "I change/drop X' (blast radius — what depends on X), and "
            "direction='upstream' for 'trace X back to its source "
            "tables' (lineage — what X depends on). Returns hop-labeled "
            "(subject, predicate, object) edges; predicate is 'calls' "
            "(procedure/function), 'writes_to', or 'references'. This is "
            "a STATIC, text-derived graph — it can't see dynamic SQL, so "
            "treat an empty result as inconclusive, not proof of no "
            "dependency."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": (
                        "A table/view/procedure/function/trigger name to "
                        "start from, e.g. "
                        "'dbo.usp_BuildReport_CustomerChurnRisk' or just "
                        "'usp_BuildReport_CustomerChurnRisk'."
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
                    "description": "Max traversal hops (default 3, max 5).",
                    "default": 3,
                },
                "extraction_method": {
                    "type": "string",
                    "enum": ["deterministic", "all"],
                    "description": (
                        "'deterministic' (default) restricts to the "
                        "regex-derived static-parse edges — the only "
                        "ones with real code-structure signal for "
                        "tracing. 'all' also includes any LLM-extracted "
                        "edges (debugging only; SQL objects shouldn't "
                        "normally have any)."
                    ),
                    "default": "deterministic",
                },
            },
            "required": ["object_name"],
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
# Tool: entity_graph_query
# ──────────────────────────────────────────────────────────────────────────────

def query_entities(
    user_id: str, entity: str, max_results: Optional[int] = None
) -> ToolResult:
    """Search entity_edges for rows involving `entity`, tenancy-scoped."""
    cap = max(1, min(int(max_results or 20), 100))

    with get_db() as db:
        jira_scopes = revalidate(db, user_id, "jira", list_accessible(db, user_id, "jira"))
        git_scopes = revalidate(db, user_id, "git", list_accessible(db, user_id, "git"))
        sql_scopes = revalidate(db, user_id, "sql", list_accessible(db, user_id, "sql"))

        clauses = []
        if jira_scopes:
            clauses.append(
                and_(EntityEdge.source == "jira", EntityEdge.resource_identifier.in_(jira_scopes))
            )
        if git_scopes:
            clauses.append(
                and_(EntityEdge.source == "git", EntityEdge.resource_identifier.in_(git_scopes))
            )
        if sql_scopes:
            clauses.append(
                and_(EntityEdge.source == "sql", EntityEdge.resource_identifier.in_(sql_scopes))
            )

        if not clauses:
            logger.info(
                "[mcp.entity_graph] user={} has no accessible Jira/Git/SQL scopes", user_id
            )
            return ToolResult(
                ok=True,
                tool="entity_graph_query",
                data={"edges": []},
                markdown="_(no accessible Jira/Git/SQL scopes — run an ingestion first)_",
                metadata={"count": 0},
            )

        pattern = f"%{entity}%"
        rows = (
            db.query(EntityEdge)
            .filter(or_(*clauses))
            .filter(or_(EntityEdge.subject.ilike(pattern), EntityEdge.object.ilike(pattern)))
            .limit(cap * 5)  # over-fetch; a repo can have many commits by the same
            .all()           # author, all producing the identical (subject, predicate, object)
        )

        # De-dupe identical (subject, predicate, object) triples in Python
        # -- a Git repo with N commits by the same author produces N edge
        # rows with identical content (see git_ingestor.py's per-commit
        # edge), since each is keyed by its own commit resource_id for
        # re-ingest replacement, not by the (subject, predicate, object)
        # triple itself. Must read attributes into plain dicts while
        # still inside this `with get_db()` block -- `rows` are ORM
        # instances that get detached once the session closes.
        seen: set[tuple[str, str, str]] = set()
        edges: list[dict[str, str]] = []
        for r in rows:
            key = (r.subject, r.predicate, r.object)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"subject": r.subject, "predicate": r.predicate, "object": r.object, "source": r.source})
            if len(edges) >= cap:
                break
    md = (
        "| Subject | Predicate | Object |\n| --- | --- | --- |\n"
        + "\n".join(f"| {e['subject']} | {e['predicate']} | {e['object']} |" for e in edges)
        if edges
        else f"_(no edges found for {entity!r})_"
    )
    logger.info(
        "[mcp.entity_graph] query_entities user={} entity={!r} -> {} edges",
        user_id,
        entity,
        len(edges),
    )
    return ToolResult(
        ok=True,
        tool="entity_graph_query",
        data={"edges": edges},
        markdown=md,
        metadata={"count": len(edges)},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool: sql_dependency_graph
# ──────────────────────────────────────────────────────────────────────────────

_MAX_HOPS_CEILING = 5


def traverse_sql_dependencies(
    user_id: str,
    object_name: str,
    direction: str = "both",
    max_hops: Optional[int] = None,
    extraction_method: str = "deterministic",
) -> ToolResult:
    """
    Multi-hop BFS over the static SQL dependency graph
    (`entity_edges` rows with `source="sql"`), tenancy-scoped to the
    user's accessible SQL databases.

    "downstream" walks backwards from `object_name` (matching it, then
    each subsequent frontier, against edges' `object` column, collecting
    the matched `subject`s as the next frontier) — this answers "what
    depends on / would break if I changed this" (blast radius).
    "upstream" walks forwards (matching against `subject`, collecting
    `object`s) — this answers "what does this depend on / trace it back
    to source". "both" runs both traversals independently.

    Matching is substring (`ILIKE %name%`), same convention as
    `query_entities` above, so a caller can pass a bare name
    ("usp_BuildReport_X") or a fully/partially qualified one
    ("dbo.usp_BuildReport_X", "MyDb.dbo.usp_BuildReport_X").

    `extraction_method` defaults to `"deterministic"` — SQL edges from
    `core.sql_dependency_extraction.find_references` are the only ones
    with real code-structure signal; an `"llm"`-tagged SQL edge is
    noise from `core.entity_extraction`'s free-text pass having run
    (before it was scoped away from `source == "sql"` — see
    `app.ingestion.base._persist_entity_edges`) over object definitions
    it was never designed to parse. Pass `"all"` to see everything,
    including any such noise still sitting in an un-re-ingested database.
    """
    if direction not in ("upstream", "downstream", "both"):
        direction = "both"
    if extraction_method not in ("deterministic", "all"):
        extraction_method = "deterministic"
    hops = max(1, min(int(max_hops or 3), _MAX_HOPS_CEILING))

    with get_db() as db:
        sql_scopes = revalidate(db, user_id, "sql", list_accessible(db, user_id, "sql"))
        if not sql_scopes:
            logger.info("[mcp.sql_dependency_graph] user={} has no accessible SQL scopes", user_id)
            return ToolResult(
                ok=True,
                tool="sql_dependency_graph",
                data={"edges": {}},
                markdown="_(no accessible SQL databases — run a SQL ingestion first)_",
                metadata={"count": 0},
            )

        clauses = [EntityEdge.source == "sql", EntityEdge.resource_identifier.in_(sql_scopes)]
        if extraction_method != "all":
            clauses.append(EntityEdge.extraction_method == "deterministic")
        base_clause = and_(*clauses)

        edge_sets: dict[str, list[dict[str, Any]]] = {}
        if direction in ("upstream", "both"):
            edge_sets["upstream"] = bfs(db, base_clause, object_name, hops, forward=True)
        if direction in ("downstream", "both"):
            edge_sets["downstream"] = bfs(db, base_clause, object_name, hops, forward=False)

    total = sum(len(rows) for rows in edge_sets.values())
    parts: list[str] = []
    for label, rows in edge_sets.items():
        if not rows:
            parts.append(f"**{label}** — _(no edges found)_")
            continue
        header = (
            f"**{label}** (from {object_name}):\n\n"
            "| Hop | Subject | Predicate | Object |\n| --- | --- | --- | --- |\n"
        )
        body = "\n".join(
            f"| {r['hop']} | {r['subject']} | {r['predicate']} | {r['object']} |"
            for r in rows
        )
        parts.append(header + body)
    markdown = "\n\n".join(parts) if parts else f"_(no edges found for {object_name!r})_"

    logger.info(
        "[mcp.sql_dependency_graph] user={} object={!r} direction={} extraction_method={} -> {} edges",
        user_id,
        object_name,
        direction,
        extraction_method,
        total,
    )
    return ToolResult(
        ok=True,
        tool="sql_dependency_graph",
        data={"edges": edge_sets},
        markdown=markdown,
        metadata={"count": total, "extraction_method": extraction_method},
    )
