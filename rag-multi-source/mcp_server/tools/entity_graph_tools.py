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
accessible Jira project keys / Git scopes and restrict the query to
rows whose `(source, resource_identifier)` matches one of them —
enforced here at the application layer, reinforced by the
`entity_edges` RLS policies (`models.migrations`) as defence-in-depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from sqlalchemy import and_, or_

from app.utils import get_db, list_accessible
from core.live_acl import revalidate
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
    }
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

        clauses = []
        if jira_scopes:
            clauses.append(
                and_(EntityEdge.source == "jira", EntityEdge.resource_identifier.in_(jira_scopes))
            )
        if git_scopes:
            clauses.append(
                and_(EntityEdge.source == "git", EntityEdge.resource_identifier.in_(git_scopes))
            )

        if not clauses:
            logger.info("[mcp.entity_graph] user={} has no accessible Jira/Git scopes", user_id)
            return ToolResult(
                ok=True,
                tool="entity_graph_query",
                data={"edges": []},
                markdown="_(no accessible Jira/Git scopes — run a Jira or Git ingestion first)_",
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
