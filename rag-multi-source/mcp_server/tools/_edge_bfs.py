"""
Shared multi-hop BFS over `entity_edges` rows.

Extracted from `entity_graph_tools.traverse_sql_dependencies` (the
original, still-only caller) so `git_dependency_tools.traverse_git_dependencies`
can reuse the exact same mechanics — the loop itself never touched
anything SQL-specific (no `source` string, no `extraction_method`
handling). Everything that *is* domain-specific — scope resolution via
`list_accessible`/`revalidate`, the `base_clause` tenancy filter, and the
markdown/tool-description text — stays in each caller, matching this
codebase's existing precedent of small per-module duplication over
cross-module coupling (`sql_tools.ToolResult` vs. `sql_schema_tools.ToolResult`
vs. `entity_graph_tools.ToolResult` are three separate identical
dataclasses, not one shared import).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Session

from models.entity_edge import EntityEdge

EDGES_PER_HOP_CAP = 500


def bfs(
    db: Session, base_clause: Any, start: str, hops: int, forward: bool
) -> list[dict[str, Any]]:
    """
    Multi-hop breadth-first walk over `EntityEdge` rows matching
    `base_clause`, starting from `start`.

    `forward=True` walks matching the `subject` column, collecting each
    hop's `object`s as the next frontier — "what does `start` depend on"
    (lineage/upstream). `forward=False` walks matching `object`,
    collecting `subject`s — "what depends on `start`"
    (blast-radius/downstream).

    Matching is substring (`ILIKE %name%`), so a caller can pass a bare
    or partially-qualified name. Returns hop-labeled
    `{"hop", "subject", "predicate", "object"}` dicts, capped at
    `EDGES_PER_HOP_CAP` rows per hop.
    """
    visited = {start.lower()}
    frontier = {start}
    results: list[dict[str, Any]] = []

    for hop in range(1, hops + 1):
        if not frontier:
            break
        match_col = EntityEdge.subject if forward else EntityEdge.object
        rows = (
            db.query(EntityEdge)
            .filter(base_clause)
            .filter(or_(*[match_col.ilike(f"%{n}%") for n in frontier]))
            .limit(EDGES_PER_HOP_CAP)
            .all()
        )
        next_frontier: set[str] = set()
        for r in rows:
            results.append(
                {"hop": hop, "subject": r.subject, "predicate": r.predicate, "object": r.object}
            )
            nxt = r.object if forward else r.subject
            if nxt.lower() not in visited:
                next_frontier.add(nxt)
                visited.add(nxt.lower())
        frontier = next_frontier

    return results
