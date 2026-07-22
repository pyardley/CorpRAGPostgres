"""
Reassemble fragmented SQL object chunks before they reach the LLM.

A stored procedure/view/function/trigger definition can be split across
several `vector_chunks` rows by the generic text splitter (see
`app/ingestion/base.py::_chunk` — chunk_size=1000 by default, no SQL
awareness). If only one fragment of a multi-chunk object wins a
retrieval slot, the LLM only ever sees a slice of it — this is exactly
how a middle stage of a stored procedure (a temp table, a join) goes
missing from a lineage-tracing answer even though the object *was*
retrieved and *did* score well enough to make the cut.

This module fetches every stored chunk for each SQL hit's resource_id
and concatenates them back into one block, so a SQL object is either
fully present or (past a size cap) left untouched — never silently
partial without the caller knowing.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import or_, select

from app.config import settings
from app.utils import get_db, set_current_user_for_rls
from core.retriever import RetrievedChunk
from models.entity_edge import EntityEdge
from models.vector_chunk import VectorChunk

_ATOMIC_OBJECT_TYPES = {"procedure", "function", "view", "trigger", "table"}


def expand_sql_chunks(
    hits: list[RetrievedChunk], user_id: Optional[str] = None
) -> list[RetrievedChunk]:
    """
    Return `hits` with every fragmented SQL object hit replaced by a
    single chunk containing its full, reassembled body.

    Resources with only one stored chunk (already atomic) or more
    chunks than `SQL_OBJECT_REASSEMBLY_MAX_CHUNKS` (too large to safely
    inline whole) are left as-is. Non-SQL hits and the synthetic
    "live-mcp" citation (no underlying vector_chunks row) pass through
    unchanged.
    """
    if not hits:
        return hits

    resource_ids = {
        hit.resource_id
        for hit in hits
        if hit.source == "sql"
        and (hit.metadata or {}).get("object_type") != "live-mcp"
    }
    if not resource_ids:
        return hits

    cap = settings.SQL_OBJECT_REASSEMBLY_MAX_CHUNKS
    reassembled: dict[str, str] = {}
    with get_db() as db:
        set_current_user_for_rls(db, user_id)
        for resource_id in resource_ids:
            rows = db.execute(
                select(VectorChunk.chunk_index, VectorChunk.text)
                .where(VectorChunk.resource_id == resource_id)
                .where(VectorChunk.source == "sql")
                .order_by(VectorChunk.chunk_index)
                .limit(cap + 1)
            ).all()
            if len(rows) <= 1:
                continue  # already atomic (or nothing found)
            if len(rows) > cap:
                continue  # too large — leave the retrieved fragment alone
            reassembled[resource_id] = "\n\n".join(r.text for r in rows)

    if not reassembled:
        return hits

    expanded: list[RetrievedChunk] = []
    seen: set[str] = set()
    for hit in hits:
        full_text = reassembled.get(hit.resource_id)
        if full_text is None:
            expanded.append(hit)
            continue
        if hit.resource_id in seen:
            # A different fragment of the same object already expanded
            # this resource_id — don't show the LLM the full body twice.
            continue
        seen.add(hit.resource_id)
        expanded.append(
            RetrievedChunk(
                resource_id=hit.resource_id,
                source=hit.source,
                chunk_index=0,
                score=hit.score,
                text=full_text,
                title=hit.title,
                url=hit.url,
                metadata=hit.metadata,
            )
        )
    return expanded


def _split_qualified(qualified: str) -> tuple[str, str]:
    """'{db_name}.{schema}.{object_name}' -> (db_name, 'schema.object_name')."""
    db_name, _, rest = qualified.partition(".")
    return db_name, rest


def expand_hits_with_dependencies(
    hits: list[RetrievedChunk],
    user_id: Optional[str] = None,
    max_hops: int = 1,
    max_extra: int = 5,
) -> list[RetrievedChunk]:
    """
    Force-include objects directly connected (via the static SQL
    dependency graph, `entity_edges`) to the SQL hits already retrieved
    — bypassing the similarity/rerank cutoff entirely.

    Closes the specific gap `expand_sql_chunks` can't: a genuinely
    relevant object (e.g. a called function) that never won a top-K
    slot at all, because it was crowded out by a loosely-related object
    that happened to score higher. No-ops unless
    `settings.SQL_DEPENDENCY_FORCED_INCLUSION_ENABLED` is set — off by
    default, since this inflates prompt size/cost/latency on every turn
    a SQL object is retrieved, worst-case badly for a "hub" object
    referenced by many procedures.

    Only SQL hits whose `object_type` is atomic (procedure/function/
    view/trigger/table) seed the traversal — the synthetic "live-mcp"
    citation and any other source pass through untouched.
    """
    if not settings.SQL_DEPENDENCY_FORCED_INCLUSION_ENABLED or not hits:
        return hits

    seed: set[tuple[str, str]] = set()
    for hit in hits:
        if hit.source != "sql":
            continue
        md = hit.metadata or {}
        if md.get("object_type") not in _ATOMIC_OBJECT_TYPES:
            continue
        db_name = md.get("db_name")
        object_name = md.get("object_name")
        if db_name and object_name:
            seed.add((db_name, object_name))

    if not seed:
        return hits

    visited = set(seed)
    frontier = set(seed)
    connected: list[tuple[str, str]] = []

    with get_db() as db:
        set_current_user_for_rls(db, user_id)
        for _hop in range(max_hops):
            if not frontier or len(connected) >= max_extra:
                break
            next_frontier: set[tuple[str, str]] = set()
            for db_name, object_name in frontier:
                qualified = f"{db_name}.{object_name}"
                rows = (
                    db.query(EntityEdge.subject, EntityEdge.object)
                    .filter(EntityEdge.source == "sql")
                    .filter(EntityEdge.resource_identifier == db_name)
                    .filter(or_(EntityEdge.subject == qualified, EntityEdge.object == qualified))
                    .limit(100)
                    .all()
                )
                for subj, obj in rows:
                    for candidate in (subj, obj):
                        key = _split_qualified(candidate)
                        if key in visited:
                            continue
                        visited.add(key)
                        next_frontier.add(key)
                        connected.append(key)
                        if len(connected) >= max_extra:
                            break
                    if len(connected) >= max_extra:
                        break
                if len(connected) >= max_extra:
                    break
            frontier = next_frontier

        if not connected:
            return hits

        existing_resource_ids = {h.resource_id for h in hits}
        extra_hits: list[RetrievedChunk] = []
        for c_db, c_object in connected[:max_extra]:
            rows = db.execute(
                select(
                    VectorChunk.resource_id,
                    VectorChunk.chunk_index,
                    VectorChunk.text,
                    VectorChunk.title,
                    VectorChunk.url,
                )
                .where(VectorChunk.source == "sql")
                .where(VectorChunk.db_name == c_db)
                .where(VectorChunk.object_name == c_object)
                .order_by(VectorChunk.chunk_index)
            ).all()
            if not rows or rows[0].resource_id in existing_resource_ids:
                continue
            extra_hits.append(
                RetrievedChunk(
                    resource_id=rows[0].resource_id,
                    source="sql",
                    chunk_index=0,
                    score=0.0,  # sentinel — force-included, not similarity-ranked
                    text="\n\n".join(r.text for r in rows),
                    title=rows[0].title or rows[0].resource_id,
                    url=rows[0].url or "",
                    metadata={
                        "db_name": c_db,
                        "object_name": c_object,
                        "object_type": "forced-dependency",
                    },
                )
            )
            existing_resource_ids.add(rows[0].resource_id)

    return hits + extra_hits
