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

from sqlalchemy import select

from app.config import settings
from app.utils import get_db, set_current_user_for_rls
from core.retriever import RetrievedChunk
from models.vector_chunk import VectorChunk


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
