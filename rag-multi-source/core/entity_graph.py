"""
Persistence for the lightweight entity graph (README "Possible
enhancements" — Lightweight entity graph, GraphRAG-inspired).

See `models.entity_edge.EntityEdge` for the schema and
`app.ingestion.base.BaseIngestor._persist_entity_edges` for how edges
are assembled (deterministic, from structured API fields, plus
optionally LLM-extracted — see `core.entity_extraction`) before being
handed to `upsert_edges`.
"""

from __future__ import annotations

from app.utils import get_db, set_current_user_for_rls
from models.entity_edge import EntityEdge


def upsert_edges(
    source: str,
    resource_identifier: str,
    source_resource_id: str,
    edges: list[tuple[str, str, str, str]],
    *,
    user_id: str | None = None,
) -> int:
    """
    Replace all edges for `source_resource_id` with `edges`.

    Delete-then-insert rather than an upsert-by-key: edges don't have a
    stable per-row identity across re-ingests the way chunks have
    `chunk_index` (an LLM extraction pass can reasonably produce a
    different edge set run to run), so a re-ingest fully replaces the
    prior edge set for that resource. `edges` is a list of
    `(subject, predicate, object, extraction_method)` tuples. Returns
    the number of rows written.
    """
    with get_db() as db:
        set_current_user_for_rls(db, user_id)
        db.query(EntityEdge).filter_by(source_resource_id=source_resource_id).delete()
        for subject, predicate, obj, extraction_method in edges:
            db.add(
                EntityEdge(
                    source=source,
                    resource_identifier=resource_identifier,
                    source_resource_id=source_resource_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    extraction_method=extraction_method,
                )
            )
    return len(edges)
