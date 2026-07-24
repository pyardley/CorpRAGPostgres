"""
Single dispatcher for the SQL impact-analysis engine switch
(``settings.SQL_IMPACT_ANALYSIS_ENGINE``).

Every call site (``app.ingestion.sql_ingestor``,
``mcp_server.tools.sql_schema_tools``, ``core.rag_chain``, ``core.mcp_chain``)
imports from this module rather than choosing an engine itself, so
"legacy" vs "sqlglot" is decided in exactly one place and both subsystems
(dependency-graph edges, JOIN-shape findings) switch together by
construction.

Imports for the ``sqlglot``-based engine are done lazily, inside each
branch, so ``"legacy"`` mode (the default) never requires ``sqlglot`` to
be installed at all.
"""

from __future__ import annotations

from app.config import settings
from core.retriever import RetrievedChunk


def find_references(
    definition: str,
    self_key: str,
    known_objects: dict[str, tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """Dependency-graph edges for one object's definition text. See
    ``core.sql_dependency_extraction.find_references`` /
    ``core.sql_dependency_extraction_sqlglot.find_references`` for the
    shared contract (predicate always ``calls``/``writes_to``/``references``)."""
    if settings.SQL_IMPACT_ANALYSIS_ENGINE == "sqlglot":
        from core.sql_dependency_extraction_sqlglot import find_references as _impl
    else:
        from core.sql_dependency_extraction import find_references as _impl
    return _impl(definition, self_key, known_objects)


def join_shape_findings(question: str, hits: list[RetrievedChunk]) -> list[str]:
    """JOIN-direction row-inclusion-risk findings for the anchor SQL hit.
    See ``core.sql_join_shape.join_shape_findings`` /
    ``core.sql_join_shape_sqlglot.join_shape_findings``."""
    if settings.SQL_IMPACT_ANALYSIS_ENGINE == "sqlglot":
        from core.sql_join_shape_sqlglot import join_shape_findings as _impl
    else:
        from core.sql_join_shape import join_shape_findings as _impl
    return _impl(question, hits)


def column_lineage_findings(question: str, hits: list[RetrievedChunk]) -> list[str]:
    """Column-level lineage findings for the anchor SQL hit. ``sqlglot``-only
    capability -- no legacy equivalent, so this always returns ``[]`` unless
    the engine is switched on."""
    if settings.SQL_IMPACT_ANALYSIS_ENGINE != "sqlglot":
        return []
    from core.sql_column_lineage import column_lineage_findings as _impl

    return _impl(question, hits)
