"""
Unit tests for core.sql_impact_engine, the single dispatcher between the
"legacy" (regex/sqlparse) and "sqlglot" SQL impact-analysis engines.

Each underlying engine already has its own exhaustive test suite
(tests/test_sql_dependency_extraction*.py, tests/test_sql_join_shape*.py,
tests/test_sql_column_lineage.py) -- these tests only confirm the
dispatcher routes to the correct implementation for a given
settings.SQL_IMPACT_ANALYSIS_ENGINE value, not correctness of either
engine's own logic.
"""

import core.sql_impact_engine as engine_module
import core.sql_dependency_extraction as legacy_deps
import core.sql_dependency_extraction_sqlglot as sqlglot_deps
import core.sql_join_shape as legacy_join
import core.sql_join_shape_sqlglot as sqlglot_join
from core.retriever import RetrievedChunk

_KNOWN_OBJECTS = {
    "dbo.foo": ("dbo.Foo", "procedure"),
    "dbo.bar": ("dbo.Bar", "table"),
}
_DEFINITION = "CREATE PROCEDURE dbo.Foo AS BEGIN SELECT * FROM dbo.Bar; END;"


def _chunk(object_name, object_type, text, score=1.0):
    return RetrievedChunk(
        resource_id=f"sql:x.{object_name}",
        source="sql",
        chunk_index=0,
        score=score,
        text=text,
        title=object_name,
        url="",
        metadata={"object_name": object_name, "object_type": object_type},
    )


_TRACING_QUESTION = "Trace TotalNetAmount back to its source tables."
_JOIN_TEXT = """
CREATE OR ALTER PROCEDURE dbo.usp_Dispatch
AS
BEGIN
    SELECT a.ID
    INTO #Result
    FROM #TempA a
    LEFT JOIN #TempB b ON b.ID = a.ID;
END;
"""


def test_find_references_routes_to_legacy_by_default(monkeypatch):
    monkeypatch.setattr(engine_module.settings, "SQL_IMPACT_ANALYSIS_ENGINE", "legacy")
    dispatched = engine_module.find_references(_DEFINITION, "dbo.foo", _KNOWN_OBJECTS)
    direct = legacy_deps.find_references(_DEFINITION, "dbo.foo", _KNOWN_OBJECTS)
    assert dispatched == direct
    assert dispatched == [("dbo.Foo", "references", "dbo.Bar")]


def test_find_references_routes_to_sqlglot_when_set(monkeypatch):
    monkeypatch.setattr(engine_module.settings, "SQL_IMPACT_ANALYSIS_ENGINE", "sqlglot")
    dispatched = engine_module.find_references(_DEFINITION, "dbo.foo", _KNOWN_OBJECTS)
    direct = sqlglot_deps.find_references(_DEFINITION, "dbo.foo", _KNOWN_OBJECTS)
    assert dispatched == direct
    assert dispatched == [("dbo.Foo", "references", "dbo.Bar")]


def test_join_shape_findings_routes_to_legacy_by_default(monkeypatch):
    monkeypatch.setattr(engine_module.settings, "SQL_IMPACT_ANALYSIS_ENGINE", "legacy")
    hits = [_chunk("dbo.usp_Dispatch", "procedure", _JOIN_TEXT)]
    dispatched = engine_module.join_shape_findings(_TRACING_QUESTION, hits)
    direct = legacy_join.join_shape_findings(_TRACING_QUESTION, hits)
    assert dispatched == direct
    assert dispatched  # sanity: the fixture text does produce a finding


def test_join_shape_findings_routes_to_sqlglot_when_set(monkeypatch):
    monkeypatch.setattr(engine_module.settings, "SQL_IMPACT_ANALYSIS_ENGINE", "sqlglot")
    hits = [_chunk("dbo.usp_Dispatch", "procedure", _JOIN_TEXT)]
    dispatched = engine_module.join_shape_findings(_TRACING_QUESTION, hits)
    direct = sqlglot_join.join_shape_findings(_TRACING_QUESTION, hits)
    assert dispatched == direct
    assert dispatched


def test_column_lineage_findings_empty_under_legacy(monkeypatch):
    """No legacy equivalent exists at all -- must return [] without ever
    importing core.sql_column_lineage (and therefore without needing
    sqlglot importable), regardless of input."""
    monkeypatch.setattr(engine_module.settings, "SQL_IMPACT_ANALYSIS_ENGINE", "legacy")
    hits = [_chunk("dbo.usp_Dispatch", "procedure", _JOIN_TEXT)]
    assert engine_module.column_lineage_findings(_TRACING_QUESTION, hits) == []


def test_column_lineage_findings_dispatches_under_sqlglot(monkeypatch):
    monkeypatch.setattr(engine_module.settings, "SQL_IMPACT_ANALYSIS_ENGINE", "sqlglot")
    text = """
CREATE OR ALTER PROCEDURE dbo.usp_Lineage
AS
BEGIN
    SELECT a.Value AS OutVal
    INTO #Result
    FROM dbo.Base a;
END;
"""
    hits = [_chunk("dbo.usp_Lineage", "procedure", text)]
    question = "Show how dbo.usp_Lineage.OutVal is derived from source tables."
    from core.sql_column_lineage import column_lineage_findings as direct_impl

    dispatched = engine_module.column_lineage_findings(question, hits)
    direct = direct_impl(question, hits)
    assert dispatched == direct
