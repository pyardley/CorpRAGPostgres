"""
Unit tests for core.trace_completeness.

Covers the anchor-based candidate-name logic (candidate_names,
candidate_callable_names) and the new missing_definition_calls check
used by core.mcp_chain's forced-retry gate for the MANDATORY
sql_object_definition rule.
"""

from core.retriever import RetrievedChunk
from core.trace_completeness import (
    _bare_object_name,
    candidate_callable_names,
    candidate_names,
    check_trace_completeness,
    missing_definition_calls,
    sql_anchor,
)


def _chunk(resource_id, object_name, object_type, text, score=1.0):
    return RetrievedChunk(
        resource_id=resource_id,
        source="sql",
        chunk_index=0,
        score=score,
        text=text,
        title=object_name,
        url="",
        metadata={"object_name": object_name, "object_type": object_type},
    )


_ANCHOR_PROC_TEXT = """
CREATE PROCEDURE dbo.usp_BuildReport_CustomerChurnRisk
AS
BEGIN
    SELECT dbo.fn_NetLineAmount(ol.Quantity, ol.UnitPrice, ol.DiscountPct) AS NetAmount
    INTO #ActiveCustomerOrders
    FROM dbo.Customers c
    JOIN dbo.OrderLines ol ON ol.OrderID = c.CustomerID;

    EXEC dbo.usp_LookupCustomerSegment;
END;
"""


def _fixture_hits():
    return [
        _chunk(
            "sql:x.dbo.usp_BuildReport_CustomerChurnRisk",
            "dbo.usp_BuildReport_CustomerChurnRisk",
            "procedure",
            _ANCHOR_PROC_TEXT,
            score=0.95,
        ),
        _chunk(
            "sql:x.dbo.fn_NetLineAmount",
            "dbo.fn_NetLineAmount",
            "function",
            "CREATE FUNCTION dbo.fn_NetLineAmount(...) RETURNS DECIMAL(12,2) ...",
            score=0.5,
        ),
        _chunk(
            "sql:x.dbo.usp_LookupCustomerSegment",
            "dbo.usp_LookupCustomerSegment",
            "procedure",
            "CREATE PROCEDURE dbo.usp_LookupCustomerSegment AS ...",
            score=0.4,
        ),
        _chunk(
            "sql:x.dbo.Customers",
            "dbo.Customers",
            "table",
            "CREATE TABLE dbo.Customers (...)",
            score=0.3,
        ),
        # Unrelated sibling, never referenced by the anchor's own text —
        # must never be flagged by anything in this module.
        _chunk(
            "sql:x.dbo.usp_BuildReport_MonthlySalesByRegion",
            "dbo.usp_BuildReport_MonthlySalesByRegion",
            "procedure",
            "CREATE PROCEDURE dbo.usp_BuildReport_MonthlySalesByRegion AS ...",
            score=0.2,
        ),
    ]


def test_candidate_callable_names_only_functions_and_procedures():
    hits = _fixture_hits()
    names = candidate_callable_names(hits)
    assert names == {"dbo.fn_NetLineAmount", "dbo.usp_LookupCustomerSegment"}


def test_candidate_callable_names_excludes_anchor_and_unrelated_sibling():
    hits = _fixture_hits()
    names = candidate_callable_names(hits)
    assert "dbo.usp_BuildReport_CustomerChurnRisk" not in names  # self
    assert "dbo.usp_BuildReport_MonthlySalesByRegion" not in names  # unrelated


def test_candidate_callable_names_drops_tables():
    hits = _fixture_hits()
    names = candidate_callable_names(hits)
    assert "dbo.Customers" not in names  # table, not procedure/function


def test_missing_definition_calls_flags_uncalled_functions():
    hits = _fixture_hits()
    missing = missing_definition_calls(hits, called_object_names=[])
    assert missing == ["dbo.fn_NetLineAmount", "dbo.usp_LookupCustomerSegment"]


def test_missing_definition_calls_matches_bare_names():
    hits = _fixture_hits()
    # A tool call might pass a bare name rather than schema-qualified.
    missing = missing_definition_calls(hits, called_object_names=["fn_NetLineAmount"])
    assert missing == ["dbo.usp_LookupCustomerSegment"]


def test_missing_definition_calls_matches_qualified_names():
    hits = _fixture_hits()
    missing = missing_definition_calls(
        hits, called_object_names=["dbo.fn_NetLineAmount", "dbo.usp_LookupCustomerSegment"]
    )
    assert missing == []


def test_bare_object_name_strips_brackets_and_qualification():
    assert _bare_object_name("dbo.fn_NetLineAmount") == "fn_netlineamount"
    assert _bare_object_name("[dbo].[fn_NetLineAmount]") == "fn_netlineamount"
    assert _bare_object_name("MyDb.dbo.fn_NetLineAmount") == "fn_netlineamount"
    assert _bare_object_name("fn_NetLineAmount") == "fn_netlineamount"


def test_check_trace_completeness_unaffected_by_new_helpers():
    """Regression: the refactor into _sql_anchor must not change
    check_trace_completeness's existing behavior."""
    hits = _fixture_hits()
    question = "Trace TotalNetAmount back to its source tables."
    answer_missing_enrichment = (
        "The procedure joins dbo.Customers and dbo.OrderLines, computing "
        "NetAmount via dbo.fn_NetLineAmount, then calls "
        "dbo.usp_LookupCustomerSegment."
    )
    missing = check_trace_completeness(question, hits, answer_missing_enrichment)
    assert "#ActiveCustomerOrders" in missing
    assert "dbo.usp_BuildReport_MonthlySalesByRegion" not in missing


def test_sql_anchor_and_candidate_names_rename_unchanged_behavior():
    """Regression: renaming _sql_anchor -> sql_anchor and _candidate_names
    -> candidate_names (to make them importable by core.sql_join_shape)
    must not change either function's behavior — pure mechanical rename,
    same call sites, same results as before."""
    hits = _fixture_hits()
    anchor = sql_anchor(hits)
    assert anchor is not None
    assert anchor.metadata["object_name"] == "dbo.usp_BuildReport_CustomerChurnRisk"

    names = candidate_names(hits)
    assert "#ActiveCustomerOrders" in names
    assert "dbo.usp_BuildReport_CustomerChurnRisk" not in names  # self excluded
    assert "dbo.usp_BuildReport_MonthlySalesByRegion" not in names  # unrelated sibling
