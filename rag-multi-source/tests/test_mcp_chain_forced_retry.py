"""
Unit test for core.mcp_chain's forced-retry gate.

Live A/B testing (documented in README.md's "remaining gaps" section)
showed the MANDATORY sql_object_definition prompt rule alone wasn't
reliably followed by the model. Since then, live runs against the
fixture have been consistently compliant on the first try, which means
the forced-retry branch itself never gets exercised end to end that
way. This test scripts a non-compliant first response directly so the
retry gate's wiring (continue-the-loop, exactly-one-retry, correct
final footer) is verified regardless of how well-behaved the live
model happens to be on any given run.
"""
from __future__ import annotations

from contextlib import contextmanager

from langchain_core.messages import AIMessage

from core.mcp_chain import answer_question_with_mcp
from core.retriever import RetrievedChunk


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


_ANCHOR_TEXT = """
CREATE PROCEDURE dbo.usp_BuildReport_CustomerChurnRisk
AS
BEGIN
    SELECT dbo.fn_NetLineAmount(ol.Quantity, ol.UnitPrice, ol.DiscountPct) AS NetAmount
    FROM dbo.Customers c
    JOIN dbo.OrderLines ol ON ol.OrderID = c.CustomerID;
END;
"""


def _hits():
    return [
        _chunk(
            "sql:x.dbo.usp_BuildReport_CustomerChurnRisk",
            "dbo.usp_BuildReport_CustomerChurnRisk",
            "procedure",
            _ANCHOR_TEXT,
            score=0.95,
        ),
        _chunk(
            "sql:x.dbo.fn_NetLineAmount",
            "dbo.fn_NetLineAmount",
            "function",
            "CREATE FUNCTION dbo.fn_NetLineAmount(...) RETURNS DECIMAL(12,2) ...",
            score=0.5,
        ),
    ]


class _ScriptedLLM:
    """Stands in for `llm.bind_tools(tools)`, replaying one AIMessage per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        response = self._responses[self.calls]
        self.calls += 1
        return response


@contextmanager
def _null_db():
    yield None


def _install_common_stubs(monkeypatch, scripted_llm):
    monkeypatch.setattr("core.mcp_chain.build_mcp_tools", lambda user_id: [])
    monkeypatch.setattr(
        "core.mcp_chain.get_llm",
        lambda: type("StubBase", (), {"bind_tools": lambda self, tools: scripted_llm})(),
    )
    monkeypatch.setattr("core.mcp_chain.get_db", _null_db)
    monkeypatch.setattr("core.mcp_chain.list_accessible", lambda db, user_id, source: [])
    monkeypatch.setattr("core.mcp_chain.revalidate", lambda db, user_id, source, ids: [])
    monkeypatch.setattr(
        "core.mcp_chain._execute_tool_call",
        lambda user_id, name, args: ("dummy tool result", {}),
    )


def test_forced_retry_fires_once_and_converges(monkeypatch):
    """
    Hop 0: model finalizes WITHOUT calling sql_object_definition on
    fn_NetLineAmount, despite it being a candidate callable in `hits`.
    Expect the gate to fire: append a corrective HumanMessage and
    `continue`, rather than accepting hop 0 as final.

    Hop 1: model complies, calling sql_object_definition.

    Hop 2: model gives a real final answer (no tool calls) that quotes
    the function's formula and names the function — this must be
    accepted immediately, with no second retry (forced_retry_used
    latches after the first).
    """
    noncompliant_final = AIMessage(
        content=(
            "TotalNetAmount comes from dbo.usp_BuildReport_CustomerChurnRisk, "
            "joining Customers and OrderLines."
        )
    )
    compliant_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "sql_object_definition",
                "args": {"db_name": "RetailReportingDemo", "object_name": "dbo.fn_NetLineAmount"},
                "id": "call_1",
            }
        ],
    )
    real_final = AIMessage(
        content=(
            "TotalNetAmount is computed by dbo.fn_NetLineAmount, which returns "
            "Quantity * UnitPrice * (1 - DiscountPct), joining dbo.Customers and "
            "dbo.OrderLines back through dbo.usp_BuildReport_CustomerChurnRisk."
        )
    )

    scripted = _ScriptedLLM([noncompliant_final, compliant_tool_call, real_final])
    _install_common_stubs(monkeypatch, scripted)

    result = answer_question_with_mcp(
        "test-user-id",
        "Show how TotalNetAmount is derived. Go right back to original source tables.",
        _hits(),
        history=[],
    )

    step_names = [s["step_name"] for s in result.steps]

    assert step_names.count("forced_retry:sql_object_definition") == 1, step_names
    assert step_names.count("llm_invocation") == 3, step_names
    assert "mcp_tool_call:sql_object_definition" in step_names, step_names
    # Real final answer must win, not the noncompliant one.
    assert result.answer == real_final.content

    post = next(s for s in result.steps if s["step_name"] == "post_processing")
    assert post["metadata"]["missing_definition_calls"] == []
    assert "never opened via" not in result.answer


def test_no_retry_when_model_complies_immediately(monkeypatch):
    """A model that calls sql_object_definition before finalizing should
    never trigger the forced-retry branch at all."""
    compliant_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "sql_object_definition",
                "args": {"db_name": "RetailReportingDemo", "object_name": "dbo.fn_NetLineAmount"},
                "id": "call_1",
            }
        ],
    )
    real_final = AIMessage(
        content=(
            "TotalNetAmount is computed by dbo.fn_NetLineAmount, which returns "
            "Quantity * UnitPrice * (1 - DiscountPct)."
        )
    )
    scripted = _ScriptedLLM([compliant_tool_call, real_final])
    _install_common_stubs(monkeypatch, scripted)

    result = answer_question_with_mcp(
        "test-user-id",
        "Show how TotalNetAmount is derived. Go right back to original source tables.",
        _hits(),
        history=[],
    )

    step_names = [s["step_name"] for s in result.steps]
    assert "forced_retry:sql_object_definition" not in step_names
    assert step_names.count("llm_invocation") == 2, step_names
    assert result.answer == real_final.content


def test_forced_retry_does_not_fire_near_hop_budget_exhaustion(monkeypatch):
    """
    The hop-reserve guard (`hop < MAX_TOOL_HOPS - _FORCED_RETRY_HOP_RESERVE`,
    MAX_TOOL_HOPS=6) means a non-compliant "final" answer arriving at
    hop 4 or later must NOT trigger a retry — there isn't enough budget
    left for both the corrective tool call and a real final answer, so
    it should fall through to the ordinary final-answer path (with the
    missing_definition_calls footer) instead of looping again.

    Script 4 tool-calling hops (0-3, all consumed by unrelated
    dependency-walk tool calls) then a noncompliant final at hop 4.
    """
    dependency_hop = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "sql_object_dependencies",
                "args": {
                    "db_name": "RetailReportingDemo",
                    "object_name": "dbo.usp_BuildReport_CustomerChurnRisk",
                    "direction": "upstream",
                },
                "id": "call_dep",
            }
        ],
    )
    noncompliant_final = AIMessage(
        content=(
            "TotalNetAmount comes from dbo.usp_BuildReport_CustomerChurnRisk, "
            "joining Customers and OrderLines."
        )
    )

    scripted = _ScriptedLLM(
        [dependency_hop, dependency_hop, dependency_hop, dependency_hop, noncompliant_final]
    )
    _install_common_stubs(monkeypatch, scripted)

    result = answer_question_with_mcp(
        "test-user-id",
        "Show how TotalNetAmount is derived. Go right back to original source tables.",
        _hits(),
        history=[],
    )

    step_names = [s["step_name"] for s in result.steps]
    assert "forced_retry:sql_object_definition" not in step_names, step_names
    assert step_names.count("llm_invocation") == 5, step_names
    assert result.answer.startswith(noncompliant_final.content)
    assert "never opened via" in result.answer

    post = next(s for s in result.steps if s["step_name"] == "post_processing")
    assert post["metadata"]["missing_definition_calls"] == ["dbo.fn_NetLineAmount"]
