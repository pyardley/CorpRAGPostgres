"""
Hybrid RAG + MCP answerer.

Used when the user has the **"Use Live SQL Table Data (MCP)"** toggle on
in the sidebar. Wires the MCP SQL tools onto the LLM via
``llm.bind_tools(...)`` and runs a small tool-calling loop:

1. Ask the LLM with the RAG context AND the bound SQL tools.
2. If the LLM emits a tool call, execute it via the MCP client and feed
   the result back as a ``ToolMessage``.
3. Loop up to ``MAX_TOOL_HOPS`` times, then return the final answer.

When MCP is actually used we synthesise an extra
:class:`core.retriever.RetrievedChunk` so the citation list shows
"Live data via MCP from <db>.<table>" alongside the RAG sources, with
``score = 1.0`` so it sorts to the top.

Because the hybrid path may invoke the LLM more than once per user
prompt, we **aggregate** token usage / cost / duration across every hop
into a single audit record (``RAGAnswer.usage``) so the chat status bar
sees one coherent number per user prompt rather than one-per-tool-call.

Step timings (``RAGAnswer.steps``) emitted by this chain:

* ``"build_context"``                    — RAG context formatting.
* ``"llm_invocation"`` (one per hop, ``metadata.hop = N``)
* ``"mcp_tool_call:sql_table_query"``    — one round-trip per call,
  with ``metadata.row_count`` / ``metadata.duration_ms`` from the
  MCP server response.
* ``"mcp_tool_call:sql_list_databases"`` — same shape.
* ``"post_processing"``                  — final text extraction.

The chat layer hands ``steps`` to ``app.utils.log_query_audit`` which
persists every entry as a ``query_step_timings`` row.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterable

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from loguru import logger

from app.config import settings
from app.utils import (
    StepTimer,
    estimate_cost,
    extract_usage,
    get_db,
    list_accessible,
    record_step_timing,
)
from core.llm import get_llm
from core.mcp_client import build_mcp_tools, get_mcp_client
from core.rag_chain import RAGAnswer, _format_context  # type: ignore[attr-defined]
from core.retriever import RetrievedChunk, deduplicate_by_resource


MAX_TOOL_HOPS = 4


HYBRID_SYSTEM_PROMPT = """You are CorporateRAG, an enterprise assistant with
TWO sources of truth and you are EXPECTED to use both:

1. **RAG context (numbered sources)** — indexed Jira tickets, Confluence
   pages, SQL Server *schemas and stored-procedure code*, and Git
   files/commits. Cite inline as [1], [2], …
2. **Live SQL tools (MCP)** — `sql_table_query` (returns rows) and
   `sql_list_databases` (lists DBs the user can query).

Tool-call policy — read carefully:

* If the user's question requires **actual rows / values / counts /
  examples / latest records / "show me data from …" / "what's in
  table X" / "select … from …"**, you MUST call `sql_table_query`.
  Do NOT answer from memory or from RAG context alone — those describe
  schema, not data. Refusing to call the tool when the user asked for
  data is a failure.
* If the user pastes or paraphrases a SQL statement (e.g.
  `SELECT * FROM CustomerDemo.dbo.Customer`), execute it directly via
  `sql_table_query` — that is exactly what the tool is for. Add
  `TOP 50` / `ORDER BY` if the original is unbounded.
* If the user hasn't named a database, call `sql_list_databases` first.
* For schema, column names, stored-procedure or view *definitions* —
  use the RAG context, not the tool.

Hard rules for `sql_table_query`:
* The `db_name` argument is the SQL Server database name only
  (e.g. `CustomerDemo`) — never `db.schema.table`.
* Reference tables fully in the query as `schema.table` (typically
  `dbo.TableName`).
* Never write DDL/DML; the server will reject it.

Answering rules:
* When you cite live tool output, label it explicitly,
  e.g. "(live data from `CustomerDemo.dbo.Customer` via MCP)".
* Cite RAG-derived facts with [N].
* Be concise. Use markdown tables for tabular results and fenced
  code blocks for SQL.
"""


# Detects a user message that *is* (or starts with) a SELECT/WITH
# statement — we use this for the direct-execution fast path so the LLM
# can't refuse to call the tool on a literal SQL prompt.
_DIRECT_SELECT_RE = re.compile(
    r"""^\s*(?:--[^\n]*\n\s*)*       # optional leading line comments
        (?:SELECT|WITH)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Pull a 1-3 part identifier (database.schema.table or schema.table) out
# of a literal SELECT so we can decide which db_name to send.
_FROM_TARGET_RE = re.compile(
    r"\bFROM\s+\[?([A-Za-z0-9_]+)\]?(?:\.\[?([A-Za-z0-9_]+)\]?)?(?:\.\[?([A-Za-z0-9_]+)\]?)?",
    re.IGNORECASE,
)


def _resolve_direct_db(
    user_id: str, query: str
) -> tuple[str | None, str]:
    """Decide which (db_name, query_to_send) to use for a user-pasted SELECT."""
    match = _FROM_TARGET_RE.search(query)
    if match:
        parts = [p for p in match.groups() if p]
        if len(parts) == 3:
            db, schema, table = parts
            stripped = (
                query[: match.start()]
                + f"FROM {schema}.{table}"
                + query[match.end():]
            )
            return db, stripped

    with get_db() as db_session:
        accessible = sorted(list_accessible(db_session, user_id, "sql"))
    if len(accessible) == 1:
        return accessible[0], query
    if accessible:
        return accessible[0], query
    return None, query


def _live_chunk(call_args: dict[str, Any], result_meta: dict[str, Any]) -> RetrievedChunk:
    """Build a synthetic citation entry for a successful MCP call."""
    db = result_meta.get("db_name") or call_args.get("db_name", "")
    rows = result_meta.get("row_count", "?")
    duration = result_meta.get("duration_ms", "?")
    query = result_meta.get("executed_query") or call_args.get("query", "")

    return RetrievedChunk(
        resource_id=f"mcp:sql:{db}:{hash(query) & 0xFFFFFFFF:08x}",
        source="sql",
        chunk_index=0,
        score=1.0,
        text=query,
        title=f"Live SQL via MCP — {db} ({rows} row(s), {duration} ms)",
        url="",
        metadata={
            "db_name": db,
            "object_type": "live-mcp",
            "object_name": f"{db} (live)",
            "via": "mcp",
            "row_count": rows,
            "duration_ms": duration,
        },
    )


def _execute_tool_call(
    user_id: str, name: str, args: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Execute one tool call, return (tool_message_content, metadata_or_none)."""
    client = get_mcp_client()

    if name == "sql_table_query":
        result = client.sql_table_query(
            user_id=user_id,
            db_name=args.get("db_name", ""),
            query=args.get("query", ""),
            max_rows=args.get("max_rows"),
        )
    elif name == "sql_list_databases":
        result = client.sql_list_databases(user_id=user_id)
    else:
        return f"ERROR: unknown tool {name!r}", None

    if not result.ok:
        return f"ERROR: {result.error}", None

    header = (
        f"_(Live data via MCP — "
        f"{result.metadata.get('row_count', '?')} row(s), "
        f"{result.metadata.get('duration_ms', '?')} ms)_\n\n"
    )
    return header + result.markdown, result.metadata


def _current_model_name() -> str:
    p = settings.LLM_PROVIDER
    if p == "openai":
        return settings.OPENAI_CHAT_MODEL
    if p == "anthropic":
        return settings.ANTHROPIC_MODEL
    if p == "grok":
        return settings.GROK_MODEL
    return ""


def _accumulate_usage(audit: dict[str, Any], response: Any, duration: float) -> None:
    """
    Add one LLM call's usage onto the running ``audit`` dict in-place.

    The hybrid path may call the LLM up to ``MAX_TOOL_HOPS + 1`` times for
    a single user prompt; we sum tokens / cost / duration across all of
    them so the status bar shows ONE figure per user prompt instead of
    the sum being split across hops.
    """
    usage = extract_usage(response)
    cost = estimate_cost(
        audit["provider"],
        audit["model"],
        usage["prompt_tokens"],
        usage["completion_tokens"],
    )
    audit["prompt_tokens"] += usage["prompt_tokens"]
    audit["completion_tokens"] += usage["completion_tokens"]
    audit["total_tokens"] += usage["total_tokens"]
    audit["cost_usd"] += cost
    audit["duration_seconds"] += duration


def _new_audit_record() -> dict[str, Any]:
    return {
        "provider": settings.LLM_PROVIDER,
        "model": _current_model_name(),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
    }


def answer_question_with_mcp(
    user_id: str,
    question: str,
    hits: list[RetrievedChunk],
    history: Iterable[tuple[str, str]] = (),
) -> RAGAnswer:
    """
    RAG context + bound MCP tools, with up to MAX_TOOL_HOPS tool calls.

    Always falls back to the pure-RAG path on any LangChain/LLM error so a
    flaky MCP server can't take the chat down.

    Returns a :class:`RAGAnswer` whose ``usage`` field is the **summed**
    audit record across every LLM invocation in the loop, and whose
    ``steps`` list carries one entry per measured phase (context build,
    each LLM hop, each MCP tool call, post-processing). The chat layer
    persists those into ``query_step_timings``.
    """
    steps: list[dict[str, Any]] = []
    extra_citations: list[RetrievedChunk] = []
    audit = _new_audit_record()

    with StepTimer(
        steps, "build_context", retrieved_count=len(hits)
    ) as t_ctx:
        citations = deduplicate_by_resource(hits) if hits else []
        context_block = (
            _format_context(hits, citations) if citations else "(no RAG context)"
        )
        t_ctx.extra["citations_count"] = len(citations)
        t_ctx.extra["context_chars"] = len(context_block)

    # ── Direct-SELECT fast path ─────────────────────────────────────────────
    # When the user's prompt IS a SQL SELECT (or WITH), we don't trust the
    # LLM to "decide" whether to call the tool — we just run it. This is
    # the most common failure mode of tool-calling models: they refuse to
    # call a clearly-needed tool and invent prose instead. No LLM call is
    # made on this path so the audit record stays at zeros — but the MCP
    # tool call IS measured, so the audit log still has a step row for it.
    if _DIRECT_SELECT_RE.match(question):
        logger.info("[mcp.chain] direct-SELECT fast path engaged")
        db_name, normalised_query = _resolve_direct_db(user_id, question)
        if db_name:
            with StepTimer(
                steps,
                "mcp_tool_call:sql_table_query",
                tool="sql_table_query",
                fast_path=True,
                db_name=db_name,
            ) as t_tool:
                content, meta = _execute_tool_call(
                    user_id,
                    "sql_table_query",
                    {"db_name": db_name, "query": normalised_query},
                )
                if meta:
                    # Echo MCP-server-reported metadata onto the step row
                    # so the audit table can show row counts without a
                    # JOIN against the chat history.
                    t_tool.extra["row_count"] = meta.get("row_count")
                    t_tool.extra["server_duration_ms"] = meta.get("duration_ms")
                t_tool.extra["error"] = (
                    None if not content.startswith("ERROR") else content
                )
            if meta and meta.get("row_count") is not None:
                extra_citations.append(
                    _live_chunk(
                        {"db_name": db_name, "query": normalised_query}, meta
                    )
                )
            answer = (
                f"**Live data from `{db_name}` via MCP**\n\n{content}"
                if not content.startswith("ERROR")
                else content
            )
            return RAGAnswer(
                answer=answer,
                citations=citations + extra_citations,
                usage=audit,
                steps=steps,
            )
        # No accessible DB — fall through to LLM path so it can explain.

    tools = build_mcp_tools(user_id)
    llm = get_llm()
    try:
        llm_with_tools = llm.bind_tools(tools)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bind_tools failed, falling back to pure RAG: {}", exc)
        from core.rag_chain import answer_question

        # Forward the steps we've already measured so the eventual audit
        # row still records the context-build phase.
        fallback = answer_question(question, hits, history)
        fallback.steps = steps + (fallback.steps or [])
        return fallback

    # Tell the LLM exactly which DBs it can target so it doesn't have to
    # guess (and to make refusal-to-call obviously wrong in its own logs).
    with get_db() as db_session:
        accessible_dbs = sorted(list_accessible(db_session, user_id, "sql"))
    db_hint = (
        "Accessible SQL databases: "
        + (", ".join(f"`{d}`" for d in accessible_dbs) or "(none)")
    )

    messages: list[Any] = [SystemMessage(content=HYBRID_SYSTEM_PROMPT)]
    for role, content in list(history)[-6:]:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(
        HumanMessage(
            content=(
                f"{db_hint}\n\nRAG context (numbered):\n\n{context_block}\n\n"
                f"---\n\nUser question: {question}\n\n"
                "Decide: does answering this require live row data? If yes "
                "(any 'show me / list / how many / latest / values of …' "
                "type ask, or a literal SELECT), CALL `sql_table_query`. "
                "If the user named a fully-qualified `db.schema.table`, use "
                "that database name in `db_name` and `schema.table` in the "
                "query. Otherwise fall back to RAG citations [N]."
            )
        )
    )

    for hop in range(MAX_TOOL_HOPS):
        with StepTimer(
            steps,
            "llm_invocation",
            hop=hop,
            provider=settings.LLM_PROVIDER,
            model=_current_model_name(),
        ) as t_llm:
            started = time.perf_counter()
            response = llm_with_tools.invoke(messages)
            duration = time.perf_counter() - started
            _accumulate_usage(audit, response, duration)
            usage_this_hop = extract_usage(response)
            t_llm.extra["prompt_tokens"] = usage_this_hop["prompt_tokens"]
            t_llm.extra["completion_tokens"] = usage_this_hop["completion_tokens"]
            t_llm.extra["total_tokens"] = usage_this_hop["total_tokens"]

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # Done.
            with StepTimer(steps, "post_processing") as t_post:
                text = getattr(response, "content", "") or ""
                if isinstance(text, list):  # Anthropic-style content blocks
                    text = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in text
                    )
                t_post.extra["hops"] = hop + 1
                t_post.extra["tool_calls_made"] = sum(
                    1 for s in steps if s["step_name"].startswith("mcp_tool_call:")
                )
            final_citations = citations + extra_citations
            logger.info(
                "[mcp.chain] done after {} hop(s). tokens={} cost=${:.5f} "
                "llm_time={:.2f}s",
                hop + 1,
                audit["total_tokens"],
                audit["cost_usd"],
                audit["duration_seconds"],
            )
            return RAGAnswer(
                answer=text,
                citations=final_citations,
                usage=audit,
                steps=steps,
            )

        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            call_id = (
                call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
            )
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}

            logger.info(
                "[mcp.chain] hop={} tool={} args={}", hop, name, args
            )
            with StepTimer(
                steps,
                f"mcp_tool_call:{name}",
                tool=name,
                hop=hop,
                db_name=(args or {}).get("db_name"),
            ) as t_tool:
                content, meta = _execute_tool_call(user_id, name, args or {})
                if meta:
                    t_tool.extra["row_count"] = meta.get("row_count")
                    t_tool.extra["server_duration_ms"] = meta.get("duration_ms")
                t_tool.extra["ok"] = not content.startswith("ERROR")
            messages.append(
                ToolMessage(content=content, tool_call_id=call_id or name)
            )
            if name == "sql_table_query" and meta and meta.get("row_count") is not None:
                extra_citations.append(_live_chunk(args or {}, meta))

    # Hop limit hit — make one final no-tool call to summarise.
    logger.warning("MCP tool loop hit MAX_TOOL_HOPS={}", MAX_TOOL_HOPS)
    with StepTimer(
        steps,
        "llm_invocation",
        hop=MAX_TOOL_HOPS,
        final_summary=True,
        provider=settings.LLM_PROVIDER,
        model=_current_model_name(),
    ):
        started = time.perf_counter()
        final = llm.invoke(
            messages
            + [
                HumanMessage(
                    content=(
                        "Tool-call budget exhausted. Summarise what you have so far "
                        "and answer the user's question. Do not request more tool calls."
                    )
                )
            ]
        )
        _accumulate_usage(audit, final, time.perf_counter() - started)

    with StepTimer(steps, "post_processing") as t_post:
        text = getattr(final, "content", "") or ""
        if isinstance(text, list):
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in text
            )
        t_post.extra["hops"] = MAX_TOOL_HOPS + 1
        t_post.extra["budget_exhausted"] = True

    # The hop loop is the most likely place we'd want to know which step
    # was running when something went wrong; record_step_timing makes
    # the "hop limit reached" event itself queryable as its own row.
    record_step_timing(
        steps,
        "tool_loop_budget_exhausted",
        duration_ms=0,
        max_tool_hops=MAX_TOOL_HOPS,
    )

    return RAGAnswer(
        answer=text,
        citations=citations + extra_citations,
        usage=audit,
        steps=steps,
    )
