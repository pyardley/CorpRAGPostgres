"""
Chat UI - turns each user message into a metadata-filtered pgvector retrieval
plus an LLM call, and renders the result with citations.

Multi-tenancy is enforced *here* at query time. The retriever does NOT inject
``user_id`` into the filter; instead we build a filter from
:class:`SelectionState` (sidebar) + the user's accessible-resources rows.

The chat layer also owns the **status-bar update** AND the **audit row**
for each turn:

* It wraps the whole answer-generation block (retrieve → reason → reply)
  with a wall-clock timer so the user-perceived latency is what shows up
  in the bar (rather than the LLM-only duration the chains report).
* Two early steps — ``"build_filter"`` and ``"vector_retrieval"`` — are
  measured here because the chains never see them. They get prepended
  to whatever steps the chain produced before persistence.
* On both success AND error the chat layer calls
  ``app.utils.log_query_audit`` with a fully-populated audit row + the
  list of step timings. The helper is wrapped so audit failures never
  surface to the user.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import streamlit as st
from loguru import logger

from app.auth import current_user
from app.config import settings
from app.sidebar import SelectionState
from app.utils import (
    StepTimer,
    get_db,
    init_session_totals,
    log_query_audit,
    record_step_timing,
    render_status_bar,
    update_session_totals,
)
from core.live_acl import revalidate
from core.query_rewrite import rewrite_query
from core.rag_chain import RAGAnswer, answer_question
from core.reranker import rerank
from core.response_cache import check_response_cache, store_response_cache
from core.retriever import RetrievedChunk, retrieve
from core.sql_object_context import expand_hits_with_dependencies, expand_sql_chunks
from core.trace_completeness import is_tracing_question
from core.vector_store import build_query_filter


def _build_filter(state: SelectionState) -> dict[str, Any]:
    """Translate the SelectionState into a vector_chunks metadata filter."""
    return build_query_filter(
        selected_sources=state.sources,
        accessible_jira_projects=state.jira_projects,
        accessible_confluence_spaces=state.confluence_spaces,
        accessible_databases=state.sql_databases,
        accessible_git_scopes=state.git_scopes,
        accessible_email_providers=state.email_providers,
        accessible_github_issue_repos=state.github_issue_repos,
    )


def _revalidate_filter(user_id: str, filter_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Intersect every source's candidate scopes in `filter_dict["by_source"]`
    with a live permission check (see `core.live_acl`). No-ops when
    `settings.LIVE_ACL_REVALIDATION_ENABLED` is False. Sources that end up
    with no confirmed scopes are dropped entirely.
    """
    by_source: dict[str, list[str]] = filter_dict.get("by_source") or {}
    if not by_source:
        return filter_dict

    with get_db() as db:
        revalidated = {
            source: revalidate(db, user_id, source, scopes)
            for source, scopes in by_source.items()
        }
    return {**filter_dict, "by_source": {s: v for s, v in revalidated.items() if v}}


def _resolve_source_type(state: SelectionState, used_mcp: bool) -> str:
    """
    Decide the ``source_type`` value for the audit row.

    ``"mcp_only"`` is reserved for the direct-SELECT fast path the user
    triggers by pasting raw SQL; "hybrid" is anything that bound the MCP
    tools to the LLM; "rag" is everything else.
    """
    if not used_mcp:
        return "rag"
    return "hybrid"


def _render_citations(citations: list[RetrievedChunk]) -> None:
    if not citations:
        return
    st.markdown("---")
    st.markdown("**Sources:**")
    for i, c in enumerate(citations, start=1):
        if c.source == "jira":
            badge = "\U0001F535 Jira"
            label = c.title or c.resource_id
        elif c.source == "confluence":
            badge = "\U0001F7E2 Confluence"
            label = c.title or c.resource_id
        elif c.source == "sql":
            obj = c.metadata.get("object_name", "")
            db = c.metadata.get("db_name", "")
            obj_type = c.metadata.get("object_type", "object").upper()
            if c.metadata.get("via") == "mcp":
                badge = "⚡ SQL (live, MCP)"
                rows = c.metadata.get("row_count", "?")
                label = f"Live data from `{db}` — {rows} row(s)"
            else:
                badge = "\U0001F7E0 SQL"
                label = f"{db} › {obj} ({obj_type})"
        elif c.source == "git":
            badge = "\U0001F7E3 Git"
            repo = c.metadata.get("repo_name", "")
            branch = c.metadata.get("branch", "")
            git_type = c.metadata.get("git_type", "")
            if git_type == "commit":
                sha = c.metadata.get("sha", "")
                label = f"{repo}@{branch} · `{sha}` {c.title}"
            else:
                file_path = c.metadata.get("file_path", "")
                label = f"{repo}@{branch} · `{file_path}`"
        elif c.source == "email":
            provider = c.metadata.get("email_provider", "email")
            sender = c.metadata.get("from", "")
            badge = (
                "\U0001F4E8 Outlook"
                if provider == "outlook"
                else "\U0001F4EC Gmail"
                if provider == "gmail"
                else "✉ Email"
            )
            subject = c.title or c.resource_id
            label = f"{subject} — _{sender}_" if sender else subject
        else:
            badge = "•"
            label = c.title or c.resource_id

        if c.url:
            st.markdown(
                f"**[{i}]** {badge} — [{label}]({c.url})  *(score {c.score:.2f})*"
            )
        else:
            st.markdown(
                f"**[{i}]** {badge} — {label}  *(score {c.score:.2f})*"
            )


def _badges(state: SelectionState) -> str:
    parts: list[str] = []
    if "jira" in state.sources and state.jira_projects:
        parts.append(f"\U0001F535 Jira ({len(state.jira_projects)})")
    if "confluence" in state.sources and state.confluence_spaces:
        parts.append(f"\U0001F7E2 Confluence ({len(state.confluence_spaces)})")
    if "sql" in state.sources and state.sql_databases:
        # Same "hybrid/MCP path is active" condition as used_mcp_path
        # below, minus the ENABLE_ENTITY_GRAPH env-flag arm — that one's
        # a pre-existing invisible admin setting and stays unreflected
        # here. fts_language, unlike that flag, is a sidebar control the
        # user just set, so silently changing answer behavior without
        # visible feedback here would be confusing.
        mcp_active = state.use_mcp_sql or state.fts_language == "simple"
        suffix = " + ⚡MCP" if mcp_active else ""
        parts.append(f"\U0001F7E0 SQL ({len(state.sql_databases)}){suffix}")
    if "git" in state.sources and state.git_scopes:
        parts.append(f"\U0001F7E3 Git ({len(state.git_scopes)})")
    if "email" in state.sources and state.email_providers:
        parts.append(f"✉ Email ({len(state.email_providers)})")
    return " · ".join(parts) if parts else ""


def render_chat(state: SelectionState) -> None:
    user = current_user()
    if not user:
        return

    # Defensive — main.py initialises this on login, but if someone
    # imports render_chat from another page or a test harness we still
    # want the bar to show zeros instead of crashing on a KeyError.
    init_session_totals()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Snapshot whether messages were empty *before* this run starts. The
    # sidebar (which has already rendered above this in main.py's call order)
    # used the same pre-run snapshot to decide whether to disable the
    # Clear-chat button. If we're about to add the very first message in this
    # run, we trigger an explicit st.rerun() at the end so the sidebar gets
    # a chance to re-render with the now-populated list and enable the button.
    _was_empty_at_start = len(st.session_state.messages) == 0

    # Note: the Clear-chat button is rendered by app/sidebar.py at the top of
    # the sidebar so it's always above the fold. Don't duplicate it here.

    st.title("\U0001F50D CorporateRAG")

    badges = _badges(state)
    if badges:
        st.caption("Searching across: " + badges)
    else:
        st.warning(
            "No queryable scopes selected. Enable a source and run an "
            "ingestion to grant yourself access, then come back here."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                _render_citations(msg["citations"])

    prompt = st.chat_input(
        "Ask anything about your Jira, Confluence, SQL, Git, or Email data…"
    )
    if not prompt:
        # Render the bar before returning so it persists across reruns
        # where no new prompt was entered.
        render_status_bar()
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Audit + step-timing setup ───────────────────────────────────────────
    # The chat layer owns the build_filter / vector_retrieval / total
    # steps; the chains contribute everything else. Both lists end up
    # in the same ``query_step_timings`` audit child rows.
    chat_steps: list[dict[str, Any]] = []
    audit_record: dict[str, Any] | None = None
    success = True
    error_message: str | None = None
    # The entity-graph MCP tool has no dedicated sidebar toggle (same
    # precedent as reranking / live-ACL / query rewriting — a pure
    # .env-level flag). Once an operator sets ENABLE_ENTITY_GRAPH=true,
    # every turn takes the hybrid/MCP path so the tool is available,
    # independent of whether the user has "Use Live SQL" on.
    #
    # "simple" search language is the user's own signal that they want
    # literal/identifier matching over SQL objects (proc names, column
    # names, error codes) rather than prose stemming — the same shape of
    # question the SQL impact-analysis MCP tools (sql_dependency_graph /
    # sql_object_definition / sql_object_dependencies) answer best. This
    # augments retrieval rather than replacing it: `hits` below is
    # computed identically either way, this only decides whether the LLM
    # also gets the MCP tools bound.
    used_mcp_path = bool(
        (state.use_mcp_sql and "sql" in state.sources)
        or settings.ENABLE_ENTITY_GRAPH
        or (state.fts_language == "simple" and "sql" in state.sources)
    )

    # Wall-clock the entire answer-generation cycle (retrieval + reasoning
    # + tool calls). This is the user-perceived latency and is what we
    # want the bar's "time" column AND the audit row's
    # ``total_duration_ms`` to reflect — not the LLM-only timing the
    # chains report separately in their log lines.
    turn_started = time.perf_counter()
    result: RAGAnswer
    with st.chat_message("assistant"):
        with st.spinner("Searching & reasoning…"):
            try:
                with StepTimer(
                    chat_steps,
                    "build_filter",
                    sources=list(state.sources),
                    use_mcp=used_mcp_path,
                    fts_language=state.fts_language,
                ) as t_filter:
                    filter_dict = _build_filter(state)
                    t_filter.extra["filter_keys"] = sorted(filter_dict.keys())
                logger.debug("Chat filter: {}", filter_dict)

                with StepTimer(chat_steps, "live_acl_check") as t_acl:
                    # No-ops (single dict lookup) unless
                    # LIVE_ACL_REVALIDATION_ENABLED is set — see
                    # core.live_acl for the source-system permission checks.
                    filter_dict = _revalidate_filter(user["id"], filter_dict)
                    t_acl.extra["filter_keys"] = sorted(filter_dict.keys())
                logger.debug("Chat filter after live ACL check: {}", filter_dict)

                with StepTimer(chat_steps, "response_cache_check") as t_cache:
                    # No-ops (returns None) unless RESPONSE_CACHE_ENABLED is
                    # set, and never checked on the MCP/hybrid path — live
                    # SQL data must never be served from a cache. See
                    # core.response_cache: a hit skips query rewriting,
                    # retrieval, and reranking entirely, not just the LLM
                    # call, since the cache key (a fingerprint of the
                    # *resolved* accessible scopes, not user_id) is already
                    # known before any of those steps run.
                    #
                    # Also skipped for lineage/impact-analysis-style
                    # questions over SQL sources: a schema re-ingestion
                    # (redefined proc, added dependency) doesn't invalidate
                    # a previously-cached trace answer, and staleness here
                    # is worse than a cache miss — see core.trace_completeness.
                    skip_cache = used_mcp_path or (
                        "sql" in state.sources and is_tracing_question(prompt)
                    )
                    cached = (
                        None
                        if skip_cache
                        else check_response_cache(
                            user["id"], prompt, filter_dict, state.fts_language
                        )
                    )
                    t_cache.extra["hit"] = cached is not None
                    t_cache.extra["skipped"] = skip_cache

                if cached is not None:
                    result = cached
                else:
                    with StepTimer(chat_steps, "query_rewrite") as t_rewrite:
                        # No-ops (returns [prompt]) unless QUERY_REWRITE_ENABLED
                        # is set — see core.query_rewrite for the decomposition
                        # LLM call.
                        sub_queries = rewrite_query(prompt)
                        t_rewrite.extra["sub_query_count"] = len(sub_queries)

                    with StepTimer(chat_steps, "vector_retrieval") as t_retr:
                        # user_id is required when ENABLE_RLS=True so the
                        # RLS policies on vector_chunks recognise the
                        # caller. The retriever no-ops the binding when
                        # RLS is disabled. When reranking is enabled we ask
                        # for a wider candidate pool (RERANK_CANDIDATE_K)
                        # than the usual TOP_K so there's something for the
                        # cross-encoder to actually rerank.
                        #
                        # One retrieve() call per sub-query (usually just one —
                        # query rewriting is off by default), run concurrently
                        # so N sub-queries cost close to one round trip of
                        # wall-clock time rather than N. Each retrieve() call
                        # opens its own DB session internally, so concurrent
                        # calls are safe. Pooled by (resource_id, chunk_index),
                        # keeping the higher score on overlap — lighter-weight
                        # than deduplicate_by_resource, which collapses to one
                        # chunk *per resource* and would be too aggressive here,
                        # before rerank even runs.
                        top_k = settings.RERANK_CANDIDATE_K if settings.RERANK_ENABLED else None
                        pooled: dict[tuple[str, int], RetrievedChunk] = {}
                        with ThreadPoolExecutor(max_workers=len(sub_queries)) as pool:
                            futures = [
                                pool.submit(
                                    retrieve,
                                    q,
                                    filter_dict,
                                    top_k=top_k,
                                    user_id=user["id"],
                                    fts_language=state.fts_language,
                                )
                                for q in sub_queries
                            ]
                            for future in futures:
                                for hit in future.result():
                                    key = (hit.resource_id, hit.chunk_index)
                                    existing = pooled.get(key)
                                    if existing is None or hit.score > existing.score:
                                        pooled[key] = hit
                        hits = list(pooled.values())
                        t_retr.extra["sub_query_count"] = len(sub_queries)
                        t_retr.extra["retrieved_count"] = len(hits)

                    with StepTimer(chat_steps, "rerank") as t_rerank:
                        # No-ops (single slice) unless RERANK_ENABLED is set —
                        # see core.reranker for the cross-encoder pass.
                        t_rerank.extra["candidate_count"] = len(hits)
                        hits = rerank(prompt, hits, top_n=settings.TOP_K)
                        t_rerank.extra["reranked_count"] = len(hits)

                    with StepTimer(chat_steps, "sql_object_expansion") as t_expand:
                        # A SQL hit may only be one fragment of a larger
                        # stored procedure/view/function/trigger body (see
                        # core.sql_object_context) — reassemble every
                        # fragmented SQL object into its full definition
                        # before the LLM ever sees it, so a lineage-tracing
                        # answer can't silently skip a middle stage just
                        # because that stage's chunk didn't win a slot.
                        hits = expand_sql_chunks(hits, user["id"])
                        t_expand.extra["expanded_count"] = len(hits)

                    with StepTimer(chat_steps, "sql_dependency_expansion") as t_dep_expand:
                        # No-ops (returns hits unchanged) unless
                        # SQL_DEPENDENCY_FORCED_INCLUSION_ENABLED is set —
                        # off by default. When on, force-includes objects
                        # directly connected via the static SQL dependency
                        # graph that never won a retrieval slot at all
                        # (expand_sql_chunks above only helps once an
                        # object already made the cut).
                        #
                        # Plain-RAG path only: on the hybrid/MCP path the
                        # LLM already has sql_object_dependencies /
                        # sql_dependency_graph as on-demand tools, so
                        # forcing extra objects into every prompt there is
                        # more likely redundant cost than value — the
                        # plain path has no such fallback, which is the
                        # actual gap this closes.
                        chars_before = sum(len(h.text) for h in hits)
                        if not used_mcp_path:
                            hits = expand_hits_with_dependencies(hits, user["id"])
                        t_dep_expand.extra["expanded_count"] = len(hits)
                        t_dep_expand.extra["chars_added"] = (
                            sum(len(h.text) for h in hits) - chars_before
                        )

                    history = [
                        (m["role"], m["content"])
                        for m in st.session_state.messages
                        if m["role"] in {"user", "assistant"}
                    ][:-1]

                    if used_mcp_path:
                        # Hybrid path: RAG context + bound SQL MCP tools.
                        from core.mcp_chain import answer_question_with_mcp

                        result = answer_question_with_mcp(
                            user_id=user["id"],
                            question=prompt,
                            hits=hits,
                            history=history,
                        )
                    else:
                        result = answer_question(prompt, hits, history)
                        # Only the plain RAG path is cached — see the
                        # core.response_cache module docstring for why the
                        # MCP/hybrid path is excluded (live data must never
                        # be served from a cache). `skip_cache` additionally
                        # excludes tracing-style SQL questions (see the
                        # response_cache_check step above) so a stale
                        # lineage answer never gets stored either.
                        if not skip_cache:
                            store_response_cache(
                                user["id"], prompt, filter_dict, state.fts_language, result
                            )

                audit_record = dict(result.usage or {})
            except Exception as exc:
                logger.exception("Chat error")
                success = False
                error_message = f"{type(exc).__name__}: {exc}"
                result = RAGAnswer(
                    answer=f"Sorry — an error occurred: {exc}",
                    citations=[],
                    usage={},
                    steps=[],
                )

        st.markdown(result.answer)
        _render_citations(result.citations)

    # Wall-clock for the whole turn — what the user actually waited for.
    total_seconds = time.perf_counter() - turn_started

    # Merge the chain's steps in after the chat-owned ones, then append
    # a final "total" step that mirrors the audit row's total duration
    # so a single SELECT against query_step_timings already shows
    # end-to-end timing without a JOIN.
    all_steps: list[dict[str, Any]] = list(chat_steps)
    all_steps.extend(result.steps or [])
    record_step_timing(
        all_steps,
        "total",
        duration_ms=int(total_seconds * 1000),
        success=success,
    )

    # Merge the user-perceived duration into the chain's audit record and
    # bump the running session totals. The chain reports its own LLM-only
    # ``duration_seconds`` for the audit log; the bar wants wall-clock.
    bar_audit = dict(audit_record or {})
    bar_audit["duration_seconds"] = total_seconds
    update_session_totals(bar_audit)

    # Persist the audit row + every step timing. Helper is best-effort
    # and never raises — a broken DB must not break the chat.
    log_query_audit(
        user_id=user["id"],
        prompt_text=prompt,
        audit=audit_record,
        total_duration_seconds=total_seconds,
        source_type=_resolve_source_type(state, used_mcp_path),
        success=success,
        error_message=error_message,
        steps=all_steps,
        fts_language=state.fts_language,
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "citations": result.citations}
    )

    # First-turn rerun: the sidebar rendered before us with messages still
    # empty, which left the Clear-chat button disabled. Force one extra
    # render so the sidebar can pick up the now-populated list. Subsequent
    # turns don't need this because the sidebar already saw a non-empty
    # list on the previous render.
    if _was_empty_at_start:
        st.rerun()

    # Render the persistent status bar at the very bottom of the chat
    # surface. This is the only place it appears — keeping it inside
    # render_chat means it's automatically scoped to the authenticated
    # chat view (anonymous users hit the login page and never get here).
    render_status_bar()
