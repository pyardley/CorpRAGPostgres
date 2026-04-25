"""
Chat UI - turns each user message into a Pinecone metadata-filtered retrieval
plus an LLM call, and renders the result with citations.

Multi-tenancy is enforced *here* at query time. The retriever does NOT inject
``user_id`` into the filter; instead we build a filter from
:class:`SelectionState` (sidebar) + the user's accessible-resources rows.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from loguru import logger

from app.auth import current_user
from app.sidebar import SelectionState
from core.rag_chain import RAGAnswer, answer_question
from core.retriever import RetrievedChunk, retrieve
from core.vector_store import build_query_filter


def _build_filter(state: SelectionState) -> dict[str, Any]:
    """Translate the SelectionState into a Pinecone metadata filter."""
    return build_query_filter(
        selected_sources=state.sources,
        accessible_jira_projects=state.jira_projects,
        accessible_confluence_spaces=state.confluence_spaces,
        accessible_databases=state.sql_databases,
        accessible_git_scopes=state.git_scopes,
    )


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
            badge = "\U0001F7E0 SQL"
            obj = c.metadata.get("object_name", "")
            db = c.metadata.get("db_name", "")
            obj_type = c.metadata.get("object_type", "object").upper()
            label = f"{db} \u203a {obj} ({obj_type})"
        elif c.source == "git":
            badge = "\U0001F7E3 Git"
            repo = c.metadata.get("repo_name", "")
            branch = c.metadata.get("branch", "")
            git_type = c.metadata.get("git_type", "")
            if git_type == "commit":
                sha = c.metadata.get("sha", "")
                label = f"{repo}@{branch} \u00b7 `{sha}` {c.title}"
            else:
                file_path = c.metadata.get("file_path", "")
                label = f"{repo}@{branch} \u00b7 `{file_path}`"
        else:
            badge = "\u2022"
            label = c.title or c.resource_id

        if c.url:
            st.markdown(
                f"**[{i}]** {badge} \u2014 [{label}]({c.url})  *(score {c.score:.2f})*"
            )
        else:
            st.markdown(
                f"**[{i}]** {badge} \u2014 {label}  *(score {c.score:.2f})*"
            )


def _badges(state: SelectionState) -> str:
    parts: list[str] = []
    if "jira" in state.sources and state.jira_projects:
        parts.append(f"\U0001F535 Jira ({len(state.jira_projects)})")
    if "confluence" in state.sources and state.confluence_spaces:
        parts.append(f"\U0001F7E2 Confluence ({len(state.confluence_spaces)})")
    if "sql" in state.sources and state.sql_databases:
        parts.append(f"\U0001F7E0 SQL ({len(state.sql_databases)})")
    if "git" in state.sources and state.git_scopes:
        parts.append(f"\U0001F7E3 Git ({len(state.git_scopes)})")
    return " \u00b7 ".join(parts) if parts else ""


def render_chat(state: SelectionState) -> None:
    user = current_user()
    if not user:
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Header row: title on the left, Clear-chat button on the right (only
    # when there is something to clear). We render this on every render so
    # the button is reachable immediately after the model has answered.
    title_col, button_col = st.columns([6, 1])
    with title_col:
        st.title("\U0001F50D CorporateRAG")
    with button_col:
        if st.session_state.messages:
            st.write("")  # spacer to vertically align with the title
            if st.button(
                "\U0001F5D1 Clear chat",
                key="clear_chat",
                use_container_width=True,
            ):
                st.session_state.messages = []
                st.rerun()

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
        "Ask anything about your Jira, Confluence, SQL, or Git data\u2026"
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching & reasoning\u2026"):
            try:
                filter_dict = _build_filter(state)
                logger.debug("Chat filter: {}", filter_dict)
                hits = retrieve(prompt, filter_dict)
                history = [
                    (m["role"], m["content"])
                    for m in st.session_state.messages
                    if m["role"] in {"user", "assistant"}
                ][:-1]
                result: RAGAnswer = answer_question(prompt, hits, history)
            except Exception as exc:
                logger.exception("Chat error")
                result = RAGAnswer(
                    answer=f"Sorry \u2014 an error occurred: {exc}", citations=[]
                )

        st.markdown(result.answer)
        _render_citations(result.citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "citations": result.citations}
    )
