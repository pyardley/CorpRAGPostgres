"""Chat interface: retrieval, LLM call, citation formatting, Streamlit rendering."""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.auth import current_user
from app.sidebar import SelectionState
from core.llm import get_llm
from core.retriever import retrieve

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are CorporateRAG, an expert assistant that answers questions using information \
from the company's Jira tickets, Confluence pages, and SQL Server database objects.

RULES:
1. Answer ONLY from the provided context. If the context does not contain the answer, \
say so clearly — do not hallucinate.
2. For every claim, cite the source inline using [N] notation (e.g. [1], [2]).
3. Do NOT add a sources list at the end — the UI displays sources automatically.
4. Write in markdown. Be concise but complete.

Context (numbered sources):
{context}
"""

_NO_RESULTS_MSG = (
    "I couldn't find any relevant information in the selected sources for your query.\n\n"
    "**Suggestions:**\n"
    "- Check that you have ingested data for the selected sources/projects.\n"
    "- Try rephrasing your question.\n"
    "- Expand your source or project selection in the sidebar."
)


# ── Citation helpers ──────────────────────────────────────────────────────────

def _format_context(docs) -> tuple[str, list[dict[str, Any]]]:
    """
    Build a numbered context string and a parallel list of source metadata dicts.
    Returns (context_text, sources_list).
    """
    parts: list[str] = []
    sources: list[dict[str, Any]] = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source_label = _build_source_label(i, meta)
        parts.append(f"[{i}]\n{source_label}\n{doc.page_content}")
        sources.append({"index": i, "label": source_label, **meta})

    return "\n\n---\n\n".join(parts), sources


def _build_source_label(n: int, meta: dict[str, Any]) -> str:
    src = meta.get("source", "")
    title = meta.get("title", "(untitled)")
    url = meta.get("url")

    if src == "jira":
        return f"Jira • {title}"
    if src == "confluence":
        return f"Confluence • {title}"
    if src == "sql":
        db = meta.get("db_name", "")
        obj = meta.get("object_name", "")
        obj_type = meta.get("object_type", "object").upper()
        return f"SQL Server • {db} • {obj} ({obj_type})"
    if src == "git":
        repo = meta.get("repo_name", "")
        git_type = meta.get("git_type", "")
        return f"Git • {repo} • {title} ({git_type})"
    return title


def _render_citations(sources: list[dict[str, Any]]) -> None:
    """Render a compact citations block below the AI answer."""
    if not sources:
        return

    st.markdown("---")
    st.markdown("**Sources used:**")

    for src in sources:
        n = src["index"]
        source_type = src.get("source", "")
        title = src.get("title", "(untitled)")
        url = src.get("url")

        if source_type == "jira":
            issue_key = src.get("issue_key", "")
            status = src.get("status", "")
            badge = f"`{status}`" if status else ""
            if url:
                st.markdown(f"**[{n}]** 🔵 [{title}]({url}) {badge}")
            else:
                st.markdown(f"**[{n}]** 🔵 {title} {badge}")

        elif source_type == "confluence":
            space = src.get("space_key", "")
            if url:
                st.markdown(f"**[{n}]** 🟢 [{title}]({url}) `{space}`")
            else:
                st.markdown(f"**[{n}]** 🟢 {title} `{space}`")

        elif source_type == "sql":
            db = src.get("db_name", "")
            obj_type = src.get("object_type", "object").upper()
            obj_name = src.get("object_name", "")
            st.markdown(f"**[{n}]** 🟠 `{db}` › `{obj_name}` ({obj_type})")

        elif source_type == "git":
            repo = src.get("repo_name", "")
            git_type = src.get("git_type", "")
            sha = src.get("sha", "")
            file_path = src.get("file_path", "")
            branch = src.get("branch", "")
            if git_type == "commit":
                label = f"`{sha}` {title}"
            else:
                label = f"`{file_path}` @ `{branch}`"
            if url:
                st.markdown(f"**[{n}]** 🟣 [{label}]({url}) `{repo}`")
            else:
                st.markdown(f"**[{n}]** 🟣 {label} `{repo}`")

        else:
            st.markdown(f"**[{n}]** {title}")


# ── Core answer function ──────────────────────────────────────────────────────

def answer_query(
    query: str,
    user_id: str,
    state: SelectionState,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Retrieve context, call the LLM, return (answer_text, sources_list).
    """
    # Build per-dimension scope lists (None = no filter = "all")
    project_keys = None if state.jira_projects == ["all"] else state.jira_projects
    space_keys = None if state.confluence_spaces == ["all"] else state.confluence_spaces
    db_names = None if state.sql_databases == ["all"] else state.sql_databases
    git_branches = None if state.git_branches == ["all"] else state.git_branches
    sources = state.sources if state.sources else None

    from app.config import settings

    docs = retrieve(
        query=query,
        user_id=user_id,
        sources=sources,
        project_keys=project_keys,
        space_keys=space_keys,
        db_names=db_names,
        git_branches=git_branches,
        top_k=settings.TOP_K,
    )

    if not docs:
        return _NO_RESULTS_MSG, []

    context, source_metas = _format_context(docs)
    prompt = _SYSTEM_PROMPT.format(context=context)

    llm = get_llm()
    messages = [SystemMessage(content=prompt), HumanMessage(content=query)]

    logger.info("Calling LLM with {} context docs.", len(docs))
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    return answer, source_metas


# ── Streamlit chat renderer ───────────────────────────────────────────────────

def render_chat(state: SelectionState) -> None:
    """Render the main chat interface."""
    user = current_user()
    if not user:
        return

    # Initialise session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("🔍 CorporateRAG")
    if state.sources:
        badge_parts = []
        if "jira" in state.sources:
            badge_parts.append("🔵 Jira")
        if "confluence" in state.sources:
            badge_parts.append("🟢 Confluence")
        if "sql" in state.sources:
            badge_parts.append("🟠 SQL Server")
        if "git" in state.sources:
            badge_parts.append("🟣 Git")
        st.caption("Searching across: " + " · ".join(badge_parts))
    else:
        st.warning("No data sources selected. Enable at least one source in the sidebar.")

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_citations(msg["sources"])

    # New input
    if prompt := st.chat_input("Ask anything about your Jira, Confluence, or SQL data…"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching & reasoning…"):
                try:
                    answer, sources = answer_query(
                        query=prompt,
                        user_id=user["id"],
                        state=state,
                    )
                except Exception as exc:
                    logger.exception("Error answering query.")
                    answer = f"Sorry, an error occurred: {exc}"
                    sources = []

            st.markdown(answer)
            if sources:
                _render_citations(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
