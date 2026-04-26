"""
Streamlit sidebar - source selection, credentials, and ingestion controls.

What's different from the old version
-------------------------------------
* The Pinecone index is shared across the whole org. The sidebar no longer
  decides "what data exists"; it decides "what the *current user* is allowed
  to query right now". The source of truth for that is the
  ``user_accessible_resources`` table.
* Scope dropdowns are populated from that table, NOT from a live API call to
  Jira/Confluence/SQL/Git. The accessible_resources rows are inserted by the
  ingestion pipeline whenever the user has successfully ingested a scope.
* The chat layer uses :class:`SelectionState` to build a Pinecone metadata
  filter (see ``core.vector_store.build_query_filter``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
from loguru import logger

from app.auth import current_user, logout_session
from app.utils import (
    get_db,
    list_accessible,
    load_all_credentials,
    revoke_access,
    save_credential,
)


# Selection state
@dataclass
class SelectionState:
    """Snapshot of what the user wants to query right now."""

    user_id: str = ""
    sources: list[str] = field(default_factory=list)
    jira_projects: list[str] = field(default_factory=list)
    confluence_spaces: list[str] = field(default_factory=list)
    sql_databases: list[str] = field(default_factory=list)
    # For git, each entry is "{repo_full_name}@{branch}", matching the
    # `git_scope` field written to chunk metadata.
    git_scopes: list[str] = field(default_factory=list)


# Credential form
def _credential_form(
    source: str,
    fields: list[tuple[str, str, bool]],
    form_key: Optional[str] = None,
) -> None:
    user = current_user()
    if not user:
        return

    key = form_key or f"creds_{source}"
    with st.form(key):
        values: dict[str, str] = {}
        with get_db() as db:
            existing = load_all_credentials(db, user["id"], source)
        for field_key, label, is_secret in fields:
            existing_val = existing.get(field_key, "")
            placeholder = "●●●●●●●●" if (is_secret and existing_val) else ""
            values[field_key] = st.text_input(
                label,
                value="" if is_secret else existing_val,
                type="password" if is_secret else "default",
                placeholder=placeholder,
                key=f"{key}_{field_key}",
            )
        if st.form_submit_button("Save credentials", use_container_width=True):
            with get_db() as db:
                for field_key, _, _ in fields:
                    val = values[field_key].strip()
                    if val:
                        save_credential(db, user["id"], source, field_key, val)
            st.success(f"{source.title()} credentials saved.")
            st.rerun()


# Ingestion trigger (synchronous, with live progress via st.status)
def _trigger_ingestion(user_id: str, source: str, mode: str, scope: str) -> None:
    """
    Run ingestion in the foreground so the user gets live progress + a clear
    completion banner. Streamlit is single-threaded per session, so this
    blocks the rest of the page until done — that's the price of getting
    real-time progress without a polling loop.
    """
    from app.ingestion.cli import _make_ingestor, ALL_SOURCES
    from app.utils import init_db

    init_db()
    sources_to_run = ALL_SOURCES if source == "all" else [source]

    label = f"Ingesting {source} (scope={scope})…"
    with st.status(label, expanded=True) as status:
        totals = {"items": 0, "vectors": 0}
        any_failed = False

        for src in sources_to_run:
            status.write(f"**▶ {src}**")
            placeholder = st.empty()

            def on_progress(snapshot: dict) -> None:
                stage = snapshot.get("stage", "")
                if stage == "starting":
                    placeholder.caption(f"Starting {snapshot['source']}…")
                elif stage == "processing":
                    placeholder.caption(
                        f"{snapshot['source']}: {snapshot['items_processed']} items, "
                        f"{snapshot['vectors_upserted']} vectors — "
                        f"latest: {snapshot['current_item'][:80]}"
                    )
                elif stage == "done":
                    placeholder.caption(
                        f"{snapshot['source']}: done — "
                        f"{snapshot['items_processed']} items, "
                        f"{snapshot['vectors_upserted']} vectors."
                    )

            try:
                with get_db() as db:
                    ingestor = _make_ingestor(db, src, user_id, scope, mode)
                    result = ingestor.run(on_progress=on_progress)
                totals["items"] += result.items_processed
                totals["vectors"] += result.vectors_upserted
                status.write(
                    f"✅ {src}: {result.items_processed} items, "
                    f"{result.vectors_upserted} vectors"
                )
            except Exception as exc:
                any_failed = True
                logger.exception("Ingestion failed for {}", src)
                status.write(f"❌ {src}: {exc}")

        if any_failed:
            status.update(
                label=(
                    f"Ingestion finished with errors — "
                    f"{totals['items']} items, {totals['vectors']} vectors total. "
                    f"See messages above."
                ),
                state="error",
                expanded=True,
            )
        else:
            status.update(
                label=(
                    f"Ingestion complete ✓ — {totals['items']} items, "
                    f"{totals['vectors']} vectors across {len(sources_to_run)} source(s)."
                ),
                state="complete",
                expanded=False,
            )


# Scope picker UI
def _scope_picker(label: str, source: str, user_id: str) -> list[str]:
    """Return the user's selected resource identifiers for `source`."""
    with get_db() as db:
        accessible = sorted(list_accessible(db, user_id, source))

    if not accessible:
        st.caption(
            f"No {source} scopes accessible yet. Configure credentials and "
            f"run an ingestion below to grant access."
        )
        return []

    select_all = st.checkbox(
        f"Use all {label.lower()}",
        value=True,
        key=f"{source}_all",
        help=f"When ticked, queries see every {source} scope you have access to.",
    )
    if select_all:
        return accessible

    return st.multiselect(
        f"Specific {label.lower()}",
        options=accessible,
        default=accessible,
        key=f"{source}_specific",
    )


# Main renderer
def render_sidebar() -> SelectionState:
    user = current_user()
    if not user:
        return SelectionState()

    state = SelectionState(user_id=user["id"])

    with st.sidebar:
        st.markdown(f"### 👤 {user['email']}")
        if st.button("Sign out", use_container_width=True):
            logout_session()
            st.rerun()

        # Clear chat — always rendered (disabled when there's nothing to clear)
        # so it's visible from the very first page load. We can't gate this on
        # `messages` being non-empty because the sidebar renders BEFORE the
        # chat layer populates messages on the first turn — Streamlit doesn't
        # auto-rerender once the chat appends, so a conditional render would
        # leave the button missing until the next user interaction.
        has_messages = bool(st.session_state.get("messages"))
        if st.button(
            "🗑 Clear chat",
            key="clear_chat_sidebar",
            use_container_width=True,
            disabled=not has_messages,
            help=None if has_messages else "No messages to clear yet.",
        ):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()

        # Source toggles + scope pickers
        st.markdown("### 📚 Data Sources")

        if st.checkbox("Jira", value=True, key="src_jira"):
            state.sources.append("jira")
            with st.expander("🔵 Jira projects", expanded=True):
                state.jira_projects = _scope_picker("Projects", "jira", user["id"])

        if st.checkbox("Confluence", value=True, key="src_confluence"):
            state.sources.append("confluence")
            with st.expander("🟢 Confluence spaces", expanded=True):
                state.confluence_spaces = _scope_picker(
                    "Spaces", "confluence", user["id"]
                )

        if st.checkbox("SQL Server", value=True, key="src_sql"):
            state.sources.append("sql")
            with st.expander("🟠 SQL databases", expanded=True):
                state.sql_databases = _scope_picker("Databases", "sql", user["id"])

        if st.checkbox("Git", value=True, key="src_git"):
            state.sources.append("git")
            with st.expander("🟣 Git scopes (repo@branch)", expanded=True):
                state.git_scopes = _scope_picker(
                    "Repos+branches", "git", user["id"]
                )

        st.divider()

        # Credentials
        with st.expander("🔑 Credentials"):
            tab_j, tab_c, tab_s, tab_g = st.tabs(
                ["Jira", "Confluence", "SQL", "Git"]
            )

            with tab_j:
                _credential_form(
                    "jira",
                    [
                        ("url", "Jira base URL (https://myorg.atlassian.net)", False),
                        ("email", "Atlassian account email", False),
                        ("api_token", "API Token", True),
                    ],
                )

            with tab_c:
                st.caption(
                    "Use your Confluence site URL - copy everything up to "
                    "(but not including) `/wiki`."
                )
                _credential_form(
                    "confluence",
                    [
                        ("url", "Confluence site URL", False),
                        ("email", "Atlassian account email", False),
                        ("api_token", "API Token", True),
                    ],
                )
                with st.expander("Additional Confluence instance (optional)"):
                    _credential_form(
                        "confluence",
                        [
                            ("url_2", "Second Confluence site URL", False),
                            ("email_2", "Email (blank = reuse primary)", False),
                            ("api_token_2", "API Token (blank = reuse primary)", True),
                        ],
                        form_key="creds_confluence_2",
                    )

            with tab_s:
                st.caption("Provide a pyodbc-compatible connection string.")
                _credential_form(
                    "sql",
                    [
                        (
                            "conn_str",
                            "ODBC connection string (DRIVER=...;SERVER=...;UID=...;PWD=...)",
                            True,
                        ),
                    ],
                )

            with tab_g:
                st.caption(
                    "Connect a GitHub repository. For private repos create a "
                    "Personal Access Token with **repo** scope at "
                    "github.com - Settings - Developer settings."
                )
                _credential_form(
                    "git",
                    [
                        ("url", "Repository URL (https://github.com/owner/repo)", False),
                        ("access_token", "Personal Access Token", True),
                        (
                            "file_extensions",
                            "File extensions, comma-separated "
                            "(blank = sensible defaults)",
                            False,
                        ),
                        ("max_commits", "Max commits per branch (default 200)", False),
                    ],
                )
                with st.expander("Additional Git repository (optional)"):
                    _credential_form(
                        "git",
                        [
                            ("url_2", "Second repo URL", False),
                            ("access_token_2", "PAT (blank = reuse primary)", True),
                        ],
                        form_key="creds_git_2",
                    )

        # Ingestion controls — incremental only.
        # Full-mode ingestion is intentionally NOT exposed in the UI: a full
        # ingest wipes by metadata filter (not by user_id), so a user with a
        # narrower view than the previous ingestor would silently delete the
        # other user's vectors. Run full ingests from the CLI on an admin
        # account that owns the union of all scopes:
        #   python -m app.ingestion.cli --source <src> --mode full --scope <key>
        with st.expander("⚙️ Ingest data"):
            ing_source = st.selectbox(
                "Source",
                ["all", "jira", "confluence", "sql", "git"],
                key="ing_source",
            )
            ing_scope = st.text_input(
                "Scope (project_key / space_key / db_name / branch, or 'all')",
                value="all",
                key="ing_scope",
            )
            st.caption(
                "Mode is **incremental** — only items changed since your last "
                "successful run are re-embedded. For full rebuilds, use the "
                "CLI from an admin account."
            )

            if st.button("Run ingestion", use_container_width=True):
                _trigger_ingestion(
                    user["id"], ing_source, "incremental", ing_scope
                )

            _render_recent_logs(user["id"])

        # Access management
        with st.expander("🛡 Manage my access"):
            st.caption(
                "Each row is a scope you can currently query. Removing a row "
                "doesn't delete vectors from Pinecone - it only revokes "
                "*your* visibility into that scope."
            )
            for src in ("jira", "confluence", "sql", "git"):
                with get_db() as db:
                    items = sorted(list_accessible(db, user["id"], src))
                if not items:
                    continue
                st.markdown(f"**{src}**")
                for item in items:
                    cols = st.columns([5, 1])
                    cols[0].caption(item)
                    if cols[1].button("✖", key=f"revoke_{src}_{item}"):
                        with get_db() as db:
                            revoke_access(db, user["id"], src, item)
                        st.rerun()

    return state


# Recent ingestion logs
def _render_recent_logs(user_id: str) -> None:
    from models.ingestion_log import IngestionLog

    with get_db() as db:
        rows = (
            db.query(IngestionLog)
            .filter_by(user_id=user_id)
            .order_by(IngestionLog.started_at.desc())
            .limit(5)
            .all()
        )
        logs = [
            {
                "status": r.status,
                "source": r.source,
                "scope": r.scope,
                "mode": r.mode,
                "items_processed": r.items_processed,
                "vectors_upserted": r.vectors_upserted,
            }
            for r in rows
        ]

    if not logs:
        return

    st.caption("Recent runs:")
    for log in logs:
        icon = {"success": "✅", "error": "❌", "running": "⏳"}.get(log["status"], "❓")
        st.caption(
            f"{icon} {log['source']}/{log['scope']} ({log['mode']}) "
            f"items={log['items_processed']} vecs={log['vectors_upserted']}"
        )
