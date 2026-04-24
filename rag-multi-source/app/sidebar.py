"""
Streamlit sidebar: source selection, credential setup, and ingestion controls.
Returns a SelectionState that the chat module uses for retrieval filtering.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
from loguru import logger

from app.auth import current_user, logout_session
from app.utils import get_db, load_all_credentials, save_credential


@dataclass
class SelectionState:
    sources: list[str] = field(default_factory=list)
    # "all" or a list of specific keys
    jira_projects: list[str] = field(default_factory=lambda: ["all"])
    confluence_spaces: list[str] = field(default_factory=lambda: ["all"])
    sql_databases: list[str] = field(default_factory=lambda: ["all"])
    git_branches: list[str] = field(default_factory=lambda: ["all"])


# ── Cache dynamic lists from APIs ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_jira_projects(user_id: str) -> list[dict]:
    try:
        from app.ingestion.jira_ingestor import JiraIngestor
        with get_db() as db:
            creds = load_all_credentials(db, user_id, "jira")
        if not creds:
            return []
        return JiraIngestor(user_id=user_id, credentials=creds).list_scopes()
    except Exception as exc:
        logger.warning("Could not load Jira projects: {}", exc)
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _cached_confluence_spaces(user_id: str) -> list[dict]:
    try:
        from app.ingestion.confluence_ingestor import ConfluenceIngestor
        with get_db() as db:
            creds = load_all_credentials(db, user_id, "confluence")
        if not creds:
            return []
        return ConfluenceIngestor(user_id=user_id, credentials=creds).list_scopes()
    except Exception as exc:
        logger.warning("Could not load Confluence spaces: {}", exc)
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _cached_sql_databases(user_id: str) -> list[dict]:
    try:
        from app.ingestion.sql_ingestor import SQLIngestor
        with get_db() as db:
            creds = load_all_credentials(db, user_id, "sql")
        if not creds:
            return []
        return SQLIngestor(user_id=user_id, credentials=creds).list_scopes()
    except Exception as exc:
        logger.warning("Could not load SQL databases: {}", exc)
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _cached_git_branches(user_id: str) -> list[dict]:
    try:
        from app.ingestion.git_ingestor import GitIngestor
        with get_db() as db:
            creds = load_all_credentials(db, user_id, "git")
        if not creds:
            return []
        return GitIngestor(user_id=user_id, credentials=creds).list_scopes()
    except Exception as exc:
        logger.warning("Could not load Git branches: {}", exc)
        return []


# ── Credential form helpers ────────────────────────────────────────────────────

def _credential_form(
    source: str,
    fields: list[tuple[str, str, bool]],
    form_key: str | None = None,
) -> None:
    """Render a credential input form and save on submit."""
    user = current_user()
    if not user:
        return

    key = form_key or f"creds_{source}"
    with st.form(key):
        values: dict[str, str] = {}
        with get_db() as db:
            for field_key, label, is_secret in fields:
                existing = load_all_credentials(db, user["id"], source).get(field_key, "")
                placeholder = "●●●●●●●●" if (is_secret and existing) else ""
                values[field_key] = st.text_input(
                    label,
                    value="" if is_secret else existing,
                    type="password" if is_secret else "default",
                    placeholder=placeholder,
                    key=f"{key}_{field_key}",
                )

        submitted = st.form_submit_button("Save credentials", use_container_width=True)
        if submitted:
            with get_db() as db:
                for field_key, _, _ in fields:
                    val = values[field_key].strip()
                    if val:  # don't overwrite with blank (placeholder trick)
                        save_credential(db, user["id"], source, field_key, val)
            st.success(f"{source.title()} credentials saved.")
            st.cache_data.clear()
            st.rerun()


# ── Ingestion trigger ─────────────────────────────────────────────────────────

def _trigger_ingestion(user_id: str, source: str, mode: str, scope: str) -> None:
    from app.ingestion.cli import run_ingestion
    from app.utils import init_db

    def _run():
        init_db()
        run_ingestion(user_id=user_id, source=source, mode=mode, scope=scope)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    st.toast(f"Ingestion started for {source} ({mode}). Check logs for progress.", icon="⚙️")


# ── Main sidebar renderer ─────────────────────────────────────────────────────

def render_sidebar() -> SelectionState:
    """Render the sidebar and return the current SelectionState."""
    user = current_user()
    if not user:
        return SelectionState()

    state = SelectionState()

    with st.sidebar:
        st.markdown(f"### 👤 {user['email']}")
        if st.button("Sign out", use_container_width=True):
            logout_session()
            st.rerun()

        st.divider()

        # ── Source selection ──────────────────────────────────────────────────
        st.markdown("### 📚 Data Sources")

        use_jira = st.checkbox("Jira", value=True)
        use_confluence = st.checkbox("Confluence", value=True)
        use_sql = st.checkbox("SQL Server", value=True)
        use_git = st.checkbox("Git", value=True)

        if use_jira:
            state.sources.append("jira")
        if use_confluence:
            state.sources.append("confluence")
        if use_sql:
            state.sources.append("sql")
        if use_git:
            state.sources.append("git")

        st.divider()

        # ── Jira scope ────────────────────────────────────────────────────────
        if use_jira:
            with st.expander("🔵 Jira scope", expanded=True):
                projects = _cached_jira_projects(user["id"])
                if projects:
                    proj_options = ["all"] + [p["key"] for p in projects]
                    proj_labels = ["All projects"] + [f"{p['key']} – {p['name']}" for p in projects]
                    idx = st.multiselect(
                        "Projects",
                        options=proj_options,
                        default=["all"],
                        format_func=lambda k: proj_labels[proj_options.index(k)],
                        key="jira_scope",
                    )
                    state.jira_projects = idx if idx else ["all"]
                else:
                    st.caption("No projects found. Configure credentials below.")

        # ── Confluence scope ──────────────────────────────────────────────────
        if use_confluence:
            with st.expander("🟢 Confluence scope", expanded=True):
                spaces = _cached_confluence_spaces(user["id"])
                if spaces:
                    space_options = ["all"] + [s["key"] for s in spaces]
                    space_labels = ["All spaces"] + [f"{s['key']} – {s['name']}" for s in spaces]
                    idx = st.multiselect(
                        "Spaces",
                        options=space_options,
                        default=["all"],
                        format_func=lambda k: space_labels[space_options.index(k)],
                        key="confluence_scope",
                    )
                    state.confluence_spaces = idx if idx else ["all"]
                else:
                    st.caption("No spaces found. Configure credentials below.")

        # ── SQL scope ─────────────────────────────────────────────────────────
        if use_sql:
            with st.expander("🟠 SQL Server scope", expanded=True):
                dbs = _cached_sql_databases(user["id"])
                if dbs:
                    db_options = ["all"] + [d["key"] for d in dbs]
                    db_labels = ["All databases"] + [d["name"] for d in dbs]
                    idx = st.multiselect(
                        "Databases",
                        options=db_options,
                        default=["all"],
                        format_func=lambda k: db_labels[db_options.index(k)],
                        key="sql_scope",
                    )
                    state.sql_databases = idx if idx else ["all"]
                else:
                    st.caption("No databases found. Configure credentials below.")

        # ── Git scope ─────────────────────────────────────────────────────────
        if use_git:
            with st.expander("🟣 Git scope", expanded=True):
                branches = _cached_git_branches(user["id"])
                if branches:
                    branch_options = ["all"] + [b["key"] for b in branches]
                    branch_labels = ["All branches"] + [b["name"] for b in branches]
                    idx = st.multiselect(
                        "Branches",
                        options=branch_options,
                        default=["all"],
                        format_func=lambda k: branch_labels[branch_options.index(k)],
                        key="git_scope",
                    )
                    state.git_branches = idx if idx else ["all"]
                else:
                    st.caption("No branches found. Configure credentials below.")

        st.divider()

        # ── Credential settings ───────────────────────────────────────────────
        with st.expander("🔑 Credentials & Settings"):
            tab_j, tab_c, tab_s, tab_g = st.tabs(["Jira", "Confluence", "SQL", "Git"])

            with tab_j:
                _credential_form(
                    "jira",
                    [
                        ("url", "Jira base URL (e.g. https://myorg.atlassian.net)", False),
                        ("email", "Atlassian account email", False),
                        ("api_token", "API Token", True),
                    ],
                )

            with tab_c:
                st.caption(
                    "Use your **Confluence** site URL — this may differ from your Jira URL. "
                    "Find it in your browser while viewing any Confluence page: copy everything "
                    "up to (but **not** including) `/wiki`. "
                    "Example: `https://yoursite-1234567890.atlassian.net`"
                )
                _credential_form(
                    "confluence",
                    [
                        (
                            "url",
                            "Confluence site URL (e.g. https://yoursite-1234567890.atlassian.net — omit /wiki)",
                            False,
                        ),
                        ("email", "Atlassian account email", False),
                        ("api_token", "API Token", True),
                    ],
                )
                with st.expander("Additional Confluence instance (optional)"):
                    st.caption(
                        "If you have a second Atlassian site (e.g. a legacy instance at a "
                        "different URL), enter its base URL here. The email and API token "
                        "from the primary instance are reused automatically — only fill in "
                        "the email/token fields if the second site uses different credentials."
                    )
                    _credential_form(
                        "confluence",
                        [
                            (
                                "url_2",
                                "Second Confluence site URL (omit /wiki)",
                                False,
                            ),
                            ("email_2", "Email (leave blank to reuse primary)", False),
                            ("api_token_2", "API Token (leave blank to reuse primary)", True),
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
                            "Connection string (e.g. DRIVER={ODBC Driver 18 for SQL Server};SERVER=...;UID=...;PWD=...)",
                            True,
                        ),
                    ],
                )

            with tab_g:
                st.caption(
                    "Connect to a GitHub repository. For **private** repos, create a "
                    "Personal Access Token (classic) at "
                    "github.com → Settings → Developer settings → Personal access tokens "
                    "with **repo** scope."
                )
                _credential_form(
                    "git",
                    [
                        ("url", "GitHub repository URL (e.g. https://github.com/owner/repo)", False),
                        ("access_token", "Personal Access Token (required for private repos)", True),
                        (
                            "file_extensions",
                            "File extensions to index, comma-separated (default: .py .md .txt .yml .yaml .json)",
                            False,
                        ),
                        ("max_commits", "Max commits to index (default: 200)", False),
                    ],
                )

        st.divider()

        # ── Ingestion controls ────────────────────────────────────────────────
        with st.expander("⚙️ Ingest data"):
            col_src, col_mode = st.columns(2)
            with col_src:
                ing_source = st.selectbox("Source", ["all", "jira", "confluence", "sql"], key="ing_source")
            with col_mode:
                ing_mode = st.selectbox("Mode", ["incremental", "full"], key="ing_mode")

            ing_scope = st.text_input("Scope (key or 'all')", value="all", key="ing_scope")

            if st.button("▶ Run ingestion", use_container_width=True):
                _trigger_ingestion(user["id"], ing_source, ing_mode, ing_scope)

            # ── Recent ingestion log ──────────────────────────────────────────
            from models.ingestion_log import IngestionLog
            with get_db() as db:
                rows = (
                    db.query(IngestionLog)
                    .filter_by(user_id=user["id"])
                    .order_by(IngestionLog.started_at.desc())
                    .limit(5)
                    .all()
                )
                # Convert to plain dicts inside the session so attributes are
                # readable after the session closes.
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
            if logs:
                st.caption("Recent runs:")
                for log in logs:
                    icon = {"success": "✅", "error": "❌", "running": "⏳"}.get(log["status"], "❓")
                    st.caption(
                        f"{icon} {log['source']}/{log['scope']} ({log['mode']}) "
                        f"items={log['items_processed']} vecs={log['vectors_upserted']}"
                    )

    return state
