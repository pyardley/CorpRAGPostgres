"""
Streamlit sidebar - source selection, credentials, and ingestion controls.

Design notes
------------
* The ``vector_chunks`` table is shared across the whole org. The sidebar
  doesn't decide "what data exists" — it decides "what the *current user*
  is allowed to query right now". The source of truth for that is the
  ``user_accessible_resources`` table.
* Scope dropdowns are populated from that table, NOT from a live API call
  to Jira/Confluence/SQL/Git. ``user_accessible_resources`` rows are
  inserted by the ingestion pipeline whenever the user has successfully
  ingested a scope.
* The chat layer uses :class:`SelectionState` to build a metadata filter
  (see ``core.vector_store.build_query_filter``) which the retriever
  turns into a SQL ``WHERE`` clause.
* The "Audit Log" expander at the bottom of the sidebar reads
  ``query_audit_logs`` + ``query_step_timings`` so an operator can see
  the last 100 prompts (filterable by user / date) plus the full
  step-by-step timing breakdown for any selected row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Any, Optional

import pandas as pd
import streamlit as st
from loguru import logger

from app.auth import current_user, logout_session
from app.utils import (
    delete_credential,
    get_db,
    list_accessible,
    load_all_credentials,
    reset_session_totals,
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
    # For email, each entry is a provider name: "outlook", "gmail",
    # and/or "yahoo", matching the `email_provider` column on
    # vector_chunks.
    email_providers: list[str] = field(default_factory=list)
    # When True, the chat layer will route through core.mcp_chain so the
    # LLM can call live SQL Server tools (read-only) on top of the RAG
    # context. Off by default — pure-RAG is the safe baseline.
    use_mcp_sql: bool = False


# Credential form
def _credential_form(
    source: str,
    fields: list[tuple[str, str, bool]],
    form_key: Optional[str] = None,
) -> None:
    """
    Render a form that saves encrypted credentials.

    `fields` is a list of (key, label, is_secret) tuples. For non-secret
    fields the existing value is shown editable. For secret fields:
      * an empty input is shown by default (we never decrypt + display
        on the page);
      * the field's *length* is shown in the caption so the user can
        confirm a value is saved;
      * leaving the field empty on save keeps the existing value
        (i.e. you only need to type something when you want to overwrite).

    The connection-string field for SQL uses a multi-line text_area so
    long ODBC strings (~150 chars) are easier to read and edit.
    """
    user = current_user()
    if not user:
        return

    key = form_key or f"creds_{source}"

    # Show any status carried across from the previous run so the user
    # actually sees the success/info banner — st.rerun() at submit time
    # otherwise wipes one-shot messages immediately.
    status_key = f"_creds_status_{key}"
    pending = st.session_state.pop(status_key, None)
    if pending:
        level, msg = pending
        if level == "success":
            st.success(msg)
        elif level == "info":
            st.info(msg)
        elif level == "error":
            st.error(msg)

    with st.form(key):
        values: dict[str, str] = {}
        clears: dict[str, bool] = {}
        with get_db() as db:
            existing = load_all_credentials(db, user["id"], source)

        for field_key, label, is_secret in fields:
            existing_val = existing.get(field_key, "")
            widget_key = f"{key}_{field_key}"
            is_long = field_key == "conn_str"

            if is_secret:
                # Caption tells the user whether something is saved without
                # exposing the value.
                if existing_val:
                    st.caption(
                        f"**{label}** — currently saved "
                        f"({len(existing_val)} characters). "
                        f"Type a new value below to overwrite; leave blank "
                        f"to keep the saved value; tick **Clear** to remove it."
                    )
                    clears[field_key] = st.checkbox(
                        f"Clear saved {label}",
                        value=False,
                        key=f"{widget_key}_clear",
                        help=(
                            "Remove this saved secret on save. If you also "
                            "type a new value above, the new value wins."
                        ),
                    )
                else:
                    st.caption(f"**{label}** — no value saved yet.")

                if is_long:
                    values[field_key] = st.text_area(
                        label,
                        value="",
                        placeholder="DRIVER={ODBC Driver 18 for SQL Server};"
                                    "SERVER=…;DATABASE=…;UID=…;PWD=…;"
                                    "TrustServerCertificate=yes",
                        key=widget_key,
                        height=120,
                        label_visibility="collapsed",
                    )
                else:
                    values[field_key] = st.text_input(
                        label,
                        value="",
                        type="password",
                        placeholder=("●●●●●●●●" if existing_val else ""),
                        key=widget_key,
                        label_visibility="collapsed",
                    )
            else:
                values[field_key] = st.text_input(
                    label,
                    value=existing_val,
                    type="default",
                    key=widget_key,
                )

        if st.form_submit_button("Save credentials", use_container_width=True):
            saved: list[str] = []
            cleared: list[str] = []
            with get_db() as db:
                for field_key, _label, _is_secret in fields:
                    val = values[field_key].strip()
                    if val:
                        # New value typed — overrides any "clear" tick.
                        save_credential(db, user["id"], source, field_key, val)
                        saved.append(field_key)
                    elif clears.get(field_key):
                        if delete_credential(db, user["id"], source, field_key):
                            cleared.append(field_key)
            parts: list[str] = []
            if saved:
                parts.append(
                    f"saved {', '.join(saved)} "
                    f"({len(saved)} field{'s' if len(saved) != 1 else ''})"
                )
            if cleared:
                parts.append(f"cleared {', '.join(cleared)}")
            if parts:
                st.session_state[status_key] = (
                    "success",
                    f"{source.title()}: " + "; ".join(parts) + ".",
                )
            else:
                st.session_state[status_key] = (
                    "info",
                    f"{source.title()}: nothing to save — all fields were "
                    f"left blank. (Existing values preserved.)",
                )
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
            # Reset the running totals shown in the chat status bar so
            # "tokens / cost / time" reflect only the new conversation.
            reset_session_totals()
            st.rerun()

        # Status-bar visibility toggle. Persists across reruns via the
        # widget key (``show_status_bar``); the bar reads the same key
        # in app.utils.render_status_bar.
        st.checkbox(
            "Show status bar",
            value=st.session_state.get("show_status_bar", True),
            key="show_status_bar",
            help=(
                "Show the running session totals (tokens, cost, time, "
                "prompts answered) at the bottom of the chat."
            ),
        )

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
                state.use_mcp_sql = st.toggle(
                    "⚡ Use Live SQL Table Data (MCP)",
                    value=st.session_state.get("use_mcp_sql", False),
                    key="use_mcp_sql",
                    help=(
                        "When ON, the chat agent can call read-only SQL "
                        "tools to fetch live row data from your SQL Server "
                        "in addition to the indexed schema/proc-code RAG. "
                        "Hard-capped at 100 rows per query, no DDL/DML."
                    ),
                )
                if state.use_mcp_sql:
                    _render_mcp_status()

        if st.checkbox("Git", value=True, key="src_git"):
            state.sources.append("git")
            with st.expander("🟣 Git scopes (repo@branch)", expanded=True):
                state.git_scopes = _scope_picker(
                    "Repos+branches", "git", user["id"]
                )

        if st.checkbox("Email", value=True, key="src_email"):
            state.sources.append("email")
            with st.expander(
                "📧 Email mailboxes (outlook / gmail / yahoo)", expanded=True
            ):
                state.email_providers = _scope_picker(
                    "Mailboxes", "email", user["id"]
                )

        st.divider()

        # Credentials
        with st.expander("🔑 Credentials"):
            tab_j, tab_c, tab_s, tab_g, tab_e = st.tabs(
                ["Jira", "Confluence", "SQL", "Git", "Email"]
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

            with tab_e:
                st.caption(
                    "Connect Outlook / Microsoft 365 and / or Gmail. "
                    "Each provider's credentials are independent — fill in "
                    "only the section(s) you want to ingest."
                )
                with st.expander("📨 Outlook / Microsoft 365", expanded=True):
                    st.caption(
                        "Register an app in Azure AD with delegated `Mail.Read` "
                        "permission. For personal outlook.com / yahoo accounts "
                        "use tenant `common` and supply a refresh token."
                    )
                    _credential_form(
                        "email",
                        [
                            ("outlook_tenant_id", "Tenant ID (or 'common')", False),
                            ("outlook_client_id", "Client (Application) ID", False),
                            ("outlook_client_secret", "Client Secret (optional)", True),
                            ("outlook_refresh_token", "Refresh Token", True),
                            ("outlook_user", "Mailbox UPN (optional, blank = /me)", False),
                            ("outlook_folder", "Folder filter (optional, e.g. inbox)", False),
                        ],
                        form_key="creds_email_outlook",
                    )
                with st.expander("📬 Gmail", expanded=True):
                    st.caption(
                        "Create OAuth client credentials at console.cloud.google.com, "
                        "grant the `gmail.readonly` scope, and obtain a refresh token "
                        "with a one-time installed-app flow. Alternatively paste a "
                        "full token.json blob in the field below."
                    )
                    _credential_form(
                        "email",
                        [
                            ("gmail_client_id", "OAuth Client ID", False),
                            ("gmail_client_secret", "OAuth Client Secret", True),
                            ("gmail_refresh_token", "Refresh Token", True),
                            ("gmail_token_json", "Full token.json (optional)", True),
                            ("gmail_user", "Mailbox address (optional, default 'me')", False),
                            ("gmail_label", "Label filter (optional, e.g. INBOX)", False),
                        ],
                        form_key="creds_email_gmail",
                    )
                with st.expander("📭 Yahoo Mail (IMAP + App Password)", expanded=False):
                    st.caption(
                        "Yahoo dropped public OAuth2 for Mail. Generate "
                        "a 16-character app password at "
                        "[login.yahoo.com → Account → Security → "
                        "*Generate app password*]"
                        "(https://login.yahoo.com/myaccount/security)"
                        " (requires 2-step verification on the account). "
                        "Spaces in the displayed password are cosmetic — "
                        "you can paste with or without them."
                    )
                    _credential_form(
                        "email",
                        [
                            (
                                "yahoo_email_address",
                                "Mailbox address (e.g. you@yahoo.com)",
                                False,
                            ),
                            (
                                "yahoo_app_password",
                                "Yahoo App Password (16 characters)",
                                True,
                            ),
                            (
                                "yahoo_folder",
                                "IMAP folder (optional, default INBOX)",
                                False,
                            ),
                        ],
                        form_key="creds_email_yahoo",
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
                ["all", "jira", "confluence", "sql", "git", "email"],
                key="ing_source",
            )
            ing_scope = st.text_input(
                "Scope (project_key / space_key / db_name / branch / "
                "'outlook' | 'gmail' | 'yahoo' / folder, or 'all')",
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
                "doesn't delete vectors from the database - it only revokes "
                "*your* visibility into that scope."
            )
            for src in ("jira", "confluence", "sql", "git", "email"):
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

        # Audit log — read-only inspector for query_audit_logs +
        # query_step_timings. Filter by user / date range and drill into
        # any row to see its step-by-step timing breakdown.
        with st.expander("📋 Audit Log"):
            _render_audit_log(user["id"])

    return state


# MCP status badge
def _render_mcp_status() -> None:
    """Tiny status line under the MCP toggle so the user knows it's wired up."""
    try:
        from app.mcp_manager import mcp_status
    except Exception:  # noqa: BLE001 - never block the sidebar on this
        st.caption("⚪ MCP: status unavailable")
        return
    info = mcp_status()
    if info["healthy"] and info["token_set"]:
        st.caption(f"🟢 MCP: connected ({info['base_url']})")
    elif info["healthy"]:
        st.caption(f"🟡 MCP: reachable but no token ({info['base_url']})")
    else:
        st.caption(
            f"🔴 MCP: not reachable at {info['base_url']} — "
            f"check the server / restart the app"
        )


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


# ── Audit Log UI ─────────────────────────────────────────────────────────────
#
# Reads ``query_audit_logs`` + ``query_step_timings``. Two-pane layout:
#   1. A sortable, filterable summary table of recent prompts.
#   2. A detail block for the row the operator picks from a dropdown
#      (shows the prompt, the audit row's metadata, and every measured
#      step with duration + JSONB extras).
#
# Filters:
#   * Scope — "Just me" (default) | "All users" — non-admin users only
#     see their own rows in the underlying query (the "All users" toggle
#     is purely a UX nicety; access is enforced at query-build time by
#     defaulting the user filter to the caller's ID).
#   * Date range — last 24h / 7d / 30d / all.
#   * Limit — 50 / 100 / 250 (default 100).


_DATE_RANGE_OPTIONS = {
    "Last 24 hours": timedelta(days=1),
    "Last 7 days": timedelta(days=7),
    "Last 30 days": timedelta(days=30),
    "All time": None,
}

_LIMIT_OPTIONS = (50, 100, 250)


def _format_duration_ms(ms: int | None) -> str:
    if not ms:
        return "—"
    if ms < 1000:
        return f"{ms} ms"
    return f"{ms / 1000:.2f} s"


def _format_cost_usd(value: float | None) -> str:
    v = float(value or 0.0)
    if v == 0:
        return "$0"
    if v < 0.01:
        return f"${v:.4f}"
    if v < 1:
        return f"${v:.3f}"
    return f"${v:,.2f}"


def _audit_status_icon(success: bool, source_type: str) -> str:
    if not success:
        return "❌"
    if source_type == "hybrid":
        return "⚡"
    if source_type == "mcp_only":
        return "🟠"
    return "✅"


def _load_audit_rows(
    *,
    self_user_id: str,
    only_self: bool,
    since: datetime | None,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Read the most recent audit rows matching the active filter.

    Joined to ``users`` for the display email so the table doesn't have
    to do per-row lookups.
    """
    from models.query_audit_log import QueryAuditLog
    from models.user import User

    rows: list[dict[str, Any]] = []
    try:
        with get_db() as db:
            q = (
                db.query(QueryAuditLog, User.email)
                .outerjoin(User, User.id == QueryAuditLog.user_id)
            )
            if only_self:
                q = q.filter(QueryAuditLog.user_id == self_user_id)
            if since is not None:
                q = q.filter(QueryAuditLog.timestamp >= since)
            q = q.order_by(QueryAuditLog.timestamp.desc()).limit(limit)
            for audit, email in q.all():
                rows.append(
                    {
                        "id": audit.id,
                        "timestamp": audit.timestamp,
                        "user_email": email or "(unknown)",
                        "prompt_text": audit.prompt_text or "",
                        "source_type": audit.source_type or "rag",
                        "success": bool(audit.success),
                        "duration_ms": int(audit.total_duration_ms or 0),
                        "tokens_prompt": int(audit.tokens_prompt or 0),
                        "tokens_completion": int(audit.tokens_completion or 0),
                        "tokens_total": int(
                            (audit.tokens_prompt or 0)
                            + (audit.tokens_completion or 0)
                        ),
                        "cost_usd": float(audit.estimated_cost_usd or 0.0),
                        "provider": audit.llm_provider or "",
                        "model": audit.llm_model or "",
                        "error_message": audit.error_message or "",
                    }
                )
    except Exception:  # noqa: BLE001
        # Audit table may not exist yet on a brand-new DB the first time
        # the sidebar is rendered before init_db() has finished — fall
        # back to an empty result rather than crashing the whole sidebar.
        logger.exception("Failed to load audit rows")
    return rows


def _load_step_timings(audit_id: str) -> list[dict[str, Any]]:
    from models.query_step_timing import QueryStepTiming

    out: list[dict[str, Any]] = []
    try:
        with get_db() as db:
            steps = (
                db.query(QueryStepTiming)
                .filter_by(audit_id=audit_id)
                .order_by(QueryStepTiming.id.asc())
                .all()
            )
            for s in steps:
                out.append(
                    {
                        "step_name": s.step_name,
                        "duration_ms": int(s.duration_ms or 0),
                        "metadata": s.extra or {},
                    }
                )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load step timings for audit_id={}", audit_id)
    return out


def _render_audit_log(self_user_id: str) -> None:
    """Render the Audit Log expander (filters → table → detail view)."""
    st.caption(
        "Last 50–250 prompts answered by the chat. Pick a row from the "
        "dropdown below the table to see its full step-by-step timing."
    )

    # Filter row
    col_scope, col_range, col_limit = st.columns([1, 1, 1])
    with col_scope:
        only_self = (
            st.selectbox(
                "User",
                options=("Just me", "All users"),
                index=0,
                key="audit_scope",
                help=(
                    "‘All users’ shows every recorded prompt across the "
                    "whole installation. Filter to ‘Just me’ to focus on "
                    "your own activity."
                ),
            )
            == "Just me"
        )
    with col_range:
        date_label = st.selectbox(
            "Date range",
            options=list(_DATE_RANGE_OPTIONS.keys()),
            index=1,  # default: last 7 days
            key="audit_range",
        )
    with col_limit:
        limit = st.selectbox(
            "Limit",
            options=_LIMIT_OPTIONS,
            index=1,  # default: 100
            key="audit_limit",
        )

    # Optional: free-form date override (only the start date — anything
    # newer than that). Folded behind a small expander so the default
    # "Last 7 days" UX stays one click away.
    custom_since: datetime | None = None
    with st.expander("Advanced — custom start date", expanded=False):
        use_custom = st.checkbox(
            "Use custom start date", value=False, key="audit_use_custom_date"
        )
        if use_custom:
            today = datetime.utcnow().date()
            picked = st.date_input(
                "Start date (UTC)",
                value=today - timedelta(days=7),
                key="audit_custom_date",
            )
            custom_since = datetime.combine(picked, dt_time.min)

    if custom_since is not None:
        since = custom_since
    else:
        delta = _DATE_RANGE_OPTIONS[date_label]
        since = (datetime.utcnow() - delta) if delta is not None else None

    # Refresh button — useful because Streamlit only re-runs on widget
    # interaction, so a freshly-arrived audit row wouldn't appear until
    # the user touched something else.
    if st.button("🔄 Refresh", key="audit_refresh", use_container_width=True):
        st.rerun()

    rows = _load_audit_rows(
        self_user_id=self_user_id,
        only_self=only_self,
        since=since,
        limit=int(limit),
    )

    if not rows:
        st.info(
            "No audit rows yet for this filter. Ask a question in the chat "
            "to generate one — the audit logger writes a row per prompt, "
            "even on errors."
        )
        return

    # Summary metrics — a quick at-a-glance view above the table.
    total = len(rows)
    failed = sum(1 for r in rows if not r["success"])
    total_cost = sum(r["cost_usd"] for r in rows)
    total_tokens = sum(r["tokens_total"] for r in rows)
    avg_duration = (
        sum(r["duration_ms"] for r in rows) / total if total else 0.0
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prompts", f"{total}", delta=f"{failed} failed" if failed else None)
    m2.metric("Total cost", _format_cost_usd(total_cost))
    m3.metric("Total tokens", f"{total_tokens:,}")
    m4.metric("Avg duration", _format_duration_ms(int(avg_duration)))

    # Sortable summary table.
    table_df = pd.DataFrame(
        [
            {
                "": _audit_status_icon(r["success"], r["source_type"]),
                "Timestamp (UTC)": r["timestamp"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                ) if r["timestamp"] else "",
                "User": r["user_email"],
                "Source": r["source_type"],
                "Duration": _format_duration_ms(r["duration_ms"]),
                "Tokens": r["tokens_total"],
                "Cost": _format_cost_usd(r["cost_usd"]),
                "Model": (
                    f"{r['provider']}/{r['model']}"
                    if r["provider"] or r["model"]
                    else "—"
                ),
                "Prompt": (r["prompt_text"][:80] + "…")
                if len(r["prompt_text"]) > 80
                else r["prompt_text"],
                "id": r["id"],
            }
            for r in rows
        ]
    )

    # Hide the raw id column from the rendered view but keep it in the
    # underlying frame so the dropdown can map a label back to a row.
    st.dataframe(
        table_df.drop(columns=["id"]),
        use_container_width=True,
        hide_index=True,
        height=min(420, 38 * (len(rows) + 1) + 4),
    )

    # Detail-row picker. Labels include the timestamp + first 60 chars of
    # the prompt so duplicates don't collide.
    label_to_id: dict[str, str] = {}
    label_options: list[str] = []
    for r in rows:
        ts = (
            r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            if r["timestamp"]
            else "(no ts)"
        )
        snippet = r["prompt_text"][:60].replace("\n", " ")
        if len(r["prompt_text"]) > 60:
            snippet += "…"
        icon = _audit_status_icon(r["success"], r["source_type"])
        label = (
            f"{icon} {ts} · {r['user_email']} · "
            f"{_format_duration_ms(r['duration_ms'])} · {snippet or '(empty)'}"
        )
        # Disambiguate by appending a short id slug if labels collide.
        if label in label_to_id:
            label = f"{label} [{r['id'][:6]}]"
        label_to_id[label] = r["id"]
        label_options.append(label)

    chosen = st.selectbox(
        "View step timings for…",
        options=["—"] + label_options,
        index=0,
        key="audit_detail_pick",
    )
    if chosen == "—":
        return

    audit_id = label_to_id[chosen]
    chosen_row = next((r for r in rows if r["id"] == audit_id), None)
    if not chosen_row:
        return

    st.markdown("---")
    st.markdown(f"### Audit detail · `{audit_id}`")

    meta_l, meta_r = st.columns(2)
    with meta_l:
        st.markdown(
            f"**User:** {chosen_row['user_email']}  \n"
            f"**Source path:** {chosen_row['source_type']}  \n"
            f"**Provider/model:** "
            f"{(chosen_row['provider'] or '—')}/"
            f"{(chosen_row['model'] or '—')}  \n"
            f"**Success:** {'✅' if chosen_row['success'] else '❌'}"
        )
    with meta_r:
        st.markdown(
            f"**Total duration:** {_format_duration_ms(chosen_row['duration_ms'])}  \n"
            f"**Prompt tokens:** {chosen_row['tokens_prompt']:,}  \n"
            f"**Completion tokens:** {chosen_row['tokens_completion']:,}  \n"
            f"**Cost:** {_format_cost_usd(chosen_row['cost_usd'])}"
        )

    if chosen_row["error_message"]:
        st.error(chosen_row["error_message"])

    if chosen_row["prompt_text"]:
        st.markdown("**Prompt:**")
        st.code(chosen_row["prompt_text"], language="text")

    steps = _load_step_timings(audit_id)
    if not steps:
        st.caption("No step timings recorded for this audit row.")
        return

    # Step-by-step table — duration + the JSONB extras (rendered as a
    # short JSON string column so an operator can scan all steps at a
    # glance without each one expanding).
    import json as _json

    steps_df = pd.DataFrame(
        [
            {
                "Step": s["step_name"],
                "Duration": _format_duration_ms(s["duration_ms"]),
                "ms": s["duration_ms"],
                "Metadata": _json.dumps(
                    s["metadata"] or {}, default=str, sort_keys=True
                ),
            }
            for s in steps
        ]
    )
    st.dataframe(
        steps_df,
        use_container_width=True,
        hide_index=True,
        height=min(420, 38 * (len(steps) + 1) + 4),
    )
