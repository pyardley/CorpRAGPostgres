"""Database session management, encryption helpers, and shared utilities.

In addition to DB / Fernet / credential / accessible-resource helpers, this
module also hosts the **chat status bar** support code:

* :func:`init_session_totals` — set up the ``st.session_state`` slot that
  accumulates running totals (tokens, cost, time, prompts answered).
* :func:`update_session_totals` — increment those totals from an audit
  record produced by the RAG / hybrid chains.
* :func:`reset_session_totals` — wipe back to zero on logout / clear-chat.
* :func:`extract_usage` — pull standardised token usage out of a LangChain
  response object regardless of provider (OpenAI / Anthropic / Grok).
* :func:`estimate_cost` — turn ``(provider, model, prompt_tokens,
  completion_tokens)`` into a USD figure using the rate table on
  ``app.config.settings``. The same call is made by both the status
  bar and the audit logger so the two figures always agree.
* :func:`render_status_bar` — render the persistent footer-style bar at
  the bottom of the chat.

…and the **audit-logging** helpers:

* :class:`StepTimer` — context-manager that measures one step with
  ``time.perf_counter()`` and stages a ``query_step_timings`` row.
* :func:`record_step_timing` — low-level "I have a duration, persist
  it" helper used when a step doesn't fit cleanly into a ``with`` block
  (e.g. summing per-hop LLM time inside the MCP loop).
* :func:`log_query_audit` — write the ``query_audit_logs`` row plus all
  staged step rows in a single short transaction, even on errors.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Iterable, Optional

import streamlit as st
from cryptography.fernet import Fernet
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from models.base import Base

# ── Database engine ───────────────────────────────────────────────────────────
#
# Single PostgreSQL connection — the app DB and the vector store live in the
# same database. `pool_pre_ping` survives idle-connection drops on managed
# Postgres providers (Supabase, Neon, RDS) which aggressively close
# connections held open across sleeps.

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    future=True,
)
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, future=True
)


def init_db() -> None:
    """Create / migrate all tables (idempotent — safe on every startup)."""
    # Delegated to models.migrations: enables the pgvector extension, creates
    # missing tables, and ensures the HNSW ANN index exists.
    from models.migrations import run_migrations

    run_migrations(engine)
    logger.info("Database tables initialised.")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Row-Level Security helpers ───────────────────────────────────────────────
#
# The RLS policies installed by ``models.migrations._apply_rls_policies``
# read ``current_setting('app.current_user_id', true)`` to decide which
# rows the current session can see. The application calls
# :func:`set_current_user_for_rls` immediately after opening a session
# (and before any SELECT against ``vector_chunks`` /
# ``user_accessible_resources``). When ``settings.ENABLE_RLS`` is False
# the helper is a cheap no-op so existing deployments behave exactly as
# before.
#
# Why ``SET LOCAL``? It's transaction-scoped — the GUC is reset when the
# transaction ends, which is critical with connection pooling (a
# subsequent checkout against the same physical connection must NOT
# inherit the previous request's user_id). Combined with SQLAlchemy's
# implicit-transaction model (a SELECT auto-begins one), every
# ``get_db()``-style block gets its own clean tenancy context.


def set_current_user_for_rls(db: Session, user_id: str | None) -> None:
    """
    Bind ``user_id`` to the current session's ``app.current_user_id``
    GUC so the RLS policies on ``vector_chunks`` /
    ``user_accessible_resources`` can recognise the caller.

    Safe to call repeatedly within a single transaction (the last value
    wins). No-op when ``settings.ENABLE_RLS`` is False or ``user_id``
    is empty/None.

    Notes
    -----
    * Uses ``SET LOCAL`` so the binding is automatically reset at
      ``COMMIT`` / ``ROLLBACK`` and never leaks to the next checkout
      from the connection pool.
    * The value is interpolated as a parameter to defeat any SQL
      injection vector in case ``user_id`` ever comes from
      attacker-controlled input — though today it's always a UUID
      pulled from the bcrypt-authenticated session.
    * The helper does NOT raise on failure: a misconfigured DB role
      (e.g. one that doesn't allow ``SET``) shouldn't take the chat
      down — it'll just see an empty result set under RLS, which is
      the correct fail-closed behaviour anyway.
    """
    if not settings.ENABLE_RLS:
        return
    if not user_id:
        # Without a bound user_id the RLS policies see NULL and return
        # zero rows. Surface this loudly — it's almost always a bug
        # (we're about to run a query that will silently return empty).
        logger.debug(
            "set_current_user_for_rls called with empty user_id; "
            "RLS policies will deny all rows."
        )
        return
    try:
        # SET LOCAL requires a bound transaction. SQLAlchemy auto-begins
        # one on the first execute() if there isn't one, so this is safe
        # even on a brand-new session.
        db.execute(
            text("SELECT set_config('app.current_user_id', :uid, true)"),
            {"uid": str(user_id)},
        )
    except Exception:  # noqa: BLE001 — best-effort
        logger.exception(
            "Could not set RLS user_id (continuing — RLS will deny rows)"
        )


@contextmanager
def get_db_for_user(user_id: str | None) -> Generator[Session, None, None]:
    """
    Convenience wrapper around :func:`get_db` that binds the current
    user_id for RLS before yielding the session.

    Prefer this over a bare ``get_db()`` for any code path that reads
    or writes ``vector_chunks`` / ``user_accessible_resources`` so the
    RLS GUC is always set. Code paths that only touch tenancy-free
    tables (``users``, ``ingestion_logs``, ``query_audit_logs``, …)
    can still use the plain ``get_db()`` if they want.
    """
    with get_db() as db:
        set_current_user_for_rls(db, user_id)
        yield db


# ── Encryption ────────────────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    key = settings.ENCRYPTION_KEY
    if not key:
        # Auto-generate a key the first time and tell the operator to persist it
        key = Fernet.generate_key().decode()
        logger.warning(
            "ENCRYPTION_KEY not set. Generated a temporary key for this session. "
            "Set ENCRYPTION_KEY={} in your .env to persist encrypted credentials.",
            key,
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _get_fernet().decrypt(ciphertext.encode()).decode()


# ── Credential helpers ────────────────────────────────────────────────────────

def save_credential(db: Session, user_id: str, source: str, key: str, value: str) -> None:
    from models.user import UserCredential

    encrypted = encrypt(value)
    existing = (
        db.query(UserCredential)
        .filter_by(user_id=user_id, source=source, credential_key=key)
        .first()
    )
    if existing:
        existing.encrypted_value = encrypted
    else:
        db.add(UserCredential(user_id=user_id, source=source, credential_key=key, encrypted_value=encrypted))
    db.flush()


def delete_credential(db: Session, user_id: str, source: str, key: str) -> bool:
    """Remove a single saved credential field. Returns True if a row was deleted."""
    from models.user import UserCredential

    deleted = (
        db.query(UserCredential)
        .filter_by(user_id=user_id, source=source, credential_key=key)
        .delete()
    )
    db.flush()
    return bool(deleted)


def load_credential(db: Session, user_id: str, source: str, key: str) -> Optional[str]:
    from models.user import UserCredential

    row = (
        db.query(UserCredential)
        .filter_by(user_id=user_id, source=source, credential_key=key)
        .first()
    )
    if row is None:
        return None
    try:
        return decrypt(row.encrypted_value)
    except Exception:
        logger.exception("Failed to decrypt credential {}/{}/{}", user_id, source, key)
        return None


def load_all_credentials(db: Session, user_id: str, source: str) -> dict[str, str]:
    from models.user import UserCredential

    rows = db.query(UserCredential).filter_by(user_id=user_id, source=source).all()
    result: dict[str, str] = {}
    for row in rows:
        try:
            result[row.credential_key] = decrypt(row.encrypted_value)
        except Exception:
            logger.warning("Could not decrypt {}/{}/{}", user_id, source, row.credential_key)
    return result


# ── Accessible-resource helpers ───────────────────────────────────────────────

def grant_access(
    db: Session, user_id: str, source: str, resource_identifier: str
) -> None:
    """Idempotently record that `user_id` can query `source/resource_identifier`."""
    from models.user_accessible_resource import UserAccessibleResource

    row = (
        db.query(UserAccessibleResource)
        .filter_by(
            user_id=user_id, source=source, resource_identifier=resource_identifier
        )
        .first()
    )
    if row:
        row.last_synced = datetime.utcnow()
    else:
        db.add(
            UserAccessibleResource(
                user_id=user_id,
                source=source,
                resource_identifier=resource_identifier,
            )
        )
    db.flush()


def list_accessible(db: Session, user_id: str, source: str) -> list[str]:
    """Return all resource_identifier values the user is allowed to query."""
    from models.user_accessible_resource import UserAccessibleResource

    rows = (
        db.query(UserAccessibleResource.resource_identifier)
        .filter_by(user_id=user_id, source=source)
        .all()
    )
    return [r[0] for r in rows]


def revoke_access(
    db: Session, user_id: str, source: str, resource_identifier: str
) -> None:
    from models.user_accessible_resource import UserAccessibleResource

    db.query(UserAccessibleResource).filter_by(
        user_id=user_id,
        source=source,
        resource_identifier=resource_identifier,
    ).delete()
    db.flush()


# ── Cost / token accounting (shared by status bar AND audit logger) ──────────

def estimate_cost(
    provider: str | None,
    model: str | None,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Estimate the USD cost of an LLM call from token counts.

    The lookup is a substring match against the configured model name so
    we don't have to keep an exhaustive list of every dated GPT/Claude
    snapshot — ``"gpt-4o-mini-2025-08-07"`` still matches ``"gpt-4o-mini"``.

    Reads its rate table from ``settings.LLM_COST_PER_1K_TOKENS`` so an
    operator can tune pricing in one place (config / env) without
    changing code. Returns 0.0 / a default rate for missing inputs
    rather than raising — the status bar and audit logger must never
    fail an answer because pricing data is incomplete.
    """
    rates = settings.LLM_DEFAULT_COST_PER_1K_TOKENS
    if provider and model:
        provider_table = settings.LLM_COST_PER_1K_TOKENS.get(provider.lower(), {})
        model_lower = model.lower()
        # Most-specific match wins (longer key matches first).
        for key in sorted(provider_table.keys(), key=len, reverse=True):
            if key in model_lower:
                rates = provider_table[key]
                break

    p_rate, c_rate = rates
    return (prompt_tokens / 1000.0) * p_rate + (completion_tokens / 1000.0) * c_rate


def extract_usage(response: Any) -> dict[str, int]:
    """
    Pull ``{prompt_tokens, completion_tokens, total_tokens}`` out of a
    LangChain chat-model response, regardless of provider.

    LangChain populates ``usage_metadata`` (``input_tokens`` /
    ``output_tokens`` / ``total_tokens``) on every message returned by
    ``BaseChatModel.invoke``; older / partial wrappers may only fill
    ``response_metadata.token_usage`` (OpenAI shape) or
    ``response_metadata.usage`` (Anthropic shape). We try all three.
    """
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Preferred: standardised usage_metadata
    um = getattr(response, "usage_metadata", None)
    if um:
        out["prompt_tokens"] = int(um.get("input_tokens", 0) or 0)
        out["completion_tokens"] = int(um.get("output_tokens", 0) or 0)
        out["total_tokens"] = int(
            um.get("total_tokens", out["prompt_tokens"] + out["completion_tokens"])
            or 0
        )
        if out["total_tokens"]:
            return out

    # Fallback: response_metadata (provider-specific)
    rm = getattr(response, "response_metadata", {}) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    if tu:
        out["prompt_tokens"] = int(
            tu.get("prompt_tokens", tu.get("input_tokens", 0)) or 0
        )
        out["completion_tokens"] = int(
            tu.get("completion_tokens", tu.get("output_tokens", 0)) or 0
        )
        out["total_tokens"] = int(
            tu.get("total_tokens", out["prompt_tokens"] + out["completion_tokens"])
            or 0
        )
    return out


# ── Status bar (running session totals) ──────────────────────────────────────
#
# The status bar reads its numbers from a single ``st.session_state`` dict so
# that **any** caller in the request lifecycle can contribute to the running
# totals without coupling the chat layer to the chains, and without us having
# to thread an extra argument through ``render_chat``.


def init_session_totals() -> None:
    """Initialise ``st.session_state["session_totals"]`` if missing.

    Idempotent — safe to call on every Streamlit rerun. Called once on
    successful login from ``app/main.py`` and defensively from each
    helper that reads/writes the totals.
    """
    if "session_totals" not in st.session_state:
        st.session_state["session_totals"] = {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
            "total_time_seconds": 0.0,
            "prompt_count": 0,
            "last_provider": "",
            "last_model": "",
        }


def reset_session_totals() -> None:
    """Wipe the running totals back to zero. Called on logout / clear-chat."""
    st.session_state["session_totals"] = {
        "total_tokens": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cost_usd": 0.0,
        "total_time_seconds": 0.0,
        "prompt_count": 0,
        "last_provider": "",
        "last_model": "",
    }


def update_session_totals(audit_record: dict[str, Any] | None) -> None:
    """
    Increment the running session totals from a single answer's audit record.

    Audit records have the same shape that ``query_audit_logs`` rows are
    built from, so the bar and the DB never need translation.
    """
    init_session_totals()
    if not audit_record:
        return

    t = st.session_state["session_totals"]
    t["total_prompt_tokens"] += int(audit_record.get("prompt_tokens", 0) or 0)
    t["total_completion_tokens"] += int(
        audit_record.get("completion_tokens", 0) or 0
    )
    t["total_tokens"] += int(audit_record.get("total_tokens", 0) or 0)
    t["total_cost_usd"] += float(audit_record.get("cost_usd", 0.0) or 0.0)
    t["total_time_seconds"] += float(
        audit_record.get("duration_seconds", 0.0) or 0.0
    )
    t["prompt_count"] += 1
    if audit_record.get("provider"):
        t["last_provider"] = str(audit_record["provider"])
    if audit_record.get("model"):
        t["last_model"] = str(audit_record["model"])


def _format_cost(value: float) -> str:
    """Format a USD figure for the bar — ``$0.0042`` for tiny, ``$1.23`` for normal."""
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}"
    return f"${value:,.2f}"


def render_status_bar() -> None:
    """
    Render the persistent bottom status bar inside the chat surface.

    Honours the ``show_status_bar`` session-state flag so the user can
    hide it from the sidebar.
    """
    if not st.session_state.get("show_status_bar", True):
        return

    init_session_totals()
    t = st.session_state["session_totals"]

    avg_seconds = (
        t["total_time_seconds"] / t["prompt_count"] if t["prompt_count"] else 0.0
    )
    model_label = (
        f"{t.get('last_provider') or '—'} / {t.get('last_model') or '—'}"
        if t.get("last_provider") or t.get("last_model")
        else "—"
    )

    st.markdown(
        f"""
        <div style="
            background: #f8f9fb;
            border: 1px solid #e3e6ec;
            border-radius: 6px;
            padding: 0.45rem 0.85rem;
            margin-top: 1rem;
            font-size: 1.17rem;
            color: #5a6470;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.85rem;
        ">
            <span><span style="opacity:0.65;">Tokens</span>&nbsp;
                <strong style="color:#2d3748;">{t['total_tokens']:,}</strong></span>
            <span><span style="opacity:0.65;">Cost</span>&nbsp;
                <strong style="color:#2d3748;">{_format_cost(t['total_cost_usd'])}</strong></span>
            <span><span style="opacity:0.65;">Time</span>&nbsp;
                <strong style="color:#2d3748;">{t['total_time_seconds']:.1f}s</strong>
                <span style="opacity:0.55;">({avg_seconds:.1f}s avg)</span></span>
            <span><span style="opacity:0.65;">Prompts</span>&nbsp;
                <strong style="color:#2d3748;">{t['prompt_count']}</strong></span>
            <span><span style="opacity:0.65;">Model</span>&nbsp;
                <strong style="color:#2d3748;">{model_label}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Audit logging helpers ────────────────────────────────────────────────────
#
# Two shapes are exposed:
#
#  1. ``StepTimer`` — context manager that measures one step and stages a
#     ``query_step_timings`` row in an in-memory list. Most call sites
#     just use ``with StepTimer(steps, "vector_retrieval") as t: ...``
#     and optionally attach extras via ``t.extra["row_count"] = …``.
#
#  2. ``record_step_timing(steps, name, duration_ms, **extra)`` — for
#     places that already have a duration measured (e.g. summing
#     per-hop LLM time inside the MCP loop, or copying the chain-level
#     ``duration_seconds`` into a ``"total"`` step).
#
# Both flavours append to a plain list of dicts. The chat layer hands
# that list to ``log_query_audit`` together with the audit-row fields,
# and the helper persists everything in one short transaction.
#
# The helpers are intentionally side-effect-free until ``log_query_audit``
# is called: a chain that raises mid-flight has no half-written audit
# state, and the chat layer can decide whether the partial timings are
# still worth keeping (they usually are, for failure-mode analysis).


class StepTimer:
    """
    Context manager that measures wall-clock for one named step and
    appends a step-timing dict to ``steps``.

    Usage::

        steps: list[dict] = []
        with StepTimer(steps, "vector_retrieval") as t:
            hits = retrieve(prompt, filter_dict)
            t.extra["retrieved_count"] = len(hits)

    The dict shape matches the eventual ``query_step_timings`` row::

        {"step_name": str, "duration_ms": int, "metadata": dict}

    On exception inside the ``with`` block we still append a row
    (with the elapsed time so far + ``metadata.error = "<exc>"``) so
    the audit log captures *what was running* when the failure happened.
    """

    def __init__(self, steps: list[dict[str, Any]], step_name: str, **extra: Any):
        self._steps = steps
        self.step_name = step_name
        self.extra: dict[str, Any] = dict(extra)
        self._started: float = 0.0
        self.duration_ms: int = 0

    def __enter__(self) -> "StepTimer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.duration_ms = int((time.perf_counter() - self._started) * 1000)
        if exc is not None:
            # Record the partial timing even on failure — the operator
            # still wants to know "we got 80ms into vector_retrieval
            # before it blew up".
            self.extra.setdefault("error", f"{type(exc).__name__}: {exc}")
        self._steps.append(
            {
                "step_name": self.step_name,
                "duration_ms": self.duration_ms,
                "metadata": dict(self.extra),
            }
        )
        # Returning False (the implicit None) re-raises any exception —
        # we never suppress, the caller decides how to handle.


def record_step_timing(
    steps: list[dict[str, Any]],
    step_name: str,
    duration_ms: int | float,
    **extra: Any,
) -> None:
    """
    Append a pre-measured step row to ``steps``.

    Use this when ``StepTimer`` doesn't fit (e.g. summed-across-hops
    LLM time inside the MCP loop, or echoing the chain-level total
    once everything else has been recorded).
    """
    steps.append(
        {
            "step_name": step_name,
            "duration_ms": int(duration_ms),
            "metadata": dict(extra),
        }
    )


def log_query_audit(
    *,
    user_id: str | None,
    prompt_text: str,
    audit: dict[str, Any] | None,
    total_duration_seconds: float,
    source_type: str,
    success: bool,
    error_message: str | None = None,
    steps: Iterable[dict[str, Any]] | None = None,
) -> str | None:
    """
    Persist one ``query_audit_logs`` row + all of its ``query_step_timings``
    children, in a single short transaction.

    Always-on by default; honours ``settings.AUDIT_LOG_ENABLED`` so the
    helper turns into a no-op when disabled.

    Returns the new audit row's UUID on success, ``None`` on failure or
    when audit logging is disabled. Never raises — audit must never
    fail an answer.

    Parameters
    ----------
    user_id
        The user who issued the prompt, or ``None`` for system-level calls.
    prompt_text
        Raw prompt — truncated at write time to
        ``settings.AUDIT_PROMPT_MAX_CHARS`` so paste-bombs don't bloat
        the table.
    audit
        The chain's audit record (``provider``, ``model``,
        ``prompt_tokens``, ``completion_tokens``, ``cost_usd``, …) or
        ``None`` if the answer never reached the LLM (e.g. the empty-hits
        early-return). Missing keys are treated as zeros / empty.
    total_duration_seconds
        Wall-clock from the chat layer's perspective, converted to
        milliseconds for storage.
    source_type
        ``"rag"`` | ``"hybrid"`` | ``"mcp_only"``.
    success
        ``False`` if the answer path raised — ``error_message`` is then
        the str(exc) and ``audit`` may be partial.
    error_message
        Truncated free-form error string, only used when
        ``success=False``.
    steps
        Iterable of dicts shaped like
        ``{"step_name": str, "duration_ms": int, "metadata": dict}``.
        Typically built up via ``StepTimer`` / ``record_step_timing``
        as the chain runs.
    """
    if not settings.AUDIT_LOG_ENABLED:
        return None

    audit = audit or {}
    cap = max(0, int(getattr(settings, "AUDIT_PROMPT_MAX_CHARS", 4000)))
    truncated_prompt = (prompt_text or "")[:cap] if cap else (prompt_text or "")
    truncated_error = (error_message or "")[:2000] if error_message else None

    try:
        from models.query_audit_log import QueryAuditLog
        from models.query_step_timing import QueryStepTiming

        with get_db() as db:
            row = QueryAuditLog(
                user_id=user_id,
                timestamp=datetime.utcnow(),
                prompt_text=truncated_prompt,
                total_duration_ms=int((total_duration_seconds or 0.0) * 1000),
                tokens_prompt=int(audit.get("prompt_tokens", 0) or 0),
                tokens_completion=int(audit.get("completion_tokens", 0) or 0),
                estimated_cost_usd=float(audit.get("cost_usd", 0.0) or 0.0),
                llm_provider=audit.get("provider") or None,
                llm_model=audit.get("model") or None,
                success=bool(success),
                error_message=truncated_error,
                source_type=source_type or "rag",
            )
            db.add(row)
            db.flush()  # populate row.id

            for step in steps or ():
                db.add(
                    QueryStepTiming(
                        audit_id=row.id,
                        step_name=str(step.get("step_name", ""))[:128],
                        duration_ms=int(step.get("duration_ms", 0) or 0),
                        extra=step.get("metadata") or {},
                    )
                )

            audit_id = row.id

        logger.info(
            "[audit] persisted id={} user={} source={} duration_ms={} "
            "tokens={}+{} cost=${:.5f} success={}",
            audit_id,
            user_id,
            source_type,
            int((total_duration_seconds or 0.0) * 1000),
            int(audit.get("prompt_tokens", 0) or 0),
            int(audit.get("completion_tokens", 0) or 0),
            float(audit.get("cost_usd", 0.0) or 0.0),
            success,
        )
        return audit_id
    except Exception:  # noqa: BLE001 — audit must never fail an answer
        logger.exception("Failed to persist audit log (non-fatal)")
        return None
