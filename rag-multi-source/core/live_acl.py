"""
Live source-system ACL re-validation (README "Possible enhancements #1").

``user_accessible_resources`` is granted once at ingestion time and never
re-checked (see ``models/user_accessible_resource.py``) — if a user loses
Jira/Confluence/SQL/GitHub access after ingestion, they keep querying that
data through chat until the next re-ingestion re-syncs the grant table.

:func:`revalidate` closes that gap: given a candidate list of resource
identifiers for one source, it calls back into the live source system and
returns only the subset still confirmed-accessible. Verdicts are cached for
``settings.LIVE_ACL_CACHE_TTL_SECONDS`` to bound the added per-query API
cost.

Fails CLOSED — any checker error (timeout, network, unknown/incomplete
credential) excludes that resource from the result rather than falling
back to the ingestion-time grant. This is a hardening feature; a silent
fail-open would defeat its purpose.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from loguru import logger
from sqlalchemy.orm import Session

from app.config import settings
from app.utils import load_all_credentials

_CacheKey = tuple[str, str, str]  # (user_id, source, resource_identifier)

_cache: dict[_CacheKey, tuple[bool, float]] = {}
_cache_lock = threading.Lock()


def _cache_get(key: _CacheKey) -> bool | None:
    with _cache_lock:
        entry = _cache.get(key)
    if entry is None:
        return None
    allowed, expires_at = entry
    if time.monotonic() >= expires_at:
        return None
    return allowed


def _cache_set(key: _CacheKey, allowed: bool) -> None:
    expires_at = time.monotonic() + settings.LIVE_ACL_CACHE_TTL_SECONDS
    with _cache_lock:
        _cache[key] = (allowed, expires_at)


def _safe_check(
    checker: Callable[[dict[str, str], str], bool],
    credentials: dict[str, str],
    identifier: str,
    source: str,
) -> bool:
    try:
        return checker(credentials, identifier)
    except Exception:
        logger.warning(
            "[live_acl] {} live check failed for {!r} — failing closed",
            source,
            identifier,
        )
        return False


def revalidate(
    db: Session, user_id: str, source: str, resource_identifiers: list[str]
) -> list[str]:
    """
    Intersect `resource_identifiers` with a live permission check against
    the source system. Returns the subset still confirmed-accessible.

    No-ops (returns the input unchanged) when
    ``settings.LIVE_ACL_REVALIDATION_ENABLED`` is False or no checker is
    registered for `source`.
    """
    if not settings.LIVE_ACL_REVALIDATION_ENABLED or not resource_identifiers:
        return resource_identifiers

    checker = _CHECKERS.get(source)
    if checker is None:
        return resource_identifiers

    credentials = load_all_credentials(db, user_id, source)
    allowed: list[str] = []
    for identifier in resource_identifiers:
        key = (user_id, source, identifier)
        verdict = _cache_get(key)
        if verdict is None:
            verdict = _safe_check(checker, credentials, identifier, source)
            _cache_set(key, verdict)
        if verdict:
            allowed.append(identifier)
    return allowed


# ──────────────────────────────────────────────────────────────────────────────
# Per-source checkers — mirror each ingestor's auth pattern (not its class)
# ──────────────────────────────────────────────────────────────────────────────

def _check_jira(credentials: dict[str, str], project_key: str) -> bool:
    import requests

    url = (credentials.get("url") or "").rstrip("/")
    email = credentials.get("email") or ""
    api_token = credentials.get("api_token") or ""
    if not (url and email and api_token):
        return False

    resp = requests.post(
        f"{url}/rest/api/3/permissions/project",
        auth=(email, api_token),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={"projectKeys": [project_key], "permissions": ["BROWSE_PROJECTS"]},
        timeout=settings.LIVE_ACL_HTTP_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    projects = resp.json().get("projects", [])
    return any(p.get("key") == project_key for p in projects)


def _check_confluence(credentials: dict[str, str], space_key: str) -> bool:
    import requests

    primary_email = (credentials.get("email") or "").strip()
    primary_token = (credentials.get("api_token") or "").strip()
    suffixes = [""] + [f"_{i}" for i in range(2, 10)]

    for suffix in suffixes:
        url = (credentials.get(f"url{suffix}") or "").strip()
        if not url:
            continue
        email = (credentials.get(f"email{suffix}") or "").strip() or primary_email
        token = (credentials.get(f"api_token{suffix}") or "").strip() or primary_token
        if not (email and token):
            continue

        resp = requests.get(
            f"{url.rstrip('/')}/wiki/rest/api/space/{space_key}",
            auth=(email, token),
            headers={"Accept": "application/json"},
            timeout=settings.LIVE_ACL_HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            return True

    return False


def _check_sql(credentials: dict[str, str], db_name: str) -> bool:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL

    conn_str = (credentials.get("conn_str") or "").strip()
    if not conn_str:
        return False

    url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT HAS_DBACCESS(:db_name)"), {"db_name": db_name}
            ).scalar()
        return result == 1
    finally:
        engine.dispose()


def _check_github(credentials: dict[str, str], owner_repo: str) -> bool:
    # `git` source identifiers are "{owner}/{repo}@{branch}"; `github_issue`
    # identifiers are already bare "{owner}/{repo}".
    owner_repo = owner_repo.split("@", 1)[0]

    import requests

    primary_token = (credentials.get("access_token") or "").strip()
    suffixes = [""] + [f"_{i}" for i in range(2, 10)]
    headers_base = {"Accept": "application/vnd.github+json"}

    for suffix in suffixes:
        # `url{suffix}` presence marks a configured instance; fall back to
        # the primary token when a secondary instance doesn't set its own.
        if suffix and not (credentials.get(f"url{suffix}") or "").strip():
            continue
        token = (credentials.get(f"access_token{suffix}") or "").strip() or primary_token
        if not token:
            continue

        headers = {**headers_base, "Authorization": f"Bearer {token}"}
        me = requests.get(
            "https://api.github.com/user",
            headers=headers,
            timeout=settings.LIVE_ACL_HTTP_TIMEOUT_SECONDS,
        )
        if me.status_code != 200:
            continue
        login = me.json().get("login")
        if not login:
            continue

        perm = requests.get(
            f"https://api.github.com/repos/{owner_repo}/collaborators/{login}/permission",
            headers=headers,
            timeout=settings.LIVE_ACL_HTTP_TIMEOUT_SECONDS,
        )
        if perm.status_code == 200 and perm.json().get("permission", "none") != "none":
            return True

    return False


_CHECKERS: dict[str, Callable[[dict[str, str], str], bool]] = {
    "jira": _check_jira,
    "confluence": _check_confluence,
    "sql": _check_sql,
    "git": _check_github,
    "github_issue": _check_github,
}
