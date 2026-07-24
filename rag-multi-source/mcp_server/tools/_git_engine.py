"""
Shared GitHub `Repository` handle cache for live git MCP tools.

Mirrors `mcp_server.tools._sql_engine.get_engine`'s cache-by-key shape,
but adapted to GitHub's actual auth model: a SQL Server connection string
is pinned to exactly one database, but a single PAT is often valid for
several repos at once, and `git_ingestor.py`'s per-repo `url{suffix}`
credential slots don't promise a 1:1 mapping between "this slot's URL" and
"which repos this slot's token can reach" (a PAT's repo access is whatever
GitHub granted it, independent of which repo the user happened to type
into that slot's `url` field). `core.live_acl._check_github` already
solved this the same way: try every configured token in turn against the
target repo, first success wins, rather than matching a specific slot by
URL. This module does the identical thing, but to build a working
`Repository` handle instead of a boolean.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.utils import get_db, load_all_credentials

_repo_cache: dict[tuple[str, str], tuple[Any, str]] = {}


def _parse_repo_path(url: str) -> str:
    """Extract ``owner/repo`` from a GitHub URL (or pass-through if already that).

    Duplicated from `app.ingestion.git_ingestor` deliberately -- this
    module must not import the ingestor (an MCP-server-side module has no
    business depending on the ingestion pipeline), and the function is a
    single, unlikely-to-drift string transform.
    """
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return url


def get_repo(user_id: str, git_scope: str) -> tuple[Any, str]:
    """
    Return (`github.Repository.Repository`, branch) for `git_scope`
    (`"owner/repo@branch"`), trying every credential the user has stored
    under `source="git"` until one can actually open the repo.

    Raises `PermissionError` if no stored credential can access it —
    same convention `_sql_engine.get_engine` uses for "no stored
    connection string", so callers can let it bubble to the route handler
    unhandled, mirroring `sql_schema_tools.py`'s `except PermissionError: raise`.
    """
    cache_key = (user_id, git_scope)
    if cache_key in _repo_cache:
        return _repo_cache[cache_key]

    if "@" not in git_scope:
        raise ValueError(f"git_scope must be 'owner/repo@branch', got {git_scope!r}")
    repo_full_name, branch = git_scope.rsplit("@", 1)

    with get_db() as db:
        credentials = load_all_credentials(db, user_id, "git")

    from github import Github

    primary_token = (credentials.get("access_token") or "").strip()
    suffixes = [""] + [f"_{i}" for i in range(2, 10)]
    tried = 0
    last_exc: Exception | None = None

    for suffix in suffixes:
        if suffix and not (credentials.get(f"url{suffix}") or "").strip():
            continue
        token = (credentials.get(f"access_token{suffix}") or "").strip() or primary_token
        if not token:
            continue
        tried += 1
        try:
            gh = Github(token)
            repo = gh.get_repo(repo_full_name)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        _repo_cache[cache_key] = (repo, branch)
        return repo, branch

    if tried == 0:
        raise PermissionError("User has no stored GitHub access token.")
    raise PermissionError(
        f"No configured GitHub credential can access '{repo_full_name}': {last_exc}"
    )


def shutdown() -> None:
    """Clear the cache on server shutdown (Repository handles hold no
    open connections to dispose, unlike a SQLAlchemy Engine -- this just
    forces fresh lookups, e.g. after a credential rotation)."""
    _repo_cache.clear()
    logger.info("[mcp.git_engine] repo cache cleared")
