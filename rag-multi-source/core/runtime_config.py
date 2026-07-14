"""
Runtime-mutable configuration overrides.

Most settings live in :mod:`app.config` and are env-driven (pydantic
``BaseSettings``) — restart the app to pick up a new value. A small
number of settings benefit from being toggleable from the UI without a
restart, notably:

* ``FTS_LANGUAGE`` — Postgres text-search configuration. ``english``
  applies stemming and stop-word removal (great for natural-language
  prose); ``simple`` lower-cases tokens but otherwise leaves them
  alone (great for code-heavy corpora where ``running`` should not
  match ``run`` and identifiers like ``OAuth2Client`` should be
  searchable verbatim).

Overrides are persisted to a tiny JSON file (see ``_RUNTIME_CONFIG_PATH``)
so the choice survives Streamlit reruns and full app restarts. If no
override is present (fresh deployment, or the file was deleted) the
helper falls back to the corresponding ``settings.<NAME>`` value, so
the existing ``.env``-only workflow keeps working unchanged.

Public API:
    get_fts_language()         -> 'english' | 'simple'
    set_fts_language(lang)     -> persist + return the stored value
    valid_fts_languages()      -> tuple of accepted values

Concurrency:
    Streamlit serialises requests per session, but the runtime-config
    file may be touched from multiple sessions / the migrations
    helper. A module-level ``Lock`` makes the read-modify-write cycle
    atomic within a single Python process; cross-process writes are
    rare (admin clicks "rebuild" once) and the file is small enough
    that the OS-level write is effectively atomic on POSIX/Windows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from app.config import settings


_VALID_FTS_LANGUAGES: tuple[str, ...] = ("english", "simple")

# Default location: a hidden JSON file next to the project root (one
# level above ``core/``). Override with the ``RUNTIME_CONFIG_PATH``
# env var if you want it somewhere persistent (e.g. a mounted volume
# in a container deployment).
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".runtime_config.json"
_RUNTIME_CONFIG_PATH = Path(os.getenv("RUNTIME_CONFIG_PATH", str(_DEFAULT_PATH)))

_LOCK = Lock()


def _load() -> dict:
    """Return the persisted overrides, or ``{}`` if the file is missing/corrupt."""
    if not _RUNTIME_CONFIG_PATH.exists():
        return {}
    try:
        raw = _RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        # Corrupt file — treat as "no override" rather than crash the app.
        return {}


def _save(data: dict) -> None:
    _RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUNTIME_CONFIG_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )


def get_fts_language() -> str:
    """
    Return the active Postgres text-search configuration name.

    Resolution order:
        1. ``fts_language`` value in the runtime-config JSON (if valid).
        2. ``settings.FTS_LANGUAGE`` from .env / pydantic defaults.
    """
    with _LOCK:
        data = _load()
    candidate = data.get("fts_language") or settings.FTS_LANGUAGE
    if candidate not in _VALID_FTS_LANGUAGES:
        return settings.FTS_LANGUAGE
    return candidate


def set_fts_language(lang: str) -> str:
    """
    Persist a new FTS language override.

    Validates against ``_VALID_FTS_LANGUAGES`` to keep the value safe
    to inline into DDL (the migration code substitutes it into
    ``to_tsvector('<lang>', …)``). Raises :class:`ValueError` on
    anything else — the UI is expected to constrain choices via a
    selectbox, so this is a defence-in-depth check, not the primary
    validation.
    """
    if lang not in _VALID_FTS_LANGUAGES:
        raise ValueError(
            f"FTS language must be one of {_VALID_FTS_LANGUAGES}; got {lang!r}"
        )
    with _LOCK:
        data = _load()
        data["fts_language"] = lang
        _save(data)
    return lang


def valid_fts_languages() -> tuple[str, ...]:
    """The whitelist of accepted FTS language values (for UI dropdowns)."""
    return _VALID_FTS_LANGUAGES


def runtime_config_path() -> Path:
    """Where the override file lives — handy for diagnostics in the UI."""
    return _RUNTIME_CONFIG_PATH
