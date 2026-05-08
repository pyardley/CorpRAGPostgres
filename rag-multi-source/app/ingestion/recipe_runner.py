"""
RecipeRunner — runs a :class:`recipes.Recipe` through the existing
ingestion pipeline.

Two execution paths
-------------------
1. **Class delegation** — when ``recipe.parser`` is a fully qualified
   ``module.ClassName`` path. The class is imported lazily (so legacy
   ingestors don't pay an import-time cost) and instantiated with the
   same kwargs the legacy CLI uses today, with the addition of
   ``recipe=recipe``. This is how ``recipes/example_jira.yaml`` wraps
   :class:`app.ingestion.jira_ingestor.JiraIngestor`.

2. **Builtin generic parser** — when ``recipe.parser == "builtin"``.
   :class:`BuiltinRecipeIngestor` is a :class:`BaseIngestor` subclass
   driven entirely by ``recipe.config``. It supports HTTP+JSON sources
   with basic / bearer / no auth, plus cursor / page / no pagination,
   and a small templating engine for resource_id / title / text / url
   / metadata. Anything more exotic (HTML scraping, custom auth) is the
   class-delegation path.

Backward compatibility
----------------------
Nothing in this module touches the legacy CLI dispatch. Recipes are
strictly additive: the legacy ``--source jira|confluence|…`` flag
continues to work via :func:`app.ingestion.cli._make_ingestor`, and
:class:`BaseIngestor` only consults ``self.recipe`` when it's set.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from typing import Any, Iterable, Optional

import requests
from loguru import logger
from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion.base import (
    BaseIngestor,
    IngestionResult,
    ProgressCallback,
    SourceResource,
)
from app.utils import load_all_credentials
from recipes import Recipe, RecipeError, get_recipe


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def run_recipe(
    db: Session,
    user_id: str,
    recipe_name: str,
    *,
    scope: str = "all",
    mode: str = "incremental",
    on_progress: Optional[ProgressCallback] = None,
    credentials_override: Optional[dict[str, str]] = None,
) -> IngestionResult:
    """Resolve ``recipe_name`` and run it. Returns the ingestion result.

    ``credentials_override`` is intended for tests; production callers
    leave it ``None`` so credentials come from ``user_credentials`` via
    :func:`app.utils.load_all_credentials` keyed by
    :attr:`Recipe.credential_key`.
    """
    recipe = get_recipe(recipe_name)
    return run_recipe_object(
        db,
        user_id,
        recipe,
        scope=scope,
        mode=mode,
        on_progress=on_progress,
        credentials_override=credentials_override,
    )


def run_recipe_object(
    db: Session,
    user_id: str,
    recipe: Recipe,
    *,
    scope: str = "all",
    mode: str = "incremental",
    on_progress: Optional[ProgressCallback] = None,
    credentials_override: Optional[dict[str, str]] = None,
) -> IngestionResult:
    """Run a Recipe instance directly (skips the registry lookup)."""
    creds = credentials_override
    if creds is None:
        creds = load_all_credentials(db, user_id, recipe.credential_key)
    if not creds:
        raise RecipeError(
            f"No credentials found for credential_key='{recipe.credential_key}' "
            f"(recipe '{recipe.name}'). Configure them in the Streamlit UI "
            f"under Credentials, or via the CLI."
        )

    ingestor = _build_ingestor(
        recipe=recipe,
        db=db,
        user_id=user_id,
        credentials=creds,
        scope=scope,
        mode=mode,
    )
    logger.info(
        "Running recipe '{}' (parser={}, source={}, scope={}, mode={})",
        recipe.name, recipe.parser, recipe.source, scope, mode,
    )
    return ingestor.run(on_progress=on_progress)


# ──────────────────────────────────────────────────────────────────────
# Ingestor construction
# ──────────────────────────────────────────────────────────────────────


def _build_ingestor(
    *,
    recipe: Recipe,
    db: Session,
    user_id: str,
    credentials: dict[str, str],
    scope: str,
    mode: str,
) -> BaseIngestor:
    if recipe.is_builtin:
        return BuiltinRecipeIngestor(
            db=db,
            user_id=user_id,
            credentials=credentials,
            scope=scope,
            mode=mode,
            recipe=recipe,
        )

    cls = _import_parser_class(recipe)
    return cls(
        db=db,
        user_id=user_id,
        credentials=credentials,
        scope=scope,
        mode=mode,
        recipe=recipe,
    )


def _import_parser_class(recipe: Recipe) -> type[BaseIngestor]:
    """Import the BaseIngestor subclass referenced by ``recipe.parser``."""
    module_path, _, class_name = recipe.parser.rpartition(".")
    if not module_path or not class_name:
        raise RecipeError(
            f"Recipe '{recipe.name}' has invalid parser={recipe.parser!r}; "
            f"expected 'module.ClassName'."
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RecipeError(
            f"Recipe '{recipe.name}': could not import {module_path}: {exc}"
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RecipeError(
            f"Recipe '{recipe.name}': {module_path} has no attribute "
            f"{class_name!r}."
        )
    if not (isinstance(cls, type) and issubclass(cls, BaseIngestor)):
        raise RecipeError(
            f"Recipe '{recipe.name}': {recipe.parser} is not a subclass of "
            f"BaseIngestor."
        )
    return cls


# ──────────────────────────────────────────────────────────────────────
# Builtin generic parser
# ──────────────────────────────────────────────────────────────────────


# scope_field → vector_chunks promoted column. ``external_id`` falls
# back onto ``object_name`` (a free-form indexed string column already
# present on vector_chunks) so brand-new sources don't require a schema
# change to participate in the per-source filter.
_SCOPE_TO_PROMOTED_COLUMN: dict[str, str] = {
    "project_key": "project_key",
    "space_key": "space_key",
    "db_name": "db_name",
    "git_scope": "git_scope",
    "email_provider": "email_provider",
    "external_id": "object_name",
}


class BuiltinRecipeIngestor(BaseIngestor):
    """A generic, recipe-driven :class:`BaseIngestor`.

    Configurable behaviour
    ----------------------
    * **Fetching** — HTTP JSON endpoints (``GET`` / ``POST``) with
      optional Basic or Bearer auth and three pagination strategies.
    * **Templating** — every resource field (``resource_id``,
      ``title``, ``text``, ``url``, plus arbitrary metadata) is built
      from a Python ``str.format``-style template. Field paths use
      dot notation: ``{a.b.c}`` resolves chained ``.get()`` calls.
    * **Incremental cutoff** — when the recipe defines
      ``incremental_param``, that query parameter is set to the ISO
      timestamp of the last successful run.
    * **Static mode** — for tests / demos, ``fetch.type: static`` reads
      a list directly out of the recipe.

    Anything more dynamic (custom auth, HTML scraping, GraphQL) should
    use the class-delegation path instead — write a tiny
    ``BaseIngestor`` subclass and point a recipe at it.
    """

    # ``source`` is set from the recipe in BaseIngestor.__init__, so
    # leave the class-level attribute blank.
    source = ""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.recipe is None:  # pragma: no cover - defensive
            raise RecipeError(
                "BuiltinRecipeIngestor requires a Recipe."
            )
        self._cfg = self.recipe.config or {}

    # ── BaseIngestor abstract API ─────────────────────────────────────

    def scope_filter(self) -> dict[str, Any]:
        """Filter dict consumed by ``vector_store.delete_by_filter``."""
        flt: dict[str, Any] = {"source": self.source}
        if self.scope and self.scope != "all":
            column = _SCOPE_TO_PROMOTED_COLUMN.get(
                self.recipe.scope_field, "object_name"  # type: ignore[union-attr]
            )
            flt[column] = self.scope
        return flt

    def resource_identifier_for(self, resource: SourceResource) -> str:
        """The value written to ``user_accessible_resources``."""
        # Whatever the recipe said the scope is, the resource's metadata
        # already carries it (we promote it during _to_resource).
        column = _SCOPE_TO_PROMOTED_COLUMN.get(
            self.recipe.scope_field, "object_name"  # type: ignore[union-attr]
        )
        return (
            resource.metadata.get(column)
            or resource.metadata.get(self.recipe.scope_field)  # type: ignore[union-attr]
            or self.scope
            or "all"
        )

    def fetch_resources(
        self, since: Optional[datetime] = None
    ) -> Iterable[SourceResource]:
        fetch = self._cfg.get("fetch") or {}
        ftype = (fetch.get("type") or "http_json").lower()
        if ftype == "static":
            yield from self._fetch_static(fetch)
        elif ftype == "http_json":
            yield from self._fetch_http_json(fetch, since=since)
        else:
            raise RecipeError(
                f"Recipe '{self.recipe.name}' has unsupported "  # type: ignore[union-attr]
                f"fetch.type={ftype!r} (supported: http_json, static)."
            )

    # ── Fetchers ──────────────────────────────────────────────────────

    def _fetch_static(self, fetch: dict[str, Any]) -> Iterable[SourceResource]:
        items = fetch.get("items") or []
        if not isinstance(items, list):
            raise RecipeError(
                f"Recipe '{self.recipe.name}': fetch.items must be a list."  # type: ignore[union-attr]
            )
        for item in items:
            yield self._to_resource(item)

    def _fetch_http_json(
        self,
        fetch: dict[str, Any],
        *,
        since: Optional[datetime] = None,
    ) -> Iterable[SourceResource]:
        url_template = fetch.get("url_template") or fetch.get("url")
        if not url_template:
            raise RecipeError(
                f"Recipe '{self.recipe.name}': fetch.url_template is required for "  # type: ignore[union-attr]
                f"http_json fetches."
            )
        method = (fetch.get("method") or "GET").upper()
        items_field = fetch.get("items_field") or "items"
        pagination = fetch.get("pagination") or {"type": "none"}
        ptype = (pagination.get("type") or "none").lower()
        page_size = int(pagination.get("page_size") or 50)
        incremental_param = self._cfg.get("incremental_param")

        session = self._build_session(fetch)

        # Render the URL once — pagination tweaks query params, not the
        # URL template, in every common API.
        rendered_url = _render_template(
            url_template, {"scope": self.scope, **self.credentials}
        )

        # Optional static base body for POSTs (we deep-copy each request
        # so we don't mutate the user's config dict).
        body_template = fetch.get("body_template")

        next_page_token: Optional[str] = None
        page = int(pagination.get("start_page") or 1)
        first_request = True

        while True:
            params: dict[str, Any] = dict(fetch.get("params") or {})
            body: Optional[dict[str, Any]] = (
                json.loads(json.dumps(body_template))
                if isinstance(body_template, dict) else None
            )

            # ── Pagination knobs ──
            if ptype == "cursor":
                token_field = pagination.get("next_token_field") or "nextPageToken"
                if next_page_token:
                    if method == "POST" and body is not None:
                        body[token_field] = next_page_token
                    else:
                        params[token_field] = next_page_token
                if method == "POST" and body is not None:
                    body.setdefault("maxResults", page_size)
                else:
                    params.setdefault("maxResults", page_size)
            elif ptype == "page":
                page_field = pagination.get("page_field") or "page"
                size_field = pagination.get("size_field") or "page_size"
                if method == "POST" and body is not None:
                    body[page_field] = page
                    body[size_field] = page_size
                else:
                    params[page_field] = page
                    params[size_field] = page_size

            # ── Incremental "since" ──
            if first_request and since is not None and incremental_param:
                params[incremental_param] = since.isoformat()

            try:
                if method == "POST":
                    response = session.post(rendered_url, params=params, json=body, timeout=30)
                else:
                    response = session.get(rendered_url, params=params, timeout=30)
            except requests.RequestException as exc:
                raise RecipeError(
                    f"Recipe '{self.recipe.name}': HTTP request to "  # type: ignore[union-attr]
                    f"{rendered_url} failed: {exc}"
                ) from exc

            if not response.ok:
                logger.error(
                    "[recipe:{}] {} {} -> {}\n  body: {}",
                    self.recipe.name,  # type: ignore[union-attr]
                    method, response.url, response.status_code,
                    (response.text or "")[:500],
                )
                response.raise_for_status()

            payload = response.json()
            items = _resolve_path(payload, items_field) or []
            if not isinstance(items, list):
                raise RecipeError(
                    f"Recipe '{self.recipe.name}': items_field "  # type: ignore[union-attr]
                    f"{items_field!r} did not resolve to a list."
                )

            for item in items:
                yield self._to_resource(item)

            # ── Next page? ──
            if ptype == "cursor":
                token_field = pagination.get("next_token_field") or "nextPageToken"
                next_page_token = (
                    payload.get(token_field)
                    if isinstance(payload, dict) else None
                )
                is_last = (
                    payload.get("isLast") is True
                    if isinstance(payload, dict) else False
                )
                if is_last or not next_page_token:
                    break
            elif ptype == "page":
                if not items:
                    break
                page += 1
            else:
                break

            first_request = False

    # ── Resource construction ─────────────────────────────────────────

    def _to_resource(self, item: dict[str, Any]) -> SourceResource:
        recipe = self.recipe
        assert recipe is not None  # for mypy

        ctx = {"item": item, "scope": self.scope, **item}

        rid_template = self._cfg.get("resource_id_template") or "{id}"
        resource_id = _render_template(rid_template, ctx)
        if recipe.source not in resource_id:
            # Namespacing convention: every existing ingestor prefixes
            # ``{source}:`` so a single string is unambiguous across
            # the whole index. Recipes that already include the prefix
            # in their template are left alone.
            resource_id = f"{recipe.source}:{resource_id}"

        title = _render_template(
            self._cfg.get("title_template") or resource_id, ctx
        )
        text = _render_template(
            self._cfg.get("text_template") or "{item}", ctx, allow_dict=True
        )
        url = _render_template(self._cfg.get("url_template") or "", ctx)

        last_updated_field = self._cfg.get("last_updated_field")
        last_updated = (
            _resolve_path(item, last_updated_field) or ""
        ) if last_updated_field else ""
        if not last_updated:
            last_updated = datetime.utcnow().isoformat()

        # ── Metadata (incl. promoted scope column) ──
        metadata: dict[str, Any] = {}
        for md_key, md_path in (self._cfg.get("metadata_fields") or {}).items():
            metadata[md_key] = _resolve_path(item, md_path)

        # Make sure the user-selected scope ends up on the right
        # promoted column so the WHERE-clause filter at query time
        # finds the row.
        promoted_col = _SCOPE_TO_PROMOTED_COLUMN.get(
            recipe.scope_field, "object_name"
        )
        if promoted_col not in metadata or metadata.get(promoted_col) in (None, ""):
            metadata[promoted_col] = self.scope or "all"

        # `object_name` is a small indexed display column — populate it
        # for the citation panel even when scope_field != external_id.
        metadata.setdefault("object_name", item.get("id") or resource_id)

        return SourceResource(
            resource_id=resource_id,
            title=title or resource_id,
            text=text or "",
            url=url,
            last_updated=str(last_updated),
            metadata=metadata,
        )

    # ── HTTP session ──────────────────────────────────────────────────

    def _build_session(self, fetch: dict[str, Any]) -> requests.Session:
        session = requests.Session()

        auth_kind = (fetch.get("auth") or "none").lower()
        creds = self.credentials
        if auth_kind == "basic":
            user = creds.get(fetch.get("auth_user_field") or "email")
            pwd = creds.get(fetch.get("auth_password_field") or "api_token")
            if user and pwd:
                session.auth = (user, pwd)
        elif auth_kind == "bearer":
            token_field = fetch.get("auth_token_field") or "access_token"
            token = creds.get(token_field)
            if token:
                session.headers["Authorization"] = f"Bearer {token}"
        # auth: none — leave session bare.

        session.headers.setdefault("Accept", "application/json")
        for k, v in (fetch.get("headers") or {}).items():
            session.headers[str(k)] = str(v)
        return session


# ──────────────────────────────────────────────────────────────────────
# Tiny templating helpers
# ──────────────────────────────────────────────────────────────────────


def _resolve_path(obj: Any, path: Optional[str]) -> Any:
    """Walk a dot-separated ``path`` through ``obj`` (dicts + list indices).

    Missing keys / out-of-range indices return ``None`` rather than
    raising — recipes are forgiving by design.
    """
    if not path:
        return obj
    cur: Any = obj
    for part in path.split("."):
        if cur is None:
            return None
        # Numeric segment → list index.
        if part.lstrip("-").isdigit() and isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (IndexError, ValueError):
                return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


def _render_template(
    template: str,
    context: dict[str, Any],
    *,
    allow_dict: bool = False,
) -> str:
    """Render a ``{a.b.c}``-style template against ``context``.

    Plain ``str.format`` doesn't support dotted lookups against nested
    dicts; we implement a tiny subset that does. Unresolved fields
    expand to an empty string so a partially-populated source item
    doesn't crash the whole batch.
    """
    if template is None:
        return ""
    if not isinstance(template, str):
        # Allow embedding a non-string template (e.g. a dict body) by
        # JSON-encoding it. This is what makes ``text_template: |
        #   {item}`` work on a non-string root.
        if allow_dict:
            try:
                return json.dumps(template)
            except TypeError:
                return str(template)
        return str(template)

    out: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "{":
            # Escape "{{" as a literal "{".
            if i + 1 < n and template[i + 1] == "{":
                out.append("{")
                i += 2
                continue
            end = template.find("}", i + 1)
            if end == -1:
                # Unterminated brace — emit the rest verbatim.
                out.append(template[i:])
                break
            field = template[i + 1: end].strip()
            value = _resolve_path(context, field)
            if value is None:
                out.append("")
            elif isinstance(value, (dict, list)):
                out.append(json.dumps(value, default=str))
            else:
                out.append(str(value))
            i = end + 1
        elif ch == "}" and i + 1 < n and template[i + 1] == "}":
            out.append("}")
            i += 2
        else:
            out.append(ch)
            i += 1

    return "".join(out)


# Re-export so callers can `from app.ingestion.recipe_runner import settings`
# style imports without pulling app.config directly. Purely a convenience.
__all__ = [
    "BuiltinRecipeIngestor",
    "run_recipe",
    "run_recipe_object",
    "settings",
]
