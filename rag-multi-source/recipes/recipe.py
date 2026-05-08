"""
Recipe dataclass + validation.

A :class:`Recipe` is the parsed in-memory representation of one YAML or
Python recipe file. The dataclass is deliberately *lenient* — anything
the bundled builtin parser doesn't understand is preserved on
``Recipe.config`` so user-authored recipes can carry custom keys for
their own ``BaseIngestor`` subclass to read at runtime.

Validation
----------
:func:`Recipe.from_dict` is the single entry point used by both the
YAML loader (``recipes.registry``) and any code building a Recipe
in-memory. It enforces the *minimum* shape required for the system to
work:

* ``name``     — non-empty string, kebab/snake_case (defaults to the
                 file stem when loading from disk).
* ``source``   — non-empty string (also written to
                 ``vector_chunks.source``).
* ``credential_key`` — non-empty string (loaded via
                 :func:`app.utils.load_all_credentials`).
* ``scope_field``    — one of the promoted columns on
                 ``vector_chunks`` (or ``"external_id"``).
* ``parser``         — either ``"builtin"`` or a fully qualified Python
                 path of the form ``"package.module.ClassName"``.
                 Mutually exclusive with ``python_module`` (which loads
                 a module from inside ``recipes/``).

Anything else (auth, pagination, templates, custom keys) lives on
``config`` and is interpreted by whichever parser the recipe targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# Promoted scope columns currently supported by ``vector_chunks``. The
# generic "external_id" pseudo-field is mapped onto the ``object_name``
# column so brand-new source types still get an indexed scope filter
# without a schema migration. Adding a new column to ``vector_chunks``
# is a separate one-line migration; see models/migrations.py.
VALID_SCOPE_FIELDS: tuple[str, ...] = (
    "project_key",
    "space_key",
    "db_name",
    "git_scope",
    "email_provider",
    # Generic / new-source path. Resolved at runtime by RecipeRunner to
    # the ``object_name`` promoted column so existing indexes still
    # apply. Recipes that target a source with no first-class column
    # should use this one.
    "external_id",
)


# ──────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────


class RecipeError(Exception):
    """Base class for every recipe-loading / validation error."""


class RecipeValidationError(RecipeError):
    """Raised when a recipe file is structurally invalid."""


class RecipeNotFoundError(RecipeError):
    """Raised when ``get_recipe(name)`` cannot find ``name``."""


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Recipe:
    """One parsed recipe — directly consumable by ``RecipeRunner``."""

    # Identity
    name: str
    source: str
    description: str = ""

    # Credentials + scope
    credential_key: str = ""
    scope_field: str = "external_id"
    scope_label: str = ""

    # Parser dispatch. Exactly one of ``parser`` / ``python_module``
    # is meaningful; ``parser`` wins when both are supplied.
    parser: str = "builtin"
    python_module: Optional[str] = None

    # Free-form parser config (templates, fetch settings, custom keys).
    # The builtin parser knows about a fixed set of keys; everything
    # else is preserved verbatim and made available to user-authored
    # ``BaseIngestor`` subclasses via ``ingestor.recipe.config``.
    config: dict[str, Any] = field(default_factory=dict)

    # Optional chunking overrides (else ``settings.CHUNK_SIZE``).
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    # Source path (filled by the registry when loaded from disk).
    # Used in error messages and in the sidebar tooltip.
    source_path: Optional[str] = None

    # ── Builders ────────────────────────────────────────────────────

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        default_name: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> "Recipe":
        """Validate ``data`` and return a fully-populated Recipe.

        ``default_name`` is used when the YAML omits ``name`` (typical:
        the registry passes the file's stem).
        """
        if not isinstance(data, dict):
            raise RecipeValidationError(
                f"Recipe must be a mapping/object, got {type(data).__name__}"
            )

        name = (data.get("name") or default_name or "").strip()
        if not name:
            raise RecipeValidationError(
                "Recipe is missing a 'name' (and no default was supplied)."
            )

        source = (data.get("source") or "").strip()
        if not source:
            raise RecipeValidationError(
                f"Recipe '{name}' is missing the required 'source' key."
            )

        credential_key = (data.get("credential_key") or source).strip()
        if not credential_key:
            raise RecipeValidationError(
                f"Recipe '{name}' has an empty 'credential_key'."
            )

        scope_field = (data.get("scope_field") or "external_id").strip()
        if scope_field not in VALID_SCOPE_FIELDS:
            raise RecipeValidationError(
                f"Recipe '{name}' has invalid scope_field={scope_field!r}. "
                f"Must be one of: {', '.join(VALID_SCOPE_FIELDS)}."
            )

        parser = (data.get("parser") or "builtin").strip()
        python_module = data.get("python_module")
        if parser != "builtin" and "." not in parser:
            raise RecipeValidationError(
                f"Recipe '{name}' has invalid parser={parser!r}. "
                f"Use 'builtin' or a fully qualified 'module.ClassName'."
            )

        config = dict(data.get("config") or {})

        # Hoist a small handful of common builtin-parser keys so that
        # short YAMLs without an explicit ``config:`` section still
        # work — anything we don't recognise stays in `config` as-is.
        for top_key in (
            "fetch",
            "resource_id_template",
            "title_template",
            "text_template",
            "url_template",
            "last_updated_field",
            "metadata_fields",
            "incremental_param",
        ):
            if top_key in data and top_key not in config:
                config[top_key] = data[top_key]

        chunk_size = data.get("chunk_size")
        chunk_overlap = data.get("chunk_overlap")
        if chunk_size is not None and not isinstance(chunk_size, int):
            raise RecipeValidationError(
                f"Recipe '{name}': chunk_size must be an integer."
            )
        if chunk_overlap is not None and not isinstance(chunk_overlap, int):
            raise RecipeValidationError(
                f"Recipe '{name}': chunk_overlap must be an integer."
            )

        return cls(
            name=name,
            source=source,
            description=str(data.get("description") or ""),
            credential_key=credential_key,
            scope_field=scope_field,
            scope_label=str(data.get("scope_label") or ""),
            parser=parser,
            python_module=python_module,
            config=config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_path=source_path,
        )

    # ── Convenience accessors ────────────────────────────────────────

    @property
    def is_builtin(self) -> bool:
        """True when this recipe should run through the generic builtin parser."""
        return self.parser == "builtin"

    @property
    def parser_class_path(self) -> Optional[str]:
        """Returns the fully qualified class path, or ``None`` for builtin."""
        return None if self.is_builtin else self.parser

    def display_label(self) -> str:
        """One-line label used in CLI listings / sidebar dropdowns."""
        if self.description:
            return f"{self.name}  —  {self.description}"
        return self.name
