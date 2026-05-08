"""
Recipe registry — discovers every YAML / Python recipe in ``recipes/``.

Discovery rules
---------------
* Any ``.yaml`` or ``.yml`` file under the ``recipes/`` package
  (recursively, except dotfiles and ``__pycache__``) is treated as a
  declarative recipe and parsed via :func:`Recipe.from_dict`.
* Any ``.py`` file under ``recipes/`` (other than ``recipe.py`` /
  ``registry.py`` / ``__init__.py``) is *imported* and inspected for a
  top-level ``recipe`` attribute. If it's a :class:`Recipe`, it's
  registered. The Python module path is recorded on
  ``recipe.python_module`` so the runner can re-import it later for
  the parser class.

The registry caches its result in process memory; call
:func:`reload_recipes` to clear the cache after editing files.

Errors during discovery are *isolated* — one bad recipe must not
prevent the others from being usable, so we log and skip.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from recipes.recipe import (
    Recipe,
    RecipeError,
    RecipeNotFoundError,
    RecipeValidationError,
)


# Module-level cache. Reset by :func:`reload_recipes`. We index by
# recipe name so two YAMLs cannot share a name (last-loaded wins, with
# a warning) — this keeps the CLI / sidebar pickers unambiguous.
_RECIPE_CACHE: Optional[dict[str, Recipe]] = None


# ──────────────────────────────────────────────────────────────────────
# YAML loading
# ──────────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    """Return the parsed YAML contents of ``path`` as a dict.

    Raises :exc:`RecipeValidationError` when the file is unparseable
    or doesn't contain a top-level mapping.
    """
    try:
        import yaml  # PyYAML — already a transitive dep via langchain
    except ImportError as exc:  # pragma: no cover - configuration error
        raise RecipeValidationError(
            "PyYAML is required to load recipes (.yaml). "
            "Install with `pip install pyyaml`."
        ) from exc

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecipeValidationError(
            f"Could not read recipe file {path}: {exc}"
        ) from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RecipeValidationError(
            f"Invalid YAML in {path}: {exc}"
        ) from exc

    if data is None:
        raise RecipeValidationError(f"Recipe file {path} is empty.")
    if not isinstance(data, dict):
        raise RecipeValidationError(
            f"Recipe file {path} must contain a YAML mapping at top level."
        )
    return data


def load_recipe_file(path: str | os.PathLike[str]) -> Recipe:
    """Parse a single recipe file (.yaml/.yml/.py) into a :class:`Recipe`.

    YAML files yield a recipe directly. Python files are imported and
    expected to expose a top-level ``recipe`` attribute that is a
    :class:`Recipe` instance.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    default_name = p.stem

    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(p)
        return Recipe.from_dict(
            data, default_name=default_name, source_path=str(p)
        )

    if suffix == ".py":
        return _import_python_recipe(p, default_name=default_name)

    raise RecipeValidationError(
        f"Unsupported recipe file extension: {p.name!r} "
        f"(supported: .yaml, .yml, .py)."
    )


def _import_python_recipe(path: Path, *, default_name: str) -> Recipe:
    """Import a recipe authored as a Python module and pluck its ``recipe`` attr."""
    module_name = f"recipes._dynamic_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RecipeValidationError(
            f"Could not build an importer for {path}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001 - surface as a recipe error
        raise RecipeValidationError(
            f"Failed to import recipe module {path}: {exc}"
        ) from exc

    recipe = getattr(module, "recipe", None)
    if not isinstance(recipe, Recipe):
        raise RecipeValidationError(
            f"Python recipe {path} must export a top-level 'recipe' "
            f"attribute that is a Recipe instance."
        )
    if not recipe.name:
        recipe.name = default_name
    if not recipe.source_path:
        recipe.source_path = str(path)
    if not recipe.python_module:
        recipe.python_module = module_name
    return recipe


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────


def _recipes_dir() -> Path:
    """The on-disk ``recipes/`` directory.

    Resolved via ``__file__`` so the package works whether installed
    in-tree or as a wheel — we never assume the CWD is the repo root.
    """
    return Path(__file__).resolve().parent


# Files that must NOT be treated as recipes even though they live
# inside the ``recipes/`` package directory.
_RESERVED_NAMES = {"__init__.py", "recipe.py", "registry.py"}


def _iter_recipe_paths(root: Path):
    """Yield every recipe-candidate file under ``root`` (recursively)."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden / cache dirs in-place.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d != "__pycache__"
        ]
        for fname in filenames:
            if fname.startswith("."):
                continue
            if fname in _RESERVED_NAMES:
                continue
            lower = fname.lower()
            if lower.endswith((".yaml", ".yml", ".py")):
                yield Path(dirpath) / fname


def _discover_all() -> dict[str, Recipe]:
    """Walk ``recipes/`` and return a name → Recipe map.

    Errors on individual files are logged and skipped so one bad
    recipe can't poison the whole registry.
    """
    out: dict[str, Recipe] = {}
    root = _recipes_dir()
    for path in _iter_recipe_paths(root):
        try:
            recipe = load_recipe_file(path)
        except RecipeError as exc:
            logger.warning("Skipping invalid recipe {}: {}", path, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - last-line defence
            logger.exception(
                "Unexpected error loading recipe {}: {}", path, exc
            )
            continue

        if recipe.name in out:
            logger.warning(
                "Duplicate recipe name '{}' — {} overrides {}.",
                recipe.name,
                path,
                out[recipe.name].source_path,
            )
        out[recipe.name] = recipe
    return out


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def list_recipes() -> list[Recipe]:
    """Return every discovered recipe, sorted by name."""
    global _RECIPE_CACHE
    if _RECIPE_CACHE is None:
        _RECIPE_CACHE = _discover_all()
    return sorted(_RECIPE_CACHE.values(), key=lambda r: r.name)


def get_recipe(name: str) -> Recipe:
    """Return the recipe with the given ``name`` or raise."""
    global _RECIPE_CACHE
    if _RECIPE_CACHE is None:
        _RECIPE_CACHE = _discover_all()
    recipe = _RECIPE_CACHE.get(name)
    if recipe is None:
        known = ", ".join(sorted(_RECIPE_CACHE.keys())) or "(none)"
        raise RecipeNotFoundError(
            f"No recipe named {name!r}. Known recipes: {known}."
        )
    return recipe


def reload_recipes() -> list[Recipe]:
    """Clear the discovery cache and re-walk the recipes directory."""
    global _RECIPE_CACHE
    _RECIPE_CACHE = None
    return list_recipes()
