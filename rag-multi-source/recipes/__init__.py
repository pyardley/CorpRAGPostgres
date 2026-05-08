"""
recipes — declarative ingestion configuration for CorporateRAG.

A *recipe* is a small YAML (or, occasionally, Python) file that describes
how to fetch resources from a corporate data source and feed them into
the existing :class:`app.ingestion.base.BaseIngestor` pipeline. Most
sources can be expressed declaratively; complex ones can still ship as
a full ``BaseIngestor`` subclass and merely *register* themselves via a
recipe so the CLI / sidebar discover them automatically.

Public surface
--------------
* :class:`Recipe` — the dataclass produced by parsing one recipe file.
* :exc:`RecipeError` / :exc:`RecipeNotFoundError` — invalid / missing
  recipe errors.
* :func:`list_recipes` — every recipe known to the registry.
* :func:`get_recipe` — fetch a recipe by name.
* :func:`load_recipe_file` — parse a single YAML/Py file.
* :func:`reload_recipes` — clear the discovery cache (useful in dev).

The registry walks the ``recipes/`` package directory at first use and
caches the result. Drop a new ``.yaml`` next to this file and it will
appear automatically the next time the CLI starts or the registry is
reloaded.
"""

from __future__ import annotations

from recipes.recipe import (  # noqa: F401  (re-export)
    Recipe,
    RecipeError,
    RecipeNotFoundError,
    RecipeValidationError,
)
from recipes.registry import (  # noqa: F401  (re-export)
    get_recipe,
    list_recipes,
    load_recipe_file,
    reload_recipes,
)

__all__ = [
    "Recipe",
    "RecipeError",
    "RecipeNotFoundError",
    "RecipeValidationError",
    "get_recipe",
    "list_recipes",
    "load_recipe_file",
    "reload_recipes",
]
