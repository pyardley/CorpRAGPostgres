"""
Unit tests for core.git_dependency_extraction.find_imports.

Uses the literal fixture files under fixtures/git/pyrepo and
fixtures/git/jsrepo — a small multi-file repo where several files import
one shared utility module, and one file that looks similar but
deliberately doesn't (mirroring fixtures/sql/06_reports.sql's shared-vs-
independent staging procedure shape, per README §9's fixture guidance).
Exercises the tree-sitter-based parser end to end, no live GitHub repo
needed.
"""

from pathlib import Path

from core.git_dependency_extraction import build_known_files, find_imports

_PYREPO = Path(__file__).parent.parent / "fixtures" / "git" / "pyrepo"
_JSREPO = Path(__file__).parent.parent / "fixtures" / "git" / "jsrepo"


def _read(repo: Path, rel_path: str) -> str:
    return (repo / rel_path).read_text(encoding="utf-8")


def _catalog(repo: Path, rel_paths: list[str]) -> frozenset[str]:
    return build_known_files(rel_paths)


_PY_PATHS = [
    "shared/__init__.py",
    "shared/utils.py",
    "feature_a.py",
    "feature_b.py",
    "feature_c.py",
    "independent.py",
    "subpkg/__init__.py",
    "subpkg/sibling.py",
    "subpkg/feature_d.py",
]

_JS_PATHS = [
    "shared/utils.js",
    "feature_a.js",
    "feature_b.js",
    "nested/feature_c.js",
    "independent.js",
]


def test_from_module_import_name_resolves_to_submodule():
    """feature_a.py: `from shared.utils import helper` -> shared/utils.py."""
    known = _catalog(_PYREPO, _PY_PATHS)
    edges = find_imports(_read(_PYREPO, "feature_a.py"), "feature_a.py", known)
    assert edges == [("feature_a.py", "imports", "shared/utils.py")]


def test_from_package_import_submodule_name_resolves():
    """feature_b.py: `from shared import utils` -> shared/utils.py (the
    imported *name* is itself a submodule, preferred over the package's
    own __init__.py)."""
    known = _catalog(_PYREPO, _PY_PATHS)
    edges = find_imports(_read(_PYREPO, "feature_b.py"), "feature_b.py", known)
    assert edges == [("feature_b.py", "imports", "shared/utils.py")]


def test_plain_import_dotted_path_resolves():
    """feature_c.py: `import shared.utils` -> shared/utils.py."""
    known = _catalog(_PYREPO, _PY_PATHS)
    edges = find_imports(_read(_PYREPO, "feature_c.py"), "feature_c.py", known)
    assert edges == [("feature_c.py", "imports", "shared/utils.py")]


def test_distractor_never_produces_an_edge():
    """independent.py deliberately does not import shared.utils, despite
    defining a same-named local `helper` function and importing `os` —
    must never be reported as importing shared/utils.py."""
    known = _catalog(_PYREPO, _PY_PATHS)
    edges = find_imports(_read(_PYREPO, "independent.py"), "independent.py", known)
    assert edges == []


def test_stdlib_import_without_known_match_produces_no_edge():
    """`import os` in independent.py must not resolve to anything (os.py
    isn't in the catalog) -- silence over noise, not a guess."""
    known = _catalog(_PYREPO, _PY_PATHS)
    edges = find_imports(_read(_PYREPO, "independent.py"), "independent.py", known)
    assert all(obj != "os.py" for _, _, obj in edges)


def test_multi_level_relative_import_resolves_up_and_across():
    """subpkg/feature_d.py: `from ..shared import utils` -> shared/utils.py
    (one level up from subpkg/, then into the shared package)."""
    known = _catalog(_PYREPO, _PY_PATHS)
    edges = find_imports(
        _read(_PYREPO, "subpkg/feature_d.py"), "subpkg/feature_d.py", known
    )
    assert ("subpkg/feature_d.py", "imports", "shared/utils.py") in edges


def test_bare_relative_import_resolves_sibling_in_same_package():
    """subpkg/feature_d.py: `from . import sibling` -> subpkg/sibling.py."""
    known = _catalog(_PYREPO, _PY_PATHS)
    edges = find_imports(
        _read(_PYREPO, "subpkg/feature_d.py"), "subpkg/feature_d.py", known
    )
    assert ("subpkg/feature_d.py", "imports", "subpkg/sibling.py") in edges
    assert len(edges) == 2  # exactly the two resolvable imports, nothing else


def test_self_import_never_appears():
    """A file can never end up importing itself in the returned edges."""
    known = _catalog(_PYREPO, _PY_PATHS)
    edges = find_imports(_read(_PYREPO, "shared/utils.py"), "shared/utils.py", known)
    assert all(subj != obj for subj, _, obj in edges)


def test_unsupported_extension_returns_empty():
    known = _catalog(_PYREPO, _PY_PATHS)
    assert find_imports("import shared.utils", "notes.md", known) == []


def test_unparseable_content_returns_empty_not_raises():
    known = _catalog(_PYREPO, _PY_PATHS)
    # Garbage bytes shouldn't crash the scan -- tree-sitter's error
    # recovery just yields ERROR nodes we don't walk into meaningfully.
    assert find_imports("\x00\x01 not python (((", "feature_a.py", known) == []


# ── JavaScript / TypeScript ─────────────────────────────────────────────


def test_js_require_relative_specifier_resolves():
    """feature_a.js: `require('./shared/utils')` -> shared/utils.js."""
    known = _catalog(_JSREPO, _JS_PATHS)
    edges = find_imports(_read(_JSREPO, "feature_a.js"), "feature_a.js", known)
    assert edges == [("feature_a.js", "imports", "shared/utils.js")]


def test_js_import_relative_specifier_with_extension_resolves():
    """feature_b.js: `import { helper } from './shared/utils.js'` ->
    shared/utils.js."""
    known = _catalog(_JSREPO, _JS_PATHS)
    edges = find_imports(_read(_JSREPO, "feature_b.js"), "feature_b.js", known)
    assert edges == [("feature_b.js", "imports", "shared/utils.js")]


def test_js_nested_file_resolves_parent_relative_import():
    """nested/feature_c.js: `import { helper } from '../shared/utils'` ->
    shared/utils.js (one level up from nested/)."""
    known = _catalog(_JSREPO, _JS_PATHS)
    edges = find_imports(
        _read(_JSREPO, "nested/feature_c.js"), "nested/feature_c.js", known
    )
    assert ("nested/feature_c.js", "imports", "shared/utils.js") in edges


def test_js_bare_npm_specifier_produces_no_edge():
    """nested/feature_c.js also does `import React from 'react'` -- a bare
    specifier is always an external package, never a repo file."""
    known = _catalog(_JSREPO, _JS_PATHS)
    edges = find_imports(
        _read(_JSREPO, "nested/feature_c.js"), "nested/feature_c.js", known
    )
    assert all(obj != "react" for _, _, obj in edges)
    assert len(edges) == 1


def test_js_distractor_never_produces_an_edge():
    """independent.js defines its own local `helper` and never imports
    shared/utils.js."""
    known = _catalog(_JSREPO, _JS_PATHS)
    edges = find_imports(_read(_JSREPO, "independent.js"), "independent.js", known)
    assert edges == []
