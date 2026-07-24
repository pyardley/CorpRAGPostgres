"""
Static Git/GitHub import-dependency extraction.

Applies README "Possible enhancements" §9's lesson list to Git-ingested
files: scans a file's content for import statements resolving to another
*ingested* file in the same repo, producing (subject, "imports", object)
edges for the entity graph (see `core.entity_graph`) — the same role
`core.sql_dependency_extraction.find_references` plays for SQL Server
objects.

Unlike the SQL module, this uses a real parser (`tree-sitter`, via
`tree_sitter_language_pack`) rather than regex, on the README's own
explicit recommendation: no single tool plays sqlglot's role for
general-purpose languages, but tree-sitter is the closest equivalent — one
host library, per-language grammars, prebuilt wheels, no per-language
toolchain. An `import_statement`/`import_from_statement`/`require(...)`
node is structurally unambiguous, so the SQL engine's keyword-collision
class of bug (`RETURNS DECIMAL(...)` word-boundary-matching a table named
`Returns` — see `sql_dependency_extraction`'s docstring) cannot happen here
by construction; there is no bare-keyword-adjacency workaround to get
right, unlike the regex engine.

Supported languages: Python (`.py`), JavaScript (`.js`/`.jsx`/`.mjs`/`.cjs`),
TypeScript (`.ts`/`.tsx`). Any other extension returns `[]` immediately —
same "silence over noise" stance as the SQL module: an unresolved or
unsupported reference produces no edge rather than a guess.

Known limitations, by design:
- Only imports that resolve to a *known, cataloged* file produce an edge.
  Bare/npm-style JS/TS specifiers (`import x from 'react'`) are always
  external packages, never repo files, and are silently skipped.
- Python absolute (non-relative) imports are resolved by *path suffix*
  against the known-file catalog, since the repo root is not necessarily
  the same directory as the top-level package root (e.g. a `src/` layout).
  A candidate matching more than one known file is treated as ambiguous
  and skipped — no edge, not a guess.
- Dynamic imports (`importlib.import_module(...)`, JS dynamic `import()`,
  `require(someVariable)`) are invisible, the same class of gap dynamic
  SQL is for `find_references`.
"""

from __future__ import annotations

import posixpath
from typing import Optional

_PY_EXTENSIONS = frozenset({".py"})
_JS_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs"})
_TS_EXTENSIONS = frozenset({".ts"})
_TSX_EXTENSIONS = frozenset({".tsx"})

# Public: reused by mcp_server.tools.git_dependency_tools to filter a
# live-fetched repo tree down to the extensions this module can parse,
# without duplicating the language list there.
SUPPORTED_EXTENSIONS = _PY_EXTENSIONS | _JS_EXTENSIONS | _TS_EXTENSIONS | _TSX_EXTENSIONS
_SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS

# Extensions tried, in order, when resolving a JS/TS relative specifier
# that doesn't match a known file verbatim (`./foo` -> `foo.js`, etc.).
_JS_RESOLVE_SUFFIXES = (
    "", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    "/index.js", "/index.jsx", "/index.ts", "/index.tsx",
)

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

_parser_cache: dict[str, object] = {}


def _ext(path: str) -> str:
    filename = path.rsplit("/", 1)[-1]
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _get_parser(language: str):
    parser = _parser_cache.get(language)
    if parser is None:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(language)  # type: ignore[assignment]
        _parser_cache[language] = parser
    return parser


def build_known_files(paths: list[str]) -> frozenset[str]:
    """
    All cataloged file paths in the repo (already filtered to indexed
    extensions) — the resolution target set `find_imports` matches
    against. Paths are repo-root-relative, forward-slash separated (the
    same shape GitHub's tree API and `git_ingestor.py` already use).
    """
    return frozenset(paths)


def find_imports(
    content: str, file_path: str, known_files: frozenset[str]
) -> list[tuple[str, str, str]]:
    """
    Find imports inside `content` (the full text of `file_path`) that
    resolve to another file in `known_files`.

    Returns (subject, "imports", object) triples where `subject` is
    `file_path` itself and `object` is the resolved target's path — both
    plain repo-relative paths, not yet `git_scope`-qualified (that
    prefixing happens at the ingestion call site, mirroring how
    `sql_ingestor.py` prefixes `find_references`' output with `db_name`
    rather than doing it inside `find_references` itself). Self-imports
    are impossible by construction (a file can't resolve to its own path
    via a relative specifier without an empty segment) but are filtered
    defensively anyway.
    """
    ext = _ext(file_path)
    if ext not in _SUPPORTED_EXTENSIONS:
        return []

    language = _LANGUAGE_BY_EXT[ext]
    try:
        parser = _get_parser(language)
        tree = parser.parse(content.encode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - unparseable content -> no edges, not a crash
        return []

    if ext in _PY_EXTENSIONS:
        targets = _python_import_targets(tree.root_node, file_path)
    else:
        targets = _js_import_targets(tree.root_node)

    edges: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for target in targets:
        resolved = (
            _resolve_python_target(target, file_path, known_files)
            if ext in _PY_EXTENSIONS
            else _resolve_js_target(target, file_path, known_files)
        )
        if resolved and resolved != file_path and resolved not in seen:
            seen.add(resolved)
            edges.append((file_path, "imports", resolved))
    return edges


# ──────────────────────────────────────────────────────────────────────────────
# Python
# ──────────────────────────────────────────────────────────────────────────────
#
# A "target" here is either:
#   ("absolute", "a.b.c")                         -- import a.b.c / from a.b.c import x
#   ("relative", level, "foo.bar", ["x", "y"])     -- from ..foo.bar import x, y
# `find_imports` resolves each into zero or more candidate paths.


def _node_text(node, default: str = "") -> str:
    if node is None:
        return default
    return node.text.decode("utf-8", errors="replace")


def _dotted_name_text(node) -> str:
    """Join a `dotted_name` node's identifier children with '.'."""
    parts = [
        _node_text(child) for child in node.children if child.type == "identifier"
    ]
    return ".".join(parts) if parts else _node_text(node)


def _python_import_targets(root, file_path: str) -> list[tuple]:
    targets: list[tuple] = []

    def walk(node):
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    targets.append(("absolute", _dotted_name_text(child)))
                elif child.type == "aliased_import":
                    dotted = child.child_by_field_name("name") or next(
                        (c for c in child.children if c.type == "dotted_name"), None
                    )
                    if dotted is not None:
                        targets.append(("absolute", _dotted_name_text(dotted)))
        elif node.type == "import_from_statement":
            module_node = None
            imported_names: list[str] = []
            for child in node.children:
                if child.type in ("dotted_name", "relative_import") and module_node is None:
                    module_node = child
                elif child.type == "dotted_name" and module_node is not None:
                    imported_names.append(_dotted_name_text(child))
                elif child.type == "aliased_import":
                    dotted = next(
                        (c for c in child.children if c.type == "dotted_name"), None
                    )
                    if dotted is not None:
                        imported_names.append(_dotted_name_text(dotted))
            if module_node is not None:
                if module_node.type == "dotted_name":
                    targets.append(
                        ("absolute_from", _dotted_name_text(module_node), imported_names)
                    )
                else:  # relative_import
                    prefix = next(
                        (c for c in module_node.children if c.type == "import_prefix"),
                        None,
                    )
                    level = len(_node_text(prefix)) if prefix is not None else 1
                    tail = next(
                        (c for c in module_node.children if c.type == "dotted_name"),
                        None,
                    )
                    tail_text = _dotted_name_text(tail) if tail is not None else ""
                    targets.append(("relative", level, tail_text, imported_names))
        for child in node.children:
            walk(child)

    walk(root)
    return targets


def _candidates_for_module_path(module_path: str) -> list[str]:
    """'a/b/c' -> ['a/b/c.py', 'a/b/c/__init__.py']."""
    if not module_path:
        return []
    return [f"{module_path}.py", f"{module_path}/__init__.py"]


def _resolve_by_suffix(candidate: str, known_files: frozenset[str]) -> Optional[str]:
    if candidate in known_files:
        return candidate
    matches = [p for p in known_files if p.endswith("/" + candidate)]
    if len(matches) == 1:
        return matches[0]
    return None  # zero or ambiguous -> no edge


def _resolve_by_exact(candidate: str, known_files: frozenset[str]) -> Optional[str]:
    return candidate if candidate in known_files else None


def _resolve_python_target(
    target: tuple, file_path: str, known_files: frozenset[str]
) -> Optional[str]:
    kind = target[0]

    if kind == "absolute":
        _, dotted = target
        module_path = dotted.replace(".", "/")
        for candidate in _candidates_for_module_path(module_path):
            resolved = _resolve_by_suffix(candidate, known_files)
            if resolved:
                return resolved
        return None

    if kind == "absolute_from":
        _, dotted, names = target
        base_path = dotted.replace(".", "/")
        # Prefer a submodule match for each imported name (e.g. `from
        # shared import utils` -> shared/utils.py); fall back to the
        # base module itself (e.g. `from shared.utils import helper`,
        # where `helper` isn't a file) at most once per statement.
        for name in names:
            for candidate in _candidates_for_module_path(f"{base_path}/{name}"):
                resolved = _resolve_by_suffix(candidate, known_files)
                if resolved:
                    return resolved
        for candidate in _candidates_for_module_path(base_path):
            resolved = _resolve_by_suffix(candidate, known_files)
            if resolved:
                return resolved
        return None

    # kind == "relative"
    _, level, tail, names = target
    base_dir = posixpath.dirname(file_path)
    for _ in range(level - 1):
        base_dir = posixpath.dirname(base_dir)
    base_path = posixpath.normpath(posixpath.join(base_dir, tail)) if tail else base_dir
    base_path = base_path.replace("\\", "/").lstrip("/")

    for name in names:
        for candidate in _candidates_for_module_path(f"{base_path}/{name}" if base_path else name):
            resolved = _resolve_by_exact(candidate, known_files)
            if resolved:
                return resolved
    if tail:
        for candidate in _candidates_for_module_path(base_path):
            resolved = _resolve_by_exact(candidate, known_files)
            if resolved:
                return resolved
    return None


# ──────────────────────────────────────────────────────────────────────────────
# JavaScript / TypeScript
# ──────────────────────────────────────────────────────────────────────────────


def _string_literal_text(node) -> Optional[str]:
    fragment = next(
        (c for c in node.children if c.type == "string_fragment"), None
    )
    if fragment is not None:
        return _node_text(fragment)
    text = _node_text(node)
    return text.strip("'\"`") if text else None


def _js_import_targets(root) -> list[str]:
    specifiers: list[str] = []

    def walk(node):
        if node.type == "import_statement":
            string_node = next((c for c in node.children if c.type == "string"), None)
            if string_node is not None:
                spec = _string_literal_text(string_node)
                if spec:
                    specifiers.append(spec)
        elif node.type == "call_expression":
            func = node.children[0] if node.children else None
            if func is not None and func.type == "identifier" and _node_text(func) == "require":
                args = next((c for c in node.children if c.type == "arguments"), None)
                if args is not None:
                    string_node = next((c for c in args.children if c.type == "string"), None)
                    if string_node is not None:
                        spec = _string_literal_text(string_node)
                        if spec:
                            specifiers.append(spec)
        for child in node.children:
            walk(child)

    walk(root)
    return specifiers


def _resolve_js_target(
    specifier: str, file_path: str, known_files: frozenset[str]
) -> Optional[str]:
    if not (specifier.startswith("./") or specifier.startswith("../")):
        return None  # bare specifier -> external package, not a repo file

    base_dir = posixpath.dirname(file_path)
    joined = posixpath.normpath(posixpath.join(base_dir, specifier)).replace("\\", "/")
    joined = joined.lstrip("/")

    for suffix in _JS_RESOLVE_SUFFIXES:
        candidate = joined + suffix
        if candidate in known_files:
            return candidate
    return None
