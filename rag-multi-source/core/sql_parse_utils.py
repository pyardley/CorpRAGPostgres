"""
Shared ``sqlglot`` parsing helpers for the sqlglot-based SQL impact-analysis
engine (``core.sql_dependency_extraction_sqlglot``,
``core.sql_join_shape_sqlglot``, ``core.sql_column_lineage``).

Centralizes the "parse T-SQL, fail silent" contract so each engine module
doesn't reimplement it -- a parser that raises on an unsupported construct
would turn a should-be-silent unparseable case into a crash, the opposite
of this codebase's "silence over noise" posture (see
``core.sql_join_shape``'s module docstring for the same philosophy applied
to the ``sqlparse``-based engine).
"""

from __future__ import annotations

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover - sqlglot is an optional dependency, only required when settings.SQL_IMPACT_ANALYSIS_ENGINE == "sqlglot"
    sqlglot = None
    exp = None


def parse_tsql(sql: str) -> list:
    """
    Parse ``sql`` as a list of top-level T-SQL expressions
    (``sqlglot.parse(sql, read="tsql")``). Returns ``[]`` on any parse
    error, or if ``sqlglot`` isn't importable, rather than raising.
    """
    if sqlglot is None:
        return []
    try:
        return sqlglot.parse(sql, read="tsql")
    except Exception:
        return []


def qualified_table_name(table: "exp.Table") -> str:
    """
    ``schema.name`` (or bare ``name`` / ``#temp``) for a ``sqlglot`` Table
    node, with any alias ignored. sqlglot's ``tsql`` reader strips the
    ``#`` off a T-SQL temp-table name into a separate ``temporary`` flag
    on the underlying Identifier rather than keeping it in ``.name`` --
    confirmed empirically against this codebase's own fixtures -- so this
    reconstructs the ``#``-prefixed display form callers expect.
    """
    name = table.name
    ident = table.this
    if isinstance(ident, exp.Identifier) and ident.args.get("temporary"):
        name = f"#{name}"
    db = table.db
    return f"{db}.{name}" if db else name


def resolve_known_object(
    qualified_name: str, known_objects: dict[str, tuple[str, str]]
) -> str | None:
    """
    Map a ``qualified_table_name()`` result to its lowercased
    ``known_objects`` key, or ``None`` if it isn't a known object.

    Mirrors ``core.sql_dependency_extraction.find_references``'s policy
    exactly: a schema-qualified name matches directly; a bare
    (non-schema-qualified) name only matches against the ``dbo`` schema
    (the overwhelmingly common unqualified-reference case) -- carried
    over as a parity choice, not something the AST forces, so switching
    engines doesn't silently change cross-schema matching behavior.
    """
    lowered = qualified_name.lower()
    if "." in lowered:
        return lowered if lowered in known_objects else None
    dbo_key = f"dbo.{lowered}"
    return dbo_key if dbo_key in known_objects else None
