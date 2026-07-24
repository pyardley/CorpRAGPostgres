"""
sqlglot-AST-based alternative to ``core.sql_dependency_extraction``.

Signature- and predicate-compatible drop-in for ``find_references`` --
selected in place of the regex/``sqlparse`` version whenever
``settings.SQL_IMPACT_ANALYSIS_ENGINE == "sqlglot"`` (see
``core.sql_impact_engine``, the single dispatcher every call site
actually imports from).

Where the legacy module does a word-boundary substring search against a
pre-built catalog (deliberately not a real parser -- see its own
docstring), this one walks a real T-SQL AST: every ``Table`` node
(``FROM``/``JOIN`` targets, ``EXEC`` targets -- confirmed empirically
that sqlglot represents an ``EXEC dbo.proc`` target as a nested ``Table``
node its own generic table walk already finds, no separate handling
needed) plus qualified scalar function calls (``dbo.fn_X(...)``, an
``Anonymous`` node whose parent is a ``Dot``). A structural improvement
this gets for free: the legacy module's documented ``RETURNS
DECIMAL(12,2)`` vs. a table literally named ``dbo.Returns`` false-positive
(a whole keyword-adjacency workaround + regression test there) cannot
happen here at all -- ``RETURNS`` is a clause keyword, never a ``Table``
node.

Known limitations, same in spirit as the legacy module:
- String literals aren't a concern here (a real parser never turns a
  quoted string into a ``Table`` node), but dynamic SQL
  (``EXEC(@sql)``, ``sp_executesql``) is still invisible either way --
  no parser can see inside a string.
- Table ``CREATE TABLE`` definitions carry no FK/REFERENCES clause in
  this app's rendered DDL, same as legacy -- only inbound edges show up
  for tables.
- Predicate priority mirrors ``core.sql_dependency_extraction`` exactly:
  a mentioned known object typed ``procedure``/``function`` is always
  ``"calls"`` (regardless of which syntax it was mentioned through),
  else ``"writes_to"`` if it's an ``INSERT``/``UPDATE``/``MERGE``/
  ``TRUNCATE TABLE`` target, else ``"references"``.
"""

from __future__ import annotations

from typing import Optional

from core.sql_parse_utils import exp, parse_tsql, qualified_table_name, resolve_known_object


def _write_target_tables(node) -> list:
    """The target ``Table`` node(s) of an INSERT/UPDATE/MERGE/TRUNCATE
    TABLE statement."""
    if isinstance(node, exp.TruncateTable):
        return [t for t in node.args.get("expressions") or [] if isinstance(t, exp.Table)]
    target = node.this
    if isinstance(target, exp.Table):
        return [target]
    # UPDATE/MERGE targets are sometimes wrapped (e.g. an aliased Table) --
    # look one level down for the underlying Table rather than guessing
    # every wrapper shape.
    if target is not None:
        return [t for t in target.find_all(exp.Table)][:1]
    return []


def _qualified_call_name(anon: "exp.Anonymous") -> Optional[str]:
    """``schema.name`` for a scalar function call node if it's qualified
    (``dbo.fn_X(...)`` parses as ``Dot(this=Identifier(dbo),
    expression=Anonymous(this=fn_X))``), else the bare function name."""
    fn_name = anon.this if isinstance(anon.this, str) else str(anon.this)
    parent = anon.parent
    if isinstance(parent, exp.Dot) and isinstance(parent.this, exp.Identifier):
        return f"{parent.this.this}.{fn_name}"
    return fn_name


def find_references(
    definition: str,
    self_key: str,
    known_objects: dict[str, tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """
    Find references to other known objects inside ``definition`` using a
    real T-SQL AST instead of regex. Same contract as
    ``core.sql_dependency_extraction.find_references``: ``self_key`` is
    the lowercased ``"schema.name"`` of the object being scanned
    (excluded from its own results); ``known_objects`` maps lowercased
    ``"schema.name"`` to ``(canonical_name, object_type)``. Returns
    ``(subject, predicate, object)`` triples, predicate one of
    ``"calls"``/``"writes_to"``/``"references"``.

    Returns ``[]`` on anything that doesn't parse cleanly, or if
    ``self_key`` isn't itself a known object -- same "silence over noise"
    posture as the legacy module, never raises on unparseable input.
    """
    self_entry = known_objects.get(self_key)
    if self_entry is None:
        return []
    self_name, _self_type = self_entry

    trees = parse_tsql(definition)
    if not trees:
        return []

    mentioned: set[str] = set()
    write_targets: set[str] = set()

    for tree in trees:
        for table in tree.find_all(exp.Table):
            key = resolve_known_object(qualified_table_name(table), known_objects)
            if key:
                mentioned.add(key)

        for anon in tree.find_all(exp.Anonymous):
            key = resolve_known_object(_qualified_call_name(anon), known_objects)
            if key:
                mentioned.add(key)

        for node in tree.find_all((exp.Insert, exp.Update, exp.Merge, exp.TruncateTable)):
            for table in _write_target_tables(node):
                key = resolve_known_object(qualified_table_name(table), known_objects)
                if key:
                    write_targets.add(key)
                    mentioned.add(key)

    mentioned.discard(self_key)
    write_targets.discard(self_key)

    edges: list[tuple[str, str, str]] = []
    for key in mentioned:
        canonical_name, object_type = known_objects[key]
        if object_type in ("procedure", "function"):
            predicate = "calls"
        elif key in write_targets:
            predicate = "writes_to"
        else:
            predicate = "references"
        edges.append((self_name, predicate, canonical_name))

    return edges
