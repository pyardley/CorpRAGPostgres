"""
Static SQL dependency extraction.

Scans a SQL Server object's definition text for references to other
known objects in the same database, producing (subject, predicate,
object) edges for the entity graph (see `core.entity_graph`).

This is a word-boundary substring search against a pre-built catalog of
known object names, not a real SQL parser — deliberately so. A real
dependency-tracking parser for arbitrary T-SQL is a large undertaking,
and SQL Server's own catalog (`sys.sql_expression_dependencies` /
`sys.dm_sql_referenced_entities`) already solves this exactly, whenever
a live connection is available. This module exists for two cases a
live DMV query can't cover: building the graph once at ingestion time
so retrieval and the entity-graph MCP tool can use it without a live
connection, and (reused verbatim) as a fallback for what those DMVs
themselves miss — dynamic SQL and some non-schema-bound references.

Known limitations, by design:
- String literals aren't excluded (only SQL comments are stripped), so
  a match inside a quoted string is a false positive.
- Can't see dynamic SQL (`EXEC(@sql)`, `sp_executesql`) or
  cross-database references.
- Table `CREATE TABLE` definitions carry no FK/REFERENCES clause in
  this app's rendered DDL (see `sql_ingestor.py::_iter_tables`), so
  scanning a table's own "definition" never yields outgoing edges —
  only inbound edges (other objects that reference the table) show up.
"""

from __future__ import annotations

import re

try:
    import sqlparse
except ImportError:  # pragma: no cover - sqlparse is an existing project dependency
    sqlparse = None

_WRITE_TARGET_RE = re.compile(
    r"\b(?:INSERT\s+INTO|UPDATE|MERGE(?:\s+INTO)?|TRUNCATE\s+TABLE)\s+"
    r"(\[?\w+\]?(?:\.\[?\w+\]?)?)",
    re.IGNORECASE,
)


def _strip_comments(definition: str) -> str:
    if sqlparse is None:
        return definition
    return sqlparse.format(definition, strip_comments=True)


def _written_targets(text: str) -> set[str]:
    """Lowercased schema.name (or bare name) tokens that are the target
    of an INSERT/UPDATE/MERGE/TRUNCATE somewhere in `text`."""
    targets: set[str] = set()
    for match in _WRITE_TARGET_RE.finditer(text):
        raw = match.group(1).replace("[", "").replace("]", "")
        targets.add(raw.lower())
    return targets


def find_references(
    definition: str,
    self_key: str,
    known_objects: dict[str, tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """
    Find references to other known objects inside `definition`.

    `self_key` is the lowercased "schema.name" of the object being
    scanned — excluded from its own results. `known_objects` maps
    lowercased "schema.name" to (canonical_name, object_type). A bare
    (non-schema-qualified) match is only attempted for objects in the
    `dbo` schema — the overwhelmingly common unqualified-reference case
    — since matching every schema's bare names risks cross-schema false
    positives for short/generic names.

    Returns (subject, predicate, object) triples where `subject` is the
    scanned object's own canonical name and `object` is the matched
    object's canonical name. Predicate is `"calls"` for procedures and
    functions, `"writes_to"` for a table/view that's the target of an
    INSERT/UPDATE/MERGE/TRUNCATE in `definition`, else `"references"`.
    """
    self_entry = known_objects.get(self_key)
    if self_entry is None:
        return []
    self_name, _self_type = self_entry

    text = _strip_comments(definition)
    write_targets = _written_targets(text)

    edges: list[tuple[str, str, str]] = []
    for key, (canonical_name, object_type) in known_objects.items():
        if key == self_key:
            continue

        schema, _, bare_name = key.partition(".")
        patterns = [key]
        if schema == "dbo":
            patterns.append(bare_name)

        if not any(
            re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text, re.IGNORECASE)
            for p in patterns
        ):
            continue

        if object_type in ("procedure", "function"):
            predicate = "calls"
        elif key in write_targets or bare_name in write_targets:
            predicate = "writes_to"
        else:
            predicate = "references"

        edges.append((self_name, predicate, canonical_name))

    return edges
