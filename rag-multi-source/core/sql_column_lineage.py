"""
Column-level lineage tracing -- ``sqlglot``-only, no legacy equivalent.

Neither ``core.sql_dependency_extraction`` (object-level edges) nor
``core.sql_join_shape`` (JOIN-direction row-inclusion risk) can answer
"which base-table column does this report column actually derive from" --
both operate at the table/object level. This module answers that
question by walking a real T-SQL AST across a stored procedure's chained
``SELECT ... INTO #TempX`` stages.

``sqlglot.lineage.lineage()`` exists but is designed for lineage *within
one query against a known schema* -- it isn't built for chaining through
several separate statements where each ``#TempX`` acts like an implicit
intermediate view with an unknown schema. Rather than fight that
mismatch, this module hand-walks each output expression's leaf
``Column`` nodes directly, resolving each leaf's table alias via that
statement's own FROM/JOIN alias map (which ``sqlglot`` gives cleanly
regardless of schema knowledge), and recurses into an earlier stage
whenever a leaf's table is itself a temp table (or CTE) this same
procedure produced. This is the "concrete fallback" the implementation
plan called out up front -- it loses ``lineage()``'s automatic
CASE/window-function branch attribution (an expression's leaf columns
are all reported together, not attributed to a specific CASE branch),
but sidesteps the schema-dict-accuracy dependency entirely.

Gated on ``is_tracing_question`` + ``sql_anchor`` like
``core.sql_join_shape``, but deliberately *not* ``candidate_names`` --
lineage only ever traces the anchor's own internal derivation chain, so
there's no cross-object noise to filter the way join-shape has.

Silence over noise throughout: any stage/column that doesn't resolve
cleanly (ambiguous unqualified reference, unsupported expression shape,
a stage that doesn't parse) is skipped rather than guessed.
"""

from __future__ import annotations

import re
from typing import Optional

from core.retriever import RetrievedChunk
from core.sql_parse_utils import exp, parse_tsql, qualified_table_name
from core.trace_completeness import is_tracing_question, sql_anchor

_MAX_LINEAGE_FINDINGS = 3
_MAX_LINEAGE_DEPTH = 6
_EXPR_SUMMARY_MAX_CHARS = 70


def _alias_table_map(select: "exp.Select") -> dict[str, str]:
    """table-alias (or bare table name if unaliased), lowercased -> this
    select's own qualified table name, for its direct FROM + JOINs only
    -- not recursing into subqueries, giving the same natural opacity
    core.sql_join_shape_sqlglot relies on."""
    mapping: dict[str, str] = {}
    tables = []
    from_ = select.args.get("from_")
    if from_ is not None and isinstance(from_.this, exp.Table):
        tables.append(from_.this)
    for j in select.args.get("joins") or []:
        if isinstance(j.this, exp.Table):
            tables.append(j.this)
    for t in tables:
        alias = t.alias or t.name
        if alias:
            mapping[alias.lower()] = qualified_table_name(t)
    return mapping


def _find_output_expr(select: "exp.Select", column_name: str):
    """The SELECT-list expression producing `column_name` -- the
    underlying expression with any alias stripped, or the bare Column
    itself for an unaliased passthrough."""
    lowered = column_name.lower()
    for e in select.expressions:
        if isinstance(e, exp.Alias):
            if e.alias and e.alias.lower() == lowered:
                return e.this
        elif isinstance(e, exp.Column):
            if e.name.lower() == lowered:
                return e
    return None


def _output_column_names(select: "exp.Select") -> list[str]:
    names = []
    for e in select.expressions:
        if isinstance(e, exp.Alias) and e.alias:
            names.append(e.alias)
        elif isinstance(e, exp.Column):
            names.append(e.name)
    return names


def _resolve_leaf(column: "exp.Column", alias_map: dict[str, str]) -> Optional[tuple[str, str]]:
    """(qualified_table, column_name) for one leaf Column reference, or
    None if it can't be confidently resolved. An unqualified column only
    resolves when the statement has exactly one table in scope --
    ambiguous otherwise, so it's silently skipped rather than guessed."""
    alias = (column.table or "").lower()
    if alias:
        qname = alias_map.get(alias)
        return (qname, column.name) if qname else None
    if len(alias_map) == 1:
        (qname,) = alias_map.values()
        return qname, column.name
    return None


def _stage_map(trees: list) -> dict[str, "exp.Select"]:
    """Every ``SELECT ... INTO #TempX`` stage and every named CTE this
    procedure defines, keyed by lowercased qualified name -> the Select
    node that produces it."""
    stages: dict[str, exp.Select] = {}
    for tree in trees:
        for select in tree.find_all(exp.Select):
            into = select.args.get("into")
            if into is not None and isinstance(into.this, exp.Table):
                stages[qualified_table_name(into.this).lower()] = select
        for cte in tree.find_all(exp.CTE):
            if cte.alias and isinstance(cte.this, exp.Select):
                stages[cte.alias.lower()] = cte.this
    return stages


def _final_stage(
    trees: list, anchor_name: Optional[str]
) -> tuple[Optional[str], Optional["exp.Select"]]:
    """(target_label, select_node) for the procedure's own final result --
    the last INSERT's consuming SELECT if there is one, else the last
    top-level Select with no INTO and no CTE ancestor."""
    last_insert = None
    for tree in trees:
        for ins in tree.find_all(exp.Insert):
            last_insert = ins
    if last_insert is not None and isinstance(last_insert.expression, exp.Select):
        target = last_insert.this
        if isinstance(target, exp.Schema):
            # INSERT INTO x (col1, col2) wraps the target Table in a
            # Schema node alongside the explicit column list.
            target = target.this
        label = qualified_table_name(target) if isinstance(target, exp.Table) else None
        return (label or anchor_name or "this object"), last_insert.expression

    candidates = []
    for tree in trees:
        for sel in tree.find_all(exp.Select):
            if sel.args.get("into") is None and sel.find_ancestor(exp.CTE) is None:
                candidates.append(sel)
    if not candidates:
        return None, None
    return (anchor_name or "this object"), candidates[-1]


def _question_target_columns(question: str, target_columns: list[str]) -> list[str]:
    lowered_q = (question or "").lower()
    return [c for c in target_columns if re.search(rf"\b{re.escape(c.lower())}\b", lowered_q)]


def _trace_hops(
    stage_key: str,
    display_name: str,
    column_name: str,
    stages: dict[str, "exp.Select"],
    depth: int,
    visited: frozenset,
) -> Optional[list[str]]:
    """
    Ordered hop-label strings from ``(stage_key, column_name)`` down to
    its base-table leaf/leaves, root-first. ``None`` if it can't be
    resolved anywhere along the way (silent skip, no partial guess).

    ``stage_key`` is lowercased (for ``stages`` dict lookups and the
    ``visited`` cycle guard); ``display_name`` carries the original
    casing through to the rendered label, since ``qualified_table_name``
    already preserves it and lowercasing is purely a lookup-key concern.
    """
    if depth > _MAX_LINEAGE_DEPTH:
        return None
    key = (stage_key, column_name.lower())
    if key in visited:
        return None
    visited = visited | {key}

    select = stages.get(stage_key)
    if select is None:
        # Not a stage this procedure produced -- a base-table leaf.
        return [f"`{display_name}.{column_name}`"]

    expr = _find_output_expr(select, column_name)
    if expr is None:
        return None

    alias_map = _alias_table_map(select)
    columns = list(expr.find_all(exp.Column))
    if not columns:
        return None

    resolved: list[tuple[str, str]] = []
    for c in columns:
        leaf = _resolve_leaf(c, alias_map)
        if leaf:
            resolved.append(leaf)
    if not resolved:
        return None

    # Suppress a redundant "(X)" annotation when the expression is just a
    # bare passthrough of a same-named column (qualified or not) -- there's
    # nothing more informative to show than the label itself already says.
    is_bare_passthrough = isinstance(expr, exp.Column) and expr.name.lower() == column_name.lower()
    if is_bare_passthrough:
        this_label = f"`{display_name}.{column_name}`"
    else:
        summary = expr.sql(dialect="tsql")
        if len(summary) > _EXPR_SUMMARY_MAX_CHARS:
            summary = summary[: _EXPR_SUMMARY_MAX_CHARS - 3] + "..."
        this_label = f"`{display_name}.{column_name}` ({summary})"

    # Continue the chain through the first resolved leaf whose table is
    # itself a known stage; any other leaves in the same expression
    # (e.g. a multi-argument function call against the same table) are
    # reported alongside it rather than each spawning their own hop.
    stage_leaf: Optional[tuple[str, str]] = None
    plain_leafs: list[str] = []
    for qname, colname in resolved:
        if stage_leaf is None and qname.lower() in stages:
            stage_leaf = (qname, colname)
        else:
            plain_leafs.append(f"`{qname}.{colname}`")

    hops = [this_label]
    if stage_leaf is not None:
        deeper = _trace_hops(
            stage_leaf[0].lower(), stage_leaf[0], stage_leaf[1], stages, depth + 1, visited
        )
        if deeper is None:
            return None
        hops.extend(deeper)
    if plain_leafs:
        hops.append(", ".join(sorted(set(plain_leafs))))
    return hops


def column_lineage_findings(question: str, hits: list[RetrievedChunk]) -> list[str]:
    """
    Column-level lineage findings for the anchor SQL hit -- an
    arrow-chain sentence per traced output column, e.g.:
    `` `Report_X.TotalNetAmount` ← `#RFM.TotalNetAmount` (SUM(NetAmount))
    ← `#ActiveCustomerOrders.NetAmount` (dbo.fn_NetLineAmount(...)) ←
    `dbo.OrderLines.Quantity`, `dbo.OrderLines.UnitPrice`,
    `dbo.OrderLines.DiscountPct` ``.

    Gated on ``is_tracing_question`` + a resolvable anchor. Traces the
    column the question names first, if any; else the anchor's first few
    output columns in declared order. Never raises -- any stage/column
    that doesn't parse or resolve cleanly is silently skipped.
    """
    if not is_tracing_question(question):
        return []
    anchor = sql_anchor(hits)
    if anchor is None:
        return []

    trees = parse_tsql(anchor.text or "")
    if not trees:
        return []

    stages = _stage_map(trees)
    anchor_name = (anchor.metadata or {}).get("object_name")
    target_label, final_select = _final_stage(trees, anchor_name)
    if final_select is None or target_label is None:
        return []

    target_columns = _output_column_names(final_select)
    if not target_columns:
        return []

    stages[target_label.lower()] = final_select
    requested = _question_target_columns(question, target_columns)
    columns_to_trace = requested or target_columns[:_MAX_LINEAGE_FINDINGS]

    findings: list[str] = []
    for column in columns_to_trace:
        if len(findings) >= _MAX_LINEAGE_FINDINGS:
            break
        hops = _trace_hops(target_label.lower(), target_label, column, stages, 0, frozenset())
        if not hops or len(hops) < 2:
            continue  # no real derivation chain found -- silent skip
        findings.append(" ← ".join(hops))

    return findings[:_MAX_LINEAGE_FINDINGS]
