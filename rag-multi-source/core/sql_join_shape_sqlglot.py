"""
sqlglot-AST-based alternative to ``core.sql_join_shape``.

Signature-compatible drop-in for ``join_shape_findings`` -- selected in
place of the ``sqlparse``-token-walking version whenever
``settings.SQL_IMPACT_ANALYSIS_ENGINE == "sqlglot"`` (see
``core.sql_impact_engine``, the single dispatcher every call site
actually imports from). Reuses ``core.trace_completeness``'s
``is_tracing_question``/``sql_anchor``/``candidate_names`` unchanged --
that gating/noise-control layer isn't a SQL-parsing concern.

Where the legacy module hand-walks a flat, non-flattened ``sqlparse``
token stream and treats ``Where``/``Parenthesis`` groups as opaque
boundaries (a hand-rolled approximation of scoping), a real AST gives
this for free: each ``sqlglot`` ``Select`` node's own ``joins`` only ever
contains joins syntactically written in *that* select's own ``FROM``
clause -- a nested subquery (e.g. inside a ``WHERE EXISTS (...)``) is a
structurally separate ``Select`` node found independently by the same
``find_all(exp.Select)`` walk, so there's no risk of it leaking into an
outer select's own chain without any special-casing needed.

Keeps the exact same INNER/LEFT finding wording as the legacy engine for
a true parity comparison, but additionally surfaces RIGHT/FULL JOIN
findings (two new templates) -- free under a real AST (``Join.side``
already distinguishes them), and the point of this engine switch is a
genuine capability comparison, not an artificially narrowed parity
shim. CTEs stay silent in both engines, same as legacy -- not extended
opportunistically (see ``plans/DataflowShapeAwarenessForSQLImpactPlan.md``
for why the feature itself never reached for CTE coverage).
"""

from __future__ import annotations

from typing import Optional

from core.retriever import RetrievedChunk
from core.sql_parse_utils import exp, parse_tsql, qualified_table_name
from core.trace_completeness import candidate_names, is_tracing_question, sql_anchor

_MAX_FINDINGS = 5

_FINDING_SIDES = {"": "INNER", "LEFT": "LEFT", "RIGHT": "RIGHT", "FULL": "FULL"}


def _cte_tainted(select: "exp.Select") -> bool:
    """True if `select` is a CTE's own definition body, is itself a
    standalone `;WITH x AS (...) SELECT ...`, or is the SELECT half of an
    `;WITH x AS (...) INSERT INTO ... SELECT ...` -- skip the whole
    statement in all three cases, matching the legacy engine's
    unconditional CTE non-goal."""
    if select.find_ancestor(exp.CTE) is not None:
        return True
    if select.args.get("with_"):
        return True
    parent = select.parent
    if isinstance(parent, exp.Insert) and parent.args.get("with_"):
        return True
    return False


def _select_target(select: "exp.Select", anchor_name: Optional[str]) -> str:
    into = select.args.get("into")
    if into is not None and isinstance(into.this, exp.Table):
        return qualified_table_name(into.this)
    parent = select.parent
    if isinstance(parent, exp.Insert) and isinstance(parent.this, exp.Table):
        return qualified_table_name(parent.this)
    return anchor_name or "this object"


def _finding_sentence(kind: str, prior: str, joined: str, target: str) -> Optional[str]:
    if kind == "INNER":
        return (
            f"`{joined}` is INNER-joined to `{prior}` when building `{target}` "
            "— a row in either table with no matching join key is silently "
            "excluded (dropped, not nulled) from what continues downstream"
        )
    if kind == "LEFT":
        return (
            f"`{prior}` LEFT-joins `{joined}` when building `{target}` — "
            f"every `{prior}` row is kept even without a match, but a "
            f"`{joined}` row with no matching key in `{prior}` is silently "
            "excluded from the result"
        )
    if kind == "RIGHT":
        return (
            f"`{joined}` RIGHT-joins `{prior}` when building `{target}` — "
            f"every `{joined}` row is kept even without a match, but a "
            f"`{prior}` row with no matching key in `{joined}` is silently "
            "excluded from the result"
        )
    if kind == "FULL":
        return (
            f"`{joined}` is FULL-joined to `{prior}` when building `{target}` "
            "— a row from either side with no match on the other is kept "
            "but arrives with the other side's columns NULLed, not "
            "excluded — a downstream filter without NULL-handling can "
            "still drop it later"
        )
    return None


def join_shape_findings(question: str, hits: list[RetrievedChunk]) -> list[str]:
    """JOIN-direction row-inclusion-risk findings for the anchor SQL hit.
    See module docstring for the engine's relationship to
    ``core.sql_join_shape.join_shape_findings``."""
    if not is_tracing_question(question):
        return []
    anchor = sql_anchor(hits)
    if anchor is None:
        return []
    candidates = candidate_names(hits)
    if not candidates:
        return []

    trees = parse_tsql(anchor.text or "")
    if not trees:
        return []

    anchor_name = (anchor.metadata or {}).get("object_name")
    findings: list[str] = []

    for tree in trees:
        for select in tree.find_all(exp.Select):
            if len(findings) >= _MAX_FINDINGS:
                return findings[:_MAX_FINDINGS]

            joins = select.args.get("joins") or []
            if not joins:
                continue
            if _cte_tainted(select):
                continue

            from_ = select.args.get("from_")
            if from_ is None or not isinstance(from_.this, exp.Table):
                # Comma-join (multiple tables) or a derived-table/subquery
                # driving source -- no finding, same "silence over noise"
                # posture the legacy engine applies to these shapes.
                continue

            chain = [qualified_table_name(from_.this)]
            kinds: list[Optional[str]] = []
            for j in joins:
                if not isinstance(j.this, exp.Table):
                    # Derived-table/subquery/table-valued-function join
                    # target -- stop extending, keep what's already parsed.
                    break
                kind = _FINDING_SIDES.get((j.side or "").upper())
                chain.append(qualified_table_name(j.this))
                kinds.append(kind)

            if len(chain) < 2:
                continue
            if not (set(chain) & candidates):
                continue

            target = _select_target(select, anchor_name)
            for idx, kind in enumerate(kinds, start=1):
                if kind is None:
                    continue
                sentence = _finding_sentence(kind, chain[idx - 1], chain[idx], target)
                if sentence:
                    findings.append(sentence)
                if len(findings) >= _MAX_FINDINGS:
                    break

    return findings[:_MAX_FINDINGS]
