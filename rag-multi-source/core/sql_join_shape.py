"""
Dataflow-shape awareness: JOIN-direction row-inclusion risk detection for
SQL impact-analysis / tracing answers.

`core.trace_completeness` catches an LLM silently *skipping* an object it
was shown. This module catches a different, structurally distinct gap: an
object's own JOINs can silently *drop rows* depending on join direction
(INNER excludes either side on no match; LEFT excludes only the
right-hand side) — a genuine impact-analysis fact ("what could make a row
disappear from this report") that name-presence checking can never surface,
because it depends on *how* two already-mentioned objects are joined, not
*whether* they're mentioned.

Not a real SQL parser — `sqlparse`'s grouped (non-flattened) token tree is
walked directly, treating `Where` / `Parenthesis` / any other group as an
opaque single token whenever it isn't specifically recognised. This is
deliberate: nested `FROM`/`WHERE` keywords inside a subquery (e.g.
`WHERE EXISTS (SELECT ... FROM ... WHERE ...)`) would otherwise be
indistinguishable from the outer chain's own tokens to a naive regex or a
flattened token scan. Anything not confidently parseable — CTEs, window
functions, comma-joins, `CROSS`/`OUTER APPLY`, derived-table subqueries —
produces no finding rather than a guessed one, same philosophy
`core.trace_completeness` already applies.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    import sqlparse
    from sqlparse import sql as S
    from sqlparse import tokens as T
except ImportError:  # pragma: no cover - sqlparse is an existing project dependency
    sqlparse = None

from core.retriever import RetrievedChunk
from core.trace_completeness import candidate_names, is_tracing_question, sql_anchor

_MAX_FINDINGS = 5

# JOIN-family keywords that produce a finding, mapped to the finding
# category. `JOIN` bare is INNER per SQL semantics.
_FINDING_JOIN_TYPES = {
    "JOIN": "INNER",
    "INNER JOIN": "INNER",
    "LEFT JOIN": "LEFT",
    "LEFT OUTER JOIN": "LEFT",
}

# Recognized (parsed without confusing the scanner) but deliberately
# produce no finding — zero fixture coverage for either, and guessing
# wording for an unvalidated case would violate "silence over noise" more
# than it would help. The chain still continues past these.
_SILENT_JOIN_KEYWORDS = {
    "RIGHT JOIN",
    "RIGHT OUTER JOIN",
    "FULL JOIN",
    "FULL OUTER JOIN",
}

# Terminator keywords — chain-walk stops here (whatever was already
# parsed is kept), same as hitting a `Where` group or `;`.
_TERMINATOR_KEYWORDS = {"GROUP BY", "ORDER BY", "HAVING"}

# `CROSS JOIN` tokenizes as one combined Keyword; `CROSS APPLY` / `OUTER
# APPLY` don't (sqlparse doesn't recognize `APPLY` as a keyword at all —
# it tokenizes as a bare Name), so those need an explicit two-token peek.
_STOP_JOIN_KEYWORDS = {"CROSS JOIN"}
_APPLY_PREFIXES = {"CROSS", "OUTER"}

_TEMP_TABLE_RE = re.compile(r"#\w{3,}")


def _norm_kw(value: str) -> str:
    return " ".join(value.split()).upper()


def _is_keyword(tok, *values: str) -> bool:
    if tok.ttype is None or tok.ttype not in T.Keyword:
        return False
    return _norm_kw(tok.value) in values


def _qualified_name(tok) -> str:
    """
    `schema.name` (or bare `name` / `#temp`) with any alias stripped,
    reconstructed from `Identifier.get_real_name()` +
    `get_parent_name()` since `get_real_name()` alone drops the schema
    prefix. Falls back to the raw token text for anything that isn't an
    `Identifier` (shouldn't happen at the call sites below, which only
    ever pass a token already confirmed to be one).
    """
    if isinstance(tok, S.Identifier):
        real = tok.get_real_name()
        if real is None:
            return str(tok).strip()
        parent = tok.get_parent_name()
        return f"{parent}.{real}" if parent else real
    return str(tok).strip()


def _find_begin_groups(token_list) -> list:
    """Recursively find every `sqlparse.sql.Begin` group anywhere in the
    tree — handles nested `IF ... BEGIN ... END`, and the fact that a
    T-SQL `CREATE PROCEDURE x AS BEGIN ... END` body ends up with its
    `Begin` group nested a level or two down (inside an `Identifier`
    sqlparse's grouper built for the `x AS ...` span), not at the
    statement's own top level."""
    found: list = []
    for tok in token_list.tokens:
        if isinstance(tok, S.Begin):
            found.append(tok)
        if tok.is_group:
            found.extend(_find_begin_groups(tok))
    return found


def _regions(statement) -> list[list]:
    """Flat, safely-scannable token regions: every `Begin` group's own
    top-level tokens, plus the statement's own top-level tokens (covers
    bare view/query text with no `BEGIN`/`END` at all)."""
    regions = [list(statement.tokens)]
    for begin in _find_begin_groups(statement):
        regions.append(list(begin.tokens))
    return regions


def _walk_chain(tokens: list, start: int) -> tuple[list[tuple[str, Optional[str]]], int]:
    """
    `tokens[start]` is the driving table's `Identifier`, immediately
    after a `FROM` keyword already confirmed valid by the caller.

    Returns `(chain, next_index)` — `chain` is `[(driving_name, None),
    (joined_name, join_type), ...]`, `join_type` one of `"INNER"` /
    `"LEFT"` / `None` (silent-but-recognized RIGHT/FULL). `next_index`
    is where the caller's outer scan should resume.
    """
    chain: list[tuple[str, Optional[str]]] = [(_qualified_name(tokens[start]), None)]
    i = start + 1
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if isinstance(tok, S.Where) or _is_keyword(tok, *_TERMINATOR_KEYWORDS):
            i += 1
            break
        if tok.ttype is T.Punctuation and tok.value == ";":
            i += 1
            break
        if _is_keyword(tok, *_STOP_JOIN_KEYWORDS):
            i += 1
            break
        if (
            tok.ttype is T.Keyword
            and _norm_kw(tok.value) in _APPLY_PREFIXES
            and i + 1 < n
            and tokens[i + 1].ttype is T.Name
            and tokens[i + 1].value.upper() == "APPLY"
        ):
            i += 1
            break

        join_kind = None
        if _is_keyword(tok, *_FINDING_JOIN_TYPES):
            join_kind = _FINDING_JOIN_TYPES[_norm_kw(tok.value)]
        elif _is_keyword(tok, *_SILENT_JOIN_KEYWORDS):
            join_kind = None  # recognized, but intentionally no finding
        else:
            i += 1
            continue

        # `tok` is a recognized JOIN-family keyword — the next
        # non-whitespace token must be the joined table's Identifier.
        j = i + 1
        while j < n and str(tokens[j]).strip() == "":
            j += 1
        if j >= n or not isinstance(tokens[j], S.Identifier):
            # Malformed/unsupported join target (derived table, function,
            # comma list) — stop extending, keep what's already parsed.
            i = j
            break
        if _is_keyword(tok, *_FINDING_JOIN_TYPES):
            chain.append((_qualified_name(tokens[j]), join_kind))
        else:
            chain.append((_qualified_name(tokens[j]), None))
        i = j + 1

    return chain, i


def _statement_target(current_target: Optional[str], anchor_name: Optional[str]) -> str:
    return current_target or anchor_name or "this object"


def _findings_for_region(
    tokens: list, candidates: set[str], anchor_name: Optional[str]
) -> list[str]:
    findings: list[str] = []
    cte_seen = False
    current_target: Optional[str] = None
    i = 0
    n = len(tokens)

    while i < n and len(findings) < _MAX_FINDINGS:
        tok = tokens[i]

        if tok.ttype is T.Punctuation and tok.value == ";":
            cte_seen = False
            current_target = None
            i += 1
            continue

        if tok.ttype is not None and tok.ttype in T.Keyword.CTE:
            cte_seen = True
            i += 1
            continue

        if _is_keyword(tok, "INTO"):
            j = i + 1
            while j < n and str(tokens[j]).strip() == "":
                j += 1
            if j < n and isinstance(tokens[j], S.Identifier):
                current_target = _qualified_name(tokens[j])
                i = j + 1
                continue
            i += 1
            continue

        if _is_keyword(tok, "FROM"):
            j = i + 1
            while j < n and str(tokens[j]).strip() == "":
                j += 1
            if j >= n or not isinstance(tokens[j], S.Identifier):
                # IdentifierList (comma-join), Parenthesis (derived
                # table), Function (table-valued function), or nothing —
                # abort this chain with no finding.
                i = j + 1
                continue

            chain, next_i = _walk_chain(tokens, j)
            i = next_i

            if not cte_seen and len(chain) > 1:
                names = {name for name, _ in chain}
                if names & candidates:
                    target = _statement_target(current_target, anchor_name)
                    for idx in range(1, len(chain)):
                        joined, kind = chain[idx]
                        prior, _ = chain[idx - 1]
                        if kind == "INNER":
                            findings.append(
                                f"`{joined}` is INNER-joined to `{prior}` when "
                                f"building `{target}` — a row in either table "
                                "with no matching join key is silently "
                                "excluded (dropped, not nulled) from what "
                                "continues downstream"
                            )
                        elif kind == "LEFT":
                            findings.append(
                                f"`{prior}` LEFT-joins `{joined}` when "
                                f"building `{target}` — every `{prior}` row "
                                f"is kept even without a match, but a "
                                f"`{joined}` row with no matching key in "
                                f"`{prior}` is silently excluded from the "
                                "result"
                            )
                        if len(findings) >= _MAX_FINDINGS:
                            break
            continue

        i += 1

    return findings


def join_shape_findings(question: str, hits: list[RetrievedChunk]) -> list[str]:
    """
    JOIN-direction row-inclusion-risk findings for the anchor SQL hit,
    restricted to chains that touch a name already in play this turn
    (`core.trace_completeness.candidate_names`) — same anchor-scoping and
    "silence over noise" posture as `check_trace_completeness`. Gated on
    `is_tracing_question` exactly like that function. Empty list on
    anything not confidently parseable (CTEs, comma-joins, window
    functions, derived tables, `CROSS`/`OUTER APPLY`, missing `sqlparse`)
    rather than a guessed finding.
    """
    if sqlparse is None:
        return []
    if not is_tracing_question(question):
        return []

    anchor = sql_anchor(hits)
    if anchor is None:
        return []

    candidates = candidate_names(hits)
    if not candidates:
        return []

    text = sqlparse.format(anchor.text or "", strip_comments=True)
    parsed = sqlparse.parse(text)
    if not parsed:
        return []

    anchor_name = (anchor.metadata or {}).get("object_name")

    findings: list[str] = []
    for region in _regions(parsed[0]):
        if len(findings) >= _MAX_FINDINGS:
            break
        findings.extend(_findings_for_region(region, candidates, anchor_name))

    return findings[:_MAX_FINDINGS]
