"""
Deterministic completeness check for lineage / impact-analysis answers.

Multi-hop tracing questions ("how is X derived", "what breaks if I change
Y") are exactly the case where an LLM can see every intermediate stage in
its context and still silently skip one when writing the final answer —
that's a generation failure, not a retrieval failure, so it can't be fixed
by retrieving more. This module catches it after the fact: it collects the
SQL object names and temp-table tokens that were actually shown to the
model, then flags any that never show up in the answer text.

No extra LLM call — plain substring matching, so it's cheap enough to run
on every turn.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from core.retriever import RetrievedChunk

_TEMP_TABLE_RE = re.compile(r"#\w{3,}")
_CALLABLE_OBJECT_TYPES = {"procedure", "function"}

# Shared with the response-cache bypass (app/chat.py) so both the
# "should we even run the completeness check" gate and the "should we skip
# the semantic cache for this turn" gate agree on what counts as a
# lineage/impact-analysis question. Deliberately conservative — false
# negatives here just mean the caveat/cache-bypass doesn't fire, which is
# safe; false positives would add noise to unrelated SQL questions.
_TRACING_KEYWORDS_RE = re.compile(
    r"\b(trace|traced|tracing|derive|derived|derivation|lineage|"
    r"depends? on|dependency|dependencies|impact|breaks? if|"
    r"affects?|affected by|where does .* come from|what references|"
    r"downstream|upstream)\b",
    re.IGNORECASE,
)


def is_tracing_question(question: str) -> bool:
    """True if `question` looks like a lineage/impact-analysis question."""
    return bool(_TRACING_KEYWORDS_RE.search(question or ""))


def sql_anchor(hits: list[RetrievedChunk]) -> Optional[RetrievedChunk]:
    """
    The single highest-scored SQL hit — `hits` arrives pre-sorted by
    score, descending, from `core.retriever.retrieve` regardless of
    whether reranking ran — excluding the synthetic "live-mcp" citation
    (no underlying object body to reference against). Treated throughout
    this module as "the object actually being explained"; every other
    candidate-name check below only counts a name if THIS hit's own text
    literally references it, precisely to avoid the false-positive noise
    a real run once produced: an unrelated, only-loosely-retrieved
    sibling procedure's own internal temp tables and its own name both
    got flagged as "missing" from an answer that had correctly ignored
    them.
    """
    sql_hits = [
        h
        for h in hits
        if h.source == "sql" and (h.metadata or {}).get("object_type") != "live-mcp"
    ]
    return sql_hits[0] if sql_hits else None


def candidate_names(hits: list[RetrievedChunk]) -> set[str]:
    """
    Candidates worth flagging if absent from the answer — deliberately
    narrow, favoring silence over noise. See `sql_anchor` for why only
    the anchor's own text is searched. Its own internal temp tables are
    always fair candidates; another retrieved object's name only counts
    if the anchor's own text literally references it. The anchor's own
    name is never a candidate against itself; not restating "I am now
    explaining X" isn't an omission.
    """
    anchor = sql_anchor(hits)
    if anchor is None:
        return set()

    anchor_text = anchor.text or ""
    anchor_name = (anchor.metadata or {}).get("object_name")

    names: set[str] = set(_TEMP_TABLE_RE.findall(anchor_text))
    for hit in hits:
        if hit.source != "sql" or (hit.metadata or {}).get("object_type") == "live-mcp":
            continue
        object_name = (hit.metadata or {}).get("object_name")
        if not object_name or object_name == anchor_name:
            continue
        if object_name in anchor_text:
            names.add(object_name)
    return names


def candidate_callable_names(hits: list[RetrievedChunk]) -> set[str]:
    """
    Same anchor-containment pattern as `candidate_names`, narrowed to
    hits whose `object_type` is `"procedure"`/`"function"` and dropping
    the temp-table half — this answers "should `sql_object_definition`
    have been called on this name" (the MANDATORY rule in
    `core.mcp_chain.HYBRID_SYSTEM_PROMPT`), not "was this name mentioned
    in the answer text" (which `candidate_names` already covers).
    """
    anchor = sql_anchor(hits)
    if anchor is None:
        return set()

    anchor_text = anchor.text or ""
    anchor_name = (anchor.metadata or {}).get("object_name")

    names: set[str] = set()
    for hit in hits:
        if hit.source != "sql" or (hit.metadata or {}).get("object_type") == "live-mcp":
            continue
        md = hit.metadata or {}
        object_name = md.get("object_name")
        if not object_name or object_name == anchor_name:
            continue
        if md.get("object_type") not in _CALLABLE_OBJECT_TYPES:
            continue
        if object_name in anchor_text:
            names.add(object_name)
    return names


def _bare_object_name(name: str) -> str:
    """
    Last dot-segment of a possibly schema/db-qualified object name,
    brackets stripped, lowercased. `object_name` metadata is always
    stored as `"schema.object_name"` (see `sql_ingestor.py`), but a
    tool-call argument might be bare (`"fn_NetLineAmount"`) or fully
    qualified (`"MyDb.dbo.fn_NetLineAmount"`) — comparing on the bare
    tail lets either form match.
    """
    return name.strip().strip("[]").split(".")[-1].strip("[]").lower()


def missing_definition_calls(
    hits: list[RetrievedChunk], called_object_names: Iterable[str]
) -> list[str]:
    """
    Names from `candidate_callable_names(hits)` not covered (by
    bare-name match) by `called_object_names` — the `object_name`
    arguments actually passed to `sql_object_definition` this turn.
    Used by `core.mcp_chain` to enforce the MANDATORY
    `sql_object_definition` rule with a forced retry rather than just a
    footer note, since prompt instruction alone wasn't reliably
    followed in live testing.
    """
    called_bare = {_bare_object_name(n) for n in called_object_names}
    return sorted(
        name
        for name in candidate_callable_names(hits)
        if _bare_object_name(name) not in called_bare
    )


def check_trace_completeness(
    question: str, hits: list[RetrievedChunk], answer: str
) -> list[str]:
    """
    Return the names of SQL objects/temp tables present in the shown
    context but never mentioned in `answer`. Empty list means either the
    question doesn't look like a lineage/impact-analysis question, no SQL
    context was involved, or everything shown was accounted for.

    Gated on `is_tracing_question` — checking this on every SQL-touching
    turn would flag plenty of harmless omissions (e.g. a simple "what does
    this proc do" answer that doesn't need to enumerate every temp table).
    """
    if not is_tracing_question(question):
        return []

    candidates = candidate_names(hits)
    if not candidates:
        return []

    answer_lower = answer.lower()
    missing = sorted(
        name for name in candidates if name.lower() not in answer_lower
    )
    return missing
