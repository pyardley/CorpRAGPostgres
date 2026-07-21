"""
Corrective retrieval — admit when nothing matches (README "Possible
enhancements" — Corrective retrieval).

`core.rag_chain.answer_question` already skips the LLM call when
`core.retriever.retrieve` returns zero hits. :func:`is_low_confidence`
extends that same judgment call to the case where retrieval *did* return
hits, but none of them are a good match for the question — Self-RAG /
Corrective RAG (CRAG) call this a "grading" step, and skipping generation
here trades a plausible-sounding hallucination for an honest "I don't
know."

Only meaningful post-rerank: `core.reranker.rerank` writes a
sigmoid-calibrated `[0, 1]` cross-encoder relevance score into `hit.score`
and sorts hits descending by it, so `hits[0].score` is the strongest
available evidence and comparable to a fixed threshold across queries.
Raw cosine similarity and RRF fusion scores aren't on that fixed scale, so
this no-ops unless `settings.RERANK_ENABLED` is also True — a
retrieval-quality knob, not an authorization boundary.
"""

from __future__ import annotations

from app.config import settings
from core.retriever import RetrievedChunk


def is_low_confidence(hits: list[RetrievedChunk]) -> bool:
    """
    True when `hits` exist but the best (post-rerank) one is a weak match.

    No-ops (returns False) when `settings.CORRECTIVE_RETRIEVAL_ENABLED` or
    `settings.RERANK_ENABLED` is False, or `hits` is empty (the caller's
    existing empty-hits branch already handles that case).
    """
    if not settings.CORRECTIVE_RETRIEVAL_ENABLED or not settings.RERANK_ENABLED or not hits:
        return False
    return hits[0].score < settings.CORRECTIVE_RETRIEVAL_SCORE_THRESHOLD
