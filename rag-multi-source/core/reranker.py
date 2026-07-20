"""
Cross-encoder reranking (README "Possible enhancements" — Reranking with
a cross-encoder).

Vector/RRF scoring never reads a candidate chunk's full text against the
query — it's a bi-encoder similarity computed independently per source.
:func:`rerank` adds a second stage: a cross-encoder jointly encodes
``(query, chunk_text)`` for every candidate and re-scores/re-sorts them,
typically surfacing the genuinely relevant chunk even when it wasn't the
top cosine hit.

Fails OPEN — unlike ``core.live_acl``, this is a ranking quality knob,
not an authorization boundary. Any error (model load failure, missing
``sentence-transformers``/``torch``, OOM) is logged and the input order
is used unchanged.
"""

from __future__ import annotations

from loguru import logger

from app.config import settings
from core.llm import get_reranker
from core.retriever import RetrievedChunk


def rerank(query: str, hits: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """
    Re-score `hits` against `query` with a cross-encoder and return the
    top `top_n`, highest-scoring first.

    No-ops (returns `hits[:top_n]` in the input order) when
    `settings.RERANK_ENABLED` is False, `hits` is empty, or the
    cross-encoder fails for any reason.
    """
    if not settings.RERANK_ENABLED or not hits:
        return hits[:top_n]

    try:
        model = get_reranker()
        scores = model.predict([(query, hit.text) for hit in hits])
        for hit, score in zip(hits, scores):
            hit.score = float(score)
        hits = sorted(hits, key=lambda h: h.score, reverse=True)
    except Exception:
        logger.exception("Reranking failed — falling back to retrieval order")

    return hits[:top_n]
