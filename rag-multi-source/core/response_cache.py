"""
Semantic response cache (README "Possible enhancements" — Semantic
response cache, in the same Postgres database).

Repeated or near-duplicate questions currently re-run the full
retrieval pipeline and pay for a full LLM call every time. This module
checks/writes `response_cache` (see `models.response_cache`) so a hit
can skip query rewriting, vector retrieval, reranking, and the LLM call
entirely.

**Tenancy design** (the reason this isn't a one-line GPTCache
integration): the cache key is `scope_fingerprint` — a hash of the
*resolved* accessible scopes actually used for retrieval
(`filter_dict["by_source"]` + `fts_language`), not `user_id`. Two users
only ever share a cache hit when their accessible data is identical,
which is provably safe (if the scopes match, the underlying retrievable
data is the same for both). `user_id` is still passed through to bind
the RLS GUC (`response_cache`'s policies are GUC-presence-only — see
`models.migrations` — so a session needs *some* authenticated user_id
to see any rows at all, but which one doesn't affect which rows match).

Fails OPEN — unlike `core.live_acl`, this is a cost/latency
optimisation, not an authorization boundary. Any error (embedding
failure, DB issue, serialization problem) is logged; `check` returns
`None` (falls through to a normal turn) and `store` just skips.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger
from sqlalchemy import select

from app.config import settings
from app.utils import get_db, set_current_user_for_rls
from core.llm import get_embeddings
from core.rag_chain import RAGAnswer
from core.retriever import RetrievedChunk
from models.response_cache import ResponseCacheEntry


def scope_fingerprint(filter_dict: dict[str, Any], fts_language: str) -> str:
    """
    Deterministic hash of the resolved accessible scopes used for
    retrieval. Two lookups only collide when their `by_source` scope
    lists (per source, sorted) and `fts_language` are identical.
    """
    by_source: dict[str, list[str]] = (filter_dict or {}).get("by_source") or {}
    canonical = {
        source: sorted(scopes) for source, scopes in sorted(by_source.items())
    }
    payload = json.dumps(
        {"by_source": canonical, "fts_language": fts_language}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _citation_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "resource_id": chunk.resource_id,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "score": chunk.score,
        "text": chunk.text,
        "title": chunk.title,
        "url": chunk.url,
        "metadata": chunk.metadata,
    }


def _dict_to_citation(data: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        resource_id=data["resource_id"],
        source=data["source"],
        chunk_index=data["chunk_index"],
        score=data["score"],
        text=data["text"],
        title=data["title"],
        url=data["url"],
        metadata=data.get("metadata") or {},
    )


def check_response_cache(
    user_id: str, question: str, filter_dict: dict[str, Any], fts_language: str
) -> Optional[RAGAnswer]:
    """
    Return a cached `RAGAnswer` for `question` if a sufficiently similar
    question was answered under the same resolved scopes within the
    TTL. Returns `None` on a miss or any error.
    """
    if not settings.RESPONSE_CACHE_ENABLED or not question.strip():
        return None

    try:
        fp = scope_fingerprint(filter_dict, fts_language)
        embedding = get_embeddings().embed_query(question)
        cutoff = datetime.utcnow() - timedelta(seconds=settings.RESPONSE_CACHE_TTL_SECONDS)

        similarity_expr = (
            1 - ResponseCacheEntry.question_embedding.cosine_distance(embedding)
        ).label("similarity")
        distance_expr = ResponseCacheEntry.question_embedding.cosine_distance(embedding)

        with get_db() as db:
            set_current_user_for_rls(db, user_id)
            stmt = (
                select(ResponseCacheEntry, similarity_expr)
                .where(ResponseCacheEntry.scope_fingerprint == fp)
                .where(ResponseCacheEntry.created_at >= cutoff)
                .where(similarity_expr >= settings.RESPONSE_CACHE_SIMILARITY_THRESHOLD)
                .order_by(distance_expr)
                .limit(1)
            )
            row = db.execute(stmt).first()
            if row is None:
                return None
            entry, similarity = row
            answer = entry.answer
            provider = settings.LLM_PROVIDER
            model = _current_model_name()
            citations = [_dict_to_citation(c) for c in entry.citations or []]

        logger.info(
            "[response_cache] hit fp={} similarity={:.4f}", fp[:8], similarity
        )
        return RAGAnswer(
            answer=answer,
            citations=citations,
            usage={
                "provider": provider,
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "duration_seconds": 0.0,
            },
            steps=[],
        )
    except Exception:
        logger.warning("[response_cache] check failed — proceeding without cache")
        return None


def store_response_cache(
    user_id: str,
    question: str,
    filter_dict: dict[str, Any],
    fts_language: str,
    result: RAGAnswer,
) -> None:
    """Persist `result` for future lookups. Best-effort — never raises."""
    if not settings.RESPONSE_CACHE_ENABLED or not question.strip():
        return

    try:
        fp = scope_fingerprint(filter_dict, fts_language)
        embedding = get_embeddings().embed_query(question)
        with get_db() as db:
            set_current_user_for_rls(db, user_id)
            db.add(
                ResponseCacheEntry(
                    user_id=user_id,
                    question=question,
                    question_embedding=embedding,
                    scope_fingerprint=fp,
                    answer=result.answer,
                    citations=[_citation_to_dict(c) for c in result.citations],
                )
            )
    except Exception:
        logger.warning("[response_cache] store failed — answer not cached")


def _current_model_name() -> str:
    provider = settings.LLM_PROVIDER
    if provider == "openai":
        return settings.OPENAI_CHAT_MODEL
    if provider == "anthropic":
        return settings.ANTHROPIC_MODEL
    if provider == "grok":
        return settings.GROK_MODEL
    return ""
