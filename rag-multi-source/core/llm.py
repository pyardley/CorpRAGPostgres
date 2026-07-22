"""LLM and embeddings factory – returns the right client based on config."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from loguru import logger

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    provider = settings.EMBEDDINGS_PROVIDER
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        logger.info("Using OpenAI embeddings: {}", settings.EMBEDDING_MODEL)
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Using HuggingFace embeddings: {}", settings.HUGGINGFACE_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(model_name=settings.HUGGINGFACE_EMBEDDING_MODEL)

    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {provider!r}")


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider = settings.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        logger.info("Using OpenAI LLM: {}", settings.OPENAI_CHAT_MODEL)
        return ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY,
            request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        logger.info("Using Anthropic LLM: {}", settings.ANTHROPIC_MODEL)
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=0.1,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            default_request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "grok":
        # xAI Grok exposes an OpenAI-compatible endpoint
        from langchain_openai import ChatOpenAI

        if not settings.GROK_API_KEY:
            raise RuntimeError("GROK_API_KEY is not set.")
        logger.info("Using Grok LLM: {}", settings.GROK_MODEL)
        return ChatOpenAI(
            model=settings.GROK_MODEL,
            temperature=0.1,
            openai_api_key=settings.GROK_API_KEY,
            openai_api_base=settings.GROK_BASE_URL,
            request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    from torch.nn import Sigmoid

    # Force sigmoid activation regardless of the configured model's own
    # default. Cross-encoders disagree on this: BAAI/bge-reranker-base
    # defaults to a sigmoid (num_labels=1), but cross-encoder/ms-marco-*
    # models default to Identity() (raw, unbounded logits) despite also
    # being num_labels=1 — their HF config predates that convention. Since
    # `core.corrective_retrieval.is_low_confidence` thresholds `hit.score`
    # against a fixed [0, 1] value, every configured model must produce
    # scores on that scale; sigmoid is a monotonic transform, so it never
    # changes the reranked *order*, only the scale, so this is free for
    # RERANK_ENABLED's own ranking purpose too.
    logger.info("Loading cross-encoder reranker: {}", settings.RERANK_MODEL)
    return CrossEncoder(settings.RERANK_MODEL, activation_fn=Sigmoid())


@lru_cache(maxsize=1)
def get_vision_llm() -> BaseChatModel:
    """
    Chat model used for image captioning (see `core.vision`).

    Falls back to `get_llm()` when `VISION_MODEL` is unset — the default
    OpenAI/Anthropic chat models are already vision-capable. Only needed
    as an override for providers (e.g. Grok) whose configured chat model
    isn't vision-capable.
    """
    if not settings.VISION_MODEL:
        return get_llm()

    provider = settings.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        logger.info("Using OpenAI vision LLM: {}", settings.VISION_MODEL)
        return ChatOpenAI(
            model=settings.VISION_MODEL,
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY,
            request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        logger.info("Using Anthropic vision LLM: {}", settings.VISION_MODEL)
        return ChatAnthropic(
            model=settings.VISION_MODEL,
            temperature=0.1,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            default_request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "grok":
        from langchain_openai import ChatOpenAI

        if not settings.GROK_API_KEY:
            raise RuntimeError("GROK_API_KEY is not set.")
        logger.info("Using Grok vision LLM: {}", settings.VISION_MODEL)
        return ChatOpenAI(
            model=settings.VISION_MODEL,
            temperature=0.1,
            openai_api_key=settings.GROK_API_KEY,
            openai_api_base=settings.GROK_BASE_URL,
            request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


@lru_cache(maxsize=1)
def get_query_rewrite_llm() -> BaseChatModel:
    """
    Chat model used for query decomposition (see `core.query_rewrite`).

    Falls back to `get_llm()` when `QUERY_REWRITE_MODEL` is unset.
    """
    if not settings.QUERY_REWRITE_MODEL:
        return get_llm()

    provider = settings.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        logger.info("Using OpenAI query-rewrite LLM: {}", settings.QUERY_REWRITE_MODEL)
        return ChatOpenAI(
            model=settings.QUERY_REWRITE_MODEL,
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY,
            request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        logger.info("Using Anthropic query-rewrite LLM: {}", settings.QUERY_REWRITE_MODEL)
        return ChatAnthropic(
            model=settings.QUERY_REWRITE_MODEL,
            temperature=0.1,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            default_request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "grok":
        from langchain_openai import ChatOpenAI

        if not settings.GROK_API_KEY:
            raise RuntimeError("GROK_API_KEY is not set.")
        logger.info("Using Grok query-rewrite LLM: {}", settings.QUERY_REWRITE_MODEL)
        return ChatOpenAI(
            model=settings.QUERY_REWRITE_MODEL,
            temperature=0.1,
            openai_api_key=settings.GROK_API_KEY,
            openai_api_base=settings.GROK_BASE_URL,
            request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


@lru_cache(maxsize=1)
def get_entity_extraction_llm() -> BaseChatModel:
    """
    Chat model used for LLM-extracted entity edges (see
    `core.entity_extraction`).

    Falls back to `get_llm()` when `ENTITY_EXTRACTION_MODEL` is unset.
    """
    if not settings.ENTITY_EXTRACTION_MODEL:
        return get_llm()

    provider = settings.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        logger.info("Using OpenAI entity-extraction LLM: {}", settings.ENTITY_EXTRACTION_MODEL)
        return ChatOpenAI(
            model=settings.ENTITY_EXTRACTION_MODEL,
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY,
            request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        logger.info("Using Anthropic entity-extraction LLM: {}", settings.ENTITY_EXTRACTION_MODEL)
        return ChatAnthropic(
            model=settings.ENTITY_EXTRACTION_MODEL,
            temperature=0.1,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            default_request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    if provider == "grok":
        from langchain_openai import ChatOpenAI

        if not settings.GROK_API_KEY:
            raise RuntimeError("GROK_API_KEY is not set.")
        logger.info("Using Grok entity-extraction LLM: {}", settings.ENTITY_EXTRACTION_MODEL)
        return ChatOpenAI(
            model=settings.ENTITY_EXTRACTION_MODEL,
            temperature=0.1,
            openai_api_key=settings.GROK_API_KEY,
            openai_api_base=settings.GROK_BASE_URL,
            request_timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
