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
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
