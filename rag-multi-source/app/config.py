from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    APP_SECRET_KEY: str = "change-me-in-production"
    ENCRYPTION_KEY: str = ""  # Fernet key; auto-generated on first run

    # ── SQLite ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./rag_system.db"

    # ── Pinecone (serverless free tier) ──────────────────────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "rag-multi-source"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_CLOUD: str = "aws"

    # ── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDINGS_PROVIDER: Literal["openai", "huggingface"] = "openai"
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536  # text-embedding-3-small default

    HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_EMBEDDING_DIMENSION: int = 384

    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["openai", "anthropic", "grok"] = "openai"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    GROK_API_KEY: Optional[str] = None
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    GROK_MODEL: str = "grok-2-1212"

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ────────────────────────────────────────────────────────────
    TOP_K: int = 8
    SCORE_THRESHOLD: float = 0.10

    @property
    def embedding_dim(self) -> int:
        if self.EMBEDDINGS_PROVIDER == "huggingface":
            return self.HF_EMBEDDING_DIMENSION
        return self.EMBEDDING_DIMENSION


# Singleton – import this everywhere
settings = Settings()
