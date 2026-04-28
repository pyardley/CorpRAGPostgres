from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    APP_SECRET_KEY: str = "change-me-in-production"
    ENCRYPTION_KEY: str = ""  # Fernet key; auto-generated on first run

    # ── Database ─────────────────────────────────────────────────────────────
    # PostgreSQL with the pgvector extension. Examples:
    #   postgresql+psycopg://user:pass@localhost:5432/rag
    #   postgresql+psycopg://postgres.<ref>:<pwd>@aws-0-eu-west-1.pooler.supabase.com:6543/postgres
    #   postgresql+psycopg://<user>:<pwd>@ep-cool-xxx.eu-central-1.aws.neon.tech/neondb
    DATABASE_URL: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/corporaterag"
    )

    # SQLAlchemy connection pool tuning. Defaults are fine for a single
    # Streamlit instance; bump pool_size for multi-process deployments.
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_PRE_PING: bool = True

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

    # ── Vector store ops ─────────────────────────────────────────────────────
    # Rows per upsert batch. We still chunk for memory + WAL pressure.
    VECTOR_UPSERT_BATCH_SIZE: int = 250

    @property
    def embedding_dim(self) -> int:
        if self.EMBEDDINGS_PROVIDER == "huggingface":
            return self.HF_EMBEDDING_DIMENSION
        return self.EMBEDDING_DIMENSION


# Singleton – import this everywhere
settings = Settings()
