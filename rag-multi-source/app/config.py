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

    # ── Audit & cost accounting ──────────────────────────────────────────────
    #
    # The chat layer always writes a row to ``query_audit_logs`` per user
    # prompt and a child row to ``query_step_timings`` per measured step.
    # These flags + the rate table below let an operator tune storage
    # growth and pricing accuracy without code changes.
    #
    # Master switch — set ``AUDIT_LOG_ENABLED=false`` in .env to disable
    # audit writes entirely (helpers no-op, the chains skip recording).
    # Useful in tests / hot benchmark loops.
    AUDIT_LOG_ENABLED: bool = True

    # Truncate stored ``prompt_text`` to this many chars. Keeps the
    # table small for users that paste long SQL or PDF excerpts.
    AUDIT_PROMPT_MAX_CHARS: int = 4000

    # Approximate USD pricing per 1K tokens — (prompt_rate, completion_rate).
    # Lookups are case-insensitive, longest-key-wins substring match
    # against the configured model name (so "gpt-4o-mini-2025-08-07"
    # still resolves to the "gpt-4o-mini" rate). Same table is consumed
    # by ``app.utils.estimate_cost`` which is called from BOTH the
    # status bar AND the audit logger so the figures never disagree.
    #
    # Update both rates here when prices move; no code change required.
    LLM_COST_PER_1K_TOKENS: dict[str, dict[str, tuple[float, float]]] = {
        "openai": {
            # GPT-4o family
            "gpt-4o-mini":   (0.000150, 0.000600),
            "gpt-4o":        (0.002500, 0.010000),
            # GPT-4 family
            "gpt-4-turbo":   (0.010000, 0.030000),
            "gpt-4.1-mini":  (0.000400, 0.001600),
            "gpt-4.1":       (0.002000, 0.008000),
            "gpt-4":         (0.030000, 0.060000),
            # GPT-3.5 family
            "gpt-3.5":       (0.000500, 0.001500),
            # o-series reasoning models
            "o1-mini":       (0.003000, 0.012000),
            "o1":            (0.015000, 0.060000),
            "o3-mini":       (0.001100, 0.004400),
        },
        "anthropic": {
            # Claude 4.x (current generation)
            "claude-opus-4":   (0.015000, 0.075000),
            "claude-sonnet-4": (0.003000, 0.015000),
            "claude-haiku-4":  (0.001000, 0.005000),
            # Claude 3.5 / 3 (kept for back-compat with older model strings)
            "claude-3-5-sonnet": (0.003000, 0.015000),
            "claude-3-5-haiku":  (0.000800, 0.004000),
            "claude-3-opus":     (0.015000, 0.075000),
            "claude-3-sonnet":   (0.003000, 0.015000),
            "claude-3-haiku":    (0.000250, 0.001250),
            # Generic family fallbacks (matched after the more specific keys
            # above thanks to longest-key-wins ordering).
            "claude-opus":   (0.015000, 0.075000),
            "claude-sonnet": (0.003000, 0.015000),
            "claude-haiku":  (0.000800, 0.004000),
        },
        "grok": {
            # xAI Grok family
            "grok-2-1212": (0.002000, 0.010000),
            "grok-2":      (0.002000, 0.010000),
            "grok-beta":   (0.005000, 0.015000),
            "grok":        (0.005000, 0.015000),
        },
    }

    # Fallback rate when the (provider, model) pair isn't in the table.
    # Conservative — better to slightly over-estimate than to silently
    # show $0 for an unknown model.
    LLM_DEFAULT_COST_PER_1K_TOKENS: tuple[float, float] = (0.001000, 0.003000)

    @property
    def embedding_dim(self) -> int:
        if self.EMBEDDINGS_PROVIDER == "huggingface":
            return self.HF_EMBEDDING_DIMENSION
        return self.EMBEDDING_DIMENSION


# Singleton – import this everywhere
settings = Settings()
