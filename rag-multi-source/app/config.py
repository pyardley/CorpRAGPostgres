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

    # HNSW runtime search effort (pgvector ``hnsw.ef_search``).
    #
    # Controls how many candidates the HNSW graph traversal explores per
    # query. The pgvector default is 40, which is fine when results are
    # tightly clustered but causes recall misses when one source has a
    # dense neighbourhood near the query embedding (e.g. lots of similar
    # marketing emails) and crowds out a higher-scoring chunk in another
    # source. Symptom: filtering to the "right" source surfaces a hit at
    # score 0.35, but enabling all sources returns only 0.32–0.34 hits
    # from the noisy source — the 0.35 chunk should rank #1 globally
    # but never makes it into the top-K candidate set.
    #
    # 200 is a comfortable default: ~5× the pgvector default, materially
    # better recall, still sub-millisecond on millions of rows. Bump to
    # 400+ if you still see the noisy-source effect; lower to 80–100 if
    # query latency matters more than recall on extremely large indexes.
    # Applied per-statement via ``SET LOCAL hnsw.ef_search`` so it only
    # affects the SELECT in ``core.retriever.retrieve`` and never leaks
    # to other sessions sharing the connection pool.
    HNSW_EF_SEARCH: int = 200

    # Per-source diversity quota.
    #
    # Even with a healthy ``HNSW_EF_SEARCH`` it's possible for one source
    # to legitimately fill all TOP_K slots — e.g. a thousand near-identical
    # marketing emails clustered just above the relevance threshold. The
    # retriever guards against that by fetching a wider candidate pool
    # (``TOP_K * RETRIEVAL_FETCH_MULTIPLIER`` rows) and then capping how
    # many chunks any single source contributes to the final top-K
    # (``MAX_HITS_PER_SOURCE``).
    #
    # The cap is soft: if there aren't enough hits from other sources to
    # fill ``TOP_K``, the retriever backfills from the overflow in
    # score order, so single-source queries (or selections where only
    # one source has matches) still return a full result set.
    #
    # Defaults: fetch 4× the candidates, cap each source at 4 of TOP_K=8.
    # Set ``MAX_HITS_PER_SOURCE`` ≥ ``TOP_K`` to disable the quota.
    RETRIEVAL_FETCH_MULTIPLIER: int = 4
    MAX_HITS_PER_SOURCE: int = 4

    # ── Hybrid retrieval (vector + keyword) ──────────────────────────────────
    #
    # Pure dense retrieval struggles with rare-token queries (error codes,
    # ticket numbers, SKUs, identifiers like "XYZ999") because the embedding
    # is dominated by the surrounding semantic context, not the literal
    # token. Hybrid retrieval runs a Postgres full-text search (FTS) in
    # parallel with the vector search and fuses the two ranked lists with
    # Reciprocal Rank Fusion (RRF). Lexical matching catches the
    # rare-token case; semantic matching catches the conceptual case.
    #
    # ``RETRIEVAL_MODE``
    #     ``"vector"`` — original behaviour, vector similarity only.
    #     ``"hybrid"`` — vector + keyword, RRF-fused. Requires the
    #     ``vector_chunks.text_search`` generated column + GIN index that
    #     ``models.migrations`` provisions automatically on first run.
    #
    # ``RRF_K``
    #     Smoothing constant for Reciprocal Rank Fusion. The standard
    #     value from the original RRF paper is 60. Lower → top-ranked
    #     items dominate the fused score; higher → lower-ranked items
    #     get a fairer share. Rarely worth tuning.
    #
    # ``FTS_LANGUAGE``
    #     Postgres text-search configuration name. ``"english"`` is the
    #     default install. Switch to ``"simple"`` to disable stemming
    #     and stop-word removal — useful for code-heavy corpora where
    #     "running" should not match "run" and "the" / "is" should still
    #     be searchable. Used by both the generated tsvector column and
    #     the query-time ``websearch_to_tsquery`` call.
    RETRIEVAL_MODE: Literal["vector", "hybrid"] = "hybrid"
    RRF_K: int = 60
    FTS_LANGUAGE: str = "english