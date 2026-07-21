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

    # Per-request timeout for every chat-model call (chat, MCP tool-loop,
    # and image captioning). Without this the underlying SDK defaults apply
    # (the OpenAI SDK's default is 600s) — a single stalled request during
    # ingestion-time image captioning can otherwise hang far longer than
    # any interactive caller would tolerate before the fail-open handler
    # in `app.ingestion.base._caption_image_chunks` ever gets a chance to
    # log it and move on.
    LLM_REQUEST_TIMEOUT_SECONDS: float = 60.0

    # ── Multi-modal ingestion (image captioning) ──────────────────────────────
    #
    # Images embedded in Confluence pages or attached to Jira tickets are
    # invisible to the indexer by default — this fetches each image and
    # captions it with a vision-capable LLM so the caption becomes a normal
    # searchable chunk. Off by default: this is the same shape of feature
    # (download binary attachment content, pay for an LLM call per item)
    # that `app.ingestion.email_ingestor` deliberately declined for email
    # attachments ("too costly + risky") — opt in once the cost for your
    # own ingestion volume (images per resource x vision-LLM pricing) is
    # understood.
    ENABLE_MULTIMODAL_INGESTION: bool = False

    # Model used to caption images. Falls back to the provider's normal
    # chat model (`OPENAI_CHAT_MODEL` / `ANTHROPIC_MODEL`) when unset —
    # both current defaults are already vision-capable. Only needed as an
    # override when `LLM_PROVIDER=grok`, since the default Grok model
    # isn't vision-capable.
    VISION_MODEL: Optional[str] = None

    # Cost/safety caps. Per-resource bounds a single huge page/issue;
    # per-run bounds a `--mode full` re-ingest of an entire space/project.
    MAX_IMAGES_PER_RESOURCE: int = 5
    MAX_IMAGE_BYTES: int = 5_000_000
    MAX_IMAGES_PER_INGESTION_RUN: int = 200

    # ── Entity graph (GraphRAG-inspired) ──────────────────────────────────────
    #
    # Pure chunk similarity can't answer relationship questions ("who else
    # has touched tickets related to this outage?", "which repos does this
    # author maintain?"). When enabled, ingestion also writes lightweight
    # (subject, predicate, object) triples to the `entity_edges` table --
    # deterministically from structured fields already fetched (Jira
    # assignee/reporter, Git commit author) at zero extra API/LLM cost.
    # Queried at chat time via the `entity_graph_query` MCP tool
    # (`mcp_server/tools/entity_graph_tools.py`), bound to the LLM whenever
    # this flag is on -- see `core.mcp_client.build_mcp_tools`.
    #
    # Off by default, same category as multi-modal ingestion: an
    # ingestion-time enrichment step with its own table + RLS policies.
    ENABLE_ENTITY_GRAPH: bool = False

    # Additional opt-in pass: also extract relationships from free text
    # (ticket descriptions/comments, commit messages) via an LLM call --
    # see `core.entity_extraction`. Requires ENABLE_ENTITY_GRAPH. Unlike
    # the deterministic edges above, this has the same cost/latency/
    # fail-open surface as reranking / query rewriting / image captioning.
    ENABLE_ENTITY_EXTRACTION_LLM: bool = False

    # Model used for LLM-extracted edges. Falls back to the provider's
    # normal chat model when unset.
    ENTITY_EXTRACTION_MODEL: Optional[str] = None

    # Cap on LLM-extracted edges per resource (deterministic edges are
    # not capped -- there are at most two, assignee + reporter, per Jira
    # issue, or one per Git commit).
    ENTITY_EXTRACTION_MAX_EDGES: int = 10

    # ── Semantic response cache ───────────────────────────────────────────────
    #
    # Repeated or near-duplicate questions currently re-run the full
    # retrieval pipeline and pay for a full LLM call every time. When
    # enabled, the chat layer checks `response_cache` (embedding
    # similarity + a fingerprint of the *resolved* accessible scopes,
    # not user_id -- see `core.response_cache`) before query rewriting /
    # vector retrieval / reranking even run. A hit skips all of that and
    # the LLM call. Two users only ever share a hit when their accessible
    # data is identical, so this is safe to share across a team, not just
    # cached per-user.
    #
    # Off by default. Never applies to the MCP/hybrid (live SQL) path --
    # live data must never be served from a cache.
    RESPONSE_CACHE_ENABLED: bool = False

    # Much stricter than SCORE_THRESHOLD (0.10): a false positive here
    # serves a wrong cached ANSWER outright, not just a weak retrieval
    # candidate that reranking can still filter out.
    RESPONSE_CACHE_SIMILARITY_THRESHOLD: float = 0.97

    # How long a cache entry is servable. Enforced lazily (a `WHERE
    # created_at > ...` at lookup time) -- there's no background sweep,
    # consistent with this being a lightweight addition, not a new piece
    # of job-scheduling infrastructure.
    RESPONSE_CACHE_TTL_SECONDS: int = 3600

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
    FTS_LANGUAGE: str = "english"

    # ── Query rewriting (multi-query decomposition) ──────────────────────────
    #
    # The chat layer sends the user's raw question straight to
    # `core.retriever.retrieve` — no rewrite pass for vague or multi-part
    # questions. When enabled, `core.query_rewrite.rewrite_query` makes one
    # LLM call that either returns the question unchanged (simple,
    # single-intent questions) or decomposes it into up to
    # QUERY_REWRITE_MAX_SUBQUERIES standalone search queries. The chat layer
    # then calls `retrieve()` once per sub-query (concurrently, via a thread
    # pool, so N sub-queries cost close to one round trip of wall-clock time)
    # and pools the results before reranking.
    #
    # Off by default — this adds a full extra LLM round trip *before*
    # retrieval even starts, on every turn, more user-facing than
    # reranking's post-retrieval latency. Fails OPEN: any rewrite error
    # (bad structured-output parse, model error, timeout) falls back to
    # `[question]` — degrades to exactly today's single-retrieve()
    # behaviour, not a broken turn. This is a retrieval-quality knob, not
    # an authorization boundary.
    QUERY_REWRITE_ENABLED: bool = False

    # Model used for the rewrite call. Falls back to the provider's normal
    # chat model (`OPENAI_CHAT_MODEL` / `ANTHROPIC_MODEL`) when unset.
    QUERY_REWRITE_MODEL: Optional[str] = None

    # Hard cap on decomposed sub-queries. Directly bounds added cost/DB
    # load: N sub-queries means N embedding API calls plus N sets of
    # per-source SQL, not just a longer answer.
    QUERY_REWRITE_MAX_SUBQUERIES: int = 3

    # ── Reranking (cross-encoder second stage) ───────────────────────────────
    #
    # Vector/RRF scoring never actually reads a candidate chunk's text
    # against the query — it's a bi-encoder similarity computed independently
    # per source. A cross-encoder jointly encodes (query, chunk_text) and is
    # typically far more accurate at picking the genuinely relevant chunk,
    # especially on "tricky" queries where the best answer isn't the top
    # cosine hit. The chat layer fetches ``RERANK_CANDIDATE_K`` candidates
    # from ``core.retriever.retrieve`` instead of the usual ``TOP_K``, reranks
    # them with ``core.reranker.rerank``, and keeps the top ``TOP_K`` — so
    # disabling this reverts to exactly the pre-reranking candidate count and
    # ordering.
    #
    # On by default. Runs a local ``sentence-transformers`` CrossEncoder in
    # -process (CPU) — no new dependency, no API key, no per-query cost. The
    # first query after enabling downloads the model weights from the
    # HuggingFace Hub (a few hundred MB for the default model). Any reranker
    # error (model load failure, missing torch, OOM) fails OPEN — logged, and
    # the un-reranked candidate order is used — this is a quality knob, not
    # an authorization boundary.
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_CANDIDATE_K: int = 50

    # ── Vector store ops ─────────────────────────────────────────────────────
    # Rows per upsert batch. We still chunk for memory + WAL pressure.
    VECTOR_UPSERT_BATCH_SIZE: int = 250

    # ── Security & deduplication (OB1-inspired enhancements) ─────────────────
    #
    # Defence-in-depth: enable PostgreSQL Row-Level Security on
    # ``vector_chunks`` and ``user_accessible_resources`` so the database
    # itself refuses to return rows the current user wasn't granted —
    # even if a future bug bypasses ``filter_to_where``. Policies key on
    # the per-session GUC ``app.current_user_id`` which the app sets
    # via ``app.utils.set_current_user_for_rls`` before any SELECT.
    #
    # Default ON. Set ``ENABLE_RLS=false`` in .env to disable for
    # backward compatibility (e.g. legacy deployments where the DB role
    # used by the app doesn't have ``ALTER TABLE`` privileges, or where
    # an external reporting tool uses the same DB role and would itself
    # be locked out by the policies). The migration runner skips the
    # ``ENABLE ROW LEVEL SECURITY`` + ``CREATE POLICY`` statements when
    # this is false; the helper turns into a cheap no-op.
    ENABLE_RLS: bool = True

    # When True, every chunk is hashed (SHA-256 over normalised text)
    # and the upsert path skips re-embedding rows whose stored
    # fingerprint already matches the new one. Always-on by default —
    # disable only for benchmarking the cost of unconditional re-embeds.
    ENABLE_CONTENT_FINGERPRINT_DEDUP: bool = True

    # ── Live ACL re-validation ────────────────────────────────────────────────
    #
    # ``user_accessible_resources`` is populated at ingestion time and never
    # re-checked — if a user loses Jira/Confluence/SQL/GitHub access after
    # ingestion, they keep querying that data until the next re-ingestion.
    # When enabled, ``core.live_acl.revalidate`` calls back into the source
    # system itself on every chat turn and intersects the live permission
    # with the granted scope, closing that gap. Adds a per-query API round
    # trip per distinct scope (bounded by ``LIVE_ACL_CACHE_TTL_SECONDS``), so
    # this defaults to off — opt in once the source systems' rate limits are
    # understood for your deployment.
    #
    # Fails CLOSED: any checker error (timeout, network, unknown credential)
    # excludes that scope from the turn's results rather than falling back
    # to the ingestion-time grant.
    LIVE_ACL_REVALIDATION_ENABLED: bool = False

    # How long a live "allowed"/"denied" verdict is cached per
    # (user, source, resource_identifier), in seconds. Short enough to
    # reflect access changes promptly, long enough that a multi-turn chat
    # session doesn't re-hit the source API on every message.
    LIVE_ACL_CACHE_TTL_SECONDS: int = 300

    # Per-request timeout for outbound live-ACL HTTP calls (Jira,
    # Confluence, GitHub). A slow/hanging source system must not stall
    # chat — it fails closed for that scope instead.
    LIVE_ACL_HTTP_TIMEOUT_SECONDS: float = 5.0

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
