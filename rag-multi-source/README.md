# CorporateRAG

A production-ready, multi-tenant Retrieval-Augmented-Generation system that
lets a whole organisation ask natural-language questions across **Jira**,
**Confluence**, **SQL Server** and **Git (GitHub)** — from a single
Streamlit chat interface, backed by a **single PostgreSQL database with the
pgvector extension**.

Vectors, metadata, user identities, encrypted credentials, and per-user
access rows all live in the same database. Retrieval is one SQL statement
that JOINs cosine-similarity ranking with the user's accessible resources —
no second hop to a managed vector DB, no two-system consistency problem.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Streamlit chat UI                              │
│                                                                       │
│   [paul@org.com]   Searching across: 🔵 Jira  🟢 Confluence  🟠 SQL    │
│                                                                       │
│   Q: What does error code ABC123 mean?                                │
│   A: ABC123 is caused by buffer overflow [1].                         │
│      Sources:                                                         │
│      [1] 🔵 Jira — SCRUM-29 Test Verification — score 0.74            │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  │ filter built from user's
                                  │ user_accessible_resources
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    PostgreSQL + pgvector                             │
│                                                                      │
│  users           ─┐                                                   │
│  user_credentials ├─ everyday SQLAlchemy tables                       │
│  ingestion_logs  ─┤                                                   │
│  user_accessible_resources  (the access table)                        │
│                                                                       │
│  vector_chunks                                                        │
│    id, resource_id, source, chunk_index, text,                        │
│    project_key | space_key | db_name | git_scope,                     │
│    metadata JSONB, embedding VECTOR(1536),                            │
│    HNSW index on embedding USING vector_cosine_ops                    │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ INSERT ... ON CONFLICT DO UPDATE
                                  │
                ┌─────────────────┴─────────────────┐
                │   Ingestion pipeline (CLI + UI)   │
                │   Jira • Confluence • SQL • Git   │
                └───────────────────────────────────┘
```

### Why PostgreSQL + pgvector?

- **One database for everything.** Auth, credentials, access rows, _and_
  vectors live side-by-side. No two-system consistency problems, no
  separate API key for a vector DB, no risk of the access table and the
  index drifting apart.
- **Real SQL alongside vector search.** Filters and rankings combine in
  one statement; you get full SQL power for hybrid search (`tsvector` BM25
  - cosine similarity) when you need it.
- **Free-tier friendly.** Runs against any Postgres ≥ 14 — local Docker,
  Supabase free, Neon free, RDS — without code changes.
- **HNSW indexing** (pgvector ≥ 0.5) gives sub-millisecond top-K cosine
  search up to several million rows on a small instance.

### Hybrid search: pick a language per query

Hybrid retrieval (`RETRIEVAL_MODE=hybrid`, the default) fuses vector
similarity with Postgres full-text search via Reciprocal Rank Fusion. Both
FTS configs are indexed **permanently and simultaneously** —
`vector_chunks.text_search_english` and `vector_chunks.text_search_simple`,
each with its own GIN index — so there's no admin toggle or rebuild step
to switch between them.

The sidebar's "🔤 Search language" picker lets any user choose, per
question:

- **english** — stemming + stop-word removal. Best for natural-language
  prose (a query for "running" also matches "run").
- **simple** — lower-cases only, no stemming. Best for code-heavy
  corpora, identifiers, and error codes, where `OAuth2Client` or `XYZ999`
  needs to match verbatim rather than being stemmed into something else.

Switch any time — no reload, no reindex, no waiting.

### Query rewriting before retrieval (multi-query decomposition)

The chat layer normally sends the user's raw question straight to
`core.retriever.retrieve` — no rewrite pass for vague or multi-part
questions. When `QUERY_REWRITE_ENABLED=true`, `core/query_rewrite.py`
adds one LLM call ahead of retrieval that either returns the question
unchanged (simple, single-intent questions) or decomposes it into up to
`QUERY_REWRITE_MAX_SUBQUERIES` standalone search queries — e.g. "compare
X's retrieval quota to Y's caching approach" becomes two independent,
self-contained queries. `app/chat.py` then calls `retrieve()` once per
sub-query, pools the results (deduped by `(resource_id, chunk_index)`,
keeping the higher score on overlap), and reranks the pool against the
*original* question — not the synthetic sub-queries.

```ini
# .env
QUERY_REWRITE_ENABLED=false          # default — off
QUERY_REWRITE_MODEL=                 # optional override; falls back to the provider's normal chat model
QUERY_REWRITE_MAX_SUBQUERIES=3       # hard cap: bounds embedding + SQL cost, not just output length
```

- **Off by default.** Unlike reranking (which adds latency *after*
  retrieval), this adds a full extra LLM round trip *before* retrieval
  even starts, on every turn — more user-facing, and worth opting into
  deliberately.
- **Concurrent sub-query retrieval.** The N `retrieve()` calls run in a
  thread pool, so N sub-queries cost close to one round trip of
  wall-clock time rather than N — each call already opens its own DB
  session internally (`app.utils.get_db()`), so concurrent calls are
  safe.
- **Fails open.** Any rewrite error (bad structured-output parse, model
  error, timeout) falls back to `[question]` — degrades to exactly
  today's single-retrieve() behaviour, not a broken turn. A quality
  knob, not an authorization boundary, same as reranking.
- **Decomposition, not HyDE.** The alternative technique — embedding a
  hypothetical answer passage (HyDE) instead of the question — was
  considered and not built: it would require threading a second "what
  to embed" parameter through `core/retriever.py` itself, since today it
  embeds and full-text-searches the same string. Decomposition reuses
  the retriever and reranker completely unchanged.

### Reranking with a cross-encoder

Vector/RRF scoring never actually reads a candidate chunk's text against
the query — it's a bi-encoder similarity computed independently per
source. When `RERANK_ENABLED=true` (the default), the chat layer adds a
second retrieval stage: it asks `core.retriever.retrieve` for
`RERANK_CANDIDATE_K` candidates (default 50, instead of the usual
`TOP_K`), then `core/reranker.py` runs a local cross-encoder
(`sentence_transformers.CrossEncoder`, model `RERANK_MODEL` — default
`cross-encoder/ms-marco-MiniLM-L-6-v2`) that jointly encodes `(query,
chunk_text)` for every candidate, re-scores and re-sorts them, and keeps
the top `TOP_K`. This is typically far more accurate than cosine
similarity alone at picking the genuinely relevant chunk, especially on
"tricky" queries where the right answer isn't the top vector hit.

```ini
# .env
RERANK_ENABLED=true                              # default — set false to skip the second stage
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # any sentence-transformers CrossEncoder repo id
RERANK_CANDIDATE_K=50                            # candidates fetched before reranking
```

- **No new dependency.** `sentence-transformers` is already required for
  the HuggingFace embeddings fallback; the reranker reuses it and runs on
  CPU in-process, cached as a singleton (`core.llm.get_reranker`) after
  first use.
- **Sized for CPU.** Cross-encoder inference cost scales roughly linearly
  with model size × candidate count. `ms-marco-MiniLM-L-6-v2` (~22M
  params) is the default specifically because it's cheap enough to run
  `RERANK_CANDIDATE_K` candidates on CPU without dominating a chat turn's
  latency — the larger, multilingual `BAAI/bge-reranker-base` (~278M
  params) it replaced benchmarked at ~200ms per `(query, chunk)` pair on a
  6-core CPU, i.e. ~10s of inference alone at the default candidate
  count. Any `sentence-transformers` CrossEncoder repo id can be
  substituted here if quality matters more than latency for your corpus.
- **Always sigmoid-activated.** `core.llm.get_reranker` explicitly passes
  `activation_fn=Sigmoid()` regardless of which model is configured —
  cross-encoders disagree on their own default here (some apply a sigmoid
  out of the box, some return raw unbounded logits), and the corrective
  retrieval feature below needs every model's `hit.score` to land on the
  same fixed `[0, 1]` scale. Sigmoid is a monotonic transform, so this
  never changes the reranked order, only the scale.
- **First-run download.** The first query after enabling downloads the
  model weights from the HuggingFace Hub — expect a one-time delay on the
  first turn.
- **Fails open.** Unlike the live-ACL check below, this is a ranking
  quality knob, not an authorization boundary — any reranker error
  (model load failure, missing torch, OOM) is logged and chat falls back
  to the un-reranked candidate order instead of failing the turn.
- **Visible in the audit trail.** Adds a `rerank` step to
  `query_step_timings` with `metadata.candidate_count` /
  `metadata.reranked_count` — see
  [Audit & Performance Logging](#audit--performance-logging).

### Corrective retrieval — admit when nothing matches

`core.rag_chain.answer_question` already skips the LLM call outright when
retrieval returns zero hits. When `CORRECTIVE_RETRIEVAL_ENABLED=true`,
that same short-circuit extends to the case where retrieval *did* return
hits, but none of them are a good match: if the best (post-rerank)
candidate's cross-encoder score falls below
`CORRECTIVE_RETRIEVAL_SCORE_THRESHOLD`, the chat layer declines to answer
— telling the user honestly that nothing confident was found and
suggesting they rephrase, widen their source selection, or re-ingest —
instead of quietly generating from marginal context and risking a
plausible-sounding hallucination. This is the "grading" step from Self-RAG
/ Corrective RAG (CRAG).

```ini
# .env
CORRECTIVE_RETRIEVAL_ENABLED=false        # off by default — opt in
CORRECTIVE_RETRIEVAL_SCORE_THRESHOLD=0.3  # on the reranker's [0, 1] scale
```

- **Depends on reranking.** `core.llm.get_reranker` forces sigmoid
  activation on whichever `RERANK_MODEL` is configured (see above), so
  `hit.score` is always a `[0, 1]` relevance score comparable across
  queries — unlike raw cosine similarity or an RRF fusion rank, neither
  of which is on a fixed scale. This check is therefore a no-op whenever
  `RERANK_ENABLED=false`, regardless of this flag.
  `CORRECTIVE_RETRIEVAL_SCORE_THRESHOLD` is a different scale from
  `SCORE_THRESHOLD` (a cosine-similarity floor applied at retrieval time)
  — don't confuse the two.
- **Off by default.** Like query rewriting and the response cache, this
  changes user-visible behaviour — an answer can be silently replaced by a
  decline — so it's opt-in until the default threshold is tuned against
  real traffic.
- **Scoped to the plain-RAG path.** The MCP/hybrid path
  (`answer_question_with_mcp`) is untouched — live SQL data can still be
  useful even when the RAG citations are weak.
- **Fails open.** Same posture as reranking: a retrieval-quality knob, not
  an authorization boundary.

### Multi-tenancy: store-once + dynamic filter

Every resource (Jira ticket, Confluence page, SQL stored proc, Git
file/commit) is stored **exactly once**. There's no `user_id` column on
`vector_chunks`. Tenancy is enforced at _query_ time:

```python
# core/vector_store.py — build_query_filter() output
filter_dict = {
    "by_source": {
        "jira":       ["PROJ", "OPS"],          # user's Jira projects
        "confluence": ["DOCS"],                 # user's Confluence spaces
        "sql":        ["customerdb"],           # user's SQL databases
        "git":        ["acme/widgets@main"],    # user's repo@branch scopes
    }
}
```

The retriever turns that into a SQL `WHERE` of the form:

```sql
WHERE (
    (source = 'jira'       AND project_key = ANY('{PROJ,OPS}'::text[]))
 OR (source = 'confluence' AND space_key   = ANY('{DOCS}'::text[]))
 OR (source = 'sql'        AND db_name     = ANY('{customerdb}'::text[]))
 OR (source = 'git'        AND git_scope   = ANY('{acme/widgets@main}'::text[]))
)
AND 1 - (embedding <=> :query_vec) >= :threshold
ORDER BY embedding <=> :query_vec
LIMIT :top_k
```

The lists come from the user's rows in `user_accessible_resources`, which
the ingestion pipeline populates whenever a user successfully ingests a
scope.

### Semantic response cache

Repeated or near-duplicate questions currently re-run the full
retrieval pipeline and pay for a full LLM call every time. When
`RESPONSE_CACHE_ENABLED=true`, `app/chat.py` checks a `response_cache`
table (`models/response_cache.py`) right after `live_acl_check` —
before query rewriting, vector retrieval, or reranking even run. A hit
skips all of that *and* the LLM call; only `core.llm.get_embeddings()`
is called (to embed the incoming question for the similarity lookup).

The interesting part is the cache key. It is **not** `user_id` — it's
`scope_fingerprint`, a SHA-256 hash of the *resolved* accessible scopes
actually used for retrieval (`filter_dict["by_source"]`, the same dict
built in [Multi-tenancy: store-once + dynamic filter](#multi-tenancy-store-once--dynamic-filter)
above, plus `fts_language`). Two users only ever share a cache hit when
their resolved scopes are byte-identical — which, by construction,
means they have access to exactly the same underlying data, so sharing
the answer is provably safe. This is what actually delivers the
scenario the feature is for ("a whole team asking the same question
over a week") rather than a cache that's only ever useful to the one
user who asked first. `user_id` is still stored on each row as an audit
trail, but it plays no role in matching.

Because the access boundary is the fingerprint match, not row
ownership, RLS on `response_cache` is deliberately **GUC-presence-only**
— `current_setting('app.current_user_id', true) IS NOT NULL` — the same
shape as `vector_chunks`' write policies, not the owner-scoped
`user_id = current_setting(...)` pattern used by
`user_accessible_resources`. A session still needs to be authenticated
to read or write the table at all; which authenticated user doesn't
change which rows a lookup matches.

```ini
# .env
RESPONSE_CACHE_ENABLED=false             # default — off
RESPONSE_CACHE_SIMILARITY_THRESHOLD=0.97 # much stricter than SCORE_THRESHOLD (0.10) —
                                          # a false positive here serves a WRONG answer outright
RESPONSE_CACHE_TTL_SECONDS=3600
```

- **Never applies to the MCP/hybrid (live SQL) path.** Live data must
  never be served from a cache — a cache check is skipped entirely
  whenever `used_mcp_path` is true.
- **Currently inert whenever `ENABLE_ENTITY_GRAPH=true`.** `used_mcp_path`
  is `true` whenever *either* the user's "Use Live SQL" toggle is on
  *or* `ENABLE_ENTITY_GRAPH` is on (see
  [Lightweight entity graph](#lightweight-entity-graph-graphrag-inspired)
  below) — the entity graph's MCP tool has to be bound on every turn to
  be available, so every turn takes the hybrid path once the flag is on,
  and the bullet above then skips the cache check on all of them. With
  both flags set, `RESPONSE_CACHE_ENABLED=true` never checks or
  populates the cache, full stop — it isn't a partial degradation, it's
  a no-op. This is deliberate, not an oversight: see
  [Possible enhancements § response cache vs. entity graph](#7-let-the-response-cache-and-entity-graph-coexist)
  for why a naive fix is unsafe and what a real one requires.
- **Strict similarity threshold, deliberately.** `SCORE_THRESHOLD`
  (0.10) governs which chunks are *candidates* for reranking — a false
  positive there just means a weak candidate gets considered.
  `RESPONSE_CACHE_SIMILARITY_THRESHOLD` (0.97) governs whether an
  entire cached *answer* gets served verbatim — a false positive there
  is a wrong answer. In testing, two genuinely different phrasings of
  the same underlying question ("what is the meaning of life" vs. "what
  is the answer to life") scored ~0.77 — comfortably below the
  threshold, confirming it isn't so loose that paraphrases collide.
- **Lazy TTL, no background sweep.** Expiry is a `WHERE created_at >
  ...` at lookup time, not a scheduled job — consistent with nothing
  else in this codebase running a job scheduler.
- **Fails open.** Any error (embedding failure, DB issue, serialization
  problem) is logged and skipped — a broken cache degrades to a normal
  retrieval+LLM turn, never a broken chat turn.

### Lightweight entity graph (GraphRAG-inspired)

Pure chunk similarity can't answer relationship questions — "who else
has touched tickets related to this outage?", "which repos does this
author maintain?". When `ENABLE_ENTITY_GRAPH=true`, ingestion also
writes `(subject, predicate, object)` triples to a new `entity_edges`
table (`models/entity_edge.py`), tenancy-scoped exactly like
`vector_chunks` — its own `source` / `resource_identifier` columns and
RLS policies (`models/migrations.py`), joined against
`user_accessible_resources` the same way.

Two sources of edges:

- **Deterministic, zero extra cost** — Jira's `assignee`/`reporter`
  (stable `accountId`, falling back to display name) and Git's
  per-commit author (email, falling back to name) are already present
  in API responses the ingestors fetch anyway; `jira_ingestor.py` /
  `git_ingestor.py` just weren't reading those fields before. No API
  call, no LLM call. Git edges use the **repo scope** as the subject
  (not the per-commit resource), so every commit by the same author
  points at one repo-level node — that's what makes "which repos does X
  maintain" a single query instead of N.
- **LLM-extracted, opt-in** (`ENABLE_ENTITY_EXTRACTION_LLM=true`) — an
  additional pass over free text (ticket descriptions/comments, commit
  messages) via `core/entity_extraction.py`, capped at
  `ENTITY_EXTRACTION_MAX_EDGES` per resource. Same fail-open philosophy
  as reranking / query rewriting / image captioning: any extraction
  error is logged and skipped, never blocks the resource's chunks or
  its deterministic edges.

```ini
# .env
ENABLE_ENTITY_GRAPH=false           # default off — new table + RLS policies + ingestion-time writes
ENABLE_ENTITY_EXTRACTION_LLM=false  # additional opt-in LLM pass over free text; requires ENABLE_ENTITY_GRAPH
ENTITY_EXTRACTION_MODEL=            # optional override; falls back to the provider's normal chat model
ENTITY_EXTRACTION_MAX_EDGES=10
```

Queried at chat time via the **`entity_graph_query`** MCP tool (see
[Hybrid RAG + MCP](#hybrid-rag--mcp-live-sql-server-table-data) below)
rather than auto-injected into every RAG context — the LLM decides when
a question actually needs graph traversal, consistent with how this
codebase already handles live/structured data. The tool is bound
whenever `ENABLE_ENTITY_GRAPH=true`, independent of the "Use Live SQL"
toggle — there's no separate sidebar control, matching the plain
`.env`-flag precedent set by reranking, live-ACL revalidation, and
query rewriting.

Side effect worth knowing: making the tool available this way means
*every* chat turn takes `app/chat.py`'s hybrid/MCP path once this flag
is on — not just questions that actually need graph traversal — because
`used_mcp_path` cannot tell "entity graph might be needed" apart from
"the user turned on live SQL." Two consequences: `sql_table_query` /
`sql_list_databases` end up bound and callable on *every* turn too (see
`core/mcp_client.py`'s `build_mcp_tools()` — they're bound
unconditionally, not gated on the "Use Live SQL" sidebar toggle at
all), and [the semantic response cache](#semantic-response-cache)
becomes entirely inert, since it never checks or stores on the hybrid
path. See
[Possible enhancements § response cache vs. entity graph](#7-let-the-response-cache-and-entity-graph-coexist)
for the fix.

### Stable resource IDs

| Source     | Format                                                                                  |
| ---------- | --------------------------------------------------------------------------------------- |
| Jira       | `jira:{ISSUE_KEY}` — e.g. `jira:PROJ-123`                                               |
| Confluence | `confluence:page-{page_id}` — e.g. `confluence:page-9876`                               |
| SQL Server | `sql:{server}.{db}.{schema}.{name}`                                                     |
| Git        | `git:{owner}/{repo}@{branch}:commit:{sha}` or `git:{owner}/{repo}@{branch}:file:{path}` |
| GitHub Issues (recipe) | `github_issue:{owner}/{repo}#{number}`                                     |

A resource that produces N chunks is keyed `(resource_id, chunk_index)` for
chunks 0..N-1. `INSERT ... ON CONFLICT DO UPDATE` makes re-ingestion an
overwrite, not a duplicate.

---

## Chat Status Bar

The chat surface carries a persistent footer-style **status bar** that shows
running session totals after every answer:

| Column      | What it shows                                                        |
| ----------- | -------------------------------------------------------------------- |
| **Tokens**  | Cumulative `prompt + completion` tokens across the session.          |
| **Cost**    | USD estimate, summed from the same price table the audit log uses.   |
| **Time**    | Wall-clock seconds spent answering, plus the per-prompt average.     |
| **Prompts** | Number of user prompts answered this session.                        |
| **Model**   | The current `provider / model` (`openai / gpt-4o-mini`, etc.).       |

Implementation in one paragraph: each call to the RAG or hybrid chain
populates a small audit record (`prompt_tokens`, `completion_tokens`,
`cost_usd`, `duration_seconds`, `provider`, `model`) on the returned
`RAGAnswer.usage`. The chat layer overrides `duration_seconds` with the
user-perceived wall-clock time and feeds it into
`app.utils.update_session_totals`, which accumulates into
`st.session_state["session_totals"]`. The bar (`render_status_bar`)
reads the same dict on every rerun so it survives reruns automatically.

The totals reset on **logout** (Streamlit clears `session_state`) and
when the **🗑 Clear chat** button is clicked. Hide the bar via the
**Show status bar** checkbox in the sidebar.

> ⚠ Costs are **estimates** based on a built-in per-1K-token price
> table — fine for dashboards, not authoritative for billing. Provider
> price changes need a one-line update to `_COST_PER_1K_TOKENS` in
> `app/utils.py`.

---

## Setup

### 1. Get a PostgreSQL ≥ 14 with pgvector

Pick whichever is easiest:

#### Option A — local Postgres in Docker

```bash
docker run -d --name corporaterag-pg \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=corporaterag \
    -p 5432:5432 \
    pgvector/pgvector:pg16
```

That's it — the `pgvector/pgvector` image ships with the extension
pre-built. Connection string:
`postgresql+psycopg://postgres:postgres@localhost:5432/corporaterag`.

#### Option B — Supabase (free tier)

1. Create a project at <https://supabase.com>.
2. **Database → Extensions → vector → Enable** (one click).
3. Get the connection string from **Project Settings → Database → Connection
   String → URI**. Replace `[YOUR-PASSWORD]` with the database password you
   set during project creation. Use the **pooler** URL (port 6543) for app
   connections.

#### Option C — Neon (free tier)

1. Create a project at <https://neon.tech>.
2. pgvector is enabled automatically. Run once on first connection:
   `CREATE EXTENSION IF NOT EXISTS vector;` (the app does this for you on
   first start if your role has `CREATE` permission).
3. Copy the connection string from the project dashboard. Append
   `?sslmode=require` if it isn't already there.

#### Option D — local Postgres without Docker

```bash
# macOS
brew install postgresql@16 pgvector

# Ubuntu
sudo apt install postgresql-16 postgresql-16-pgvector

# Then in psql:
CREATE DATABASE corporaterag;
\c corporaterag
CREATE EXTENSION vector;
```

### 2. Python environment

Python 3.11 or 3.12.

```bash
cd rag-multi-source
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

### 3. Configure `.env`

```bash
cp .env.example .env
```

Fill in at minimum:

- `DATABASE_URL` — the URL from step 1.
- `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GROK_API_KEY`).
- `ENCRYPTION_KEY` — generate with
  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.

### 4. Launch

```bash
streamlit run app/main.py
```

On first start the app:

1. Connects to Postgres with your `DATABASE_URL`.
2. Runs `CREATE EXTENSION IF NOT EXISTS vector` (idempotent).
3. Creates the tables (`users`, `user_credentials`, `ingestion_logs`,
   `user_accessible_resources`, `vector_chunks`).
4. Builds the HNSW ANN index on `vector_chunks.embedding`.

Sign up with any email, enter your Jira / Confluence / SQL credentials in
the sidebar (they're encrypted at rest with Fernet), and run an ingestion.

---

## Ingestion

### From the UI

Sidebar → ⚙️ **Ingest data** →

- **Source**: jira / confluence / sql / git / all
- **Scope**: project key, space key, db name, git branch, or `all`
- **Mode**: incremental (the only choice in the UI by design — see
  ["Mode semantics"](#mode-semantics) below).

The run is foreground with a live `st.status` progress block; you'll see
each source tick through items + vectors and a green ✓ banner when done.

### From the CLI

```bash
# Incremental ingest of a single Jira project
python -m app.ingestion.cli --source jira --mode incremental --scope PROJ \
    --email me@org.com

# Full re-ingest of one Confluence space (wipes existing chunks for that space first)
python -m app.ingestion.cli --source confluence --mode full --scope DOCS \
    --email me@org.com

# Index a Git branch (default branch when scope=all)
python -m app.ingestion.cli --source git --mode incremental --scope main \
    --email me@org.com

# Email — defaults to the previous calendar month if there's no prior log
python -m app.ingestion.cli --source email --mode incremental \
    --email me@org.com

# Email — pin to a specific calendar month
python -m app.ingestion.cli --source email --mode month --month 2025-03 \
    --email me@org.com

# Email — only one provider (outlook | gmail)
python -m app.ingestion.cli --source email --mode incremental --scope outlook \
    --email me@org.com

# Everything for everything
python -m app.ingestion.cli --source all --mode incremental --scope all \
    --email me@org.com
```

### Mode semantics

| Mode          | What happens                                                                                                                                                                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `full`        | `DELETE FROM vector_chunks WHERE source = ... AND <scope-key> IN (...)`, then re-fetch + re-embed.                                                                                                                                                |
| `incremental` | Fetches resources updated since this user's last successful run; overwrites their chunks via `INSERT ... ON CONFLICT DO UPDATE`. For the **email** source the very first run defaults to the **previous calendar month** instead of full history. |
| `month`       | **Email source only.** Combine with `--month YYYY-MM` to ingest exactly that calendar month — useful for backfills.                                                                                                                               |

In both modes, on success the user is granted access to every
`(source, resource_identifier)` they ingested.

**Full mode is intentionally not exposed in the UI.** A full ingest by a
user with a narrower view than the previous ingestor would silently delete
chunks of resources only the previous ingestor could see. Run full ingests
from the CLI on an admin account that owns the union of all scopes.

---

### Multi-modal ingestion (image captioning)

Images embedded in Confluence pages or attached to Jira tickets —
architecture diagrams, error screenshots — are invisible to the indexer
by default. When `ENABLE_MULTIMODAL_INGESTION=true`, each ingestor
discovers image attachments (Confluence: `get_attachments_from_content`;
Jira: the issue `attachment` field), captions them with a vision-capable
LLM (`core/vision.py`, `core.llm.get_vision_llm()`), and stores the
caption as one extra searchable chunk per image
(`metadata={"type": "image_caption", ...}`) — alongside the page/issue's
normal text chunks, not in a parallel pipeline.

```ini
# .env
ENABLE_MULTIMODAL_INGESTION=false   # default off — see below
VISION_MODEL=                       # optional override; falls back to the provider's normal chat model
MAX_IMAGES_PER_RESOURCE=5           # cap per Confluence page / Jira issue
MAX_IMAGE_BYTES=5000000             # ~5MB; larger images are skipped, not truncated
MAX_IMAGES_PER_INGESTION_RUN=200    # hard cap on vision-LLM calls per CLI invocation
```

- **Off by default, deliberately.** `app/ingestion/email_ingestor.py`
  already made this call for email attachments — it collects filenames
  but never downloads the bytes, with the comment "too costly + risky."
  Image captioning is the same shape of feature (fetch binary content,
  pay for a vision-LLM call per item) applied to Confluence/Jira instead;
  the same caution applies. Opt in once the cost for your own ingestion
  volume — images per resource × vision-LLM pricing — is understood.
- **Fails open per image**, never per resource or per run. A caption
  failure (bad model, oversized/corrupt image, network error) is logged
  and that one image is skipped; the resource's text chunks still ingest
  normally.
- **Two independent cost caps.** `MAX_IMAGES_PER_RESOURCE` bounds a
  single huge page/issue; `MAX_IMAGES_PER_INGESTION_RUN` bounds an
  entire `--mode full` re-ingest of a whole space/project.
- **No new dependency.** `VISION_MODEL` only needs to be set when
  `LLM_PROVIDER=grok`, since the default Grok model isn't vision-capable
  — the default OpenAI/Anthropic chat models already are.

### Email source — obtaining credentials

The email ingestor supports three providers in a single run: **Outlook /
Microsoft 365** (incl. personal `outlook.com`, `live.com`, `hotmail.com`,
and federated yahoo accounts that route through Outlook), **Gmail**, and
**Yahoo Mail** (native, via IMAP + App Password). Each provider is
independent — fill in only the section(s) you actually want to ingest.
All values are stored encrypted in `user_credentials` under
`source = "email"`.

UI: **Sidebar → 🔑 Credentials → Email → Outlook / Gmail / Yahoo
sub-form**.

#### Outlook / Microsoft 365 (e.g. `paul_r_yardley@yahoo.co.uk` linked to

outlook.com)

We use the **delegated public-client flow** with a long-lived refresh
token. This works for both organisational tenants and personal
Microsoft accounts.

1. **Register a public-client app in Microsoft Entra ID**
   ([entra.microsoft.com](https://entra.microsoft.com) → Identity → App
   registrations → New registration).
   - **Supported account types**: pick _"Accounts in any organizational
     directory and personal Microsoft accounts"_.
   - **Redirect URI**: select **Public client / native** and use
     `http://localhost`.
   - After creation, on the **Authentication** blade enable
     _"Allow public client flows"_.
2. **Grant delegated permissions** under _API permissions_:
   - `Microsoft Graph → Delegated → Mail.Read`
   - `Microsoft Graph → Delegated → offline_access` (required to receive
     a `refresh_token`).
   - Click _Grant admin consent_ if you have admin rights — otherwise
     consent will happen interactively on first sign-in.
3. **Capture a refresh token once** with a short helper script (run it
   on your laptop, paste the output into the UI). Drop the snippet
   below into `scripts/get_outlook_refresh_token.py`:

   ```python
   import msal, webbrowser
   CLIENT_ID = "<application-id-from-step-1>"
   AUTHORITY = "https://login.microsoftonline.com/common"
   SCOPES    = ["Mail.Read", "offline_access"]
   app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
   flow = app.initiate_device_flow(scopes=SCOPES)
   print(flow["message"])              # → open URL, enter code
   webbrowser.open(flow["verification_uri"])
   result = app.acquire_token_by_device_flow(flow)
   print("REFRESH TOKEN:\n", result["refresh_token"])
   ```

   `pip install msal && python scripts/get_outlook_refresh_token.py`,
   sign in as `paul_r_yardley@yahoo.co.uk`, approve the prompt, and copy
   the printed refresh token.

4. **Paste into the UI** under _Outlook / Microsoft 365_:
   - **Tenant ID** → `common` (use your tenant GUID for org accounts only)
   - **Client (Application) ID** → from step 1
   - **Client Secret** → leave blank (public-client flow doesn't need one)
   - **Refresh Token** → from step 3
   - **Mailbox UPN** → leave blank (the `/me` endpoint resolves itself)
   - **Folder filter** → e.g. `inbox` to limit to Inbox; blank = all mail

#### Gmail (e.g. `pr.yardley@gmail.com`)

We use the standard **installed-app OAuth 2.0 flow** to obtain a
refresh token, then store either the discrete keys or the full
`token.json` blob.

> **Note on Google's UI (current as of April 2026).** The OAuth
> configuration moved out of the legacy "APIs & Services → OAuth
> consent screen" page into a dedicated **Google Auth Platform**
> console with sub-tabs **Overview / Branding / Audience / Clients /
> Data access / Verification centre / Settings**. The instructions
> below match this layout. The legacy single-page UI is still served
> for some older projects and works the same way conceptually — just
> with everything on one screen.

1. **Enable the Gmail API.**
   In the Cloud Console for your project go to
   _APIs & Services → Library_ → search "Gmail API" → **Enable**.

2. **Configure the OAuth consent screen.**
   Open _Google Auth Platform_ (the shield icon in APIs & Services, or
   directly from the App selector — once configured for the project
   it sits at the top-level left nav).
   - **Branding** — set an app name (e.g. `CorporateRAG-Gmail`),
     a user-support email, and a developer contact email.
   - **Audience** — keep **User type: External** unless your project
     belongs to a Google Workspace org and you only need internal users
     (in which case "Make internal" gives long-lived refresh tokens).
   - **Test users** (only matters for External, Testing-mode apps).
     - You **do not need to add yourself if you own the Cloud project**
       — project Owners/Editors are implicitly allow-listed. Adding
       them here is rejected with a misleading "Ineligible accounts"
       error; that's expected, just skip the step.
     - Add only _other_ Gmail addresses you want to ingest from while
       in Testing mode (max 100). The address must be a real Google
       Account in lowercase canonical form (Gmail's dot-insensitivity
       does **not** apply here).
   - **Data access** — you don't need to pre-list scopes while in
     Testing; the helper script declares `gmail.readonly` dynamically
     and the consent screen surfaces it on first sign-in. (You only
     need to add it explicitly if you ever submit for verification.)

3. **Create the OAuth client.**
   _Google Auth Platform → Clients → + Create client_:
   - **Application type: Desktop app**
   - Give it a name (e.g. `CorporateRAG Desktop`).
   - On creation, the dialog shows the **Client ID** and a
     one-time **Client secret**. Click **Download JSON** and save it
     to the repo root as `client_secret.json` (gitignored).

   > 🔁 **Rotating the secret later.** Google no longer offers a
   > "Reset secret" button or a re-downloadable JSON. To rotate:
   > _Clients → \[your client] → Client secrets → **+ Add secret**_.
   > A row appears with the **plaintext shown exactly once** —
   > copy it immediately ("Viewing and downloading client secrets is
   > no longer available"). You can hold at most **two** active
   > secrets, so to rotate again you must **Disable** then **Delete**
   > the old one. Update `client_secret.json` by hand: open the file
   > and replace the `installed.client_secret` value with the new
   > `GOCSPX-…` string; everything else (`client_id`, `project_id`,
   > `auth_uri`, `token_uri`, `redirect_uris`) stays the same.

4. **Run the one-time installed-app flow** to mint a refresh token.
   A ready-made helper is included at
   [`scripts/gmail_get_refresh_token.py`](scripts/gmail_get_refresh_token.py):

   ```bash
   # google-auth-oauthlib is already in requirements.txt
   python scripts/gmail_get_refresh_token.py
   ```

   Useful flags:
   - `--client-secret <path>` if your JSON isn't at `./client_secret.json`
   - `--out token.json` to also write the resulting blob to disk
   - `--port <n>` to pin the loopback redirect to a specific port

   The browser opens automatically — sign in as the mailbox owner
   (e.g. `pr.yardley@gmail.com`), approve **Read your email**, and the
   tab closes itself. The terminal then prints two blocks:

   ```
   ============================================================
   REFRESH TOKEN (paste into 'Refresh Token' field in the sidebar):
   ============================================================
   1//09…
   ============================================================
   FULL token.json (paste into 'Full token.json (optional)' instead,
   if you'd rather use a single field):
   ============================================================
   {
     "token": "ya29.…",
     "refresh_token": "1//09…",
     …
   }
   ```

   The script forces `prompt=consent` + `access_type=offline`, so a
   refresh token is guaranteed even on re-runs.

5. **Paste into the UI** under _Sidebar → 🔑 Credentials → Email →
   📬 Gmail_. Pick **one** of the two routes — don't fill both:

   | Route                         | Fields to populate                                                              |
   | ----------------------------- | ------------------------------------------------------------------------------- |
   | **Single blob (recommended)** | only **Full token.json (optional)** — embeds client id/secret/scope/`token_uri` |
   | **Discrete keys**             | **OAuth Client ID**, **OAuth Client Secret**, **Refresh Token**                 |

   Optional in either case:
   - **Mailbox address** → blank or `me` (resolves to the authenticated
     account)
   - **Label filter** → e.g. `INBOX` to limit to Inbox; blank = all mail

   > 🧹 **Clearing a saved secret.** Each saved secret field renders a
   > "Clear saved …" checkbox above the input. Tick it, leave the
   > input blank, click **Save credentials** to delete the value.
   > Typing a new value always wins over the Clear tick.

6. **Token longevity.** While the consent screen is in **Testing**
   mode and the user is `@gmail.com` (not Workspace-internal),
   refresh tokens expire after **7 days of inactivity**. If a future
   ingestion fails with `invalid_grant`, just re-run the helper script
   and update the credential. To get long-lived tokens click
   **Publish app** under _Audience_ — for `gmail.readonly` on a
   personal account with no other users, Google won't actually require
   verification.

#### Yahoo Mail (e.g. `pr.yardley@yahoo.com`)

Yahoo deprecated public OAuth2 access to Mail content for new
third-party apps — the Yahoo Developer Console no longer offers a
**Mail (read)** permission, and `mail-r`-scoped tokens are silently
rejected at the IMAP gateway. Yahoo's
[official IMAP documentation](https://uk.help.yahoo.com/kb/new-yahoo-mail/imap-server-settings-yahoo-mail-sln4075.html)
now instructs users to authenticate with a **mailbox-scoped 16-character
app password** instead. We follow that path: plain
[`imaplib.IMAP4_SSL.login()`](https://docs.python.org/3/library/imaplib.html#imaplib.IMAP4.login)
against `imap.mail.yahoo.com:993`, no helper script, no refresh-token
dance, no Yahoo Developer Console.

1. **Enable 2-step verification** (one-time, prerequisite for app
   passwords). Sign in to Yahoo, click your avatar →
   **Account info → Account Security**, scroll to _2-step
   verification_ → **Turn on** → verify with SMS or an authenticator
   app of your choice.

2. **Generate the app password.** On the same Account Security page
   (Yahoo's current UI as of 2026):
   1. Sign in to your [Yahoo Account Security page](https://login.yahoo.com/myaccount/security).
   2. Under **External connections**, click **Create app password**.
      (On older accounts the section may be labelled _Other ways to
      sign in → Generate app password_ — it's the same feature.)
   3. Enter your app's name in the text field — anything goes, e.g.
      `CorporateRAG`. The string is cosmetic; Yahoo doesn't validate
      it against any registry.
   4. Click **Generate password**.
   5. Yahoo displays a 16-character one-time password in groups of
      four (e.g. `xxxx xxxx xxxx xxxx`). **Copy it now.** It's shown
      exactly once; you cannot re-fetch it later, only revoke and
      regenerate.
   6. Click **Done**.

   You can manage / delete app passwords from the same screen later;
   each is independent of every other and of your account password.

3. **Smoke-test before saving** (optional but recommended). One-liner
   from the project venv:

   ```powershell
   python -c "import imaplib; m=imaplib.IMAP4_SSL('imap.mail.yahoo.com',993,timeout=30); m.login('you@yahoo.com','xxxxxxxxxxxxxxxx'); print(m.select('INBOX',readonly=True)); m.logout()"
   ```

   Expected: `('OK', [b'<message-count>'])`. If you get
   `imaplib.IMAP4.error: AUTHENTICATE failed.`, the password is wrong
   or 2-step verification was disabled (which auto-revokes app
   passwords). Spaces in the displayed password are cosmetic — IMAP
   accepts it with or without them, and the sidebar form strips them
   on save anyway.

4. **Paste into the UI.** _Sidebar → 🔑 Credentials → Email →
   📭 Yahoo Mail (IMAP + App Password)_:

   | Field                                  | Value                                                                |
   | -------------------------------------- | -------------------------------------------------------------------- |
   | **Mailbox address**                    | the Yahoo email you generated the password for, e.g. `you@yahoo.com` |
   | **Yahoo App Password (16 characters)** | the password Yahoo showed you (with or without spaces)               |
   | **IMAP folder**                        | leave blank for `INBOX`, or e.g. `Sent`, `Bulk Mail`, `Archive`      |

   Click **Save credentials**.

5. **Run ingestion.** Sidebar → ⚙️ Ingest data → Source: `email`,
   Scope: `yahoo` → **Run ingestion**. Or via CLI:

   ```bash
   python -m app.ingestion.cli \
       --source email --mode incremental --scope yahoo \
       --email <your CorporateRAG account email>
   ```

   Healthy log signature:

   ```
   [email] Yahoo provider registered.
   [email/yahoo] selected folder INBOX (1234 messages total)
   [email/yahoo] 47 messages match window ['SINCE', '1-Mar-2026', 'BEFORE', '2-Apr-2026']
   ```

   After completion, `yahoo` appears under **Email mailboxes** and
   under **🛡 Manage my access → email**, so it can be selected as a
   query scope or revoked independently of Outlook / Gmail.

> 🔁 **No helper script for Yahoo.** Earlier revisions of this repo
> shipped `scripts/yahoo_get_refresh_token.py` to drive an OAuth2
> consent flow against Yahoo. That script has been **removed** because
> Yahoo no longer issues `mail-r` tokens to new apps. If you have a
> legacy Yahoo Developer app that still has the permission and want
> XOAUTH2 instead of an app password, restore the previous revision
> of [`app/ingestion/email_ingestor.py`](app/ingestion/email_ingestor.py)
> from git history — but for any new account, the app-password flow
> above is the supported path.

> 🔐 **Credential storage.** Outlook / Gmail refresh tokens and the
> Yahoo app password are all encrypted at rest with the same Fernet
> key (`ENCRYPTION_KEY`) used for every other connector credential.
> Treat them like passwords — anyone holding a refresh token can read
> the mailbox until it's revoked from the provider's account-security
> page; the same applies to the Yahoo app password (revoke at
> _Account Security → External connections → Delete app password_).

---

## Project layout

```
rag-multi-source/
├── README.md                             # ← you are here
├── requirements.txt
├── .env.example
├── .streamlit/config.toml                # watcher tweaks + hide Deploy button
├── app/
│   ├── main.py                           # Streamlit entry point
│   ├── auth.py                           # email + bcrypt login
│   ├── config.py                         # pydantic-settings
│   ├── chat.py                           # builds dynamic filter, calls retriever + LLM
│   ├── sidebar.py                        # source toggles + scope pickers + ingest UI
│   ├── utils.py                          # DB session, Fernet, credential & access helpers
│   └── ingestion/
│       ├── base.py                       # BaseIngestor — chunking, dedup, full vs incremental
│       ├── jira_ingestor.py
│       ├── confluence_ingestor.py
│       ├── sql_ingestor.py
│       ├── git_ingestor.py
│       ├── email_ingestor.py                 # Outlook (Graph) + Gmail
│       └── cli.py
├── core/
│   ├── llm.py                            # embeddings + LLM factory
│   ├── vector_store.py                   # ★ pgvector upsert / delete / filter builder
│   ├── retriever.py                      # ★ pgvector cosine ANN query, filter-aware
│   └── rag_chain.py                      # prompt + LLM call + citation prep
├── models/
│   ├── base.py
│   ├── user.py
│   ├── ingestion_log.py
│   ├── user_accessible_resource.py
│   ├── vector_chunk.py                   # ★ pgvector-backed chunk table
│   └── migrations.py                     # idempotent schema + extension bootstrap
└── scripts/
    └── vector_store_admin.py             # report / purge-source / nuke
```

(★ = the three modules where vector storage lives.)

---

## Recipes System — Add New Sources Easily

The **Recipes system** is a lightweight, declarative layer on top of the
existing `BaseIngestor` architecture. It lets you add a new corporate data
source — Notion, ServiceNow, Linear, an internal REST API, a static folder
of Markdown, anything that returns JSON or text — by writing a **single
YAML file** instead of a full Python ingestor.

### Why recipes?

The classic ingestors (`jira_ingestor.py`, `confluence_ingestor.py`, …) are
~250 lines each. Most of that code is boilerplate: HTTP auth, pagination,
mapping JSON fields onto `SourceResource`, computing a stable
`resource_id`. Recipes capture that boilerplate as configuration.

| Feature                              | Without recipes               | With recipes              |
| ------------------------------------ | ----------------------------- | ------------------------- |
| Add a simple REST source             | New Python class (~150 lines) | New `.yaml` file (~30 lines) |
| Try a new mapping locally            | Edit + reimport               | Edit YAML, rerun           |
| Share a connector with the community | Fork + PR                     | Drop a YAML in `recipes/` |
| Reuse complex Python logic           | n/a — already a class         | Recipe references the class |

Existing hardcoded ingestors keep working unchanged. Recipes are
**purely additive**.

### Where recipes live

```
rag-multi-source/
├── recipes/
│   ├── __init__.py
│   ├── recipe.py                 # Recipe dataclass + validation
│   ├── registry.py               # Auto-discovery of YAML / .py recipes
│   ├── example_jira.yaml         # Re-implements the Jira ingestor declaratively
│   └── (your_source.yaml)        # Drop new YAML files here
└── app/
    └── ingestion/
        └── recipe_runner.py      # Loads a recipe and runs it
```

Drop a `.yaml` file into `recipes/` (or any subfolder) and it appears
automatically in:

- The CLI: `python -m app.ingestion.cli --list-recipes`
- The Streamlit sidebar: under **⚙️ Ingest data → Recipe**

No registration step, no config edit.

### Recipe YAML schema

```yaml
# Minimum recipe — declarative, no Python code needed.

name: notion_pages              # Unique identifier (kebab/snake_case).
source: notion                  # Logical source key. Goes into vector_chunks.source.
description: "Notion pages by workspace database."
credential_key: notion          # Key into user_credentials (`source` column).

# How user-selected scope translates to a filter column on vector_chunks.
# Use one of the existing promoted fields:
#   project_key | space_key | db_name | git_scope | email_provider
# …or "external_id" for sources that don't fit the existing buckets.
scope_field: external_id
scope_label: "Notion workspace"

# Either delegate to a Python class …
parser: app.ingestion.notion_ingestor.NotionIngestor

# … or use the built-in generic HTTP/JSON parser (parser: builtin).
config:
  fetch:
    type: http_json
    url_template: "https://api.notion.com/v1/databases/{scope}/query"
    method: POST
    auth: bearer                        # bearer | basic | none
    auth_token_field: integration_token # which credential field to send
    headers:
      Notion-Version: "2022-06-28"
    pagination:
      type: cursor                      # cursor | page | none
      next_token_field: next_cursor
      items_field: results
      page_size: 100

  resource_id_template: "notion:{id}"
  title_template:        "{properties.Name.title.0.plain_text}"
  text_template: |
    # {properties.Name.title.0.plain_text}

    {properties.Body.rich_text.0.plain_text}
  url_template:           "https://notion.so/{id}"
  last_updated_field:     "last_edited_time"
  metadata_fields:
    object_name: "id"
    page_status: "properties.Status.select.name"

# Optional chunking overrides (defaults to settings.CHUNK_SIZE / OVERLAP).
chunk_size: 800
chunk_overlap: 150
```

The Notion example above is illustrative — for a real, working `parser:
builtin` recipe you can actually run, see
[`recipes/github_issues.yaml`](recipes/github_issues.yaml). It also
demonstrates reusing an existing source's stored credential
(`credential_key: git`) so a new source doesn't always need its own
credential form.

#### Field reference

| Key                         | Required | Description                                                                                  |
| --------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `name`                      | yes      | Unique recipe id. Defaults to the YAML filename (stem) if omitted.                           |
| `source`                    | yes      | The logical source key (`vector_chunks.source`). Multiple recipes can share a source.        |
| `description`               | no       | Shown in CLI listings and the sidebar dropdown.                                              |
| `credential_key`            | yes      | The `user_credentials.source` row to load. Often equals `source`.                            |
| `scope_field`               | yes      | Which promoted column the user-supplied scope maps to.                                       |
| `scope_label`               | no       | Friendly label for the sidebar scope picker.                                                 |
| `parser`                    | yes      | Either `"builtin"` or `"<module_path>.<ClassName>"` (subclass of `BaseIngestor`).            |
| `python_module`             | no       | Alternative to `parser` — load a `.py` file in `recipes/` and use its exported `recipe`/class. |
| `config`                    | when builtin | Builtin-parser config (see below).                                                       |
| `chunk_size` / `chunk_overlap` | no    | Overrides the global chunking settings for this recipe only.                                 |

#### Builtin parser config (`parser: builtin`)

| Key                          | Description                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `fetch.type`                 | `http_json` (REST) or `static` (a list embedded in the recipe — handy for testing).            |
| `fetch.url_template`         | URL with `{credential_field}` and `{scope}` placeholders.                                      |
| `fetch.method`               | `GET` or `POST` (default `GET`).                                                               |
| `fetch.auth`                 | `basic` (uses `email`/`api_token`), `bearer` (uses `auth_token_field`), or `none`.             |
| `fetch.headers`              | Static headers added to every request.                                                         |
| `fetch.pagination`           | `none`, `cursor` (next_token_field), or `page` (start_field, page_size).                       |
| `fetch.items_field`          | Path to the array of items in the response (e.g. `"results"`, `"data.items"`).                 |
| `fetch.body_template`        | (POST only) Optional JSON body template.                                                       |
| `resource_id_template`       | Format string. Field paths use dot notation: `{id}`, `{fields.summary}`, `{a.b.0.c}`.          |
| `title_template`             | Likewise. Falls back to `resource_id` when blank.                                              |
| `text_template`              | Multi-line template used as the chunk text.                                                    |
| `url_template`               | URL to display in citations.                                                                   |
| `last_updated_field`         | Path to the ISO-8601 timestamp used for incremental cutoff.                                    |
| `metadata_fields`            | Dict mapping `metadata_key → field_path`. Anything mapped to `object_name` / `title` / `url` is promoted to its own column on `vector_chunks` automatically. |
| `incremental_param`          | Name of the query parameter the API expects for "since" filtering (optional).                  |

### Three ways to author a recipe

1. **Pure YAML (`parser: builtin`)** — the fast path. Good for any source
   whose REST API returns JSON arrays.
2. **YAML pointing at a Python class (`parser: app.ingestion.…`)** — wraps
   an existing `BaseIngestor` subclass so it shows up in the recipe list.
   This is how the bundled `example_jira.yaml` works.
3. **A `.py` file in `recipes/`** — exports a top-level `recipe = Recipe(…)`
   plus optionally a `BaseIngestor` subclass. Useful when the parser logic
   needs Python (HTML scraping, custom auth) but you still want it to be
   discoverable like a YAML recipe.

### CLI usage

```bash
# List every available recipe (built-in or user-authored).
python -m app.ingestion.cli --list-recipes

# Run a recipe by name.
python -m app.ingestion.cli --recipe notion_pages \
    --mode incremental --scope my-workspace-id \
    --email me@org.com

# Recipes coexist with the legacy --source flag — pick whichever you prefer.
python -m app.ingestion.cli --source jira --mode incremental --scope PROJ
```

### Backward compatibility

- Every existing ingestor (`JiraIngestor`, `ConfluenceIngestor`, …) works
  exactly as before. Recipes never replace them — they wrap them.
- The legacy `--source jira|confluence|sql|git|email` CLI flag continues
  to function and is the default path.
- `BaseIngestor` accepts a new optional `recipe` keyword. Subclasses that
  don't pass one behave identically to the previous version.
- The Streamlit sidebar still shows the existing source toggles + scope
  pickers. The Recipe selector is an *additional* control alongside
  them, not a replacement.

See the **Next Steps** at the end of this file for a worked example of
authoring a new recipe.

---

## Operations

### Inspect the index

```bash
python -m scripts.vector_store_admin --strategy report
```

Output:

```
INFO     vector_chunks per-source counts: {'confluence': 64, 'git': 213, 'jira': 12, 'sql': 13, '__total__': 302}
INFO     Storage: table=4128 kB HNSW index=2752 kB
```

### Purge a single source

```bash
python -m scripts.vector_store_admin --strategy purge-source --source jira
```

### Wipe everything (last resort)

```bash
python -m scripts.vector_store_admin --strategy nuke \
    --confirm-i-know-what-im-doing
```

After a nuke every user must re-ingest before the chat can answer
anything.

### Useful ad-hoc queries

```sql
-- Top-10 most-chunked resources
SELECT resource_id, COUNT(*) AS chunks
FROM vector_chunks
GROUP BY resource_id
ORDER BY chunks DESC
LIMIT 10;

-- Who has access to what?
SELECT u.email, uar.source, uar.resource_identifier, uar.last_synced
FROM user_accessible_resources uar
JOIN users u ON u.id = uar.user_id
ORDER BY u.email, uar.source, uar.resource_identifier;

-- Which scopes are indexed but nobody has access to (orphans)?
SELECT vc.source, COALESCE(vc.project_key, vc.space_key, vc.db_name, vc.git_scope) AS scope,
       COUNT(*) AS chunks
FROM vector_chunks vc
LEFT JOIN user_accessible_resources uar
  ON uar.source = vc.source
 AND uar.resource_identifier =
       COALESCE(vc.project_key, vc.space_key, vc.db_name, vc.git_scope)
WHERE uar.id IS NULL
GROUP BY vc.source, scope;
```

---

## Free-tier friendly

- **PostgreSQL** — free local Docker / Supabase free tier (500 MB DB,
  2 GB bandwidth) / Neon free tier (3 GB storage, autoscale-to-zero).
- **OpenAI `text-embedding-3-small`** at 1536 dims is cheap (~$0.02 / 1M
  tokens). Swap to `EMBEDDINGS_PROVIDER=huggingface` for zero cost (slower).
- **Streamlit** for the UI.
- **Fernet** for encrypted credential storage.

---

## Security notes

- User passwords are bcrypt-hashed (12 rounds).
- Source credentials (`api_token`, `conn_str`, …) are Fernet-encrypted at
  rest in `user_credentials.encrypted_value` using the `ENCRYPTION_KEY`
  env var.
- The `vector_chunks` table has no `user_id` column; access is enforced
  solely by the `WHERE` clause built from `user_accessible_resources`.
  Treat that table as the security boundary — anyone able to write into it
  can grant themselves access to any resource that's been ingested.

### Access model — current behaviour

Access is **granted at ingestion time**, not **re-checked at query time
against the source system**. Concretely:

- A user gains access to a scope (`jira:PROJ`, `confluence:DOCS`,
  `sql:mydb`, `git:owner/repo@main`) the moment they successfully run an
  ingestion under their own account. The ingestor calls
  `grant_access(user_id, source, resource_identifier)` for every resource
  it processes.
- That access **persists until they revoke it** via the sidebar's
  _🛡 Manage my access_ panel. If their permissions are later removed at
  the source, CorporateRAG won't notice — they'll continue to retrieve
  any chunks already ingested for that scope.
- Conversely, a user who has full access at the source but has never run
  an ingestion themselves cannot query the data — there's no
  `user_accessible_resources` row for the retriever to use.

This is still the default behaviour. An opt-in hardening layer —
[live source-system ACL re-validation](#3-live-source-system-acl-re-validation) —
closes the "permissions changed at the source" half of this gap by
calling back into Jira/Confluence/SQL Server/GitHub on every query.

---

## Security & Deduplication Enhancements

Three upgrades sit on top of the original multi-tenancy model:

1. **Content fingerprint deduplication** — every chunk carries a
   SHA-256 hex digest of its normalised body. The upsert path uses
   the fingerprint to skip re-embedding any row whose content hasn't
   changed since the previous ingest. Embedding API calls dominate
   the cost of an incremental re-ingest, so dedup pays for itself
   the first time you re-run an ingestor against an unchanged corpus.

2. **PostgreSQL Row-Level Security** as defence-in-depth on top of
   the existing application-layer `WHERE` clause. The database
   itself refuses to surface chunks the active user wasn't granted —
   a regression in `filter_to_where`, an admin running a naked
   `SELECT *` from psql against the wrong role, or a future MCP tool
   that forgets to apply the access table all fail closed.

3. **Live source-system ACL re-validation** (opt-in) — on every chat
   turn, intersect the granted scopes with a live permission check
   against Jira, Confluence, SQL Server, or GitHub, closing the
   "permissions changed at the source but `user_accessible_resources`
   wasn't re-synced" gap that (1) and (2) don't address.

### 1. Content fingerprint deduplication

A `content_fingerprint CHAR(64)` column on `vector_chunks` stores a
SHA-256 hex digest of the chunk text after normalisation
(lower-cased, whitespace-collapsed, NUL-stripped). The fingerprint is
computed in `BaseIngestor._chunk()` and consumed in
`core.vector_store.upsert_chunks()`:

```python
# What the upsert path does for every batch
existing = _existing_fingerprints(db, [(c.resource_id, c.chunk_index) for c in batch])
to_process = [c for c in batch if existing.get((c.resource_id, c.chunk_index)) != c.content_fingerprint]
# Only chunks that survived the dedup are sent to OpenAI / HF for embedding.
```

Properties:

- **Stable across re-ingests.** Whitespace churn (CRLF↔LF, double
  spaces, trailing newlines) doesn't change the hash, so a Confluence
  page re-rendered through markdownify with slightly different
  spacing still dedups against itself.
- **Cross-source aware.** The same paragraph copy-pasted into Jira
  and Confluence hashes to the same digest — useful diagnostic for
  finding accidental content duplication across systems.
- **Cheap.** SHA-256 over a 1 KB chunk is microseconds; the saving
  is one OpenAI embedding API hit per skipped row.
- **Indexed.** A partial index `ix_vector_chunks_content_fingerprint`
  covers `WHERE content_fingerprint IS NOT NULL` for ad-hoc dedup
  audits (`SELECT content_fingerprint, COUNT(*) FROM vector_chunks
  GROUP BY 1 HAVING COUNT(*) > 1`).
- **Opt-in to disable.** Set `ENABLE_CONTENT_FINGERPRINT_DEDUP=false`
  in `.env` to force re-embeds on every upsert (useful for
  benchmarking embedding latency).

When fingerprint dedup is on, incremental re-ingests skip the
per-resource `delete_resource()` call and instead use a narrower
`_prune_orphan_tail_chunks()` that only removes chunk rows whose
`chunk_index >= len(new_chunks)`. This preserves "shrinking
documents leave no orphans" semantics without nuking rows that the
dedup short-circuit could have saved.

### 2. PostgreSQL Row-Level Security (RLS)

The migration runner enables RLS on `vector_chunks` and
`user_accessible_resources` and installs SELECT / INSERT / UPDATE /
DELETE policies keyed on a per-session GUC,
`app.current_user_id`. The application binds the GUC immediately
after opening any tenancy-sensitive session via
`app.utils.set_current_user_for_rls(db, user_id)` —
implemented as `SET LOCAL` so the binding is automatically reset at
COMMIT/ROLLBACK and never leaks across pooled connections.

```sql
-- vector_chunks SELECT policy (installed automatically)
CREATE POLICY vector_chunks_tenant_select ON vector_chunks
FOR SELECT USING (
    current_setting('app.current_user_id', true) IS NOT NULL
    AND EXISTS (
        SELECT 1 FROM user_accessible_resources uar
        WHERE uar.user_id = current_setting('app.current_user_id', true)
          AND uar.source  = vector_chunks.source
          AND uar.resource_identifier = COALESCE(
              vector_chunks.project_key,
              vector_chunks.space_key,
              vector_chunks.db_name,
              vector_chunks.git_scope,
              vector_chunks.email_provider
          )
    )
);
```

The application-layer `WHERE` clause stays in place — RLS is purely
additional. If both layers agree, performance is identical (the
planner short-circuits the EXISTS check when the WHERE clause has
already eliminated rows). If they ever disagree, RLS wins and the
chat surfaces an empty result instead of leaking a chunk.

#### Enable / disable

RLS is **on by default**. Toggle with:

```ini
# .env
ENABLE_RLS=true   # default — install + enforce policies
ENABLE_RLS=false  # legacy / admin-tool deployments — skip the policy install
```

When `ENABLE_RLS=false`:

- The migration runner skips `_apply_rls_policies()` entirely.
- `set_current_user_for_rls()` becomes a no-op, so existing call
  sites have zero overhead.
- Existing deployments upgrade with **zero behavioural change** —
  the new column is created, but tenancy stays application-only.

To back the policies out of an already-migrated database (e.g.
because `scripts/vector_store_admin.py` is being run from a non-app
DB role that doesn't set the GUC):

```python
from app.utils import engine
from models.migrations import drop_rls_policies
drop_rls_policies(engine)
```

#### Verifying it works

After enabling, a quick psql session as the app's DB role:

```sql
-- Without binding the GUC: zero rows (RLS denying)
SELECT COUNT(*) FROM vector_chunks;
-- count: 0

-- After binding to a real user: their accessible chunks only
SELECT set_config('app.current_user_id', '<their-uuid>', true);
SELECT COUNT(*) FROM vector_chunks;
-- count: <their actual count>

-- A user_id with no grants returns zero
SELECT set_config('app.current_user_id', '00000000-0000-0000-0000-000000000000', true);
SELECT COUNT(*) FROM vector_chunks;
-- count: 0
```

#### Operational notes

- **Admin scripts** (`scripts/vector_store_admin.py`) connect as the
  same DB role as the app. If you need to run `--strategy report` /
  `purge-source` / `nuke` against a database with RLS on, either run
  the script as a Postgres superuser (RLS is bypassed for `BYPASSRLS`
  and superuser roles) or temporarily unset the policies via
  `drop_rls_policies(engine)` from a Python REPL.
- **Connection pooling.** `set_current_user_for_rls()` uses
  `set_config(..., true)` which is the SQL equivalent of `SET
  LOCAL` — transaction-scoped, automatically reset at COMMIT /
  ROLLBACK. Safe with PgBouncer in transaction mode.
- **Monitoring.** Every policy is named `*_select / *_insert /
  *_update / *_delete` so `SELECT * FROM pg_policies` gives an
  auditor a one-screen view of what's enforced.

### 3. Live source-system ACL re-validation

`user_accessible_resources` is granted once at ingestion time and never
re-checked (see [Access model — current behaviour](#access-model--current-behaviour)).
When `LIVE_ACL_REVALIDATION_ENABLED=true`, `core/live_acl.py` closes that
gap by calling back into the source system itself on every chat turn and
intersecting the live permission with the granted scope:

| Source | Live check |
|---|---|
| Jira | `POST /rest/api/3/permissions/project` (`BROWSE_PROJECTS`) |
| Confluence | `GET /wiki/rest/api/space/{spaceKey}` against each configured instance |
| SQL Server | `SELECT HAS_DBACCESS(:db_name)` |
| Git / GitHub Issues | `GET /repos/{owner}/{repo}/collaborators/{login}/permission` |

Wired into two call sites:

- **Chat retrieval** — `app/chat.py`'s `live_acl_check` step prunes
  `filter_dict["by_source"]` after `build_filter`, before the vector
  search runs.
- **Live SQL via MCP** — `mcp_server/tools/sql_tools.py`'s `table_query`
  and `list_databases` re-validate before executing or advertising a
  database, since a stale grant there means live query execution
  against a real database, not just stale indexed chunks.

Properties:

- **Fails closed.** Any checker error (timeout, network, unreachable
  host, incomplete/revoked credential) excludes that scope from the
  turn's results rather than falling back to the ingestion-time grant.
  A slow or unreachable source system shrinks that source's results
  instead of blocking chat.
- **Cached.** Verdicts are cached in-process per
  `(user_id, source, resource_identifier)` for
  `LIVE_ACL_CACHE_TTL_SECONDS` (default 300s) so a multi-turn chat
  session doesn't re-hit the source API on every message.
- **Off by default.** This adds a per-query API round trip per distinct
  scope not already cached — opt in once the source systems' rate
  limits are understood for your deployment:

  ```ini
  # .env
  LIVE_ACL_REVALIDATION_ENABLED=true
  LIVE_ACL_CACHE_TTL_SECONDS=300
  LIVE_ACL_HTTP_TIMEOUT_SECONDS=5.0
  ```

- **No checker, no-op.** Sources without a registered checker (e.g.
  `email`) pass through unchanged — the feature only ever narrows
  results for sources it knows how to verify.

---

## Audit & Performance Logging

Every chat prompt produces two pieces of durable telemetry:

1. **`query_audit_logs`** — one high-level row per user prompt:
   `id`, `user_id`, `timestamp`, `prompt_text` (truncated), `total_duration_ms`,
   `tokens_prompt`, `tokens_completion`, `estimated_cost_usd`, `llm_provider`,
   `llm_model`, `success`, `error_message`, `source_type`
   (`"rag"` | `"hybrid"` | `"mcp_only"`), `fts_language` (`"english"` |
   `"simple"` | `NULL` — whichever the sidebar's "🔤 Search language" picker
   was set to for that turn; `NULL` for rows logged before this column
   existed), plus a nine-column snapshot of the feature flags that were
   live in `app.config.settings` when the row was written —
   `response_cache_enabled`, `entity_graph_enabled`,
   `query_rewrite_enabled`, `multimodal_ingestion_enabled`,
   `rerank_enabled`, `corrective_retrieval_enabled`,
   `live_acl_revalidation_enabled`, `sql_dependency_graph_enabled`,
   `sql_dependency_mcp_tools_enabled` (all `BOOLEAN NULL`, populated by
   `app.utils.log_query_audit` reading `settings` directly rather than
   being passed in, since — unlike `fts_language` — they're global env
   config, not a per-turn user choice; `NULL` for rows logged before
   these columns existed). Lets a later look back at an old answer
   correlate its behaviour with which optional pipeline stages were
   actually active, without cross-referencing `.env` history.

2. **`query_step_timings`** — N child rows per audit entry, one per measured
   step in the answer pipeline. Each row carries `step_name`, `duration_ms`,
   and a JSONB `metadata` blob with step-specific extras (retrieved-chunk
   counts, MCP row counts, hop indices, token counts per LLM hop, …).

### What gets measured

The pipeline is timed end-to-end with `time.perf_counter()` from the chat
layer. Steps emitted on a typical RAG turn:

```
build_filter          -- translate sidebar selection into the metadata filter
live_acl_check        -- (opt-in) intersect with a live source-system permission check
response_cache_check  -- (opt-in) check for a cached answer; a hit skips every step below
query_rewrite         -- (opt-in) decompose the question into sub-queries
vector_retrieval      -- pgvector top-K cosine + de-dup (one call per sub-query, pooled)
rerank                -- (opt-out) cross-encoder second pass over the candidates
build_context         -- group hits by resource, build numbered LLM context
llm_invocation        -- single llm.invoke(messages)
post_processing       -- extract text + assemble the audit record
total                 -- mirror of total_duration_ms (no JOIN required)
```

`live_acl_check` is a single dict lookup (sub-millisecond) unless
`LIVE_ACL_REVALIDATION_ENABLED=true` — see
[Live source-system ACL re-validation](#3-live-source-system-acl-re-validation)
below. `response_cache_check` is a no-op unless
`RESPONSE_CACHE_ENABLED=true` — see
[Semantic response cache](#semantic-response-cache) above; on a hit,
`query_rewrite` / `vector_retrieval` / `rerank` don't run at all (they
simply won't appear in that turn's step list), and the audit row shows
`cost_usd = 0`. `query_rewrite` is a single-item list return
(sub-millisecond) unless `QUERY_REWRITE_ENABLED=true` — see
[Query rewriting before retrieval](#query-rewriting-before-retrieval-multi-query-decomposition)
above. `rerank` runs the cross-encoder by default — see
[Reranking with a cross-encoder](#reranking-with-a-cross-encoder) above;
set `RERANK_ENABLED=false` to make it a no-op slice instead.

The hybrid (MCP) path additionally emits one `llm_invocation` per hop
(with `metadata.hop = N`) and one `mcp_tool_call:<tool_name>` per
round-trip, with `metadata.row_count` / `metadata.server_duration_ms`
copied from the MCP-server response. The direct-SELECT fast path skips
the LLM entirely and emits a single `mcp_tool_call:sql_table_query` step
with `metadata.fast_path = true`.

### Cost estimation

Token-to-USD conversion uses the rate table on
`settings.LLM_COST_PER_1K_TOKENS` in [`app/config.py`](app/config.py).
Lookups are case-insensitive, longest-key-wins substring matches, so
`"gpt-4o-mini-2025-08-07"` resolves to the `gpt-4o-mini` rate without
the table needing to know about every dated snapshot. Rates ship for
**OpenAI** (GPT-4o, GPT-4.1, GPT-4 turbo, GPT-3.5, o1/o3 reasoning),
**Anthropic** (Claude 4 Opus / Sonnet / Haiku, 3.5 / 3 family), and
**xAI Grok** (grok-2, grok-beta). When a (provider, model) pair isn't
in the table, `LLM_DEFAULT_COST_PER_1K_TOKENS` is used as a
conservative fallback so the audit row never silently shows `$0` for
an unknown model.

The same rate table is consumed by the chat status bar, so the
in-session counter and the persisted audit row always agree.

### Always-on, never-fatal

`app.utils.log_query_audit` is wrapped in a `try / except` that swallows
all errors — a broken DB connection or a missing column must never
take the chat down. Even on the failure path (LLM timeout, MCP server
down, retrieval exception) the chat layer still calls the helper with
`success=False` and the partial step timings, so the audit table
captures _what was running_ when the failure happened.

To disable audit logging entirely (e.g. for hot benchmark loops or
test runs), set `AUDIT_LOG_ENABLED=false` in `.env`. The `prompt_text`
field is truncated at write time to `AUDIT_PROMPT_MAX_CHARS` (default
4000) so paste-bombs don't bloat the table.

### How to view it

A new **📋 Audit Log** expander at the bottom of the sidebar surfaces
the audit data without a SQL client:

* **Filters** — User scope (Just me / All users), Date range
  (24h / 7d / 30d / All time, plus a custom-start-date override), and
  Limit (50 / 100 / 250).
* **Summary metrics** — total prompts, failed prompts, total cost,
  total tokens, average duration across the active filter.
* **Sortable table** — Timestamp · User · Source · Language · Duration ·
  Tokens · Cost · Model · Prompt-snippet, with a status icon (✅ rag /
  ⚡ hybrid / 🟠 mcp-only / ❌ failed). Language is the FTS config
  (`english` / `simple`) the query used, or `—` for rows logged before
  that column existed.
* **Detail view** — pick any row from the dropdown below the table
  and the page expands a panel with the full prompt, the audit
  metadata, the error (if any), and a **step-by-step timing table**
  showing every `query_step_timings` row for that prompt with its
  JSONB metadata.

### Querying the data directly

The tables are plain Postgres — you can also analyse them ad-hoc:

```sql
-- Slowest steps over the last 24 hours
SELECT step_name,
       COUNT(*)                AS calls,
       AVG(duration_ms)::int   AS avg_ms,
       MAX(duration_ms)        AS max_ms
FROM   query_step_timings t
JOIN   query_audit_logs   a ON a.id = t.audit_id
WHERE  a.timestamp > NOW() - INTERVAL '24 hours'
GROUP BY step_name
ORDER BY avg_ms DESC;

-- Per-user spend this month
SELECT u.email,
       COUNT(*)                          AS prompts,
       SUM(a.estimated_cost_usd)::numeric(10,4) AS spend_usd,
       SUM(a.tokens_prompt + a.tokens_completion) AS tokens
FROM   query_audit_logs a
JOIN   users u ON u.id = a.user_id
WHERE  a.timestamp >= DATE_TRUNC('month', NOW())
GROUP BY u.email
ORDER BY spend_usd DESC;
```

### Privacy & storage growth

Prompts are stored in cleartext (truncated, but not encrypted).
Operationally:

* **PII / sensitive content.** Treat `query_audit_logs.prompt_text` as
  user-content of the same sensitivity as the chat history. If the
  installation handles regulated data, lower `AUDIT_PROMPT_MAX_CHARS`
  to a snippet length (e.g. 200) or set it to `0` to skip the prompt
  entirely; the timing/cost columns remain useful on their own.
* **Retention.** Both tables are append-only. A per-month estimate at
  ~1 KB per audit row + ~6 step rows × ~0.3 KB each gives roughly
  3 KB per prompt. 10 000 prompts/month ≈ 30 MB. Add a nightly
  `DELETE FROM query_audit_logs WHERE timestamp < NOW() - INTERVAL
  '90 days'` (the FK on `query_step_timings` cascades) once the
  archive policy is decided.
* **Encryption at rest.** If your Postgres host doesn't already
  encrypt at rest, the same `cryptography.Fernet` machinery used for
  credentials in [`app/utils.py`](app/utils.py) can be applied to
  `prompt_text` — keep the audit row queryable for cost/duration
  metrics while keeping the prompt body opaque to a casual DB read.

---

## Hybrid RAG + MCP (Live SQL Server table data)

CorporateRAG also ships a **Model-Context-Protocol-style server** that
gives the chat agent safe, read-only, _live_ access to SQL Server table
rows — on top of the existing RAG over schemas / stored-procedure code.

```
┌──────────── Streamlit chat ─────────────┐
│  Q: "Show me the last 5 orders for      │
│      customer 12345"                    │
│  ┌─────────────────────────────────┐    │
│  │ RAG retriever (pgvector)        │────┼──▶ schema/proc context
│  │ + MCP tools (bound to LLM)      │────┼──▶ live row data
│  └─────────────────────────────────┘    │
│              │                          │
│              ▼  bind_tools()            │
│           OpenAI / Claude / Grok        │
└─────────────────────────────────────────┘
                │
                ▼  HTTP+token (loopback)
┌──────────── mcp_server (FastAPI) ───────┐
│  POST /mcp/tools/sql_table_query        │
│   • single-statement SELECT validator   │
│   • TOP-N injection (≤ MCP_SQL_MAX_ROWS)│
│   • per-user ODBC creds + USE [db]      │
│   • per-statement timeout, full audit   │
│  POST /mcp/tools/sql_list_databases     │
└─────────────────────────────────────────┘
```

The same server also exposes `POST /mcp/tools/entity_graph_query`
(bound to the LLM whenever `ENABLE_ENTITY_GRAPH=true`) for relationship
questions — see
[Lightweight entity graph](#lightweight-entity-graph-graphrag-inspired)
above.

### What changed (code map)

| Path                                       | Purpose                                        |
| ------------------------------------------ | ----------------------------------------------- |
| `mcp_server/server.py`                     | FastAPI app exposing MCP-style tool endpoints  |
| `mcp_server/tools/sql_tools.py`            | Read-only SELECT validator + executor          |
| `mcp_server/tools/entity_graph_tools.py`   | Entity-graph search (see [Lightweight entity graph](#lightweight-entity-graph-graphrag-inspired)) |
| `mcp_server/config.py`                     | `MCP_HOST` / `MCP_PORT` / row caps / token     |
| `core/mcp_client.py`                       | `httpx` client + LangChain `StructuredTool`s   |
| `app/mcp_manager.py`                       | Spawns/health-checks the MCP child process     |
| `core/mcp_chain.py`                        | Hybrid answerer: RAG context + bound MCP tools |
| `app/sidebar.py`                           | "⚡ Use Live SQL Table Data (MCP)" toggle      |
| `app/chat.py`                              | Routes through `core.mcp_chain` when toggle on, or when `ENABLE_ENTITY_GRAPH=true` |
| `app/main.py`                              | `ensure_mcp_running()` on first auth load      |

### Safety guarantees (enforced server-side)

- **Read-only.** Queries are tokenised with `sqlparse`; only a single
  `SELECT` (or `WITH ... SELECT` CTE) is accepted. Any of
  `INSERT/UPDATE/DELETE/MERGE/DROP/CREATE/ALTER/TRUNCATE/EXEC/EXECUTE/GRANT/REVOKE/BACKUP/RESTORE/sp_*/xp_*/fn_*/USE/GO/DBCC` rejects the request.
- **Hard row cap.** `TOP (n)` is injected if missing, and the whole
  query is wrapped in `SELECT TOP (n) * FROM (...)` to defeat
  cleverness. `n ≤ MCP_SQL_MAX_ROWS` (default 100).
- **Per-statement timeout.** `MCP_SQL_QUERY_TIMEOUT_SECONDS` (default 15s).
- **Tenancy.** Every call requires `user_id`; the tool refuses to run
  unless the user has a row for that database in
  `user_accessible_resources` (i.e. they ran a SQL ingestion under
  their account).
- **Auditable.** Every call is logged via loguru with user, database,
  query, row count and duration.
- **Token-protected.** All `/mcp/*` endpoints require an `X-MCP-Token`
  header. The Streamlit manager auto-generates one per process and
  pins it to `.streamlit/mcp/token`.

### Running the MCP server

The Streamlit app starts it for you on first authenticated load. To run
it standalone (e.g. for a separate microservice deployment):

```bash
# Set a stable token so other clients can connect
export MCP_SHARED_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# Start the server
python -m mcp_server.server
# or:
uvicorn mcp_server.server:app --host 127.0.0.1 --port 8765
```

Health check:

```bash
curl http://127.0.0.1:8765/healthz
```

Tool discovery:

```bash
curl -H "X-MCP-Token: $MCP_SHARED_TOKEN" http://127.0.0.1:8765/mcp/tools
```

Manual table query:

```bash
curl -s -X POST http://127.0.0.1:8765/mcp/tools/sql_table_query \
    -H "X-MCP-Token: $MCP_SHARED_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
          "user_id": "<user-uuid>",
          "db_name": "customerdb",
          "query": "SELECT TOP 5 * FROM dbo.Orders ORDER BY OrderDate DESC",
          "max_rows": 5
        }' | jq
```

### Using it from the chat

1. Sign in to Streamlit.
2. Sidebar → **SQL Server** → tick **Use Live SQL Table Data (MCP)**.
3. Confirm the green "🟢 MCP: connected …" status line.
4. Ask a row-data question, e.g.
   _"What were the last 10 errors in dbo.AuditLog?"_ — the agent will
   call `sql_table_query`, you'll see a markdown result table inline,
   and the Sources list will show a "⚡ SQL (live, MCP)" entry alongside
   any RAG citations.

The toggle is **off by default** so the existing pure-RAG behaviour is
unchanged for everyone who doesn't opt in.

### Configuration (`.env`)

| Variable                        | Default       | Purpose                                     |
| ------------------------------- | ------------- | ------------------------------------------- |
| `MCP_HOST`                      | `127.0.0.1`   | Bind address                                |
| `MCP_PORT`                      | `8765`        | Bind port                                   |
| `MCP_SHARED_TOKEN`              | _(generated)_ | Stable token; if unset, auto per-process    |
| `MCP_SQL_MAX_ROWS`              | `100`         | Hard row cap                                |
| `MCP_SQL_DEFAULT_ROWS`          | `50`          | Default row cap when caller doesn't specify |
| `MCP_SQL_QUERY_TIMEOUT_SECONDS` | `15`          | Per-statement timeout                       |
| `MCP_LOG_LEVEL`                 | `INFO`        | loguru level for the MCP server             |

### Future-proofing: adding MCP for Git / Confluence / Jira

The package layout is deliberately source-pluggable. To add a new
source:

1. Create `mcp_server/tools/<source>_tools.py` exposing
   `TOOL_SPECS` + handler functions (mirror `sql_tools.py`).
2. Wire two endpoints into `mcp_server/server.py` —
   `POST /mcp/tools/<source>_<tool>`.
3. Add typed methods on `core.mcp_client.MCPClient` and corresponding
   `StructuredTool` factories in `build_mcp_tools()`.
4. Optional: add a sidebar toggle on the same pattern as the SQL one
   (`SelectionState.use_mcp_<source>` + chat-layer switch).

The on-the-wire format is already MCP-compatible (tools with
`name` / `description` / `input_schema`), so swapping HTTP for
`langchain-mcp-adapters` / Anthropic's stdio transport later is a
transport change, not a redesign.

---

## Possible enhancements

Things this codebase deliberately does not do today, listed in roughly
ascending order of effort. (Live source-system ACL re-validation at
query time, cross-encoder reranking, multi-modal image captioning,
query rewriting via multi-query decomposition, a lightweight entity
graph, a semantic response cache, and corrective retrieval — formerly
listed here — now ship as flags; see
[Live source-system ACL re-validation](#3-live-source-system-acl-re-validation),
[Reranking with a cross-encoder](#reranking-with-a-cross-encoder),
[Multi-modal ingestion (image captioning)](#multi-modal-ingestion-image-captioning),
[Query rewriting before retrieval](#query-rewriting-before-retrieval-multi-query-decomposition),
[Lightweight entity graph](#lightweight-entity-graph-graphrag-inspired),
[Semantic response cache](#semantic-response-cache),
and [Corrective retrieval](#corrective-retrieval--admit-when-nothing-matches).
HyDE — the other technique named in the query-rewriting entry — was
considered and not built; see that section for why.)

### 1. Per-chunk visibility (issue / page-level granularity)

The current model is at _project / space / database / repo+branch_
granularity. For chunk-level controls (e.g. a single Jira issue restricted
inside an otherwise-public project), capture per-resource ACL during
ingestion and AND it into the retriever filter.

### 2. Background / scheduled ingestion

A small scheduler (cron, Celery beat, GitHub Actions, the Cowork
`schedule` skill) running
`python -m app.ingestion.cli --source all --mode incremental` keeps the
index fresh without anybody clicking buttons.

### 3. Streaming LLM responses

`render_chat` currently calls `llm.invoke()` and renders the full answer
once it returns. Switching to `llm.stream()` and `st.write_stream()`
gives a typewriter-style response and noticeably improves perceived
latency on long answers.

### 4. Citation-level relevance feedback

Enterprise search products (Glean's promote/demote, Copilot's 👍/👎)
close the loop between what users actually found useful and what
retrieval surfaces next time. Add a thumbs up/down per citation in
`_render_citations` (`app/chat.py`), persist it against the existing
`query_audit_logs` row, and fold accumulated feedback into
`core.retriever`'s scoring as a small per-`(query pattern, resource_id)`
boost/penalty — turning the audit trail this project already logs into
a genuine learning-to-rank signal instead of a read-only log.

### 5. Ingestion-time secret / PII scanning

Jira comments, Confluence pages, and email bodies routinely contain
pasted API keys, connection strings, or customer PII — today they're
embedded and stored verbatim like any other text. GitHub's push
secret-scanning and enterprise DLP features (Microsoft Purview, Glean)
scan content before it's indexed; running chunk text through a
regex/entropy check (or the `detect-secrets` library) inside
`BaseIngestor._chunk()` before it reaches `upsert_chunks` would let the
system redact or quarantine a match instead of silently vectorizing a
leaked credential — relevant given this same codebase already treats
*its own* stored credentials as sensitive enough to Fernet-encrypt.

### 6. Offline retrieval/answer-quality eval harness

Tuning knobs like `HNSW_EF_SEARCH`, `RRF_K`, `RERANK_CANDIDATE_K`, and
`MAX_HITS_PER_SOURCE` are currently tuned by eyeballing results. RAGAS,
TruLens, and Langfuse popularized scoring RAG pipelines against a fixed
golden question set — retrieval precision/recall@K, answer
faithfulness, citation accuracy. A `scripts/eval_harness.py`, following
the same shape as the existing `scripts/vector_store_admin.py`, running
a curated `(question, expected resource_id)` set through `retrieve()`
and `answer_question()` and tracking the scores over time, would turn
retrieval tuning into something measurable instead of anecdotal.

### 7. Let the response cache and entity graph coexist

Today `RESPONSE_CACHE_ENABLED=true` and `ENABLE_ENTITY_GRAPH=true` are
effectively mutually exclusive — see the callouts in
[Semantic response cache](#semantic-response-cache) and
[Lightweight entity graph](#lightweight-entity-graph-graphrag-inspired)
above. `used_mcp_path` in `app/chat.py` is `true` whenever *either* the
"Use Live SQL" toggle is on *or* `ENABLE_ENTITY_GRAPH` is on, and the
response cache unconditionally skips both checking and storing whenever
`used_mcp_path` is `true`. With entity graph enabled, every turn takes
that path, so the cache never engages — not degraded, entirely inert.

The tempting fix — bypass the cache only when the user's SQL toggle is
genuinely on, and let it run for turns where only the entity graph
forced the hybrid path — is unsafe as stated, because that distinction
doesn't actually exist yet at the tool level: `core/mcp_client.py`'s
`build_mcp_tools()` binds `sql_table_query` / `sql_list_databases`
*unconditionally* whenever the hybrid path runs, regardless of whether
the "Use Live SQL" toggle is on, and `HYBRID_SYSTEM_PROMPT`
(`core/mcp_chain.py`) actively instructs the model it's "EXPECTED to
use both" sources and "MUST call `sql_table_query`" for anything that
smells like a data request. So an "entity-graph-only" turn can still
invoke live SQL today — caching its answer risks serving live data back
to a different user, for up to `RESPONSE_CACHE_TTL_SECONDS`, which is
exactly what the response cache's own docstring rules out.

A safe version needs two changes, in order:

1. **Gate SQL tool binding on the toggle.** Thread `state.use_mcp_sql`
   (or an equivalent per-request flag) into `build_mcp_tools()` so
   `sql_table_query` / `sql_list_databases` are only bound when the user
   has genuinely opted into live SQL — not just whenever the hybrid path
   is entered for entity-graph reasons. Only after this can an
   "entity-graph-only" turn be trusted to never touch live data.
2. **Split the single `used_mcp_path` flag into two.** One flag (say,
   `live_sql_path = state.use_mcp_sql and "sql" in state.sources`) keeps
   gating the response-cache bypass, on both the check *and* the store
   side — `store_response_cache` needs the same narrower condition, or
   the check will simply miss forever since nothing populates the
   bucket. The broader `used_mcp_path` keeps deciding whether
   `answer_question_with_mcp()` (vs. plain `answer_question()`) is
   called, unchanged, since the entity graph tool still needs to be
   bound whenever `ENABLE_ENTITY_GRAPH` is on.

Worth validating afterwards: `RESPONSE_CACHE_SIMILARITY_THRESHOLD`
(0.97) was tuned against plain-RAG answer phrasing, not MCP/tool-
augmented answers, which can have more variable structure (tool-output
headers, inline "(live data from ...)" labels) — check it still holds
before relying on it for the newly-cacheable hybrid answers.

### 8. SQL Server impact analysis: gaps found and closed

The static dependency graph (`core.sql_dependency_extraction`,
`sql_dependency_graph`) and live tools (`sql_object_definition`,
`sql_object_dependencies`) closed the main gaps found while testing
impact-analysis questions ("what breaks if I change X", "trace Y back
to source") against a purpose-built fixture (see
`fixtures/sql-server-impact-analysis-prompt.md`). Four follow-on gaps
surfaced during further live testing; all four are now closed:

- **`ENABLE_ENTITY_EXTRACTION_LLM`'s free-text pass didn't check
  `source` before running on SQL objects.** `BaseIngestor._persist_entity_edges`
  gated its *deterministic* edges per-source, but the LLM-extraction
  branch underneath it ran unconditionally whenever
  `ENABLE_ENTITY_GRAPH` (or `ENABLE_SQL_DEPENDENCY_GRAPH`) was on and
  `ENABLE_ENTITY_EXTRACTION_LLM` was also on — meaning it also ran over
  stored-procedure/function definitions, producing noisy,
  inconsistently-named predicates (`has_column`, `defines`, `executes`,
  `related_to`, …) written to `entity_edges` alongside the clean,
  deterministic `calls`/`references`/`writes_to` edges from
  `find_references`. Confirmed in one ingestion run of the fixture
  database: 457 LLM-tagged edges against 67 deterministic ones for the
  same 38 objects. **Fixed**: the LLM pass is now scoped away from
  `source == "sql"` entirely (`app/ingestion/base.py`); a normal
  re-ingest purges any previously-written `"llm"`-tagged SQL edges via
  `upsert_edges`'s delete-then-insert semantics.
- **No way to filter `entity_edges` by `extraction_method`.** Even
  after the fix above, historical noise persists until re-ingested, and
  there was no way for a caller to request deterministic-only edges.
  **Fixed**: `traverse_sql_dependencies` (and the `sql_dependency_graph`
  MCP tool/client wrapper) now accepts `extraction_method:
  "deterministic" | "all"`, defaulting to `"deterministic"`.
- **Forced-inclusion into the plain (non-MCP) RAG path shipped off by
  default and was never validated against a real schema.**
  `core.sql_object_context.expand_hits_with_dependencies`
  (`SQL_DEPENDENCY_FORCED_INCLUSION_ENABLED`) force-includes objects
  connected via the dependency graph, bypassing the similarity/rerank
  cutoff. **Measured**: against the fixture's worst case
  (`dbo.usp_StageCompletedOrderLines`, shared by 3 of 5 report procs),
  asking "What breaks if I change dbo.usp_StageCompletedOrderLines?"
  via the plain-RAG path — off, retrieval already surfaced all 3
  dependent procedures unaided (6 hits, 13.2k context chars, 3760
  prompt tokens, zero incomplete traces). On, forced-inclusion pulled
  in 5 more objects (+2255 chars, +18.1% prompt tokens) but named the
  same 3 procedures with no completeness gain. Left off by default
  (see the measured numbers recorded in `app/config.py`'s comment);
  the flag is now also gated to the plain-RAG path only, since the
  hybrid/MCP path already has `sql_object_dependencies` as an on-demand
  tool and gained nothing from forcing the same objects into every
  prompt.
- **The MANDATORY `sql_object_definition` prompt rule wasn't reliably
  followed.** Confirmed via live A/B testing: the same tracing question,
  asked twice, sometimes got the called function's formula opened up
  and sometimes didn't — `core.trace_completeness` could only ever
  append a caveat footer after the fact, never correct the model.
  **Fixed**: `core.mcp_chain` now runs a bounded forced-retry —
  `missing_definition_calls` (in `core.trace_completeness`) detects any
  named procedure/function the model never opened via
  `sql_object_definition`, and if hop budget allows (`MAX_TOOL_HOPS`
  raised 4→6, with 2 hops reserved for the corrective round), injects
  one corrective instruction naming the gap before accepting a final
  answer. Verified via scripted unit tests
  (`tests/test_mcp_chain_forced_retry.py`) covering: retry fires once
  and converges; no retry when already compliant; no retry attempted
  too close to hop exhaustion (falls back to the honest footer
  instead). Live testing after the fix showed the model consistently
  compliant on the first try, so the retry path is a safety net that
  hasn't needed to fire in practice yet, not the primary mechanism.

### 9. Apply the SQL impact-analysis lessons to GitHub repositories

**Implemented.** Git ingestion previously pulled in full file contents
and commit metadata chunked the same generic way as every other source
— no import/reference awareness at all, and the only `entity_edges` it
wrote were repo-level `modified_by` triples from commit authorship.
This closes that gap, applying the shape of the SQL work (§8/§11) to
Git rather than rediscovering it — with one deliberate departure from
how that work started: it goes straight to a real parser instead of
regex, on the theory that inheriting SQL's original keyword-collision
mistake would just mean re-fixing it later.

- **Static import graph** (`core.git_dependency_extraction`): a
  two-pass scan, same shape as `sql_ingestor.py`'s catalog-then-scan —
  `git_ingestor.py`'s `_iter_files` now catalogs every filtered file
  path from the single `get_git_tree(recursive=True)` call it already
  made (cheap: paths only, no content) before fetching any file body,
  then each file's content is scanned for import statements resolving
  to another *cataloged* file, emitting `(subject, "imports", object)`
  edges — written to `entity_edges` (`source="git"`) whenever
  `ENABLE_GIT_DEPENDENCY_GRAPH=true` (default on, same "deterministic,
  zero-marginal-cost parse pass" reasoning as `ENABLE_SQL_DEPENDENCY_GRAPH`).
  Runs correctly in both `full` and `incremental` mode, since the
  catalog only needs the path list, not content, so an import in a file
  that wasn't re-fetched this run still resolves against the full
  catalog.
- **A real parser, not regex** — [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
  via `tree-sitter-language-pack` (prebuilt wheels, no per-language
  toolchain), covering Python (`import`/`from ... import`, including
  relative imports) and JavaScript/TypeScript (`import ... from`,
  `require(...)`). An `import_statement`/`import_from_statement`/
  `call_expression` node is structurally unambiguous, so the
  keyword-collision bug class SQL's regex engine had to fix
  after the fact (`RETURNS` colliding with a table named `Returns`, §8)
  cannot happen here at all — there's no bare-keyword-adjacency
  workaround to get right. Python absolute imports resolve by *path
  suffix* against the known-file catalog (the repo root isn't
  necessarily the top-level package root); a suffix matching more than
  one file is treated as ambiguous and skipped, not guessed — same
  "silence over noise" stance `find_references` documents for SQL.
  Bare/npm-style JS/TS specifiers (`import x from 'react'`) are always
  external packages and are silently skipped, never guessed at.
- **Test fixture with the same deliberate structure** as the SQL
  fixture's shared-vs-independent staging procedure:
  [`fixtures/git/pyrepo/`](fixtures/git/pyrepo/) and
  [`fixtures/git/jsrepo/`](fixtures/git/jsrepo/) — a shared `utils`
  module imported by three-plus feature files across `import`,
  `from X import Y`, `from X import submodule`, and multi-level
  relative-import forms, plus an `independent.py`/`independent.js` file
  that defines a same-named local helper but deliberately never imports
  the shared module. `tests/test_git_dependency_extraction.py` (15
  cases) asserts exact edge sets against these fixtures — no live
  GitHub repo needed to test the parsing logic, unlike SQL Server.
- **No DMV equivalent, so the live tool split by direction, not by
  primary/fallback.** GitHub's API has nothing like
  `sys.dm_sql_referenced_entities`. `mcp_server/tools/git_dependency_tools.py`
  ships three tools (parity with the SQL trio): `git_dependency_graph`
  (static BFS over the ingested graph, both directions — reuses the
  same `bfs()` helper `sql_dependency_graph`'s traversal was refactored
  into, `mcp_server/tools/_edge_bfs.py`, since the loop itself had no
  SQL-specific logic to begin with), `github_file_content` (fetches a
  file's current content straight from GitHub — parity with
  `sql_object_definition`), and `github_file_dependencies`, which
  genuinely splits by direction rather than layering a fallback: the
  *upstream* direction (what a file imports) is live-primary — fetch
  the file fresh and re-scan it, recursing hop-by-hop — since that's
  cheap and always current; the *downstream* direction (what imports a
  file — blast radius) has no live mechanism at all without scanning
  the whole repo, so it defers entirely to the static graph, with
  results tagged `"evidence": "live-scan"` vs `"evidence":
  "static-graph"` so the caller can tell which is which. Both live
  tools sit behind `ENABLE_GIT_DEPENDENCY_MCP_TOOLS` (default on — a
  git PAT that can ingest can also read live content, so unlike
  `ENABLE_SQL_DEPENDENCY_MCP_TOOLS` there's no permission mismatch to
  guard against, but the flag stays for symmetry and to let an operator
  disable live per-turn GitHub calls independently of the static graph).
- **Prompt and forced-retry tuning, not just the tool** — the same
  lesson §8's fourth bullet learned the hard way for SQL applied
  proactively here instead of waiting to discover it live:
  `HYBRID_SYSTEM_PROMPT` (`core/mcp_chain.py`) gained a Git tracing
  section and a MANDATORY rule that any function/class named as
  defining behavior in a Python/JS/TS file must have had
  `github_file_content` called on it, and the existing forced-retry
  loop (`missing_definition_calls` → `sql_object_definition`) now has a
  Git-shaped sibling (`core.trace_completeness.missing_content_opens` →
  `github_file_content`) wired into the *same* single-retry-per-turn
  latch — a turn that skips both a MANDATORY SQL definition and a
  MANDATORY file open still only gets one corrective round, naming both
  gaps in one message, not two separate retries.
  `tests/test_mcp_chain_forced_retry.py` covers the Git-only and
  combined-SQL-and-Git cases with the same scripted-LLM scaffold used
  for the original SQL test.

Not yet done: live end-to-end verification against a real GitHub repo +
running MCP server (this session had no such infrastructure reachable,
the same gap §11's live comparison hit for a real SQL Server) — what's
verified today is the parser/extraction logic against the fixtures
(unit tests) and the forced-retry wiring against a scripted LLM (unit
tests), not a real model's tool-calling behavior against these prompts.

### 10. Full SQL AST parsing with `sqlglot` (column-level lineage)

**Implemented and compared — see [§11](#11-sqlglot-vs-legacy-sql-impact-analysis-engine-built-and-compared)
for what was actually found.** The discussion below is the original
speculative cost/benefit analysis; it held up well against real testing,
but read §11 for the measured outcome and the reasoning behind the
default that was actually chosen.

Both `core.sql_dependency_extraction` (the static dependency graph) and
the join-shape awareness work in
`plans/DataflowShapeAwarenessForSQLImpactPlan.md` deliberately parse
SQL with `sqlparse` — a tokenizer/grouper, not a real parser — trading
structural precision for a dependency already pinned in this codebase
and a design that stays silent rather than guesses on anything it
can't cleanly handle (CTEs, comma-joins, window functions, `APPLY`).

`sqlglot` would remove that ceiling. It's a full parser that builds a
typed AST per statement — `Select`, `Join`, `From`, `Where` as real
node classes with structured arguments — so join type/table/condition
come from `select.args["joins"]` directly instead of hand-walking a
token stream and bailing at anything unrecognized. It understands
dialect-specific SQL Server syntax (a `tsql` dialect) including
`#temp` tables, `MERGE`/`OUTPUT`, `PIVOT`, CTEs, and window functions
as first-class structure rather than constructs to detect-and-skip.
Most notably, its `sqlglot.optimizer`/`lineage` modules can resolve
**column-level lineage** — which output column derives from which
input column, `*` expansion, alias resolution — a materially different
(and more useful) capability than the table/JOIN-level structure this
codebase currently extracts.

Costs: it's a second SQL-parsing dependency alongside the
already-pinned `sqlparse` (used in `core/sql_dependency_extraction.py`,
`mcp_server/tools/sql_tools.py`), maintained indefinitely rather than
reused; a much heavier package (parser generator + dialect tables +
optimizer) than `sqlparse`'s lightweight tokenizer; and a
dialect-fidelity risk — its T-SQL support, while solid, isn't
guaranteed against every construct in this codebase's live schemas and
fixtures, and a parser that throws or mis-groups on an edge case is
worse here than `sqlparse`'s tolerant grouping, given this codebase's
"silence over noise" design goal (see `core/trace_completeness.py`,
`_candidate_names`'s docstring). Adopting it "properly" would also
invite migrating the existing `sqlparse` call sites rather than
running two parsers side by side — a materially bigger change than any
single feature that would motivate it. Worth it only if a future
feature genuinely needs column-level lineage rather than table/JOIN-level
structure — see the "not needed either" call in
`DataflowShapeAwarenessForSQLImpactPlan.md`'s scope-decisions section
for why the join-shape feature itself didn't reach for it.

### 11. sqlglot vs. legacy SQL impact-analysis engine: built and compared

`settings.SQL_IMPACT_ANALYSIS_ENGINE` (`app/config.py`) switches between
the original regex/`sqlparse` engine (`"legacy"`, default) and a new
`sqlglot`-AST-based one (`"sqlglot"`) for both `core.sql_dependency_extraction`
and `core.sql_join_shape` together, plus a new capability neither engine
had before: real column-level lineage (`core.sql_column_lineage`,
`sqlglot`-only). Every call site imports from `core.sql_impact_engine`,
the single dispatcher that reads the setting, rather than choosing an
engine itself.

**Spike findings** (against all 5 real report-building procedures in
`fixtures/sql/06_reports.sql`, before writing any of the engine code):
`sqlglot`'s `tsql` dialect parsed every one of them cleanly on the first
try — `CREATE OR ALTER PROCEDURE ... AS BEGIN ... END`, `SELECT ... INTO
#TempTable`, `;WITH CteName AS (...)` leading-semicolon CTEs, `EXEC
dbo.proc`, `IF OBJECT_ID(...) IS NOT NULL DROP TABLE ...`, `TRUNCATE
TABLE`, and `ROW_NUMBER()`/`LAG() OVER (PARTITION BY ...)` window
functions all produced correct, structured AST nodes (`Select`, `Join`,
`Where`, `With`, `Into`, `Execute`, `TruncateTable`), a materially
cleaner result than `sqlparse` gave the join-shape feature (whose own
`Where`-grouping was found to swallow far more than expected, crossing
statement boundaries in undocumented ways — see §10's plan reference).
One quirk found and designed around rather than fought: a single-line
`IF cond stmt;` with no `BEGIN`/`END` gets parsed with every *subsequent*
statement nested inside its `true` branch, recursively — harmless for a
tree-walking approach (`find_all(...)` finds every `Select`/`Join`/etc.
node regardless of nesting depth), but would have broken a design that
assumed a flat top-level statement list.

**Dependency-graph parity** (`core/sql_dependency_extraction_sqlglot.py`):
all 8 of `tests/test_sql_dependency_extraction.py`'s cases mirrored
exactly against the sqlglot engine, including the exact-set edge
assertion on the churn-risk fixture — identical results. One structural
improvement free of charge: the legacy engine's documented `RETURNS
DECIMAL(12,2)` vs. a table literally named `dbo.Returns` false-positive
(a whole keyword-adjacency workaround + regression test) cannot occur
under a real AST at all, since `RETURNS` is a clause keyword and never
an `exp.Table` node.

**JOIN-shape parity + extra coverage** (`core/sql_join_shape_sqlglot.py`):
11 of `tests/test_sql_join_shape.py`'s 12 cases mirrored exactly (the
omitted one, tracing the procedure's own trailing CTE in isolation, is
subsumed by the deliberate-CTE-skip test also mirrored), byte-identical
finding wording on every overlapping case — confirmed not
just in unit tests but live, through a real LLM call (see below), where
both engines produced the identical JOIN-risk footer sentence for the
same question against the same procedure. The sqlglot engine additionally
surfaces RIGHT/FULL JOIN findings (two new templates) that the legacy
engine recognizes but deliberately stays silent on — free under a real
AST (`Join.side` already distinguishes LEFT/RIGHT/FULL), so there was no
reason to withhold it just to stay artificially parity-locked. CTEs stay
silent in both engines, unchanged.

**Column-level lineage** (`core/sql_column_lineage.py`, new capability):
rather than use `sqlglot.lineage.lineage()` directly — designed for
lineage *within one query against a known schema*, not for chaining
through several separate `SELECT ... INTO #Temp` statements with no
external schema available — this hand-walks each output expression's
leaf `Column` nodes, resolving each one's table alias via that
statement's own FROM/JOIN alias map and recursing into an earlier stage
whenever a leaf's table is itself a temp table (or CTE) the same
procedure produced. Successfully chained the full 5-hop case on the
first working version: tracing `Report_CustomerChurnRisk.TotalNetAmount`
correctly walks back through the `Scored` CTE → `#Enriched` → `#RFM` →
`#ActiveCustomerOrders` → the `dbo.fn_NetLineAmount(...)` call → the
real base-table columns behind it
(`dbo.OrderLines.{Quantity,UnitPrice,DiscountPct}`), rendered as one
arrow-chain sentence. 7 tests cover this, from a simple 2-hop sanity
case up through the full CTE-spanning chain (exact-string asserted) and
silent-failure cases (unparseable text, no temp-table stages, ambiguous
unqualified columns). Known limitation, by design: an expression's leaf
columns are reported together, not attributed to a specific `CASE`
branch or window-function partition — `lineage()`'s automatic branch
attribution is traded away for not depending on schema-dict accuracy.

**Live end-to-end comparison**: the same real tracing question used for
the join-shape feature's own live verification — *"Show how
Report_CustomerChurnRisk.TotalNetAmount is derived. Go right back to
original source tables."* — run through `core.rag_chain.answer_question`
twice against the identical literal procedure text, once per engine
setting, with a real `gpt-4o-mini` call each time:

| | `legacy` | `sqlglot` |
|---|---|---|
| `join_shape_finding_count` | 1 | 1 (identical wording) |
| `column_lineage_finding_count` | 0 (no capability) | 1 |

Both engines produced the byte-identical JOIN-risk footer sentence. The
`sqlglot` run additionally appended a third footer: *"Note: column-level
lineage (sqlglot) — `dbo.Report_CustomerChurnRisk.TotalNetAmount` ←
`Scored.TotalNetAmount` ← `#Enriched.TotalNetAmount`
(ISNULL(r.TotalNetAmount, 0)) ← `#RFM.TotalNetAmount` (SUM(NetAmount)) ←
`#ActiveCustomerOrders.NetAmount`
(dbo.fn_NetLineAmount(ol.Quantity, ol.UnitPrice, ol.DiscountPct)) ←
`dbo.OrderLines.DiscountPct`, `dbo.OrderLines.Quantity`,
`dbo.OrderLines.UnitPrice`."* (Live testing against the MCP/hybrid path
with a real SQL Server + MCP server was not attempted — that
infrastructure wasn't reachable in this session, same gap the join-shape
feature's own live verification hit; the wiring is code-identical to the
proven `rag_chain.py` path and covered by `tests/test_sql_impact_engine.py`'s
dispatcher-routing tests.)

**Default kept at `"legacy"`**, despite the sqlglot engine handling
every real fixture procedure cleanly and matching (or exceeding) the
legacy engine on every measured dimension. Reasoning: "handled this
one fixture cleanly" is not the same evidence base `RETRIEVAL_MODE`'s
`"hybrid"` default rests on (measured recall improvement across real
production traffic), and this flag's own comment in `app/config.py`
already sets that bar — dialect fidelity against *your own* schema and
T-SQL idioms, not just this codebase's fixture, is what should justify
flipping the default. Flip it once you've run both engines against your
own production schema and confirmed the same parity/improvement holds.
