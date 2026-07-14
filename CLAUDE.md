# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CorporateRAG is a multi-tenant RAG system backed by PostgreSQL + pgvector. Users ingest data from Jira, Confluence, SQL Server, Git, and email sources, and chat with it through a Streamlit UI. All vectors, auth, credentials, and access control live in one PostgreSQL database — no separate vector DB.

The entire application lives under `rag-multi-source/`. All paths below are relative to that directory.

## Commands

```bash
cd rag-multi-source

# Set up
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run the app
streamlit run app/main.py

# Ingest data (CLI)
python -m app.ingestion.cli --source jira --mode incremental --scope PROJ --email me@org.com
python -m app.ingestion.cli --source all --mode incremental --scope all --email me@org.com
python -m app.ingestion.cli --list-recipes

# Admin
python -m scripts.vector_store_admin --strategy report
python -m scripts.vector_store_admin --strategy purge-source --source jira
python -m scripts.vector_store_admin --strategy nuke --confirm-i-know-what-im-doing

# MCP server (standalone)
python -m mcp_server.server
```

There is no test suite yet.

## Architecture

### Core data flow

1. **Ingestion**: `app/ingestion/<source>_ingestor.py` subclasses `BaseIngestor` (in `app/ingestion/base.py`), which handles chunking, SHA-256 content fingerprinting for dedup, and upsert via `core.vector_store.upsert_chunks`.
2. **Chat**: `app/chat.py` calls `build_query_filter` (using the user's accessible resources from `user_accessible_resources`) → `core.retriever.retrieve` → `core.rag_chain.answer_question` (or `core.mcp_chain` for hybrid SQL).
3. **Retrieval**: `core/retriever.py` runs one vector SELECT (and one FTS SELECT in hybrid mode) **per source**, then pools and RRF-fuses the results. This prevents a dense source (e.g. thousands of emails) from crowding out sparse sources.

### Multi-tenancy

Vectors in `vector_chunks` have no `user_id`. Access is enforced at query time by a `WHERE (source = X AND <scope_col> IN (...)) OR ...` clause built from `user_accessible_resources`. Two layers:
- **Application layer**: `core.vector_store.filter_to_where_for_source` builds the WHERE clause.
- **PostgreSQL RLS** (default on): `app.utils.set_current_user_for_rls` binds `app.current_user_id` as a `SET LOCAL` GUC before each SELECT; policies on `vector_chunks` enforce the same access rule at the DB level as defence-in-depth.

Access is **granted at ingestion time**, not re-checked live. The `user_accessible_resources` table is the security boundary — writes to it grant query access.

### Key modules

| Module | Role |
|---|---|
| `core/vector_store.py` | `upsert_chunks`, `delete_resource`, `build_query_filter`, content fingerprint dedup |
| `core/retriever.py` | Per-source HNSW + FTS queries, RRF fusion, per-source quota |
| `core/rag_chain.py` | Prompt assembly, LLM call, citation prep |
| `core/mcp_chain.py` | Hybrid RAG + live SQL via bound MCP tools |
| `app/config.py` | All settings (pydantic-settings); singleton `settings` imported everywhere |
| `app/utils.py` | DB session (`get_db`), Fernet encryption, `StepTimer`, audit logging, session totals, status bar |
| `models/migrations.py` | Idempotent schema bootstrap: pgvector extension, HNSW index, additive ALTERs, RLS policies |
| `mcp_server/server.py` | FastAPI app exposing read-only SQL tools; auto-started by Streamlit on first auth load |
| `core/runtime_config.py` | JSON-persisted overrides for settings that need to change without restart (e.g. `FTS_LANGUAGE`) |

### Schema landmarks

- `vector_chunks`: `(resource_id, chunk_index)` unique key; `embedding VECTOR(1536)` with HNSW index; promoted scope columns (`project_key`, `space_key`, `db_name`, `git_scope`, `email_provider`); `content_fingerprint CHAR(64)`; generated `text_search tsvector` for FTS.
- Stable resource ID format: `jira:PROJ-123`, `confluence:page-9876`, `sql:server.db.schema.name`, `git:owner/repo@branch:file:path`.

### Retrieval modes

Controlled by `settings.RETRIEVAL_MODE` (default `"hybrid"`):
- `"vector"`: pure cosine similarity over HNSW.
- `"hybrid"`: vector + Postgres FTS via `websearch_to_tsquery`, fused with Reciprocal Rank Fusion. Requires the `text_search` generated column (provisioned automatically by migrations).

`HNSW_EF_SEARCH` (default 200) controls recall vs. speed per-statement via `SET LOCAL`. `MAX_HITS_PER_SOURCE` (default 4) soft-caps how many chunks any single source contributes to the final top-K.

### Audit & cost tracking

Every chat turn writes a `query_audit_logs` row and N `query_step_timings` child rows. `app.utils.StepTimer` is the context manager used to measure steps. The same cost rate table in `settings.LLM_COST_PER_1K_TOKENS` drives both the persistent status bar and the audit log, so they always agree.

### Recipes system

Drop a `.yaml` file into `recipes/` to add a new data source without writing Python. YAML recipes auto-discover at startup and appear in the CLI (`--list-recipes`) and sidebar. See `recipes/example_jira.yaml` as a reference.

## Configuration

All settings are in `app/config.py` (pydantic-settings, reads from `.env`). Key flags:
- `ENABLE_RLS` — PostgreSQL Row-Level Security (default `true`)
- `ENABLE_CONTENT_FINGERPRINT_DEDUP` — skip re-embedding unchanged chunks (default `true`)
- `RETRIEVAL_MODE` — `"vector"` or `"hybrid"` (default `"hybrid"`)
- `AUDIT_LOG_ENABLED` — set `false` for benchmarking (default `true`)
- `AUDIT_PROMPT_MAX_CHARS` — truncation limit for stored prompt text

Runtime-mutable overrides (survive restarts without `.env` changes) are stored in `.runtime_config.json` and managed via `core/runtime_config.py`. Currently only `FTS_LANGUAGE` is runtime-mutable.

## Code Conventions

- Use `round()` instead of `int()` when displaying timing/duration values to avoid truncating sub-millisecond results to 0.
- LLM cost lookups use case-insensitive, longest-key-wins substring matching against the model name — add new models to `LLM_COST_PER_1K_TOKENS` in `app/config.py`, no code changes needed elsewhere.
- The `extra` attribute on `VectorChunk` maps to the DB column named `metadata` (JSONB) — the column was renamed but the Python attribute kept for historical reasons; don't rename it.
- `filter_to_where_for_source` (not `filter_to_where`) is used in the retriever since moving to per-source fetching — use the right variant when adding new retrieval paths.
