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

* **One database for everything.** Auth, credentials, access rows, *and*
  vectors live side-by-side. No two-system consistency problems, no
  separate API key for a vector DB, no risk of the access table and the
  index drifting apart.
* **Real SQL alongside vector search.** Filters and rankings combine in
  one statement; you get full SQL power for hybrid search (`tsvector` BM25
  + cosine similarity) when you need it.
* **Free-tier friendly.** Runs against any Postgres ≥ 14 — local Docker,
  Supabase free, Neon free, RDS — without code changes.
* **HNSW indexing** (pgvector ≥ 0.5) gives sub-millisecond top-K cosine
  search up to several million rows on a small instance.

### Multi-tenancy: store-once + dynamic filter

Every resource (Jira ticket, Confluence page, SQL stored proc, Git
file/commit) is stored **exactly once**. There's no `user_id` column on
`vector_chunks`. Tenancy is enforced at *query* time:

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

### Stable resource IDs

| Source     | Format                                                                              |
| ---------- | ----------------------------------------------------------------------------------- |
| Jira       | `jira:{ISSUE_KEY}` — e.g. `jira:PROJ-123`                                           |
| Confluence | `confluence:page-{page_id}` — e.g. `confluence:page-9876`                           |
| SQL Server | `sql:{server}.{db}.{schema}.{name}`                                                 |
| Git        | `git:{owner}/{repo}@{branch}:commit:{sha}` or `git:{owner}/{repo}@{branch}:file:{path}` |

A resource that produces N chunks is keyed `(resource_id, chunk_index)` for
chunks 0..N-1. `INSERT ... ON CONFLICT DO UPDATE` makes re-ingestion an
overwrite, not a duplicate.

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

* `DATABASE_URL` — the URL from step 1.
* `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GROK_API_KEY`).
* `ENCRYPTION_KEY` — generate with
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

* **Source**: jira / confluence / sql / git / all
* **Scope**: project key, space key, db name, git branch, or `all`
* **Mode**: incremental (the only choice in the UI by design — see
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

# Everything for everything
python -m app.ingestion.cli --source all --mode incremental --scope all \
    --email me@org.com
```

### Mode semantics

| Mode          | What happens                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------- |
| `full`        | `DELETE FROM vector_chunks WHERE source = ... AND <scope-key> IN (...)`, then re-fetch + re-embed.          |
| `incremental` | Fetches resources updated since this user's last successful run; overwrites their chunks via `INSERT ... ON CONFLICT DO UPDATE`. |

In both modes, on success the user is granted access to every
`(source, resource_identifier)` they ingested.

**Full mode is intentionally not exposed in the UI.** A full ingest by a
user with a narrower view than the previous ingestor would silently delete
chunks of resources only the previous ingestor could see. Run full ingests
from the CLI on an admin account that owns the union of all scopes.

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

* **PostgreSQL** — free local Docker / Supabase free tier (500 MB DB,
  2 GB bandwidth) / Neon free tier (3 GB storage, autoscale-to-zero).
* **OpenAI `text-embedding-3-small`** at 1536 dims is cheap (~$0.02 / 1M
  tokens). Swap to `EMBEDDINGS_PROVIDER=huggingface` for zero cost (slower).
* **Streamlit** for the UI.
* **Fernet** for encrypted credential storage.

---

## Security notes

* User passwords are bcrypt-hashed (12 rounds).
* Source credentials (`api_token`, `conn_str`, …) are Fernet-encrypted at
  rest in `user_credentials.encrypted_value` using the `ENCRYPTION_KEY`
  env var.
* The `vector_chunks` table has no `user_id` column; access is enforced
  solely by the `WHERE` clause built from `user_accessible_resources`.
  Treat that table as the security boundary — anyone able to write into it
  can grant themselves access to any resource that's been ingested.

### Access model — current behaviour

Access is **granted at ingestion time**, not **re-checked at query time
against the source system**. Concretely:

* A user gains access to a scope (`jira:PROJ`, `confluence:DOCS`,
  `sql:mydb`, `git:owner/repo@main`) the moment they successfully run an
  ingestion under their own account. The ingestor calls
  `grant_access(user_id, source, resource_identifier)` for every resource
  it processes.
* That access **persists until they revoke it** via the sidebar's
  *🛡 Manage my access* panel. If their permissions are later removed at
  the source, CorporateRAG won't notice — they'll continue to retrieve
  any chunks already ingested for that scope.
* Conversely, a user who has full access at the source but has never run
  an ingestion themselves cannot query the data — there's no
  `user_accessible_resources` row for the retriever to use.

### Optional: PostgreSQL Row-Level Security

If you'd rather have the database itself enforce per-user filtering rather
than rely on the application building the right `WHERE` clause, you can
opt into Postgres RLS:

```sql
-- 1. Add a session-scoped current-user setting.
ALTER TABLE vector_chunks ENABLE ROW LEVEL SECURITY;

-- 2. Allow users to read only chunks whose source/key combination they
--    have a row for in user_accessible_resources.
CREATE POLICY vc_select_policy ON vector_chunks FOR SELECT
USING (
    EXISTS (
        SELECT 1
        FROM user_accessible_resources uar
        WHERE uar.user_id = current_setting('app.user_id')::text
          AND uar.source = vector_chunks.source
          AND uar.resource_identifier = COALESCE(
              vector_chunks.project_key,
              vector_chunks.space_key,
              vector_chunks.db_name,
              vector_chunks.git_scope
          )
    )
);

-- 3. The app sets the current user per session:
--    SET app.user_id = '<uuid>';
```

The app-level filter approach (default) and RLS are not mutually exclusive
— RLS gives you defence-in-depth at the cost of having to set
`app.user_id` on every connection (use SQLAlchemy `event.listen("connect", ...)`
or wrap `get_db()` to do it).

---

## Possible enhancements

Things this codebase deliberately does not do today, listed in roughly
ascending order of effort.

### 1. Live source-system ACL re-validation at query time

Today the access table is the only thing standing between users and
resources. A natural hardening is to have the chat layer call back into
each source on every query and intersect the live ACL with the granted
scopes (Jira `/rest/api/3/permissions/project`, Confluence `space`
permissions, SQL Server `HAS_DBACCESS`, GitHub `/repos/.../collaborators/.../permission`).
Adds a per-query API hit (use a short-TTL session cache) but closes the
"permissions changed at the source" gap.

### 2. Per-chunk visibility (issue / page-level granularity)

The current model is at *project / space / database / repo+branch*
granularity. For chunk-level controls (e.g. a single Jira issue restricted
inside an otherwise-public project), capture per-resource ACL during
ingestion and AND it into the retriever filter.

### 3. Background / scheduled ingestion

A small scheduler (cron, Celery beat, GitHub Actions, the Cowork
`schedule` skill) running
`python -m app.ingestion.cli --source all --mode incremental` keeps the
index fresh without anybody clicking buttons.

### 4. Audit log of queries

Log every `(user_id, query, filter_dict, citations)` to a `query_logs`
table — useful for compliance reviews and for debugging "why did the
model say X?".

### 5. Hybrid search (BM25 + cosine)

Postgres makes this trivial: add a `tsvector` column on `vector_chunks`,
GIN-index it, and combine `ts_rank` + `1 - (embedding <=> :q)` as the
final score. Often a measurable improvement for keyword-heavy queries
(error codes, IDs, exact phrases).

### 6. Reranking with a cross-encoder

A second-stage cross-encoder reranker (e.g. `bge-reranker-base`) over the
top-50 from pgvector, returning the top-8 to the LLM. Tens of milliseconds
of latency; meaningful accuracy bump on tricky retrievals.

### 7. Streaming LLM responses

`render_chat` currently calls `llm.invoke()` and renders the full answer
once it returns. Switching to `llm.stream()` and `st.write_stream()`
gives a typewriter-style response and noticeably improves perceived
latency on long answers.
