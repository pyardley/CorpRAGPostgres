# CorporateRAG (Enhanced)

A production-ready, multi-tenant RAG system that lets a whole organisation
ask natural-language questions across **Jira**, **Confluence**, **SQL Server**
and **Git (GitHub)** — from a single Streamlit chat interface, backed by a
single shared Pinecone index.

---

## What changed in this version

The previous architecture stored **a separate copy of every chunk per user**
in Pinecone, with `user_id` baked into vector metadata. That works, but it
duplicates storage roughly N× (N = number of users) and re-runs the same
embedding cost on every team member.

This version implements the **store-once + dynamic-filter** architecture:

| Concern             | Before                                 | Now                                                              |
| ------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| Pinecone index      | Shared, but rows duplicated per user   | Shared, **one row per (resource, chunk)**                        |
| Vector primary key  | Random / opaque                        | Stable: `f"{resource_id}::chunk-{i}"`                            |
| `user_id` in vector | Always present                         | **Never** — never written, must be removed from legacy rows      |
| Tenancy enforcement | Hard-coded into every retrieval filter | Built dynamically at query time from `user_accessible_resources` |
| Re-ingest cost      | Pay per user                           | Pay once per resource, every user benefits                       |

### Stable resource IDs

| Source     | Format                                                                                  |
| ---------- | --------------------------------------------------------------------------------------- |
| Jira       | `jira:{ISSUE_KEY}` — e.g. `jira:PROJ-123`                                               |
| Confluence | `confluence:page-{page_id}` — e.g. `confluence:page-9876`                               |
| SQL Server | `sql:{server}.{db}.{schema}.{name}`                                                     |
| Git        | `git:{owner}/{repo}@{branch}:commit:{sha}` or `git:{owner}/{repo}@{branch}:file:{path}` |

A resource that produces N chunks gets vector IDs
`{resource_id}::chunk-0`, `{resource_id}::chunk-1`, … so re-upserting is a
clean overwrite and full-resource deletion is just a prefix scan.

### How tenancy works at query time

When a user types in the chat, `app/chat.py` builds a Pinecone metadata
filter from their sidebar selection plus the `user_accessible_resources`
table:

```python
filter_dict = {
    "$or": [
        {"source": "jira",       "project_key": {"$in": user_project_keys}},
        {"source": "confluence", "space_key":   {"$in": user_space_keys}},
        {"source": "sql",        "db_name":     {"$in": user_db_names}},
        {"source": "git",        "git_scope":   {"$in": user_git_scopes}},
    ]
}
```

For Git, `git_scope` is `{repo_full_name}@{branch}` — e.g. `acme/widgets@main` —
which is also the value stored in `user_accessible_resources.resource_identifier`
for git rows.

That filter is the _only_ thing standing between users and resources they
shouldn't see — so the `user_accessible_resources` table is the source of
truth for "what is this user allowed to query?".

### New SQLite table

```sql
CREATE TABLE user_accessible_resources (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id),
    source              TEXT NOT NULL,           -- 'jira' | 'confluence' | 'sql'
    resource_identifier TEXT NOT NULL,           -- project_key | space_key | db_name
    last_synced         DATETIME NOT NULL,
    UNIQUE (user_id, source, resource_identifier)
);
```

The table is created automatically on first start by `models.migrations.run_migrations()`,
which `app/utils.init_db()` calls on every Streamlit / CLI launch.

---

## Setup

### 1. Python environment

Python 3.11 or 3.12.

```bash
cd rag-multi-source
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Fill in at minimum:

- `PINECONE_API_KEY` — free serverless tier is fine.
- `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GROK_API_KEY`).
- `ENCRYPTION_KEY` — generate with
  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.

### 3. Launch

```bash
streamlit run app/main.py
```

Sign up with any email, then enter your Jira / Confluence / SQL credentials
in the sidebar (they're encrypted at rest with Fernet), and run an
ingestion.

---

## Ingestion

### From the UI

Sidebar → ⚙️ **Ingest data** →

- **Source**: jira / confluence / sql / git / all
- **Mode**: incremental (default) / full
- **Scope**: project key, space key, db name, git branch, or `all`

### From the CLI

```bash
# Incremental ingest of a single Jira project
python -m app.ingestion.cli --source jira --mode incremental --scope PROJ \
    --email me@org.com

# Full re-ingest of one Confluence space (wipes existing chunks for that space first)
python -m app.ingestion.cli --source confluence --mode full --scope DOCS \
    --email me@org.com

# Index a git branch (default branch when scope=all)
python -m app.ingestion.cli --source git --mode incremental --scope main \
    --email me@org.com

# Everything for everything
python -m app.ingestion.cli --source all --mode incremental --scope all \
    --email me@org.com
```

### Mode semantics

| Mode          | What happens                                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| `full`        | Pinecone deletes vectors matching the scope filter (e.g. `source=jira AND project_key=PROJ`), then rebuilds.   |
| `incremental` | Fetches resources updated since the last successful run for `(user, source, scope)`, overwriting their chunks. |

In both modes, on success the user is granted access to every
`(source, resource_identifier)` they ingested.

---

## Project layout

```
rag-multi-source/
├── README.md                 # ← you are here
├── requirements.txt
├── .env.example
├── app/
│   ├── main.py               # Streamlit entry point
│   ├── auth.py               # email + bcrypt login
│   ├── config.py             # pydantic-settings
│   ├── chat.py               # builds dynamic Pinecone filter, calls retriever + LLM
│   ├── sidebar.py            # source toggles + scope pickers (uses user_accessible_resources)
│   ├── utils.py              # DB session, Fernet, credential & access helpers
│   └── ingestion/
│       ├── base.py           # BaseIngestor — chunking, dedup, full vs incremental
│       ├── jira_ingestor.py
│       ├── confluence_ingestor.py
│       ├── sql_ingestor.py
│       ├── git_ingestor.py
│       └── cli.py
├── core/
│   ├── llm.py                # embeddings + LLM factory
│   ├── vector_store.py       # ★ shared Pinecone index, ResourceChunk, build_query_filter
│   ├── retriever.py          # ★ filter-aware similarity search
│   └── rag_chain.py          # prompt + LLM call + citation prep
├── models/
│   ├── base.py
│   ├── user.py
│   ├── ingestion_log.py
│   ├── user_accessible_resource.py   # ← new
│   └── migrations.py                  # ← new (idempotent schema bootstrap)
└── scripts/
    └── cleanup_legacy_vectors.py     # one-shot Pinecone cleanup tool
```

---

## Migration guide (existing Pinecone data)

If you previously ran the duplicated-per-user version, you need to either
nuke and re-ingest, or strip `user_id` from existing vectors. There is no
in-place automatic migration — running the new app against the old index
just hides duplicates from the user; it doesn't remove them.

### Option A — recommended: nuke and re-ingest

Cheapest and simplest if your index is small (≲ 100k vectors).

```bash
# 1. Inspect what you've got
python -m scripts.cleanup_legacy_vectors --strategy report

# 2. Delete EVERYTHING in the index
python -m scripts.cleanup_legacy_vectors --strategy nuke \
    --confirm-i-know-what-im-doing

# 3. Re-ingest each source under one user (the others will inherit access
#    once they ingest their own scopes — vectors will be reused, not duplicated)
python -m app.ingestion.cli --source all --mode full --scope all \
    --email admin@org.com
```

### Option B — in-place strip-user-id

If the index is large enough that re-embedding would cost real money:

```bash
# Dry run: report what would be deleted / stripped
python -m scripts.cleanup_legacy_vectors --strategy strip-user-id

# Apply for real
python -m scripts.cleanup_legacy_vectors --strategy strip-user-id --apply
```

This walks the index, groups vectors by `resource_id` (chunks added under
the new ingestor will already have one), keeps one copy per resource, and
removes the `user_id` field from kept copies. Vectors that pre-date the
`resource_id` change are deleted (you'll need to re-ingest those resources).

After either option, every user must run an ingestion (even just
`--mode incremental`) once so the `user_accessible_resources` table is
populated for them — that's how their queries get scoped.

### Sanity checks

```bash
# After migration, the index should look like this:
python -m scripts.cleanup_legacy_vectors --strategy report
# → No `user_id` field anywhere in metadata.
# → Total vector count ≈ (sum of chunks across resources), NOT × users.
```

---

## Free-tier friendly

- **Pinecone serverless** (1 free index, ≤ 2GB).
- **OpenAI `text-embedding-3-small`** at 1536 dims is cheap (~$0.02 / 1M tokens).
  Swap to `EMBEDDINGS_PROVIDER=huggingface` for zero cost (slower).
- **SQLite** for the app DB.
- **Streamlit** for the UI.
- **Fernet** for encrypted credential storage.

---

## Security notes

- User passwords are bcrypt-hashed (12 rounds).
- Source credentials (`api_token`, `conn_str`, …) are Fernet-encrypted at
  rest in `user_credentials.encrypted_value` using the `ENCRYPTION_KEY`
  env var.
- The Pinecone index has no `user_id` metadata; access is enforced solely
  by the dynamic filter built from `user_accessible_resources`. Treat that
  table as the security boundary — anyone able to write into it can grant
  themselves access to any resource that's been ingested.
