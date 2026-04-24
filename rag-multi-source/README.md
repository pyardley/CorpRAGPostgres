# CorporateRAG

A production-ready, multi-source RAG (Retrieval-Augmented Generation) system that lets your team ask natural-language questions across **Jira**, **Confluence**, **SQL Server**, and **Git repositories** — all from a clean Streamlit chat interface.

---

## Architecture at a Glance

```
User → Streamlit UI
         │
         ├─ Sidebar: source/project/branch selection, credential setup, ingestion controls
         │
         └─ Chat: query → Pinecone retrieval (filtered by user + source + scope)
                         → LLM (OpenAI / Anthropic / Grok)
                         → Answer with clickable source citations

Ingestion CLI → Jira API / Confluence API / SQL Server / GitHub API
              → Chunk + embed (OpenAI or HuggingFace)
              → Pinecone serverless index (one shared index, metadata-filtered)
              → SQLite ingestion log (tracks last-modified timestamps)
```

**Multi-tenancy:** every vector stored in Pinecone carries a `user_id` metadata field. All retrieval queries are hard-filtered by `user_id`, so users can never see each other's data.

---

## Prerequisites

| Requirement                                                                 | Free tier?            |
| --------------------------------------------------------------------------- | --------------------- |
| Python 3.11+ (3.12 recommended)                                             | ✅                    |
| [Pinecone account](https://www.pinecone.io/) – Serverless Starter           | ✅ Free               |
| OpenAI API key (embeddings + LLM) **or** HuggingFace (zero-cost embeddings) | 💰 ~$0.10 / 1M tokens |
| Atlassian Cloud account (Jira + Confluence)                                 | Existing              |
| SQL Server (any edition, including Express)                                 | Existing              |
| ODBC Driver 17 or 18 for SQL Server                                         | ✅ Free               |
| GitHub account + Personal Access Token (for Git source)                     | ✅ Free               |

---

### Pinecone API key

1. Sign up at [pinecone.io](https://www.pinecone.io/) (free Serverless Starter tier is sufficient).
2. In the Pinecone console, go to **API Keys** and copy the default key.
3. Paste it into `.env` as `PINECONE_API_KEY=...`.

The app will create the index automatically on first run.

---

### OpenAI API key

1. Sign up or log in at [platform.openai.com](https://platform.openai.com/).
2. Go to **API keys** → **Create new secret key**.
3. Paste it into `.env` as `OPENAI_API_KEY=sk-...`.

OpenAI usage for this app is minimal — embeddings cost ~$0.02 per 1 M tokens with
`text-embedding-3-small`. A typical ingestion of a few hundred pages costs well under $0.10.
If you prefer zero-cost embeddings, see the HuggingFace option in the Setup section.

---

### Atlassian API token (Jira + Confluence)

1. Log in to your Atlassian account at [id.atlassian.com](https://id.atlassian.com/).
2. Go to **Security** → **API tokens** → **Create API token**.
3. Give it a label (e.g. "CorporateRAG") and copy the token — you won't see it again.
4. Use this same token for both the Jira and Confluence credential fields in the sidebar.

**Finding the correct Confluence URL:**
Your Confluence URL is visible in the browser when you open any Confluence page.
Copy everything up to (but **not** including) `/wiki`, for example:
```
https://yoursite-1234567890.atlassian.net
```
This may be different from your Jira URL — do not assume they are the same.

**Multiple Confluence instances:**
If your content spans two Atlassian sites (e.g. a legacy instance and a newer one), expand
**Additional Confluence instance (optional)** in the sidebar's Confluence credentials tab and
enter the second site's URL. The primary email and API token are reused automatically unless
the second site requires different credentials.

---

### GitHub Personal Access Token (Git source)

Required for private repositories. Public repositories can be indexed without a token, but
providing one avoids rate limiting.

1. Go to github.com → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**.
2. Click **Generate new token (classic)**.
3. Give it a descriptive name (e.g. "CorporateRAG") and set an expiry.
4. Under **Select scopes**, tick **repo** (grants read access to private repos).
5. Click **Generate token** and copy it immediately — it won't be shown again.
6. Paste it into the **Git** credentials tab in the sidebar as **Personal Access Token**.

For fine-grained tokens: create under **Fine-grained tokens**, select the target repository,
and grant **Contents: Read-only** permission.

---

### SQL Server ODBC driver

The app connects to SQL Server via pyodbc and requires Microsoft's ODBC driver to be installed locally.

**Check whether a driver is already installed (PowerShell):**

```powershell
Get-OdbcDriver | Where-Object { $_.Name -like "*SQL*" } | Select-Object Name
```

If you see `ODBC Driver 17 for SQL Server` or `ODBC Driver 18 for SQL Server` you are ready.

**Install via winget (Windows 10/11):**

```powershell
winget install Microsoft.ODBCDriverForSQLServer
```

This installs ODBC Driver 18. If winget reports the package is already installed, the driver is present.

**Connection string format:**

Windows Authentication (no username/password needed):
```
DRIVER={ODBC Driver 18 for SQL Server};SERVER=MYPC\SQLEXPRESS;Trusted_Connection=yes;TrustServerCertificate=yes
```

SQL Server Authentication:
```
DRIVER={ODBC Driver 18 for SQL Server};SERVER=myserver.example.com;UID=myuser;PWD=mypassword;TrustServerCertificate=yes
```

Use `ODBC Driver 17` in the string if that is the version you have installed. The driver name must exactly match what `Get-OdbcDriver` reports.

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url> corporaterag
cd corporaterag/rag-multi-source

# Python 3.12 is strongly recommended. The LangChain/Pinecone ecosystem
# does not yet support Python 3.13+. Install 3.12 with:
#   winget install Python.Python.3.12   (Windows, then reopen terminal)

py -3.12 -m venv .venv          # Windows (py launcher)
# python3.12 -m venv .venv      # macOS / Linux

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD: .venv\Scripts\activate.bat
# macOS / Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:

| Variable           | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `PINECONE_API_KEY` | From your [Pinecone console](https://app.pinecone.io/)   |
| `OPENAI_API_KEY`   | From [platform.openai.com](https://platform.openai.com/) |
| `ENCRYPTION_KEY`   | Generate with the command below (only needed once)       |

**Generate an encryption key:**

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the output into `.env` as `ENCRYPTION_KEY=...`.

**To use HuggingFace embeddings instead of OpenAI** (zero cost, slower):

```env
EMBEDDINGS_PROVIDER=huggingface
EMBEDDING_DIMENSION=384
```

The first run will download the model (~90 MB) automatically.

**To use Anthropic Claude as the LLM:**

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

---

## Running the App

> **Each time you open a new terminal**, activate the virtual environment first or `streamlit` won't be found:
> ```powershell
> .venv\Scripts\Activate.ps1
> ```

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**First time:**

1. Click "Create account" and register with your email.
2. Expand **🔑 Credentials & Settings** in the sidebar.
3. Enter credentials for each source you want to use:

   **Jira tab**
   - Site URL: `https://yourorg.atlassian.net`
   - Atlassian account email
   - API token (see Prerequisites above)

   **Confluence tab**
   - Site URL: everything up to but **not** including `/wiki`, e.g. `https://yourorg-1234567890.atlassian.net`
   - Atlassian account email and API token
   - If you have a second Atlassian instance, expand **Additional Confluence instance** and enter only its URL (credentials are reused automatically)

   **SQL tab**
   - Full pyodbc connection string, e.g.:
     ```
     DRIVER={ODBC Driver 18 for SQL Server};SERVER=MYPC\SQLEXPRESS;Trusted_Connection=yes;TrustServerCertificate=yes
     ```

   **Git tab**
   - GitHub repository URL, e.g. `https://github.com/yourorg/yourrepo`
   - Personal Access Token with **repo** scope (required for private repos)
   - File extensions to index (optional — defaults to `.py .md .txt .yml .yaml .json .js .ts .sh .sql`)
   - Max commits to index (optional — defaults to 200)

4. Run an initial ingestion (see below).
5. Start chatting!

---

## Ingestion Commands

Ingestion is run via the CLI or triggered from the Streamlit UI sidebar (⚙️ Ingest data).

### Full load (re-indexes everything)

```bash
# All sources
python -m app.ingestion.cli --source all --mode full --email you@company.com

# One specific source
python -m app.ingestion.cli --source jira       --mode full --email you@company.com
python -m app.ingestion.cli --source confluence --mode full --email you@company.com
python -m app.ingestion.cli --source sql        --mode full --email you@company.com
python -m app.ingestion.cli --source git        --mode full --email you@company.com

# Narrow to a specific project / space / database / branch
python -m app.ingestion.cli --source jira       --mode full --scope PROJ     --email you@company.com
python -m app.ingestion.cli --source confluence --mode full --scope MYSPACE  --email you@company.com
python -m app.ingestion.cli --source sql        --mode full --scope mydb     --email you@company.com
python -m app.ingestion.cli --source git        --mode full --scope main     --email you@company.com
```

### Incremental update (only new/changed items)

```bash
python -m app.ingestion.cli --source all --mode incremental --email you@company.com
```

Incremental mode reads the `last_item_updated_at` timestamp from the most recent successful ingestion log and fetches only items modified after that point.

For the Git source, incremental mode fetches only commits pushed after the last ingestion.
File contents are always re-indexed in full (files have no reliable modification timestamp
via the GitHub API without per-file commit history lookups).

### Scheduled ingestion (Render.com free tier)

1. Create a free **Background Worker** on [Render.com](https://render.com/).
2. Set the start command to:
   ```
   python -m app.ingestion.cli --source all --mode incremental --email $INGEST_EMAIL --password $INGEST_PASSWORD
   ```
3. Add a cron schedule (e.g. every hour) via Render's **Cron Jobs** (free plan supports one cron job).

---

## Deployment

### Option A: Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app".
3. Set **Main file path** to `rag-multi-source/app/main.py`.
4. Add all `.env` variables as **Secrets** in the Streamlit Cloud UI.
5. Deploy. The free tier provides 1 GB RAM and sleeps after inactivity.

**Note:** SQLite is ephemeral on Streamlit Cloud. For persistent credentials/logs, either:

- Commit an initial `rag_system.db` to the repo (not recommended for secrets), or
- Replace `DATABASE_URL` with a free PostgreSQL instance (e.g. [Supabase free tier](https://supabase.com/)) and swap `sqlite` → `postgresql+psycopg2` in `requirements.txt`.

### Option B: Docker (self-hosted or cloud VM)

```bash
cp .env.example .env   # fill in values
mkdir -p data
docker compose up -d
```

Access at `http://localhost:8501`.

### Option C: Render.com Web Service

1. Create a **Web Service** on Render, point to your GitHub repo.
2. Set build command: `pip install -r rag-multi-source/requirements.txt`
3. Set start command: `streamlit run rag-multi-source/app/main.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
4. Add all env vars in the Render dashboard.
5. For ODBC driver on Render, add a `build.sh` that installs `msodbcsql18` (see Dockerfile for the apt commands).

---

## Environment Variables Reference

| Variable              | Required        | Description                              |
| --------------------- | --------------- | ---------------------------------------- |
| `APP_SECRET_KEY`      | ✅              | Random string for session signing        |
| `ENCRYPTION_KEY`      | ✅              | Fernet key for credential encryption     |
| `DATABASE_URL`        | ✅              | SQLite or Postgres URL                   |
| `PINECONE_API_KEY`    | ✅              | Pinecone API key                         |
| `PINECONE_INDEX_NAME` |                 | Index name (default: `rag-multi-source`) |
| `PINECONE_REGION`     |                 | Serverless region (default: `us-east-1`) |
| `PINECONE_CLOUD`      |                 | Serverless cloud (default: `aws`)        |
| `EMBEDDINGS_PROVIDER` |                 | `openai` or `huggingface`                |
| `OPENAI_API_KEY`      | ✅ if OpenAI    | OpenAI API key                           |
| `EMBEDDING_MODEL`     |                 | Default: `text-embedding-3-small`        |
| `EMBEDDING_DIMENSION` |                 | 1536 for OpenAI, 384 for MiniLM          |
| `LLM_PROVIDER`        |                 | `openai`, `anthropic`, or `grok`         |
| `OPENAI_CHAT_MODEL`   |                 | Default: `gpt-4o-mini`                   |
| `ANTHROPIC_API_KEY`   | ✅ if Anthropic | Anthropic API key                        |
| `ANTHROPIC_MODEL`     |                 | Default: `claude-sonnet-4-6`             |
| `GROK_API_KEY`        | ✅ if Grok      | xAI API key                              |
| `CHUNK_SIZE`          |                 | Characters per chunk (default: 1000)     |
| `CHUNK_OVERLAP`       |                 | Overlap between chunks (default: 200)    |
| `TOP_K`               |                 | Docs to retrieve per query (default: 8)  |
| `SCORE_THRESHOLD`     |                 | Min cosine similarity (default: 0.35)    |

**Source credentials** (stored encrypted in SQLite, entered via the sidebar — not in `.env`):

| Source     | Credential keys                                                  |
| ---------- | ---------------------------------------------------------------- |
| Jira       | `url`, `email`, `api_token`                                      |
| Confluence | `url`, `email`, `api_token` + optional `url_2` for a second site |
| SQL Server | `conn_str` (full pyodbc connection string)                       |
| Git        | `url`, `access_token`, `file_extensions`, `max_commits`         |

---

## Adding a New Data Source

1. Create `app/ingestion/myservice_ingestor.py` extending `BaseIngestor`.
2. Implement `list_scopes()` and `load_documents()`.
   - Each `Document.metadata` must include `user_id`, `source`, `title`, `url`, `last_updated`.
3. Add a `_scope_filter()` override for full-load deletion.
4. Register the new source in `app/ingestion/cli.py` (`_get_ingestor` function and `ALL_SOURCES` list).
5. Add a checkbox and scope expander in `app/sidebar.py`, a `_cached_<source>_scopes()` function, and a credentials tab.
6. Add a `SelectionState` field and pass it through `answer_query` → `retrieve` → `build_filter` in `app/chat.py` and `core/retriever.py`.
7. Add citation label and rendering logic in `app/chat.py` (`_build_source_label` and `_render_citations`).

---

## Project Structure

```
CorporateRAG/
├── .gitignore
└── rag-multi-source/
    ├── app/
    │   ├── main.py                    # Streamlit entry point
    │   ├── config.py                  # All settings (pydantic-settings + .env)
    │   ├── auth.py                    # Register / login / session
    │   ├── sidebar.py                 # Source selection, credentials, ingestion UI
    │   ├── chat.py                    # Chat logic, retrieval, LLM, citations
    │   ├── utils.py                   # DB session, Fernet encryption helpers
    │   └── ingestion/
    │       ├── base.py                # Abstract BaseIngestor (chunk, embed, upsert)
    │       ├── cli.py                 # python -m app.ingestion.cli
    │       ├── jira_ingestor.py       # Jira issues + comments (ADF body parsing)
    │       ├── confluence_ingestor.py # Confluence pages; multi-instance support
    │       ├── sql_ingestor.py        # Stored procs, functions, views, table schemas
    │       └── git_ingestor.py        # GitHub commits + file contents via PyGithub
    ├── core/
    │   ├── vector_store.py            # Pinecone index lifecycle + helpers
    │   ├── retriever.py               # Metadata-filtered similarity search
    │   └── llm.py                     # LLM + embeddings factory
    ├── models/
    │   ├── base.py                    # SQLAlchemy declarative base
    │   ├── user.py                    # User + UserCredential (encrypted creds)
    │   └── ingestion_log.py           # Ingestion run log + timestamps
    ├── requirements.txt
    ├── .env.example
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Security Notes

- **Credentials** (API tokens, SQL connection strings, GitHub PATs) are encrypted with Fernet (AES-128-CBC + HMAC) before being stored in SQLite. The encryption key lives only in your `.env` / environment.
- **Multi-tenancy**: every Pinecone vector carries `user_id` in metadata, and every retrieval query includes `{"user_id": {"$eq": current_user_id}}` — users are cryptographically isolated at the vector store level.
- **Passwords** are hashed with bcrypt (cost factor 12).
- Keep `ENCRYPTION_KEY` and `APP_SECRET_KEY` out of version control. The `.gitignore` excludes `.env` and `*.db` by default.
