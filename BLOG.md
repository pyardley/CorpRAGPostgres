# CorporateRAG: Ask Your Entire Tech Stack a Question

Most companies store their knowledge in at least four places at once: a project tracker, a wiki, a database, and version control. Answering a question like _"What changed in the authentication service last month, and is there a Jira ticket for it?"_ normally means opening four tabs, copying text between them, and doing the synthesis yourself.

CorporateRAG solves this by pulling all four sources into a single vector index and putting a chat interface on top. You type a question in plain English, and the system retrieves the most relevant chunks from whichever sources you have enabled — Jira, Confluence, SQL Server, or GitHub — then uses an LLM to write a grounded, cited answer.

---

## What It Does

CorporateRAG is a **Retrieval-Augmented Generation (RAG)** application with four data connectors:

| Source         | What gets indexed                                                          |
| -------------- | -------------------------------------------------------------------------- |
| **Jira**       | Issue summaries, descriptions, comments, status, priority, assignee        |
| **Confluence** | Page bodies (HTML → markdown), across multiple Atlassian instances         |
| **SQL Server** | Stored procedure definitions, function bodies, view queries, table schemas |
| **GitHub**     | Commit messages with file lists, source/doc file contents by branch        |

When you ask a question, the app:

1. Embeds the query using the same model used at ingestion time
2. Runs a metadata-filtered nearest-neighbour search in Pinecone, scoped to your user ID and any source/scope filters you have set in the sidebar
3. Passes the top-k retrieved chunks to an LLM as numbered context
4. Returns a markdown answer with inline `[N]` citations linked to the original source

Every user's vectors are isolated in Pinecone via a hard `user_id` filter, so in a shared deployment no user can see another's data.

---

## Why It's Useful

### The problem it solves

Developers and project managers constantly context-switch between tools. A typical investigation might involve:

- Checking Jira to understand the scope of a bug
- Reading the Confluence design doc to understand the intended behaviour
- Querying the database to find the stored procedure that implements it
- Checking the Git log to see what changed and when

CorporateRAG compresses this into a single question. Because retrieval is semantic rather than keyword-based, queries like _"buffer overflow error in the payment service"_ will match a Jira comment that says _"stack corruption in the billing module"_ without needing exact wording.

### Concrete use cases

- **Onboarding** — a new developer asks _"How does the user authentication flow work?"_ and gets a synthesised answer drawing from design docs, code, and relevant tickets.
- **Incident response** — _"Has this SQL timeout error been seen before?"_ searches Jira history, Confluence runbooks, and recent commit messages simultaneously.
- **Code review** — _"What does the `sp_ProcessPayment` stored procedure do?"_ returns the procedure definition plus any related Confluence documentation.
- **Sprint planning** — _"What Jira tickets are currently open in the SCRUM project with high priority?"_ surfaces the right context without manually filtering the board.

---

## Tech Stack

| Layer                  | Technology                                                                        | Role                                                               |
| ---------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **UI**                 | [Streamlit](https://streamlit.io/)                                                | Chat interface, sidebar controls, ingestion triggers               |
| **Orchestration**      | [LangChain](https://www.langchain.com/)                                           | Document chunking, embedding abstraction, vector store integration |
| **Vector store**       | [Pinecone](https://www.pinecone.io/) Serverless                                   | Storing and querying embeddings with metadata filtering            |
| **Embeddings**         | OpenAI `text-embedding-3-small` or HuggingFace `all-MiniLM-L6-v2`                 | Turning text chunks into vectors                                   |
| **LLM**                | OpenAI GPT-4o-mini / Anthropic Claude / xAI Grok                                  | Synthesising answers from retrieved context                        |
| **Jira + Confluence**  | [atlassian-python-api](https://atlassian-python-api.readthedocs.io/)              | Fetching issues and pages via REST API                             |
| **GitHub**             | [PyGithub](https://pygithub.readthedocs.io/)                                      | Fetching commits and file contents via GitHub API                  |
| **SQL Server**         | pyodbc + SQLAlchemy                                                               | Introspecting stored procedures, functions, views, and tables      |
| **Credential storage** | SQLite + [Fernet](https://cryptography.io/en/latest/fernet/) (AES-128-CBC + HMAC) | Encrypting API tokens and connection strings at rest               |
| **Auth**               | bcrypt (cost 12)                                                                  | Hashing user passwords                                             |
| **Settings**           | pydantic-settings + `.env`                                                        | Typed configuration with environment variable overrides            |
| **Retry logic**        | [tenacity](https://tenacity.readthedocs.io/)                                      | Exponential backoff on API calls                                   |
| **Logging**            | [loguru](https://loguru.readthedocs.io/)                                          | Structured, coloured console logs                                  |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                       │
│  ┌──────────────┐          ┌────────────────────┐   │
│  │   Sidebar    │          │     Chat panel     │   │
│  │  · Checkboxes│          │  · Chat history    │   │
│  │  · Scopes    │          │  · Input box       │   │
│  │  · Creds     │          │  · Citations       │   │
│  │  · Ingest    │          └────────────────────┘   │
│  └──────────────┘                                   │
└───────────────────┬─────────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │   answer_query()   │
          │   chat.py          │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │   retrieve()       │   ← metadata filter:
          │   retriever.py     │     user_id + source +
          └─────────┬──────────┘     project/space/branch
                    │
          ┌─────────▼──────────┐
          │ Pinecone Serverless│
          │ similarity search  │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │   LLM (OpenAI /    │
          │   Anthropic / Grok)│
          └────────────────────┘

Ingestion pipeline (CLI or sidebar trigger):
  Jira API ──────────┐
  Confluence API ────┤──► BaseIngestor.run()
  SQL Server ────────┤     · chunk text
  GitHub API ────────┘     · embed
                           · upsert to Pinecone
                           · log to SQLite
```

---

## How to Set It Up

### Prerequisites

- Python 3.12 (3.13+ not yet supported by the LangChain/Pinecone ecosystem)
- A free [Pinecone](https://www.pinecone.io/) Serverless account
- An [OpenAI](https://platform.openai.com/) API key (or HuggingFace for zero-cost embeddings)
- Atlassian Cloud account with an API tokens for Jira and Confluence
- SQL Server with ODBC Driver 17 or 18 installed locally
- A GitHub Personal Access Token with `repo` scope for private repositories

### Installation

```bash
git clone https://github.com/pyardley/CorporateRAG corporaterag
cd corporaterag/rag-multi-source

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Minimum required values in `.env`:

```env
PINECONE_API_KEY=pcsk_...
OPENAI_API_KEY=sk-...
ENCRYPTION_KEY=<generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">
APP_SECRET_KEY=any-random-string
DATABASE_URL=sqlite:///rag_system.db
```

To switch to Anthropic Claude as the LLM:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

### Running the app

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501), create an account, then enter credentials for each source in the **🔑 Credentials & Settings** sidebar expander.

---

## How to Run the Ingestion Suite

Ingestion is the process of fetching data from each source, chunking it, embedding it, and upserting it into Pinecone. It can be triggered from the sidebar UI or run via the CLI.

### Full load — re-index everything from scratch

```bash
# All sources at once
python -m app.ingestion.cli --source all --mode full --email you@company.com

# Individual sources
python -m app.ingestion.cli --source jira       --mode full --email you@company.com
python -m app.ingestion.cli --source confluence --mode full --email you@company.com
python -m app.ingestion.cli --source sql        --mode full --email you@company.com
python -m app.ingestion.cli --source git        --mode full --email you@company.com

# Narrow to a specific project, space, database, or branch
python -m app.ingestion.cli --source jira       --mode full --scope SCRUM  --email you@company.com
python -m app.ingestion.cli --source confluence --mode full --scope TS     --email you@company.com
python -m app.ingestion.cli --source git        --mode full --scope main   --email you@company.com
```

### Incremental — only new or changed items

```bash
python -m app.ingestion.cli --source all --mode incremental --email you@company.com
```

Incremental mode reads the `last_item_updated_at` timestamp from the most recent successful run in the ingestion log and fetches only items modified after that point. For Jira this uses JQL `updated >= '...'`; for Confluence it compares `version.when` client-side; for Git it uses the GitHub commits API `since` parameter.

### What happens during ingestion

```
Source API
    │
    ▼
load_documents()          # fetch raw text + metadata per item
    │
    ▼
RecursiveCharacterTextSplitter  # chunk_size=1000, overlap=200 (configurable)
    │
    ▼
OpenAIEmbeddings / HuggingFaceEmbeddings  # embed each chunk
    │
    ▼
Pinecone.upsert()         # vector id = SHA256(user_id + source + item_id + chunk_index)
    │
    ▼
IngestionLog (SQLite)     # status, items_processed, vectors_upserted, timestamps
```

---

## Code Walkthrough

### Multi-tenant metadata filter

Every vector carries a `user_id` field. All retrieval queries start with a hard equality filter on it, making cross-user data leakage impossible at the vector store level regardless of how the application layer behaves.

```python
# core/retriever.py
def build_filter(user_id, sources=None, project_keys=None,
                 space_keys=None, db_names=None, git_branches=None):
    filt = {"user_id": {"$eq": user_id}}

    if sources:
        filt["source"] = {"$in": sources}

    sub_conditions = []
    if project_keys:
        sub_conditions.append({"project_key": {"$in": project_keys}})
    if space_keys:
        sub_conditions.append({"space_key": {"$in": space_keys}})
    if db_names:
        sub_conditions.append({"db_name": {"$in": db_names}})
    if git_branches:
        sub_conditions.append({"branch": {"$in": git_branches}})

    if len(sub_conditions) == 1:
        filt.update(sub_conditions[0])
    elif sub_conditions:
        filt["$or"] = sub_conditions

    return filt
```

### Jira ADF body parsing

Jira Cloud returns issue descriptions and comments as **Atlassian Document Format (ADF)** — a nested JSON structure, not plain text. The ingestor recursively converts this to plain text before embedding:

````python
# app/ingestion/jira_ingestor.py
def _adf_to_text(node) -> str:
    if not node:
        return ""
    if isinstance(node, str):
        return node
    node_type = node.get("type", "")
    if node_type == "text":
        return node.get("text", "")
    if node_type == "hardBreak":
        return "\n"
    children = [_adf_to_text(c) for c in node.get("content", [])]
    if node_type in ("paragraph", "heading"):
        return "".join(children).strip()
    if node_type in ("bulletList", "orderedList"):
        return "\n".join(f"• {c.strip()}" for c in children if c.strip())
    if node_type == "codeBlock":
        return "```\n" + "".join(children) + "\n```"
    return "\n".join(c for c in children if c.strip())
````

Without this, every Jira issue with a formatted description would produce an empty embedding — the comment bodies would just read `[object Object]`.

### Multiple Confluence instances

Organisations sometimes span more than one Atlassian site. The Confluence ingestor supports this natively by reading `url`/`url_2`/`url_3` credential keys and creating one API client per instance. Secondary instances fall back to the primary email and token unless their own are supplied:

```python
# app/ingestion/confluence_ingestor.py
def _make_instances(self):
    primary_email = self.credentials.get("email", "")
    primary_token = self.credentials.get("api_token", "")

    for suffix in ["", "_2", "_3", "_4", "_5"]:
        url = self.credentials.get(f"url{suffix}", "").strip()
        if not url:
            continue
        email = self.credentials.get(f"email{suffix}", "") or primary_email
        token = self.credentials.get(f"api_token{suffix}", "") or primary_token
        client = Confluence(url=url, username=email, password=token, cloud=True)
        self._instances.append((client, url.rstrip("/")))
```

### Idempotent vector IDs

Rather than letting Pinecone generate random IDs, each chunk gets a deterministic SHA-256 ID based on its source coordinates. Re-running a full ingestion overwrites existing vectors rather than duplicating them:

```python
# app/ingestion/base.py (simplified)
chunk_id = hashlib.sha256(
    f"{self.user_id}:{self.SOURCE}:{item_id}:{chunk_index}".encode()
).hexdigest()
```

---

## Examples in Use

### Example 1 — Cross-source query

**Prompt:** `Explain how app.ingestion.cli works.`

> The ingestion CLI (`app/ingestion/cli.py`) is the command-line entry point for the
> CorporateRAG data pipeline [1]. It accepts four arguments: `--source` (jira, confluence,
> sql, git, or all), `--mode` (full or incremental), `--scope` (a project key, space key,
> branch name, or "all"), and `--email` for authentication [1].
>
> In **full mode** the ingestor deletes all existing Pinecone vectors matching the scope
> filter before re-indexing, ensuring stale content is removed [1]. In **incremental mode**
> it reads the `last_item_updated_at` timestamp from the SQLite ingestion log and only
> fetches items modified since that point [2].
>
> Sources: **[1]** 🟣 `rag-multi-source/app/ingestion/cli.py` @ `master`  
> **[2]** 🟣 `rag-multi-source/app/ingestion/base.py` @ `master`

---

### Example 2 — Jira ticket lookup

**Prompt:** `What does error code ABC123 mean?`

> Error code ABC123 is caused by a buffer overflow [1]. The issue was reported in ticket
> SCRUM-29 and the root cause was identified in a comment by the assignee.
>
> Sources: **[1]** 🔵 [SCRUM-29: Investigate error code ABC123](https://yourorg.atlassian.net/browse/SCRUM-29) `In Progress`

---

### Example 3 — Confluence knowledge base

**Prompt:** `What is Robin Gherkin's favourite colour?`

> According to the Favourite Colours page in Confluence, Robin Gherkin's favourite colour
> is blue [1].
>
> Sources: **[1]** 🟢 [Favourite Colours](https://yourorg.atlassian.net/wiki/spaces/SD/pages/123) `SD`

---

### Example 4 — SQL Server schema exploration

**Prompt:** `What does the sp_ProcessPayment stored procedure do?`

> `sp_ProcessPayment` validates the incoming payment record, checks the account balance,
> inserts a transaction row into `dbo.Transactions`, and updates `dbo.AccountBalances` in
> a single transaction [1]. It returns error code 1001 if the account has insufficient
> funds and 1002 if the payment reference already exists.
>
> Sources: **[1]** 🟠 `PaymentsDB` › `sp_ProcessPayment` (PROCEDURE)

---

### Example 5 — Git history

**Prompt:** `What changed in the ingestion pipeline recently?`

> Three commits touched the ingestion pipeline in the last week [1][2][3]:
>
> - `a538526` added the Git data source with `git_ingestor.py` and wired it into the CLI,
>   sidebar, and retriever.
> - `1121f5c` updated the README with Git prerequisites and ingestion command examples.
> - `d86b3ad` was the initial commit establishing the Jira, Confluence, and SQL ingestors.
>
> Sources: **[1]** 🟣 [`a538526`](https://github.com/pyardley/CorporateRAG/commit/a538526) `pyardley/CorporateRAG`  
> **[2]** 🟣 [`1121f5c`](https://github.com/pyardley/CorporateRAG/commit/1121f5c) `pyardley/CorporateRAG`  
> **[3]** 🟣 [`d86b3ad`](https://github.com/pyardley/CorporateRAG/commit/d86b3ad) `pyardley/CorporateRAG`

---

## Alternative Technologies

### Vector store: Pinecone vs alternatives

| Alternative                         | Advantages                                                                       | Disadvantages                                                             |
| ----------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Pinecone** _(current)_            | Fully managed, serverless free tier, fast metadata filtering, no ops overhead    | Proprietary, cost at scale, data leaves your infrastructure               |
| **Weaviate**                        | Open source, self-hostable, native hybrid search (BM25 + vector), GraphQL API    | More complex to operate, self-managed at scale                            |
| **Qdrant**                          | Open source, Rust-based (fast), self-hostable, strong filtering, free cloud tier | Smaller ecosystem than Pinecone, fewer managed-cloud options              |
| **pgvector** (PostgreSQL extension) | Runs in existing Postgres, no extra infrastructure, SQL-native                   | Slower at scale (no HNSW by default until pg16), needs tuning             |
| **Chroma**                          | Extremely simple, local file-based, great for prototyping                        | Not designed for multi-user/production; persistence and filtering limited |

**Recommendation:** Pinecone is the right choice for a managed, zero-ops deployment. Switch to Qdrant if data residency or cost at scale becomes a concern.

---

### Embeddings: OpenAI vs alternatives

| Alternative                                     | Advantages                                                       | Disadvantages                                        |
| ----------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------- |
| **OpenAI `text-embedding-3-small`** _(current)_ | Best-in-class quality, cheap (~$0.02/1M tokens), 1536 dimensions | API cost, data sent to OpenAI, requires internet     |
| **HuggingFace `all-MiniLM-L6-v2`**              | Free, runs locally, 384 dimensions (smaller index)               | Lower quality on domain-specific text, slower on CPU |
| **Cohere Embed v3**                             | Strong multilingual support, 1024 dimensions, reranking API      | Additional vendor dependency, cost                   |
| **Voyage AI**                                   | State-of-the-art on code and technical retrieval benchmarks      | Newer service, cost, less community tooling          |

**Recommendation:** OpenAI for production quality; HuggingFace if you need air-gapped or zero-cost operation.

---

### LLM: OpenAI vs alternatives

| Alternative                                | Advantages                                                          | Disadvantages                                             |
| ------------------------------------------ | ------------------------------------------------------------------- | --------------------------------------------------------- |
| **OpenAI GPT-4o-mini** _(current default)_ | Fast, cheap, strong instruction following                           | Proprietary, cost, data leaves your infrastructure        |
| **Anthropic Claude Sonnet** _(supported)_  | Excellent at long-context synthesis, strong reasoning, 200K context | Cost, proprietary                                         |
| **Ollama (Llama 3, Mistral, etc.)**        | Fully local, free, no data egress                                   | Requires capable GPU, lower quality on complex reasoning  |
| **Azure OpenAI**                           | Same models as OpenAI but within your Azure tenant (data residency) | Azure subscription, more setup overhead                   |
| **Google Gemini**                          | Competitive quality, large context window, Google Cloud integration | Not currently integrated; would require a new LLM adapter |

---

### UI: Streamlit vs alternatives

| Alternative               | Advantages                                                             | Disadvantages                                                          |
| ------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Streamlit** _(current)_ | Fast to build, pure Python, easy deployment, built-in session state    | Limited layout flexibility, not suitable for complex multi-page apps   |
| **Gradio**                | Even simpler for ML demos, built-in sharing via Hugging Face Spaces    | Even less layout control, weaker for production                        |
| **FastAPI + React**       | Full control over UI, proper REST API, scales to a real product        | Significant additional complexity and build tooling                    |
| **Chainlit**              | Chat-first UI built for LLM apps, markdown + citation support built in | Smaller community, less flexible for non-chat views (e.g. the sidebar) |

---

### Orchestration: LangChain vs alternatives

| Alternative               | Advantages                                                                 | Disadvantages                                                        |
| ------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **LangChain** _(current)_ | Large ecosystem, consistent abstractions for embeddings/vector stores/LLMs | Heavy dependency tree, abstractions can obscure bugs                 |
| **LlamaIndex**            | Stronger focus on RAG-specific patterns, good multi-document indexing      | Overlapping with LangChain, similar complexity                       |
| **Haystack**              | Production-focused, strong pipeline abstractions, good evaluation tooling  | Steeper learning curve, less Pinecone-native integration             |
| **Plain Python**          | Zero abstraction overhead, full control, easier debugging                  | Must re-implement chunking, retry logic, provider switching yourself |

---

## What's Next

Some natural extensions to the current system:

- **Reranking** — add a cross-encoder reranker (Cohere Rerank or a local model) after the initial Pinecone retrieval to improve result ordering before passing to the LLM
- **Hybrid search** — combine vector similarity with BM25 keyword search for better recall on exact names (ticket IDs, function names, error codes)
- **Streaming responses** — stream the LLM output token-by-token using Streamlit's `st.write_stream` to improve perceived latency
- **Slack / Teams integration** — expose `answer_query()` behind a bot webhook so users can query without opening the Streamlit app
- **Evaluation harness** — use [RAGAS](https://docs.ragas.io/) to score faithfulness and answer relevancy against a ground-truth question set
- **More data sources** — SharePoint, Notion, Google Drive, email (the `BaseIngestor` interface makes this straightforward to add)

---

## Summary

CorporateRAG demonstrates that a genuinely useful enterprise RAG system doesn't require a large team or complex infrastructure. The core loop — ingest, chunk, embed, filter, retrieve, synthesise — is implemented in roughly 1,500 lines of Python across a clean set of modules. The multi-tenant architecture, encrypted credential storage, and incremental ingestion make it deployable to a real team without significant rework.

The full source code is available at [https://github.com/pyardley/CorporateRAG](https://github.com/pyardley/CorporateRAG).
