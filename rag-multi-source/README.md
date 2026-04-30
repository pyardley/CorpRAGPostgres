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

### Stable resource IDs

| Source     | Format                                                                                  |
| ---------- | --------------------------------------------------------------------------------------- |
| Jira       | `jira:{ISSUE_KEY}` — e.g. `jira:PROJ-123`                                               |
| Confluence | `confluence:page-{page_id}` — e.g. `confluence:page-9876`                               |
| SQL Server | `sql:{server}.{db}.{schema}.{name}`                                                     |
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

### What changed (code map)

| Path                            | Purpose                                        |
| ------------------------------- | ---------------------------------------------- |
| `mcp_server/server.py`          | FastAPI app exposing MCP-style tool endpoints  |
| `mcp_server/tools/sql_tools.py` | Read-only SELECT validator + executor          |
| `mcp_server/config.py`          | `MCP_HOST` / `MCP_PORT` / row caps / token     |
| `core/mcp_client.py`            | `httpx` client + LangChain `StructuredTool`s   |
| `app/mcp_manager.py`            | Spawns/health-checks the MCP child process     |
| `core/mcp_chain.py`             | Hybrid answerer: RAG context + bound MCP tools |
| `app/sidebar.py`                | "⚡ Use Live SQL Table Data (MCP)" toggle      |
| `app/chat.py`                   | Routes through `core.mcp_chain` when toggle on |
| `app/main.py`                   | `ensure_mcp_running()` on first auth load      |

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

The current model is at _project / space / database / repo+branch_
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
