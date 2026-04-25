"""
Git ingestor — indexes commits and file contents from one or more GitHub
repositories.

resource_id format
------------------
    git:{repo_full_name}@{branch}:commit:{sha}
    git:{repo_full_name}@{branch}:file:{path}

resource_identifier (for the access table) is ``{repo_full_name}@{branch}``,
e.g. ``acme/widgets@main``. This is also written to each chunk's
``git_scope`` metadata field, which the dynamic Pinecone filter targets at
query time.

Scope semantics
---------------
* ``--scope all`` → every configured repo's *default* branch.
* ``--scope <branch>`` → that branch on every configured repo (a repo without
  the branch is silently skipped).

Required credential keys
------------------------
Primary repo:
  url            GitHub repo URL (https://github.com/owner/repo)
  access_token   PAT with `repo` scope (required for private repos)
Optional per-repo overrides — same as Confluence's multi-instance pattern:
  url_2, access_token_2, …  (and url_3, etc.)

Optional shared knobs (apply to every repo):
  file_extensions  Comma-separated extensions to index
                   (default: .py .md .txt .yml .yaml .json .js .ts .html .css
                             .sh .sql .toml .ini .cfg .rst)
  max_commits      Max recent commits per branch (default: 200)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Iterable, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingestion.base import BaseIngestor, SourceResource


_DEFAULT_EXTENSIONS = frozenset(
    {
        ".py", ".md", ".txt", ".yml", ".yaml", ".json", ".js", ".ts",
        ".html", ".css", ".sh", ".sql", ".toml", ".ini", ".cfg", ".rst",
    }
)
_MAX_FILE_BYTES = 150_000  # skip files larger than ~150KB


def _parse_repo_path(url: str) -> str:
    """Extract ``owner/repo`` from a GitHub URL (or pass-through if already that)."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    for prefix in (
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
    ):
        if url.startswith(prefix):
            return url[len(prefix):]
    return url  # already owner/repo


class GitIngestor(BaseIngestor):
    source = "git"
    RATE_LIMIT_SLEEP = 0.1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._repos = self._make_repos()
        self._extensions = self._parse_extensions()
        self._max_commits = self._parse_max_commits()

    # ── Repo construction (multi-repo support) ───────────────────────────────

    def _make_repos(self) -> list:
        """Return a list of `github.Repository` handles, one per configured repo."""
        from github import Github

        repos: list = []
        primary_token = (self.credentials.get("access_token") or "").strip()
        suffixes = [""] + [f"_{i}" for i in range(2, 10)]

        for suffix in suffixes:
            url = (self.credentials.get(f"url{suffix}") or "").strip()
            if not url:
                continue
            token = (
                self.credentials.get(f"access_token{suffix}") or ""
            ).strip() or primary_token

            try:
                gh = Github(token) if token else Github()
                repo_path = _parse_repo_path(url)
                repo = gh.get_repo(repo_path)
                logger.info(
                    "[git] Registered repo: {} (default branch: {})",
                    repo.full_name,
                    repo.default_branch,
                )
                repos.append(repo)
            except Exception as exc:
                logger.warning("[git] Could not register {!r}: {}", url, exc)

        if not repos:
            raise ValueError(
                "Git credentials incomplete: need at least one `url` (and "
                "`access_token` for private repos)."
            )
        return repos

    def _parse_extensions(self) -> frozenset[str]:
        raw = (self.credentials.get("file_extensions") or "").strip()
        if not raw:
            return _DEFAULT_EXTENSIONS
        out = set()
        for piece in raw.replace(",", " ").split():
            piece = piece.strip().lower()
            if not piece:
                continue
            out.add(piece if piece.startswith(".") else f".{piece}")
        return frozenset(out) or _DEFAULT_EXTENSIONS

    def _parse_max_commits(self) -> int:
        try:
            return int((self.credentials.get("max_commits") or "200").strip())
        except ValueError:
            return 200

    # ── Abstract API ─────────────────────────────────────────────────────────

    def scope_filter(self) -> dict[str, Any]:
        """
        Filter used to wipe the scope in ``--mode full``.

        For ``--scope all`` we wipe every (repo @ default-branch) we know about,
        because that's what we're going to re-ingest. Any branches outside that
        set are left untouched on purpose — full-mode here means "rebuild what
        I'm about to ingest", not "delete the entire index".
        """
        if self.scope == "all":
            scopes = [f"{r.full_name}@{r.default_branch}" for r in self._repos]
        else:
            scopes = [
                f"{r.full_name}@{self.scope}"
                for r in self._repos
                if _branch_exists(r, self.scope)
            ]
        if not scopes:
            return {"source": "git", "git_scope": {"$in": ["__none__"]}}
        return {"source": "git", "git_scope": {"$in": scopes}}

    def resource_identifier_for(self, resource: SourceResource) -> str:
        return resource.metadata["git_scope"]

    def fetch_resources(
        self, since: Optional[datetime] = None
    ) -> Iterable[SourceResource]:
        for repo in self._repos:
            if self.scope == "all":
                branch = repo.default_branch
            else:
                branch = self.scope
                if not _branch_exists(repo, branch):
                    logger.info(
                        "[git] Branch {!r} not on {} — skipping.",
                        branch,
                        repo.full_name,
                    )
                    continue
            yield from self._iter_commits(repo, branch, since)
            yield from self._iter_files(repo, branch)

    # ── Commits ──────────────────────────────────────────────────────────────

    def _iter_commits(
        self, repo, branch: str, since: Optional[datetime]
    ) -> Iterable[SourceResource]:
        kwargs: dict[str, Any] = {"sha": branch}
        if since:
            # PyGithub expects a tz-aware datetime; assume UTC if naive.
            if since.tzinfo is None:
                from datetime import timezone

                since = since.replace(tzinfo=timezone.utc)
            kwargs["since"] = since

        try:
            paged = repo.get_commits(**kwargs)
        except Exception as exc:
            logger.warning(
                "[git] Could not list commits on {}/{}: {}",
                repo.full_name,
                branch,
                exc,
            )
            return

        for i, commit in enumerate(paged):
            if i >= self._max_commits:
                break
            resource = self._commit_to_resource(commit, repo, branch)
            if resource:
                yield resource
            if i and i % 50 == 0:
                time.sleep(1)  # be gentle with the GH API

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _commit_to_resource(
        self, commit, repo, branch: str
    ) -> Optional[SourceResource]:
        try:
            sha = commit.sha
            message = (commit.commit.message or "").strip()
            author_obj = commit.commit.author
            author = author_obj.name if author_obj else "Unknown"
            date = (
                author_obj.date.isoformat()
                if author_obj and author_obj.date
                else datetime.utcnow().isoformat()
            )

            try:
                files_changed = [f.filename for f in commit.files][:30]
            except Exception:
                files_changed = []

            short_msg = message.split("\n")[0][:200]
            text = (
                f"Git Commit: {sha[:8]}\n"
                f"Repository: {repo.full_name}\n"
                f"Branch: {branch}\n"
                f"Author: {author}\n"
                f"Date: {date}\n"
                f"Message:\n{message}\n"
            )
            if files_changed:
                text += f"\nFiles changed: {', '.join(files_changed)}"

            git_scope = f"{repo.full_name}@{branch}"
            return SourceResource(
                resource_id=f"git:{git_scope}:commit:{sha}",
                title=f"{repo.name}: {short_msg or sha[:8]}",
                text=text,
                url=commit.html_url,
                last_updated=date,
                metadata={
                    "git_scope": git_scope,
                    "repo_name": repo.full_name,
                    "branch": branch,
                    "git_type": "commit",
                    "sha": sha[:8],
                    "object_name": sha[:8],
                },
            )
        except Exception as exc:
            logger.warning("[git] Could not convert commit: {}", exc)
            return None

    # ── Files ────────────────────────────────────────────────────────────────

    def _iter_files(self, repo, branch: str) -> Iterable[SourceResource]:
        try:
            tree = repo.get_git_tree(branch, recursive=True)
        except Exception as exc:
            logger.warning(
                "[git] Could not get tree {}/{}: {}", repo.full_name, branch, exc
            )
            return

        for item in tree.tree:
            if item.type != "blob":
                continue
            path = item.path
            filename = path.rsplit("/", 1)[-1]
            ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
            if ext not in self._extensions:
                continue

            resource = self._file_to_resource(repo, branch, path)
            if resource:
                yield resource
            time.sleep(self.RATE_LIMIT_SLEEP)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _file_to_resource(
        self, repo, branch: str, path: str
    ) -> Optional[SourceResource]:
        try:
            file_obj = repo.get_contents(path, ref=branch)
            if isinstance(file_obj, list):  # directory — shouldn't happen for blobs
                return None
            if file_obj.size and file_obj.size > _MAX_FILE_BYTES:
                logger.debug(
                    "[git] Skipping large file {} ({} bytes)", path, file_obj.size
                )
                return None
            content = file_obj.decoded_content.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("[git] Could not read {}@{}: {}", path, branch, exc)
            return None

        url = f"https://github.com/{repo.full_name}/blob/{branch}/{path}"
        text = (
            f"Git File: {path}\n"
            f"Repository: {repo.full_name}\n"
            f"Branch: {branch}\n\n"
            f"{content}"
        )

        git_scope = f"{repo.full_name}@{branch}"
        return SourceResource(
            resource_id=f"git:{git_scope}:file:{path}",
            title=f"{repo.name}/{path}",
            text=text,
            url=url,
            # GH doesn't return last-modified on get_contents cheaply; use ingest
            # time as a baseline so incremental mode at least has a high-water mark.
            last_updated=datetime.utcnow().isoformat(),
            metadata={
                "git_scope": git_scope,
                "repo_name": repo.full_name,
                "branch": branch,
                "git_type": "file",
                "file_path": path,
                "object_name": path,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _branch_exists(repo, branch: str) -> bool:
    try:
        repo.get_branch(branch)
        return True
    except Exception:
        return False
