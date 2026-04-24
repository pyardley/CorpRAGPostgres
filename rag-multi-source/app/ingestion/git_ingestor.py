"""
Git ingestor — indexes commits and file contents from a GitHub repository.

Credentials stored under source key "git":
  url              GitHub repo URL (e.g. https://github.com/owner/repo)
  access_token     Personal access token with 'repo' scope (required for private repos)
  file_extensions  Comma-separated extensions to index (default: .py .md .txt .yml .yaml .json)
  max_commits      Max recent commits to index per branch (default: 200)

Scopes correspond to branch names.  scope_key="all" indexes the repo's default branch.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.documents import Document
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingestion.base import BaseIngestor

_DEFAULT_EXTENSIONS = frozenset(
    {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".js", ".ts",
     ".html", ".css", ".sh", ".sql", ".toml", ".ini", ".cfg", ".rst"}
)
_MAX_FILE_BYTES = 150_000  # skip files larger than 150 KB


def _parse_repo_path(url: str) -> str:
    """Extract 'owner/repo' from a GitHub URL."""
    url = url.strip().rstrip("/").removesuffix(".git")
    for prefix in (
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
    ):
        if url.startswith(prefix):
            return url[len(prefix):]
    # Assume it's already owner/repo
    return url


class GitIngestor(BaseIngestor):
    SOURCE = "git"

    def __init__(self, user_id: str, credentials: dict[str, str]) -> None:
        super().__init__(user_id, credentials)
        self._gh, self._repo = self._make_client()

    def _make_client(self):
        from github import Github

        url = self.credentials.get("url", "").strip()
        token = self.credentials.get("access_token", "").strip()

        if not url:
            raise ValueError("Git credentials incomplete: need url.")

        g = Github(token) if token else Github()
        repo_path = _parse_repo_path(url)
        repo = g.get_repo(repo_path)
        logger.info("[git] Connected to repo: {} (default branch: {})", repo.full_name, repo.default_branch)
        return g, repo

    # ── Scope = branch ────────────────────────────────────────────────────────

    def list_scopes(self) -> list[dict[str, str]]:
        """Return all branches as scopes."""
        scopes = []
        for branch in self._repo.get_branches():
            scopes.append({"key": branch.name, "name": branch.name})
        return scopes

    def load_documents(
        self, scope_key: str, since: Optional[str] = None
    ) -> list[Document]:
        branch = self._repo.default_branch if scope_key == "all" else scope_key
        logger.info("[git] Loading repo '{}' branch '{}' (since={})", self._repo.full_name, branch, since or "all")
        docs: list[Document] = []
        docs.extend(self._load_commits(branch, since))
        docs.extend(self._load_files(branch))
        return docs

    # ── Commit loader ─────────────────────────────────────────────────────────

    def _load_commits(self, branch: str, since: Optional[str]) -> list[Document]:
        from datetime import datetime, timezone

        max_commits = int(self.credentials.get("max_commits", "200"))
        kwargs: dict[str, Any] = {"sha": branch}
        if since:
            dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            kwargs["since"] = dt

        docs: list[Document] = []
        try:
            commits_paged = self._repo.get_commits(**kwargs)
            for i, commit in enumerate(commits_paged):
                if i >= max_commits:
                    break
                doc = self._commit_to_document(commit, branch)
                if doc:
                    docs.append(doc)
                if i > 0 and i % 50 == 0:
                    time.sleep(1)
        except Exception as exc:
            logger.warning("[git] Could not load commits for branch '{}': {}", branch, exc)

        logger.info("[git] Loaded {} commits from '{}/{}''", len(docs), self._repo.full_name, branch)
        return docs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _commit_to_document(self, commit, branch: str) -> Optional[Document]:
        try:
            sha = commit.sha
            message = commit.commit.message or ""
            author_obj = commit.commit.author
            author = author_obj.name if author_obj else "Unknown"
            date = author_obj.date.isoformat() if author_obj else ""

            try:
                files_changed = [f.filename for f in commit.files][:30]
            except Exception:
                files_changed = []

            short_msg = message.split("\n")[0][:200]
            text = (
                f"Git Commit: {sha[:8]}\n"
                f"Repository: {self._repo.full_name}\n"
                f"Branch: {branch}\n"
                f"Author: {author}\n"
                f"Date: {date}\n"
                f"Message:\n{message}\n"
            )
            if files_changed:
                text += f"\nFiles changed: {', '.join(files_changed)}"

            return Document(
                page_content=text,
                metadata={
                    "user_id": self.user_id,
                    "source": "git",
                    "repo_name": self._repo.full_name,
                    "branch": branch,
                    "git_type": "commit",
                    "sha": sha[:8],
                    "title": f"{self._repo.name}: {short_msg}",
                    "url": commit.html_url,
                    "last_updated": date,
                },
            )
        except Exception as exc:
            logger.warning("[git] Could not convert commit: {}", exc)
            return None

    # ── File loader ───────────────────────────────────────────────────────────

    def _load_files(self, branch: str) -> list[Document]:
        raw_ext = self.credentials.get("file_extensions", "").strip()
        if raw_ext:
            extensions = frozenset(
                e.strip() if e.strip().startswith(".") else f".{e.strip()}"
                for e in raw_ext.split(",") if e.strip()
            )
        else:
            extensions = _DEFAULT_EXTENSIONS

        docs: list[Document] = []
        try:
            tree = self._repo.get_git_tree(branch, recursive=True)
        except Exception as exc:
            logger.warning("[git] Could not get file tree for branch '{}': {}", branch, exc)
            return docs

        items = [item for item in tree.tree if item.type == "blob"]
        for item in items:
            path = item.path
            filename = path.rsplit("/", 1)[-1]
            ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
            if ext not in extensions:
                continue

            doc = self._file_to_document(path, branch)
            if doc:
                docs.append(doc)
            time.sleep(self.RATE_LIMIT_SLEEP)

        logger.info("[git] Loaded {} files from '{}/{}'", len(docs), self._repo.full_name, branch)
        return docs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _file_to_document(self, path: str, branch: str) -> Optional[Document]:
        try:
            file_obj = self._repo.get_contents(path, ref=branch)
            # get_contents can return a list for directories — skip those
            if isinstance(file_obj, list):
                return None
            if file_obj.size > _MAX_FILE_BYTES:
                logger.debug("[git] Skipping large file '{}' ({} bytes)", path, file_obj.size)
                return None
            content = file_obj.decoded_content.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("[git] Could not read file '{}': {}", path, exc)
            return None

        url = f"https://github.com/{self._repo.full_name}/blob/{branch}/{path}"
        text = (
            f"Git File: {path}\n"
            f"Repository: {self._repo.full_name}\n"
            f"Branch: {branch}\n\n"
            f"{content}"
        )

        return Document(
            page_content=text,
            metadata={
                "user_id": self.user_id,
                "source": "git",
                "repo_name": self._repo.full_name,
                "branch": branch,
                "git_type": "file",
                "file_path": path,
                "title": f"{self._repo.name}/{path}",
                "url": url,
                "last_updated": "",
            },
        )

    # ── Scope filter for full-load deletion ──────────────────────────────────

    def _scope_filter(self, scope_key: str) -> dict:
        filt: dict = {"source": "git", "repo_name": self._repo.full_name}
        if scope_key != "all":
            filt["branch"] = scope_key
        return filt
