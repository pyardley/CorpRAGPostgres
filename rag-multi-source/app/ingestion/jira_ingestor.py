"""Jira ingestor – fetches issues (with comments) and indexes them."""

from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.documents import Document
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingestion.base import BaseIngestor


def _adf_to_text(node) -> str:
    """
    Recursively convert an Atlassian Document Format (ADF) node to plain text.
    Jira Cloud returns description and comment bodies as ADF dicts, not strings.
    Falls back gracefully if the value is already a string or empty.
    """
    if not node:
        return ""
    if isinstance(node, str):
        return node

    if not isinstance(node, dict):
        return str(node)

    node_type = node.get("type", "")

    if node_type == "text":
        return node.get("text", "")

    if node_type == "hardBreak":
        return "\n"

    if node_type == "mention":
        attrs = node.get("attrs", {})
        return attrs.get("text", "@mention")

    if node_type == "inlineCard":
        attrs = node.get("attrs", {})
        return attrs.get("url", "")

    children = [_adf_to_text(c) for c in node.get("content", [])]

    if node_type in ("paragraph", "heading"):
        return "".join(children).strip()

    if node_type in ("bulletList", "orderedList"):
        return "\n".join(f"• {c.strip()}" for c in children if c.strip())

    if node_type == "listItem":
        return "".join(children).strip()

    if node_type == "codeBlock":
        return "```\n" + "".join(children) + "\n```"

    if node_type == "blockquote":
        return "> " + "\n> ".join("".join(children).splitlines())

    # doc, table, tableRow, tableCell, etc. — join non-empty children
    return "\n".join(c for c in children if c.strip())


class JiraIngestor(BaseIngestor):
    SOURCE = "jira"

    # Fields to include in issue text
    _INCLUDE_FIELDS = "summary,description,status,priority,issuetype,assignee,reporter,updated,comment"

    def __init__(self, user_id: str, credentials: dict[str, str]) -> None:
        super().__init__(user_id, credentials)
        self._client = self._make_client()

    def _make_client(self):
        from atlassian import Jira

        url = self.credentials.get("url", "")
        email = self.credentials.get("email", "")
        token = self.credentials.get("api_token", "")

        if not all([url, email, token]):
            raise ValueError("Jira credentials incomplete: need url, email, api_token.")

        return Jira(url=url, username=email, password=token, cloud=True)

    # ── Abstract implementations ──────────────────────────────────────────────

    def list_scopes(self) -> list[dict[str, str]]:
        """Return all projects the authenticated user can see."""
        projects = self._client.get_all_projects()
        return [{"key": p["key"], "name": p["name"]} for p in projects]

    def load_documents(
        self, scope_key: str, since: Optional[str] = None
    ) -> list[Document]:
        """
        Load all issues from *scope_key* (or all projects if "all").
        *since* is an ISO-8601 datetime string for incremental mode.
        """
        if scope_key == "all":
            scopes = [s["key"] for s in self.list_scopes()]
        else:
            scopes = [scope_key]

        docs: list[Document] = []
        for proj_key in scopes:
            docs.extend(self._load_project(proj_key, since))

        return docs

    # ── Per-project loading ───────────────────────────────────────────────────

    def _load_project(self, project_key: str, since: Optional[str]) -> list[Document]:
        jql = f"project = {project_key} ORDER BY updated DESC"
        if since:
            # Jira uses "updated >= 'YYYY-MM-DD HH:MM'" format
            since_fmt = since[:16].replace("T", " ")
            jql = f"project = {project_key} AND updated >= '{since_fmt}' ORDER BY updated DESC"

        docs: list[Document] = []
        start = 0
        page_size = 50

        while True:
            logger.debug("[jira] JQL={!r} start={}", jql, start)
            response = self._safe_jql(jql, start, page_size)

            issues = response.get("issues", [])
            if not issues:
                break

            for issue in issues:
                doc = self._issue_to_document(issue, project_key)
                if doc:
                    docs.append(doc)

            total = response.get("total", 0)
            start += len(issues)
            if start >= total:
                break

            time.sleep(self.RATE_LIMIT_SLEEP)

        logger.info("[jira] Loaded {} issues from project {}.", len(docs), project_key)
        return docs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _safe_jql(self, jql: str, start: int, limit: int) -> dict:
        return self._client.jql(jql, limit=limit, start=start, fields=self._INCLUDE_FIELDS)

    # ── Document conversion ───────────────────────────────────────────────────

    def _issue_to_document(self, issue: dict, project_key: str) -> Optional[Document]:
        fields = issue.get("fields", {})
        key = issue.get("key", "")
        summary = fields.get("summary", "(no summary)")
        description = _adf_to_text(fields.get("description") or "")
        status = (fields.get("status") or {}).get("name", "")
        priority = (fields.get("priority") or {}).get("name", "")
        issue_type = (fields.get("issuetype") or {}).get("name", "")
        assignee_obj = fields.get("assignee") or {}
        assignee = assignee_obj.get("displayName", "Unassigned")
        reporter_obj = fields.get("reporter") or {}
        reporter = reporter_obj.get("displayName", "")
        updated = fields.get("updated", "")

        # Collect comments
        comment_texts: list[str] = []
        comments_container = (fields.get("comment") or {})
        for comment in comments_container.get("comments", []):
            author = (comment.get("author") or {}).get("displayName", "")
            body = _adf_to_text(comment.get("body", ""))
            if body:
                comment_texts.append(f"[Comment by {author}]: {body}")

        base_url = self.credentials.get("url", "").rstrip("/")
        url = f"{base_url}/browse/{key}"

        text_parts = [
            f"Jira Issue: {key}",
            f"Summary: {summary}",
            f"Type: {issue_type}  Status: {status}  Priority: {priority}",
            f"Reporter: {reporter}  Assignee: {assignee}",
            f"Description:\n{description}",
        ]
        if comment_texts:
            text_parts.append("Comments:\n" + "\n".join(comment_texts))

        page_content = "\n\n".join(text_parts)

        metadata: dict[str, Any] = {
            "user_id": self.user_id,
            "source": "jira",
            "project_key": project_key,
            "issue_key": key,
            "title": f"{key}: {summary}",
            "url": url,
            "status": status,
            "priority": priority,
            "last_updated": updated,
        }

        return Document(page_content=page_content, metadata=metadata)

    # ── Scope filter for full-load deletion ───────────────────────────────────

    def _scope_filter(self, scope_key: str) -> dict:
        filt: dict = {"source": "jira"}
        if scope_key != "all":
            filt["project_key"] = scope_key
        return filt
