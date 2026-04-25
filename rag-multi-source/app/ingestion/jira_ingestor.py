"""
Jira Cloud ingestor.

resource_id format: ``jira:{ISSUE_KEY}``  (e.g. ``jira:PROJ-123``)
resource_identifier (for the access table): the project key (e.g. ``PROJ``).

Required credential keys (stored encrypted in user_credentials):
  url        - e.g. ``https://your-org.atlassian.net``
  email      - Atlassian account email
  api_token  - https://id.atlassian.com/manage-profile/security/api-tokens
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Optional

import requests
from loguru import logger

from app.ingestion.base import BaseIngestor, SourceResource


class JiraIngestor(BaseIngestor):
    source = "jira"
    PAGE_SIZE = 50
    # Cap how many comments we embed per issue. Inline `comment` field returns
    # at most 50 by default; we'll fetch more from the per-issue comment
    # endpoint if there's overflow.
    MAX_COMMENTS_PER_ISSUE = 200

    # HTTP helpers
    def _session(self) -> requests.Session:
        session = requests.Session()
        session.auth = (self.credentials["email"], self.credentials["api_token"])
        session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )
        return session

    def _base_url(self) -> str:
        return self.credentials["url"].rstrip("/")

    # Abstract API
    def scope_filter(self) -> dict[str, Any]:
        if self.scope == "all":
            return {"source": "jira"}
        return {"source": "jira", "project_key": self.scope}

    def resource_identifier_for(self, resource: SourceResource) -> str:
        return resource.metadata["project_key"]

    def fetch_resources(
        self, since: Optional[datetime] = None
    ) -> Iterable[SourceResource]:
        """
        Iterate over matching issues using the new ``POST /rest/api/3/search/jql``
        endpoint. Atlassian retired the legacy ``GET /rest/api/3/search`` (and
        its ``startAt`` pagination) for Jira Cloud in May 2025 - calls return
        ``410 Gone``. The new endpoint uses cursor pagination via
        ``nextPageToken``.
        """
        session = self._session()
        jql_parts: list[str] = []
        if self.scope != "all":
            jql_parts.append(f'project = "{self.scope}"')
        if since:
            iso = since.strftime("%Y-%m-%d %H:%M")
            jql_parts.append(f'updated >= "{iso}"')
        if not jql_parts:
            jql_parts.append('created >= "1970-01-01"')

        jql = " AND ".join(jql_parts) + " ORDER BY updated DESC"

        url = f"{self._base_url()}/rest/api/3/search/jql"
        fields = [
            "summary", "description", "status", "issuetype", "project",
            "updated", "assignee", "reporter", "labels",
            # `comment` returns the first ~50 comments inline. For tickets
            # with more, _fetch_overflow_comments() fills in the rest from
            # /rest/api/3/issue/{key}/comment.
            "comment",
        ]

        next_page_token: Optional[str] = None
        while True:
            body: dict[str, Any] = {
                "jql": jql,
                "fields": fields,
                "maxResults": self.PAGE_SIZE,
            }
            if next_page_token:
                body["nextPageToken"] = next_page_token

            response = session.post(url, json=body, timeout=30)
            if not response.ok:
                logger.error(
                    "[jira] {} {} -> {}\n  request body: {}\n  response body: {}",
                    response.request.method,
                    response.url,
                    response.status_code,
                    body,
                    response.text[:1000],
                )
                response.raise_for_status()
            payload = response.json()

            issues = payload.get("issues", []) or []
            for issue in issues:
                yield self._issue_to_resource(issue, session)

            next_page_token = payload.get("nextPageToken")
            is_last = payload.get("isLast")
            if is_last is True or not next_page_token:
                break

    # Helpers
    def _issue_to_resource(
        self, issue: dict[str, Any], session: requests.Session
    ) -> SourceResource:
        key = issue["key"]
        fields = issue.get("fields", {})
        summary = fields.get("summary") or ""
        description = _extract_adf_text(fields.get("description")) or ""
        status = (fields.get("status") or {}).get("name") or ""
        issue_type = (fields.get("issuetype") or {}).get("name") or ""
        project_key = ((fields.get("project") or {}).get("key")) or key.split("-")[0]
        updated = fields.get("updated") or datetime.utcnow().isoformat()
        assignee = ((fields.get("assignee") or {}).get("displayName")) or "Unassigned"
        reporter = ((fields.get("reporter") or {}).get("displayName")) or ""
        labels = ", ".join(fields.get("labels") or [])

        comments_block = self._format_comments(key, fields, session)
        comment_count = comments_block.count("\n--- comment ")

        text = (
            f"# {key}: {summary}\n\n"
            f"Type: {issue_type}\n"
            f"Status: {status}\n"
            f"Assignee: {assignee}\n"
            f"Reporter: {reporter}\n"
            f"Labels: {labels}\n\n"
            f"## Description\n{description or '(no description)'}\n"
            f"{comments_block}"
        )

        return SourceResource(
            resource_id=f"jira:{key}",
            title=f"{key} - {summary}",
            text=text,
            url=f"{self._base_url()}/browse/{key}",
            last_updated=updated,
            metadata={
                "project_key": project_key,
                "object_name": key,
                "issue_type": issue_type,
                "status": status,
                "comment_count": comment_count,
            },
        )

    # Comments
    def _format_comments(
        self, issue_key: str, fields: dict[str, Any], session: requests.Session
    ) -> str:
        """
        Return a markdown-ish block summarising the issue's comments, ready
        to append to the chunk text. Empty string if no comments.
        """
        comment_field = fields.get("comment") or {}
        comments = list(comment_field.get("comments") or [])
        total = comment_field.get("total")

        if (
            isinstance(total, int)
            and total > len(comments)
            and len(comments) < self.MAX_COMMENTS_PER_ISSUE
        ):
            comments.extend(
                self._fetch_overflow_comments(
                    issue_key,
                    session,
                    skip=len(comments),
                    cap=self.MAX_COMMENTS_PER_ISSUE - len(comments),
                )
            )

        if not comments:
            return ""

        comments = comments[: self.MAX_COMMENTS_PER_ISSUE]
        out_lines = ["\n## Comments"]
        for i, c in enumerate(comments, start=1):
            author = ((c.get("author") or {}).get("displayName")) or "Unknown"
            created = c.get("created") or ""
            body = _extract_adf_text(c.get("body")).strip()
            if not body:
                continue
            out_lines.append(f"\n--- comment {i} by {author} ({created}) ---\n{body}")
        return "\n".join(out_lines)

    def _fetch_overflow_comments(
        self,
        issue_key: str,
        session: requests.Session,
        skip: int,
        cap: int,
    ) -> list[dict[str, Any]]:
        """Paginate /rest/api/3/issue/{key}/comment for comments beyond the inline batch."""
        out: list[dict[str, Any]] = []
        url = f"{self._base_url()}/rest/api/3/issue/{issue_key}/comment"
        start_at = skip
        page_size = 50
        while len(out) < cap:
            try:
                r = session.get(
                    url,
                    params={"startAt": start_at, "maxResults": page_size},
                    timeout=30,
                )
                r.raise_for_status()
                payload = r.json()
            except Exception as exc:
                logger.warning(
                    "[jira] overflow-comments fetch failed for {}: {}", issue_key, exc
                )
                break
            batch = payload.get("comments") or []
            if not batch:
                break
            out.extend(batch)
            total = payload.get("total")
            start_at += len(batch)
            if isinstance(total, int) and start_at >= total:
                break
        return out[:cap]


# ADF -> plain text
def _extract_adf_text(node: Any) -> str:
    """Walk an ADF document and concatenate its text leaves."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_extract_adf_text(n) for n in node)
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        out = _extract_adf_text(node.get("content"))
        if node.get("type") in {"paragraph", "heading", "bulletList", "orderedList"}:
            out += "\n"
        return out
    return ""
