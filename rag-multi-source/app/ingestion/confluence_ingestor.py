"""
Confluence ingestor — supports one or more Confluence Cloud instances.

resource_id format: ``confluence:page-{page_id}`` (page_id is globally unique)
resource_identifier (for the access table): the space_key.

Required credential keys (per instance):
  Primary:   url, email, api_token
  Secondary: url_2, email_2, api_token_2  (and url_3 / _3 / _3, etc.)
Secondary instances fall back to the primary email/token if not set.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Iterable, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingestion.base import BaseIngestor, SourceResource


def _html_to_text(html: str) -> str:
    """Confluence storage-format HTML → markdown-ish plain text."""
    try:
        import markdownify

        return markdownify.markdownify(
            html, heading_style="ATX", strip=["script", "style"]
        )
    except Exception:
        try:
            from bs4 import BeautifulSoup

            return BeautifulSoup(html, "lxml").get_text(separator="\n", strip=True)
        except Exception:
            return html


class ConfluenceIngestor(BaseIngestor):
    source = "confluence"
    RATE_LIMIT_SLEEP = 0.25

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._instances = self._make_instances()

    # ── Instance construction ────────────────────────────────────────────────

    def _make_instances(self) -> list[tuple]:
        from atlassian import Confluence

        instances: list[tuple] = []
        suffixes = [""] + [f"_{i}" for i in range(2, 10)]

        primary_email = (self.credentials.get("email") or "").strip()
        primary_token = (self.credentials.get("api_token") or "").strip()

        for suffix in suffixes:
            url = (self.credentials.get(f"url{suffix}") or "").strip()
            if not url:
                continue
            email = (
                self.credentials.get(f"email{suffix}") or ""
            ).strip() or primary_email
            token = (
                self.credentials.get(f"api_token{suffix}") or ""
            ).strip() or primary_token
            if not (email and token):
                logger.warning(
                    "[confluence] Instance '{}' missing email/token — skipping.", url
                )
                continue
            try:
                client = Confluence(
                    url=url, username=email, password=token, cloud=True
                )
                instances.append((client, url.rstrip("/")))
                logger.info("[confluence] Registered instance: {}", url)
            except Exception as exc:
                logger.warning(
                    "[confluence] Could not create client for '{}': {}", url, exc
                )

        if not instances:
            raise ValueError(
                "Confluence credentials incomplete: need url, email, api_token."
            )
        return instances

    # ── Abstract API ─────────────────────────────────────────────────────────

    def scope_filter(self) -> dict[str, Any]:
        if self.scope == "all":
            return {"source": "confluence"}
        return {"source": "confluence", "space_key": self.scope}

    def resource_identifier_for(self, resource: SourceResource) -> str:
        return resource.metadata["space_key"]

    def fetch_resources(
        self, since: Optional[datetime] = None
    ) -> Iterable[SourceResource]:
        since_iso = since.isoformat() if since else None
        seen_page_ids: set[str] = set()

        for client, base_url in self._instances:
            spaces = self._discover_spaces(client) if self.scope == "all" else [self.scope]
            if not spaces:
                logger.warning(
                    "[confluence] No accessible spaces for instance {}", base_url
                )
                continue

            for space_key in spaces:
                yield from self._iter_space_pages(
                    space_key, since_iso, client, base_url, seen_page_ids
                )

    # ── Space discovery ──────────────────────────────────────────────────────

    def _discover_spaces(self, client) -> list[str]:
        spaces: dict[str, str] = {}
        try:
            start = 0
            limit = 50
            while True:
                result = client.get_all_spaces(start=start, limit=limit)
                items = (
                    result.get("results", [])
                    if isinstance(result, dict)
                    else (result or [])
                )
                if not items:
                    break
                for space in items:
                    key = space.get("key", "")
                    if key:
                        spaces[key] = space.get("name", "")
                if len(items) < limit:
                    break
                start += len(items)
        except Exception as exc:
            logger.debug("[confluence] get_all_spaces failed: {}", exc)
        return sorted(spaces.keys())

    # ── Per-space iterator ───────────────────────────────────────────────────

    def _iter_space_pages(
        self,
        space_key: str,
        since: Optional[str],
        client,
        base_url: str,
        seen_page_ids: set[str],
    ) -> Iterable[SourceResource]:
        start = 0
        limit = 50

        while True:
            try:
                batch = client.get_all_pages_from_space(
                    space_key, start=start, limit=limit, expand="version"
                )
            except Exception as exc:
                logger.warning(
                    "[confluence] List failed for space '{}' on {}: {}",
                    space_key,
                    base_url,
                    exc,
                )
                break
            if not batch:
                break

            for page in batch:
                last_modified = page.get("version", {}).get("when", "") or ""
                if since and last_modified and last_modified < since:
                    continue
                page_id = str(page.get("id", ""))
                if not page_id:
                    continue
                dedup_key = f"{base_url}:{page_id}"
                if dedup_key in seen_page_ids:
                    continue
                seen_page_ids.add(dedup_key)

                resource = self._fetch_page_resource(
                    page_id, space_key, last_modified, client, base_url
                )
                if resource:
                    yield resource
                time.sleep(self.RATE_LIMIT_SLEEP)

            if len(batch) < limit:
                break
            start += len(batch)

    # ── Page fetch ───────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _fetch_page_resource(
        self,
        page_id: str,
        space_key: str,
        last_modified: str,
        client,
        base_url: str,
    ) -> Optional[SourceResource]:
        try:
            page = client.get_page_by_id(
                page_id, expand="body.storage,version,space"
            )
        except Exception as exc:
            logger.warning("[confluence] Page {} fetch failed: {}", page_id, exc)
            return None

        title = page.get("title", "(no title)")
        html_body = page.get("body", {}).get("storage", {}).get("value", "")
        plain_text = _html_to_text(html_body)
        if not plain_text.strip():
            logger.warning(
                "[confluence] Page '{}' empty after conversion — skipping.", title
            )
            return None

        actual_space_key = page.get("space", {}).get("key", space_key)
        version_when = page.get("version", {}).get("when", last_modified)
        url = f"{base_url}/wiki/spaces/{actual_space_key}/pages/{page_id}"

        text = (
            f"Confluence Page: {title}\nSpace: {actual_space_key}\n\n{plain_text}"
        )

        return SourceResource(
            resource_id=f"confluence:page-{page_id}",
            title=title,
            text=text,
            url=url,
            last_updated=version_when or datetime.utcnow().isoformat(),
            metadata={
                "space_key": actual_space_key,
                "page_id": page_id,
                "object_name": title,
            },
        )
