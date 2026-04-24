"""
Confluence ingestor – supports multiple Confluence instances.

Page listing uses GET /rest/api/space/{key}/content/page (via
get_all_pages_from_space) rather than CQL search.  CQL relies on a search
index that can lag for new pages and omits spaces with certain restrictions.
The direct content endpoint is always up-to-date and honours API-token auth.

Multiple instances are configured by storing extra credential keys:
  Primary:    url, email, api_token
  Secondary:  url_2, email_2, api_token_2
  Tertiary:   url_3, email_3, api_token_3  (and so on)

During "all" ingestion every instance is queried independently so pages that
exist only on a secondary instance (e.g. a legacy Atlassian site) are captured.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.documents import Document
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingestion.base import BaseIngestor


def _html_to_text(html: str) -> str:
    """
    Convert Confluence storage-format HTML to markdown-ish plain text.
    markdownify preserves table structure so embeddings can correctly answer
    questions like "What is Robin Gherkin's favourite colour?" from a table.
    """
    try:
        import markdownify
        return markdownify.markdownify(html, heading_style="ATX", strip=["script", "style"])
    except Exception:
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(html, "lxml").get_text(separator="\n", strip=True)
        except Exception:
            return html


class ConfluenceIngestor(BaseIngestor):
    SOURCE = "confluence"

    def __init__(self, user_id: str, credentials: dict[str, str]) -> None:
        super().__init__(user_id, credentials)
        self._instances = self._make_instances()

    # ── Instance construction ─────────────────────────────────────────────────

    def _make_instances(self) -> list[tuple]:
        """
        Return a list of (client, base_url) tuples, one per configured instance.
        Credentials are keyed as url/email/api_token (primary) and
        url_2/email_2/api_token_2 (secondary), etc.
        """
        from atlassian import Confluence

        instances: list[tuple] = []
        suffixes = [""] + [f"_{i}" for i in range(2, 10)]

        primary_email = self.credentials.get("email", "").strip()
        primary_token = self.credentials.get("api_token", "").strip()

        for suffix in suffixes:
            url = self.credentials.get(f"url{suffix}", "").strip()
            if not url:
                continue
            # Secondary instances fall back to primary credentials if their own aren't set
            email = self.credentials.get(f"email{suffix}", "").strip() or primary_email
            token = self.credentials.get(f"api_token{suffix}", "").strip() or primary_token

            if not all([email, token]):
                logger.warning(
                    "[confluence] Instance '{}' missing email or api_token — skipping.", url
                )
                continue

            try:
                client = Confluence(url=url, username=email, password=token, cloud=True)
                instances.append((client, url.rstrip("/")))
                logger.info("[confluence] Registered instance: {}", url)
            except Exception as exc:
                logger.warning("[confluence] Could not create client for '{}': {}", url, exc)

        if not instances:
            raise ValueError("Confluence credentials incomplete: need url, email, api_token.")

        return instances

    # ── Abstract implementations ──────────────────────────────────────────────

    def list_scopes(self) -> list[dict[str, str]]:
        """
        Discover all accessible spaces across every configured instance.
        Spaces with the same key on different instances are merged (name from
        the first instance that returns it).
        """
        spaces: dict[str, str] = {}
        for client, _base_url in self._instances:
            self._discover_spaces_for_client(client, spaces)
        scope_list = [{"key": k, "name": v} for k, v in sorted(spaces.items())]
        logger.info("[confluence] Accessible spaces across all instances: {}", [s["key"] for s in scope_list])
        return scope_list

    def load_documents(
        self, scope_key: str, since: Optional[str] = None
    ) -> list[Document]:
        """
        Load pages from all configured Confluence instances.

        For scope_key == "all": discover every instance's spaces independently
        and load all of their pages (handles same space key on different instances
        by loading from both).

        For a specific scope_key: query every instance for that space and combine
        results (deduplicates by page_id).
        """
        docs: list[Document] = []
        seen_page_ids: set[str] = set()

        if scope_key == "all":
            for client, base_url in self._instances:
                instance_spaces: dict[str, str] = {}
                self._discover_spaces_for_client(client, instance_spaces)
                if not instance_spaces:
                    logger.warning("[confluence] No accessible spaces for instance {}", base_url)
                    continue
                logger.info(
                    "[confluence] Instance {}: loading pages from spaces {} (since={})",
                    base_url,
                    list(instance_spaces.keys()),
                    since or "all",
                )
                for space_key in instance_spaces:
                    for doc in self._load_space_pages(space_key, since, client, base_url):
                        page_id = doc.metadata.get("page_id", "")
                        dedup_key = f"{base_url}:{page_id}"
                        if dedup_key not in seen_page_ids:
                            seen_page_ids.add(dedup_key)
                            docs.append(doc)
        else:
            for client, base_url in self._instances:
                for doc in self._load_space_pages(scope_key, since, client, base_url):
                    page_id = doc.metadata.get("page_id", "")
                    dedup_key = f"{base_url}:{page_id}"
                    if dedup_key not in seen_page_ids:
                        seen_page_ids.add(dedup_key)
                        docs.append(doc)

        return docs

    # ── Space discovery ───────────────────────────────────────────────────────

    def _discover_spaces_for_client(self, client, spaces: dict[str, str]) -> None:
        """Populate *spaces* dict from a single Confluence client instance."""
        # Direct spaces API
        try:
            start = 0
            limit = 50
            while True:
                result = client.get_all_spaces(start=start, limit=limit)
                items = result.get("results", []) if isinstance(result, dict) else (result or [])
                if not items:
                    break
                for space in items:
                    key = space.get("key", "")
                    name = space.get("name", "")
                    if key and key not in spaces:
                        spaces[key] = name
                if len(items) < limit:
                    break
                start += len(items)
        except Exception as exc:
            logger.debug("[confluence] Direct spaces API unavailable for {}: {}", client.url, exc)

        # CQL page-scan (catches spaces not in the API listing)
        try:
            start = 0
            limit = 50
            while True:
                result = self._safe_cql_for_client(client, "type = page", start, limit, expand="space")
                items = result.get("results", [])
                for item in items:
                    space = item.get("content", {}).get("space", {})
                    key = space.get("key", "")
                    name = space.get("name", "")
                    if key and key not in spaces:
                        spaces[key] = name
                total_size = result.get("totalSize", 0)
                start += len(items)
                if len(items) < limit or start >= total_size:
                    break
        except Exception as exc:
            logger.debug("[confluence] CQL space discovery unavailable for {}: {}", client.url, exc)

    # ── Per-space loader (direct content API) ─────────────────────────────────

    def _load_space_pages(
        self,
        space_key: str,
        since: Optional[str],
        client,
        base_url: str,
    ) -> list[Document]:
        """
        List all pages in *space_key* via GET /rest/api/space/{key}/content/page,
        then filter by version.when >= since for incremental mode.
        """
        page_stubs: list[dict] = []
        start = 0
        limit = 50

        while True:
            try:
                batch = client.get_all_pages_from_space(
                    space_key, start=start, limit=limit, expand="version"
                )
            except Exception as exc:
                logger.warning(
                    "[confluence] Could not list pages for space '{}' on {}: {}",
                    space_key,
                    base_url,
                    exc,
                )
                break

            if not batch:
                break

            for page in batch:
                if since:
                    last_modified = page.get("version", {}).get("when", "")
                    if last_modified and last_modified < since:
                        continue
                page_stubs.append(page)

            if len(batch) < limit:
                break
            start += len(batch)

        logger.info(
            "[confluence] Space '{}' on {}: {} pages to process (since={})",
            space_key,
            base_url,
            len(page_stubs),
            since or "all",
        )

        docs: list[Document] = []
        for page in page_stubs:
            page_id = str(page.get("id", ""))
            last_modified = page.get("version", {}).get("when", "")

            if not page_id:
                continue

            doc = self._fetch_page_document(page_id, space_key, last_modified, client, base_url)
            if doc:
                docs.append(doc)
            time.sleep(self.RATE_LIMIT_SLEEP)

        return docs

    # ── Page fetch ────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_page_document(
        self,
        page_id: str,
        space_key: str,
        last_modified: str,
        client,
        base_url: str,
    ) -> Optional[Document]:
        """Fetch a single page's full body via the v1 content API."""
        try:
            page = client.get_page_by_id(
                page_id, expand="body.storage,version,space"
            )
        except Exception as exc:
            logger.warning("[confluence] Could not fetch page {}: {}", page_id, exc)
            return None

        title = page.get("title", "(no title)")
        html_body = page.get("body", {}).get("storage", {}).get("value", "")
        plain_text = _html_to_text(html_body)

        version_when = page.get("version", {}).get("when", last_modified)
        actual_space_key = page.get("space", {}).get("key", space_key)

        url = f"{base_url}/wiki/spaces/{actual_space_key}/pages/{page_id}"

        logger.debug(
            "[confluence] Page '{}' (space={}) body chars: {}",
            title,
            actual_space_key,
            len(html_body),
        )

        if not plain_text.strip():
            logger.warning(
                "[confluence] Page '{}' has empty body after conversion — skipping.", title
            )
            return None

        text = f"Confluence Page: {title}\nSpace: {actual_space_key}\n\n{plain_text}"

        metadata: dict[str, Any] = {
            "user_id": self.user_id,
            "source": "confluence",
            "space_key": actual_space_key,
            "page_id": page_id,
            "title": title,
            "url": url,
            "last_updated": version_when,
        }

        return Document(page_content=text, metadata=metadata)

    # ── CQL helpers ───────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _safe_cql_for_client(
        self, client, cql: str, start: int, limit: int, expand: str = ""
    ) -> dict:
        return client.cql(
            cql,
            start=start,
            limit=limit,
            expand=expand if expand else None,
        )

    # ── Scope filter for full-load deletion ──────────────────────────────────

    def _scope_filter(self, scope_key: str) -> dict:
        filt: dict = {"source": "confluence"}
        if scope_key != "all":
            filt["space_key"] = scope_key
        return filt
