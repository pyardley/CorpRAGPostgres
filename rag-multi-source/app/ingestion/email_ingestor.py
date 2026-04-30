"""
Email ingestor — supports Outlook / Microsoft 365 (Graph API), Gmail
(Google API), and Yahoo Mail (IMAP + App Password), in any combination,
from a single ingestion run.

resource_id format
------------------
    email:outlook:message:{graph_message_id}
    email:gmail:message:{gmail_message_id}
    email:yahoo:message:{rfc822_message_id_or_uid_hash}

resource_identifier (for the access table) is the **provider name**
(``outlook`` / ``gmail`` / ``yahoo``), so users see and can revoke each
mailbox independently. The same value is also written to the
``email_provider`` column on ``vector_chunks`` for query-time filtering.

Scope semantics
---------------
* ``--scope all`` (default) → every configured provider.
* ``--scope outlook``       → Outlook / M365 mailbox only.
* ``--scope gmail``         → Gmail mailbox only.
* ``--scope yahoo``         → Yahoo Mail mailbox only.
* Any other value           → treated as a folder/label name applied to
  every configured provider (Outlook well-known folder name, e.g.
  ``inbox``, ``sentitems``; Gmail label name, e.g. ``INBOX``,
  ``IMPORTANT``; or Yahoo IMAP folder, e.g. ``INBOX``, ``Sent``).

Modes
-----
The :class:`BaseIngestor` contract only knows ``full`` / ``incremental``.
We accept a third logical mode — ``month`` — and translate it to
``incremental`` for the base-class state machine while applying an
explicit ``[start, end)`` date window to the provider iterators.

* ``incremental`` (default) — since = last successful run's high-water
  mark; if there is none, fall back to the **previous calendar month**
  (the requirement-driven default for a brand-new mailbox).
* ``full``                  — no date floor; ingest every message the
  account can see (CLI-only, dangerous for big mailboxes).
* ``month`` + ``--month YYYY-MM`` — exactly that calendar month.

Required credentials (stored encrypted under source=``email``)
--------------------------------------------------------------
Outlook (any one of the two grant types):
  outlook_tenant_id           Azure AD tenant (or "common" for personal
                              accounts / outlook.com)
  outlook_client_id           App registration's application ID
  outlook_client_secret       (optional) for client-credentials grant
  outlook_refresh_token       (preferred) delegated refresh token for
                              the user — used with the public-client flow
  outlook_user                Mailbox UPN, e.g. paul@contoso.com.
                              When using ``/me`` flows, leave blank.
  outlook_folder              (optional) well-known folder filter
                              (default: no folder filter — all mail).

Gmail:
  gmail_client_id
  gmail_client_secret
  gmail_refresh_token         OAuth2 refresh token from a one-time
                              installed-app flow (recommended).
  gmail_token_json            (optional) full token.json blob — used
                              instead of the discrete keys above when
                              present.
  gmail_user                  Mailbox address, e.g. me@gmail.com
                              (defaults to "me", which Google resolves
                              to the authenticated account).
  gmail_label                 (optional) label filter (default: no label
                              filter — all mail; use "INBOX" to limit).

Yahoo Mail (IMAP + App Password):
  yahoo_email_address         Mailbox to authenticate as, e.g.
                              ``you@yahoo.com``. Used as the IMAP
                              ``LOGIN`` username.
  yahoo_app_password          16-character mailbox-scoped app password
                              generated at
                              login.yahoo.com → Account → Security →
                              "Generate app password". Yahoo deprecated
                              public OAuth2 access to Mail content for
                              new third-party apps; app passwords are
                              the supported integration today (see
                              uk.help.yahoo.com/kb/sln4075). Treat like
                              a real password — anyone holding it has
                              full read access to the mailbox until it
                              is revoked from the same screen.
  yahoo_folder                (optional) IMAP folder name, default
                              ``INBOX``. Common values: ``INBOX``,
                              ``Sent``, ``Bulk Mail``, ``Archive``.

Rate-limiting and reliability
-----------------------------
* Tenacity exponential backoff on every transient HTTP failure.
* Small inter-call sleeps to be polite to both APIs.
* Body content is HTML-stripped to plain text via :mod:`markdownify` (with
  a BeautifulSoup fallback) so RAG quality stays consistent with
  Confluence.
"""

from __future__ import annotations

import base64
import email as email_lib
import hashlib
import imaplib
import json
import os
import time
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Iterable, Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingestion.base import BaseIngestor, SourceResource


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """Strip HTML to readable plain text (markdownify → BS4 → raw)."""
    if not html:
        return ""
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


def _previous_calendar_month_window(
    today: Optional[datetime] = None,
) -> tuple[datetime, datetime]:
    """Return (start_of_prev_month_utc, start_of_this_month_utc)."""
    today = today or datetime.now(timezone.utc)
    first_of_this = today.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    last_of_prev = first_of_this - timedelta(seconds=1)
    first_of_prev = last_of_prev.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    return first_of_prev, first_of_this


def _month_window(month: str) -> tuple[datetime, datetime]:
    """``YYYY-MM`` → (first_of_month_utc, first_of_next_month_utc)."""
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    days = monthrange(start.year, start.month)[1]
    end = start + timedelta(days=days)
    return start, end


def _format_addresses(values: Iterable[Any]) -> str:
    """Render Graph/Gmail address objects as ``Name <email>; …``."""
    out: list[str] = []
    for v in values or []:
        if isinstance(v, dict):
            ea = v.get("emailAddress") or {}
            name = ea.get("name") or v.get("name") or ""
            addr = ea.get("address") or v.get("address") or ""
            if name and addr:
                out.append(f"{name} <{addr}>")
            elif addr:
                out.append(addr)
        elif isinstance(v, str):
            out.append(v.strip())
    return "; ".join(p for p in out if p)


# ──────────────────────────────────────────────────────────────────────────────
# Outlook / Microsoft Graph provider
# ──────────────────────────────────────────────────────────────────────────────

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_GRAPH_SCOPE = "https://graph.microsoft.com/.default"
_OAUTH2_TOKEN_URL = (
    "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
)


class _OutlookProvider:
    """Thin Microsoft Graph wrapper that yields :class:`SourceResource`s."""

    name = "outlook"
    PAGE_SIZE = 50
    RATE_LIMIT_SLEEP = 0.1

    def __init__(self, credentials: dict[str, str]) -> None:
        self.tenant_id = (credentials.get("outlook_tenant_id") or "common").strip()
        self.client_id = (credentials.get("outlook_client_id") or "").strip()
        self.client_secret = (
            credentials.get("outlook_client_secret") or ""
        ).strip()
        self.refresh_token = (
            credentials.get("outlook_refresh_token") or ""
        ).strip()
        self.user = (credentials.get("outlook_user") or "").strip()
        self.folder = (credentials.get("outlook_folder") or "").strip()

        if not self.client_id:
            raise ValueError("outlook_client_id is required")
        if not (self.refresh_token or self.client_secret):
            raise ValueError(
                "Outlook needs either outlook_refresh_token (delegated) or "
                "outlook_client_secret (app-only)."
            )

        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    # ── Auth ────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _refresh_access_token(self) -> None:
        url = _OAUTH2_TOKEN_URL.format(tenant=self.tenant_id)
        if self.refresh_token:
            body = {
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "scope": (
                    "offline_access https://graph.microsoft.com/Mail.Read"
                ),
            }
            if self.client_secret:
                body["client_secret"] = self.client_secret
        else:
            body = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                "scope": _GRAPH_SCOPE,
            }
        resp = requests.post(url, data=body, timeout=30)
        if not resp.ok:
            logger.error(
                "[email/outlook] token request failed: {} {}",
                resp.status_code,
                resp.text[:500],
            )
            resp.raise_for_status()
        payload = resp.json()
        self._access_token = payload["access_token"]
        # Refresh ~1 minute before expiry.
        self._token_expires_at = time.time() + int(payload.get("expires_in", 3600)) - 60
        # Microsoft sometimes rotates the refresh token on use.
        if "refresh_token" in payload:
            self.refresh_token = payload["refresh_token"]

    def _auth_headers(self) -> dict[str, str]:
        if not self._access_token or time.time() >= self._token_expires_at:
            self._refresh_access_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    def _mailbox_root(self) -> str:
        if self.user and self.client_secret and not self.refresh_token:
            # App-only flow → must address the mailbox by UPN.
            return f"/users/{self.user}"
        return "/me"

    # ── Iteration ───────────────────────────────────────────────────────────

    def fetch(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        folder_override: Optional[str] = None,
    ) -> Iterable[SourceResource]:
        folder = (folder_override or self.folder or "").strip()
        path = self._mailbox_root()
        if folder:
            path += f"/mailFolders/{folder}/messages"
        else:
            path += "/messages"

        params: dict[str, Any] = {
            "$top": self.PAGE_SIZE,
            "$select": (
                "id,internetMessageId,subject,from,toRecipients,ccRecipients,"
                "bccRecipients,receivedDateTime,bodyPreview,body,"
                "hasAttachments,webLink,parentFolderId,conversationId"
            ),
            "$orderby": "receivedDateTime desc",
        }
        filters: list[str] = []
        if start:
            filters.append(
                f"receivedDateTime ge {start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )
        if end:
            filters.append(
                f"receivedDateTime lt {end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )
        if filters:
            params["$filter"] = " and ".join(filters)

        url = f"{_GRAPH_BASE}{path}"
        next_url: Optional[str] = None

        while True:
            resp = self._get(next_url or url, params=None if next_url else params)
            data = resp.json()
            for msg in data.get("value", []) or []:
                resource = self._message_to_resource(msg)
                if resource:
                    yield resource
                time.sleep(self.RATE_LIMIT_SLEEP)
            next_url = data.get("@odata.nextLink")
            if not next_url:
                break

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        resp = requests.get(
            url, headers=self._auth_headers(), params=params, timeout=30
        )
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            logger.warning(
                "[email/outlook] 429 throttled — sleeping {}s", retry_after
            )
            time.sleep(retry_after)
            resp.raise_for_status()
        if not resp.ok:
            logger.error(
                "[email/outlook] {} -> {}: {}",
                url,
                resp.status_code,
                resp.text[:500],
            )
            resp.raise_for_status()
        return resp

    # ── Message conversion ──────────────────────────────────────────────────

    def _message_to_resource(self, msg: dict[str, Any]) -> Optional[SourceResource]:
        msg_id = msg.get("id")
        if not msg_id:
            return None

        subject = msg.get("subject") or "(no subject)"
        received = msg.get("receivedDateTime") or datetime.now(
            timezone.utc
        ).isoformat()
        sender = _format_addresses([msg.get("from")]) or "(unknown)"
        to_list = _format_addresses(msg.get("toRecipients") or [])
        cc_list = _format_addresses(msg.get("ccRecipients") or [])

        body = msg.get("body") or {}
        content_type = (body.get("contentType") or "").lower()
        raw = body.get("content") or msg.get("bodyPreview") or ""
        plain = _html_to_text(raw) if content_type == "html" else raw
        plain = (plain or "").strip()

        attachments_summary = ""
        if msg.get("hasAttachments"):
            attachments_summary = "[has attachments]"

        text = (
            f"Email (Outlook): {subject}\n"
            f"From: {sender}\n"
            f"To: {to_list}\n"
            + (f"Cc: {cc_list}\n" if cc_list else "")
            + f"Date: {received}\n"
            + (f"{attachments_summary}\n" if attachments_summary else "")
            + "\n"
            + plain
        )

        url = msg.get("webLink") or ""
        return SourceResource(
            resource_id=f"email:outlook:message:{msg_id}",
            title=subject,
            text=text,
            url=url,
            last_updated=received,
            metadata={
                "email_provider": "outlook",
                "object_name": subject,
                "from": sender,
                "to": to_list,
                "cc": cc_list,
                "message_id": msg_id,
                "internet_message_id": msg.get("internetMessageId", ""),
                "conversation_id": msg.get("conversationId", ""),
                "folder_id": msg.get("parentFolderId", ""),
                "has_attachments": bool(msg.get("hasAttachments")),
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# Gmail provider
# ──────────────────────────────────────────────────────────────────────────────

_GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class _GmailProvider:
    """Gmail API wrapper that yields :class:`SourceResource`s."""

    name = "gmail"
    PAGE_SIZE = 100
    RATE_LIMIT_SLEEP = 0.05

    def __init__(self, credentials: dict[str, str]) -> None:
        self.client_id = (credentials.get("gmail_client_id") or "").strip()
        self.client_secret = (credentials.get("gmail_client_secret") or "").strip()
        self.refresh_token = (credentials.get("gmail_refresh_token") or "").strip()
        self.token_json = (credentials.get("gmail_token_json") or "").strip()
        self.user = (credentials.get("gmail_user") or "me").strip() or "me"
        self.label = (credentials.get("gmail_label") or "").strip()

        if not self.token_json and not (
            self.client_id and self.client_secret and self.refresh_token
        ):
            raise ValueError(
                "Gmail needs either gmail_token_json or "
                "(gmail_client_id, gmail_client_secret, gmail_refresh_token)."
            )

        self._service = None  # lazy

    # ── Auth + service construction ─────────────────────────────────────────

    def _build_service(self):
        if self._service is not None:
            return self._service

        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        if self.token_json:
            try:
                info = json.loads(self.token_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"gmail_token_json is not valid JSON: {exc}")
            creds = Credentials.from_authorized_user_info(info, _GMAIL_SCOPES)
        else:
            creds = Credentials(
                token=None,
                refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=_GMAIL_SCOPES,
            )

        if not creds.valid:
            from google.auth.transport.requests import Request

            creds.refresh(Request())

        self._service = build(
            "gmail", "v1", credentials=creds, cache_discovery=False
        )
        return self._service

    # ── Iteration ───────────────────────────────────────────────────────────

    def fetch(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        folder_override: Optional[str] = None,
    ) -> Iterable[SourceResource]:
        service = self._build_service()
        label = (folder_override or self.label or "").strip()

        # Gmail's `q` parameter accepts `after:` and `before:` with epoch
        # seconds (also accepts YYYY/MM/DD; epoch is more precise).
        q_parts: list[str] = []
        if start:
            q_parts.append(f"after:{int(start.timestamp())}")
        if end:
            q_parts.append(f"before:{int(end.timestamp())}")
        q = " ".join(q_parts) if q_parts else None

        list_kwargs: dict[str, Any] = {
            "userId": self.user,
            "maxResults": self.PAGE_SIZE,
        }
        if q:
            list_kwargs["q"] = q
        if label:
            list_kwargs["labelIds"] = [label]

        page_token: Optional[str] = None
        while True:
            if page_token:
                list_kwargs["pageToken"] = page_token
            resp = self._with_retry(
                lambda: service.users().messages().list(**list_kwargs).execute()
            )
            for stub in resp.get("messages", []) or []:
                msg = self._with_retry(
                    lambda: service.users()
                    .messages()
                    .get(userId=self.user, id=stub["id"], format="full")
                    .execute()
                )
                resource = self._message_to_resource(msg)
                if resource:
                    yield resource
                time.sleep(self.RATE_LIMIT_SLEEP)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _with_retry(self, fn):
        try:
            return fn()
        except Exception as exc:
            from googleapiclient.errors import HttpError

            if isinstance(exc, HttpError) and exc.resp.status in (
                429, 500, 502, 503, 504,
            ):
                logger.warning(
                    "[email/gmail] transient {}: backing off…", exc.resp.status
                )
                raise
            raise

    # ── Message conversion ──────────────────────────────────────────────────

    def _message_to_resource(
        self, msg: dict[str, Any]
    ) -> Optional[SourceResource]:
        msg_id = msg.get("id")
        if not msg_id:
            return None

        headers = {
            h.get("name", "").lower(): h.get("value", "")
            for h in (msg.get("payload", {}).get("headers") or [])
        }
        subject = headers.get("subject") or "(no subject)"
        sender = headers.get("from") or "(unknown)"
        to_list = headers.get("to") or ""
        cc_list = headers.get("cc") or ""

        # internalDate is ms since epoch as a string.
        try:
            internal_ms = int(msg.get("internalDate") or "0")
        except ValueError:
            internal_ms = 0
        received_dt = datetime.fromtimestamp(internal_ms / 1000, tz=timezone.utc)
        received_iso = received_dt.isoformat()

        plain, html, attachments = self._extract_body_and_attachments(
            msg.get("payload") or {}
        )
        body_text = plain or _html_to_text(html)
        body_text = (body_text or "").strip()

        attachments_summary = (
            f"[{len(attachments)} attachment(s): "
            f"{', '.join(a[:60] for a in attachments[:5])}"
            f"{'…' if len(attachments) > 5 else ''}]"
            if attachments
            else ""
        )

        text = (
            f"Email (Gmail): {subject}\n"
            f"From: {sender}\n"
            f"To: {to_list}\n"
            + (f"Cc: {cc_list}\n" if cc_list else "")
            + f"Date: {received_iso}\n"
            + (f"{attachments_summary}\n" if attachments_summary else "")
            + "\n"
            + body_text
        )

        url = (
            f"https://mail.google.com/mail/u/0/#all/"
            f"{msg.get('threadId') or msg_id}"
        )

        return SourceResource(
            resource_id=f"email:gmail:message:{msg_id}",
            title=subject,
            text=text,
            url=url,
            last_updated=received_iso,
            metadata={
                "email_provider": "gmail",
                "object_name": subject,
                "from": sender,
                "to": to_list,
                "cc": cc_list,
                "message_id": msg_id,
                "thread_id": msg.get("threadId", ""),
                "internet_message_id": headers.get("message-id", ""),
                "labels": msg.get("labelIds") or [],
                "has_attachments": bool(attachments),
            },
        )

    @staticmethod
    def _extract_body_and_attachments(
        payload: dict[str, Any],
    ) -> tuple[str, str, list[str]]:
        """Walk the MIME tree; return (plain, html, attachment_filenames)."""
        plain_parts: list[str] = []
        html_parts: list[str] = []
        attachments: list[str] = []

        def walk(part: dict[str, Any]) -> None:
            mime = part.get("mimeType", "")
            filename = part.get("filename") or ""
            body = part.get("body") or {}
            data = body.get("data")
            if filename:
                attachments.append(filename)
                # We don't pull attachment bytes — too costly + risky.
                return
            if mime == "text/plain" and data:
                plain_parts.append(_b64url_decode(data))
            elif mime == "text/html" and data:
                html_parts.append(_b64url_decode(data))
            for child in part.get("parts") or []:
                walk(child)

        walk(payload)
        return "\n".join(plain_parts), "\n".join(html_parts), attachments


def _b64url_decode(s: str) -> str:
    pad = "=" * (-len(s) % 4)
    try:
        return base64.urlsafe_b64decode(s + pad).decode("utf-8", errors="replace")
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Yahoo Mail provider (IMAP + App Password)
# ──────────────────────────────────────────────────────────────────────────────

_YAHOO_IMAP_HOST = "imap.mail.yahoo.com"
_YAHOO_IMAP_PORT = 993


class _YahooProvider:
    """
    Yahoo Mail wrapper using IMAP4 + a mailbox-scoped app password.

    Why an app password and not OAuth
    ---------------------------------
    Yahoo discontinued public OAuth2 access to Mail content for new
    third-party apps. The Yahoo Developer Console no longer exposes
    the ``mail-r`` scope under *API Permissions*, and Yahoo's official
    IMAP help (uk.help.yahoo.com/kb/sln4075) instructs users to
    authenticate with a 16-character app password generated under
    *Account → Security → Generate app password*. We follow that
    documented path:

    1. Open ``imap.mail.yahoo.com:993`` over SSL.
    2. Plain ``LOGIN <email> <app_password>`` (RFC 3501 §6.2.3). The
       app password is mailbox-scoped, revocable independently of the
       user's account password, and only valid for IMAP/SMTP/POP — it
       cannot be used to sign in to the web UI, so the blast radius is
       contained even if it leaks.
    3. ``SELECT`` the requested folder read-only.
    4. ``UID SEARCH`` with ``SINCE``/``BEFORE`` to honour the date
       window from :class:`EmailIngestor._compute_window`.
    5. ``UID FETCH (RFC822)`` and parse with the stdlib :mod:`email`
       module — same body / attachment extraction story as Gmail.

    Note on legacy OAuth2 (XOAUTH2)
    -------------------------------
    A previous revision of this module supported XOAUTH2 against
    Yahoo's ``oauth2/get_token`` endpoint. That code was removed
    because it could not authenticate any newly-registered Yahoo
    Developer app — Yahoo silently rejects bearer tokens whose client
    doesn't have the legacy ``mail-r`` permission. If you have a
    legacy app that still works that way, the integration option is to
    expose IMAP via an account-level alias rather than re-introducing
    XOAUTH2 here.
    """

    name = "yahoo"
    BATCH_SIZE = 50
    RATE_LIMIT_SLEEP = 0.05

    def __init__(self, credentials: dict[str, str]) -> None:
        self.email_address = (
            credentials.get("yahoo_email_address") or ""
        ).strip()
        # Yahoo's UI displays app passwords with whitespace in groups
        # of four (``xxxx xxxx xxxx xxxx``). The IMAP server doesn't
        # care about the spaces, but normalising here removes a
        # paste-error footgun.
        self.app_password = (
            (credentials.get("yahoo_app_password") or "")
            .replace(" ", "")
            .strip()
        )
        self.folder = (credentials.get("yahoo_folder") or "INBOX").strip() or "INBOX"

        if not self.email_address or "@" not in self.email_address:
            raise ValueError(
                "yahoo_email_address must be the full mailbox address, "
                "e.g. 'you@yahoo.com'."
            )
        if not self.app_password:
            raise ValueError(
                "yahoo_app_password is required. Generate one at "
                "login.yahoo.com → Account → Security → 'Generate app "
                "password'. (Yahoo deprecated OAuth2 for Mail; app "
                "passwords are the supported credential.)"
            )

        self._mail: Optional[imaplib.IMAP4_SSL] = None

    # ── IMAP connection ─────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _connect_and_auth(self) -> imaplib.IMAP4_SSL:
        """
        Open a fresh IMAP SSL connection and authenticate via plain
        ``LOGIN``. Yahoo's gateway accepts the SASL ``LOGIN``/``PLAIN``
        mechanisms with the email address as the username and the
        16-character app password as the password.

        The retry decorator handles transient TLS or DNS hiccups; auth
        errors (bad credentials) are NOT retried — re-issuing the same
        wrong password just locks the account out faster.
        """
        try:
            mail = imaplib.IMAP4_SSL(
                _YAHOO_IMAP_HOST, _YAHOO_IMAP_PORT, timeout=30
            )
        except (OSError, imaplib.IMAP4.error) as exc:
            logger.warning(
                "[email/yahoo] could not connect to {}:{}: {}",
                _YAHOO_IMAP_HOST,
                _YAHOO_IMAP_PORT,
                exc,
            )
            raise

        try:
            mail.login(self.email_address, self.app_password)
        except imaplib.IMAP4.error as exc:
            # Auth failures are terminal — surface them clearly so the
            # user sees "AUTHENTICATE failed" / "Invalid credentials"
            # in the ingestion log instead of a generic IMAP error.
            try:
                mail.logout()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"Yahoo IMAP login failed for {self.email_address}: "
                f"{exc}. Check that the app password is correct and "
                f"hasn't been revoked, and that the mailbox address "
                f"matches the one that issued it."
            ) from exc

        self._mail = mail
        return mail

    def _close(self) -> None:
        if not self._mail:
            return
        try:
            self._mail.close()
        except Exception:  # noqa: BLE001 - already in teardown
            pass
        try:
            self._mail.logout()
        except Exception:  # noqa: BLE001
            pass
        self._mail = None

    # ── Iteration ───────────────────────────────────────────────────────────

    def fetch(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
        folder_override: Optional[str] = None,
    ) -> Iterable[SourceResource]:
        folder = (folder_override or self.folder).strip() or "INBOX"
        mail = self._connect_and_auth()
        try:
            # SELECT in read-only mode — we never want to mutate the
            # remote mailbox state (no \Seen flag updates, etc.).
            typ, data = mail.select(self._encode_folder(folder), readonly=True)
            if typ != "OK":
                raise RuntimeError(
                    f"IMAP SELECT {folder!r} failed: {data}"
                )
            logger.info(
                "[email/yahoo] selected folder {} ({} messages total)",
                folder,
                (data[0].decode() if data and data[0] else "?"),
            )

            criteria = self._build_search_criteria(start, end)
            typ, search_data = mail.uid("SEARCH", None, *criteria)
            if typ != "OK":
                raise RuntimeError(
                    f"IMAP SEARCH {criteria} failed: {search_data}"
                )

            uids: list[bytes] = (
                search_data[0].split() if search_data and search_data[0] else []
            )
            logger.info(
                "[email/yahoo] {} messages match window {}",
                len(uids),
                criteria,
            )

            # Stream UIDs in modest batches so a mid-run failure leaves
            # most of the work durably committed by the upstream
            # vector-store batcher.
            for i in range(0, len(uids), self.BATCH_SIZE):
                batch = uids[i : i + self.BATCH_SIZE]
                for uid in batch:
                    try:
                        resource = self._fetch_one(mail, uid, folder)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[email/yahoo] uid={} fetch failed: {}",
                            uid.decode(errors="replace"),
                            exc,
                        )
                        continue
                    if resource:
                        yield resource
                    time.sleep(self.RATE_LIMIT_SLEEP)
        finally:
            self._close()

    @staticmethod
    def _build_search_criteria(
        start: Optional[datetime], end: Optional[datetime]
    ) -> list[str]:
        """
        Translate a ``[start, end)`` window into IMAP SEARCH tokens.

        IMAP date format is ``DD-Mon-YYYY`` (RFC 3501 §6.4.4). The
        ``SINCE`` keyword is *inclusive of the day*, so for a half-open
        window we widen the BEFORE bound by one day to keep the same
        semantics the Graph and Gmail providers use.
        """
        criteria: list[str] = []
        if start is not None:
            criteria += ["SINCE", start.strftime("%d-%b-%Y")]
        if end is not None:
            # +1 day so [start, end) maps to SINCE/BEFORE inclusively.
            end_inclusive = end + timedelta(days=1)
            criteria += ["BEFORE", end_inclusive.strftime("%d-%b-%Y")]
        if not criteria:
            criteria = ["ALL"]
        return criteria

    @staticmethod
    def _encode_folder(folder: str) -> str:
        """
        IMAP folder names with spaces or non-ASCII characters need
        quoting. Yahoo's well-known names like ``Bulk Mail`` are the
        common case worth handling here.
        """
        if " " in folder or '"' in folder:
            escaped = folder.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return folder

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _fetch_one(
        self, mail: imaplib.IMAP4_SSL, uid: bytes, folder: str
    ) -> Optional[SourceResource]:
        typ, data = mail.uid("FETCH", uid, "(RFC822)")
        if typ != "OK" or not data or not data[0]:
            logger.debug("[email/yahoo] uid={} returned {}", uid, typ)
            return None
        # data is e.g. [(b'1 (RFC822 {N}', <bytes>), b')']
        for entry in data:
            if isinstance(entry, tuple) and len(entry) >= 2:
                raw = entry[1]
                if isinstance(raw, bytes) and raw:
                    return self._parse_message(raw, uid, folder)
        return None

    # ── Message conversion ──────────────────────────────────────────────────

    def _parse_message(
        self, raw: bytes, uid: bytes, folder: str
    ) -> Optional[SourceResource]:
        msg = email_lib.message_from_bytes(raw)

        subject = self._decoded_header(msg, "Subject") or "(no subject)"
        sender = self._decoded_header(msg, "From") or "(unknown)"
        to_list = self._decoded_header(msg, "To") or ""
        cc_list = self._decoded_header(msg, "Cc") or ""
        message_id = (self._decoded_header(msg, "Message-ID") or "").strip()
        date_hdr = self._decoded_header(msg, "Date") or ""

        try:
            received_dt = parsedate_to_datetime(date_hdr) if date_hdr else None
        except (TypeError, ValueError):
            received_dt = None
        if received_dt is None:
            received_dt = datetime.now(timezone.utc)
        if received_dt.tzinfo is None:
            received_dt = received_dt.replace(tzinfo=timezone.utc)
        received_iso = received_dt.isoformat()

        plain, html, attachments = self._extract_body_and_attachments(msg)
        body_text = (plain or _html_to_text(html) or "").strip()

        attachments_summary = (
            f"[{len(attachments)} attachment(s): "
            f"{', '.join(a[:60] for a in attachments[:5])}"
            f"{'…' if len(attachments) > 5 else ''}]"
            if attachments
            else ""
        )

        text = (
            f"Email (Yahoo): {subject}\n"
            f"From: {sender}\n"
            f"To: {to_list}\n"
            + (f"Cc: {cc_list}\n" if cc_list else "")
            + f"Date: {received_iso}\n"
            + (f"{attachments_summary}\n" if attachments_summary else "")
            + "\n"
            + body_text
        )

        # Prefer the globally-unique RFC822 Message-ID; fall back to a
        # deterministic hash of (folder, uid, date) so that re-running
        # against the same mailbox produces the same resource_id and
        # the upstream INSERT … ON CONFLICT dedupe path still works.
        if message_id:
            stable_id = message_id.strip("<>")
        else:
            seed = f"{folder}|{uid.decode(errors='replace')}|{received_iso}"
            stable_id = hashlib.sha1(seed.encode()).hexdigest()

        # Yahoo doesn't expose a deep-link URL the way Outlook/Graph
        # does; the folder-rooted webmail URL is the closest analogue.
        url = "https://mail.yahoo.com/d/folders/1"

        return SourceResource(
            resource_id=f"email:yahoo:message:{stable_id}",
            title=subject,
            text=text,
            url=url,
            last_updated=received_iso,
            metadata={
                "email_provider": "yahoo",
                "object_name": subject,
                "from": sender,
                "to": to_list,
                "cc": cc_list,
                "message_id": stable_id,
                "internet_message_id": message_id,
                "imap_uid": uid.decode(errors="replace"),
                "folder": folder,
                "has_attachments": bool(attachments),
            },
        )

    @staticmethod
    def _decoded_header(msg: "email_lib.message.Message", name: str) -> str:
        """
        RFC 2047-decode a header (``=?utf-8?B?…?=`` etc.) into a single
        Python string. Returns "" if absent.
        """
        from email.header import decode_header, make_header

        raw = msg.get(name)
        if raw is None:
            return ""
        try:
            return str(make_header(decode_header(raw)))
        except Exception:  # noqa: BLE001
            return raw if isinstance(raw, str) else raw.decode(errors="replace")

    @staticmethod
    def _extract_body_and_attachments(
        msg: "email_lib.message.Message",
    ) -> tuple[str, str, list[str]]:
        """
        Walk the MIME tree the same way the Gmail provider does.
        Returns ``(plain_text, html_text, attachment_filenames)``.
        Attachment bytes are intentionally NOT downloaded — too costly
        and risky to embed unsanitised binary content into RAG.
        """
        plain_parts: list[str] = []
        html_parts: list[str] = []
        attachments: list[str] = []

        def _decode(part: "email_lib.message.Message") -> str:
            payload = part.get_payload(decode=True)
            if not payload:
                return ""
            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="replace")
            except (LookupError, TypeError):
                return payload.decode("utf-8", errors="replace")

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disposition = (str(part.get("Content-Disposition") or "")).lower()
                filename = part.get_filename()
                if filename or "attachment" in disposition:
                    attachments.append(filename or "(unnamed)")
                    continue
                if ctype == "text/plain":
                    txt = _decode(part)
                    if txt:
                        plain_parts.append(txt)
                elif ctype == "text/html":
                    txt = _decode(part)
                    if txt:
                        html_parts.append(txt)
        else:
            ctype = msg.get_content_type()
            txt = _decode(msg)
            if ctype == "text/html":
                html_parts.append(txt)
            else:
                plain_parts.append(txt)

        return "\n".join(plain_parts), "\n".join(html_parts), attachments


# ──────────────────────────────────────────────────────────────────────────────
# Ingestor
# ──────────────────────────────────────────────────────────────────────────────

_VALID_EMAIL_MODES = {"full", "incremental", "month"}


class EmailIngestor(BaseIngestor):
    source = "email"

    def __init__(
        self,
        db,
        user_id: str,
        credentials: dict[str, str],
        scope: str,
        mode: str = "incremental",
        month: Optional[str] = None,
    ) -> None:
        if mode not in _VALID_EMAIL_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_EMAIL_MODES}, got {mode!r}"
            )

        # The base class only validates {full, incremental}. "month" is a
        # logical refinement of incremental — same state-machine semantics
        # (no destructive wipe), but a fixed date window.
        base_mode = "incremental" if mode == "month" else mode
        super().__init__(db, user_id, credentials, scope, base_mode)

        self._email_mode = mode
        self._month = month

        # `scope` may be one of: "all", "outlook", "gmail", "yahoo",
        # or a folder/label/IMAP-folder name applied to every configured
        # provider.
        self._scope_provider: Optional[str] = None
        self._folder_override: Optional[str] = None
        if scope in {"outlook", "gmail", "yahoo"}:
            self._scope_provider = scope
        elif scope and scope != "all":
            # Treat anything else as a folder/label name.
            self._folder_override = scope

        self._providers = self._build_providers()

    # ── Provider construction ───────────────────────────────────────────────

    def _build_providers(self) -> list:
        provs: list = []
        wanted = (
            {self._scope_provider}
            if self._scope_provider
            else {"outlook", "gmail", "yahoo"}
        )

        if "outlook" in wanted and self._has_outlook_creds():
            try:
                provs.append(_OutlookProvider(self.credentials))
                logger.info("[email] Outlook provider registered.")
            except Exception as exc:
                logger.warning(
                    "[email] Outlook credentials present but invalid: {}", exc
                )

        if "gmail" in wanted and self._has_gmail_creds():
            try:
                provs.append(_GmailProvider(self.credentials))
                logger.info("[email] Gmail provider registered.")
            except Exception as exc:
                logger.warning(
                    "[email] Gmail credentials present but invalid: {}", exc
                )

        if "yahoo" in wanted and self._has_yahoo_creds():
            try:
                provs.append(_YahooProvider(self.credentials))
                logger.info("[email] Yahoo provider registered.")
            except Exception as exc:
                logger.warning(
                    "[email] Yahoo credentials present but invalid: {}", exc
                )

        if not provs:
            raise ValueError(
                "No usable email providers — supply outlook_*, gmail_* "
                "and/or yahoo_* credentials in Settings → Credentials → "
                "Email. Configured providers: outlook, gmail, yahoo."
            )
        return provs

    def _has_outlook_creds(self) -> bool:
        return bool(self.credentials.get("outlook_client_id")) and bool(
            self.credentials.get("outlook_refresh_token")
            or self.credentials.get("outlook_client_secret")
        )

    def _has_gmail_creds(self) -> bool:
        if self.credentials.get("gmail_token_json"):
            return True
        return all(
            self.credentials.get(k)
            for k in ("gmail_client_id", "gmail_client_secret", "gmail_refresh_token")
        )

    def _has_yahoo_creds(self) -> bool:
        # Yahoo dropped public OAuth2 for Mail; the supported integration
        # is mailbox-scoped app passwords (see _YahooProvider docstring).
        return bool(
            self.credentials.get("yahoo_email_address")
            and self.credentials.get("yahoo_app_password")
        )

    # ── BaseIngestor abstract API ───────────────────────────────────────────

    def scope_filter(self) -> dict[str, Any]:
        if self._scope_provider:
            return {"source": "email", "email_provider": self._scope_provider}
        # "all" or folder-scoped: wipe the providers we're about to ingest
        # (folder filters don't change the wipe target — they narrow ingestion
        # but full-mode rebuilds the whole provider's slice).
        provider_names = [p.name for p in self._providers]
        return {"source": "email", "email_provider": {"$in": provider_names}}

    def resource_identifier_for(self, resource: SourceResource) -> str:
        return resource.metadata["email_provider"]

    def fetch_resources(
        self, since: Optional[datetime] = None
    ) -> Iterable[SourceResource]:
        start, end = self._compute_window(since)
        if start or end:
            logger.info(
                "[email] window: {} → {}",
                start.isoformat() if start else "∞",
                end.isoformat() if end else "now",
            )
        for prov in self._providers:
            try:
                yield from prov.fetch(
                    start=start, end=end, folder_override=self._folder_override
                )
            except Exception as exc:
                logger.exception(
                    "[email/{}] provider failed: {}", prov.name, exc
                )
                # Don't kill the whole run if one provider hiccups — let
                # the other one still complete. Re-raise only if every
                # provider failed (BaseIngestor would mark the run errored).
                continue

    # ── Window selection ────────────────────────────────────────────────────

    def _compute_window(
        self, since: Optional[datetime]
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        if self._email_mode == "full":
            return None, None
        if self._email_mode == "month":
            if not self._month:
                raise ValueError("--mode month requires --month YYYY-MM")
            return _month_window(self._month)
        # Incremental
        if since is not None:
            return self._ensure_aware(since), None
        # No prior high-water mark → previous calendar month (the
        # documented default to keep first-ever runs sane).
        logger.info(
            "[email] No prior ingestion log — defaulting to previous "
            "calendar month."
        )
        return _previous_calendar_month_window()

    @staticmethod
    def _ensure_aware(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
