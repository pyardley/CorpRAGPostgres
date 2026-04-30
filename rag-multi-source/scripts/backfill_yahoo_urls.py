"""
One-shot backfill: rewrite the ``url`` column on Yahoo email chunks to
the search-by-subject+from+date URL produced by
:func:`app.ingestion.email_ingestor._build_yahoo_search_url`.

Why this exists
---------------
Yahoo chunks ingested before the URL builder was hardened got a
placeholder URL pointing at the inbox root. Re-ingestion would fix
them but would also re-embed every message (~7,000 rows × OpenAI
text-embedding-3-small = real $$$). This script does the metadata-side
fix without touching the embedding column.

The URL pattern landed on after experimentation::

    https://mail.yahoo.com/n/search/keyword=<URL-encoded>
    where keyword = "<subject> from:<sender_email> after:<YYYY-MM-DD>"

Yahoo's web UI lands on a search results page with the target message
visible as the first result. (Yahoo doesn't index the RFC 5322
Message-ID, so search-by-MID returns nothing — subject + from + date
is the most-unique combination available from metadata.)

What it does
------------
For each row in ``vector_chunks`` where:
  * ``source = 'email'``
  * ``email_provider = 'yahoo'``
  * ``url`` is missing, blank, the legacy ``/d/folders/1`` placeholder,
    the new ``/n/folders/1`` fallback, or the older Message-ID-search
    pattern — all of which were emitted by earlier ingest revisions —

…it reads ``title``, ``last_updated``, and
``metadata->>'from'`` / ``metadata->>'sender'``, hands them to the
canonical URL builder, and UPDATEs the row. Rows whose subject AND
sender are both empty (extremely rare) get the inbox-root fallback —
better than a broken URL.

Usage
-----
From the project root with the venv active:

    python scripts/backfill_yahoo_urls.py
    python scripts/backfill_yahoo_urls.py --dry-run     # show, don't write
    python scripts/backfill_yahoo_urls.py --limit 10    # stop early for sanity

Idempotent: re-running on an already-fixed table is a no-op (the WHERE
clause skips rows whose URL already starts with the search prefix).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

# Path bootstrap so this works as a standalone script from the project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from loguru import logger
from sqlalchemy import text

from app.utils import get_db, init_db
from app.ingestion.email_ingestor import (
    _YAHOO_INBOX_FALLBACK,
    _build_yahoo_search_url,
)


# Every URL flavour earlier ingestor revisions ever emitted for Yahoo,
# so a single re-run cleans up any chunk that hasn't been migrated yet.
# We DON'T match the new ``/n/search/keyword=`` prefix — that's how we
# detect rows that have already been migrated and skip them.
_LEGACY_PLACEHOLDERS = (
    "https://mail.yahoo.com/d/folders/1",
    "https://mail.yahoo.com/n/folders/1",
    _YAHOO_INBOX_FALLBACK,  # belt-and-braces in case the constant changes
)


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended updates without writing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows then stop (useful for sanity checks).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="UPDATE statements per intermediate commit (default: 500).",
    )
    args = parser.parse_args()

    init_db()

    select_sql = text(
        """
        SELECT id,
               url,
               title,
               last_updated,
               metadata->>'from'   AS sender_raw,
               metadata->>'sender' AS sender_alt
        FROM vector_chunks
        WHERE source = 'email'
          AND email_provider = 'yahoo'
          AND (
              url IS NULL
              OR url = ''
              OR url = ANY(:placeholders)
              OR url LIKE 'https://mail.yahoo.com/d/search/keyword=%'
          )
        ORDER BY last_updated DESC NULLS LAST, id ASC
        """
    )
    update_sql = text("UPDATE vector_chunks SET url = :url WHERE id = :id")

    examined = 0
    updated = 0
    fallback = 0

    with get_db() as db:
        rows = db.execute(
            select_sql, {"placeholders": list(_LEGACY_PLACEHOLDERS)}
        ).all()
        if args.limit:
            rows = rows[: args.limit]
        logger.info(
            "Found {} Yahoo chunks needing URL backfill (limit={}).",
            len(rows),
            args.limit or "none",
        )

        for row in rows:
            examined += 1
            chunk_id, _old_url, title, last_updated, sender_raw, sender_alt = row
            received = _parse_iso(last_updated or "")
            sender = (sender_raw or sender_alt or "").strip()
            new_url = _build_yahoo_search_url(
                subject=title or "",
                sender=sender,
                received_dt=received or datetime.now(timezone.utc),
            )
            if new_url == _YAHOO_INBOX_FALLBACK:
                fallback += 1

            if args.dry_run:
                if updated < 5:
                    logger.info(
                        "[dry-run] chunk_id={} title={!r} from={!r} "
                        "last_updated={} -> {}",
                        chunk_id,
                        (title or "")[:60],
                        sender[:60],
                        last_updated,
                        new_url,
                    )
                updated += 1
                continue

            db.execute(update_sql, {"url": new_url, "id": chunk_id})
            updated += 1
            if updated % args.batch_size == 0:
                db.commit()
                logger.info("Committed {} updates so far…", updated)

        if not args.dry_run:
            db.commit()

    logger.info(
        "{} examined={} updated={} fell_back_to_inbox={}",
        "[dry-run]" if args.dry_run else "Done.",
        examined,
        updated,
        fallback,
    )

    # Diagnostic tail count so the user can see at a glance whether
    # anything got missed (will only be the inbox-fallback rows).
    with get_db() as db:
        remaining = db.execute(
            text(
                """
                SELECT count(*) FROM vector_chunks
                WHERE source = 'email' AND email_provider = 'yahoo'
                  AND (
                      url IS NULL
                      OR url = ''
                      OR url = ANY(:placeholders)
                      OR url LIKE 'https://mail.yahoo.com/d/search/keyword=%'
                  )
                """
            ),
            {"placeholders": list(_LEGACY_PLACEHOLDERS)},
        ).scalar_one()
    logger.info(
        "Yahoo chunks still on a legacy/placeholder URL: {} (re-run "
        "without --limit if non-zero).",
        remaining,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
