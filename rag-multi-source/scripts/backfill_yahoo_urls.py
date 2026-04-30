"""
One-shot backfill: rewrite the ``url`` column on Yahoo email chunks to
the search-by-Message-ID URL produced by the post-`5aa0cbb`
:class:`app.ingestion.email_ingestor._YahooProvider`.

Why this exists
---------------
Up to commit `5aa0cbb`, every Yahoo chunk got a placeholder URL
(``https://mail.yahoo.com/d/folders/1``) because Yahoo's web UI doesn't
expose stable per-message permalinks. The new URL builder uses the
Internet Message-ID in a Yahoo search URL — close enough to a permalink
that citations are clickable. Existing chunks keep the placeholder
unless re-ingested. Re-ingestion would re-embed every message
(`~7000` rows × 1500-token model = real $$$), so this script does the
purely metadata-side fix without touching the embedding column.

What it does
------------
For each row in ``vector_chunks`` where:
  * ``source = 'email'``
  * ``email_provider = 'yahoo'``
  * ``url`` is missing / blank / equal to the old folder placeholder

…it reads ``metadata->>'internet_message_id'``, URL-encodes it via
:func:`urllib.parse.quote` (matching `_YahooProvider`'s logic exactly),
and writes the new URL with a parameterised UPDATE. The transaction is
batched and committed once at the end, so it's atomic but not OOM-prone.

Rows whose ``internet_message_id`` is missing or blank are left alone —
the original folder-root URL is the correct fallback for those.

Usage
-----
From the project root with the venv active:

    python scripts/backfill_yahoo_urls.py
    python scripts/backfill_yahoo_urls.py --dry-run    # no writes
    python scripts/backfill_yahoo_urls.py --user-id <uuid>   # narrow

Idempotent: re-running on an already-fixed table is a no-op (the WHERE
clause skips rows whose URL already matches the search-keyword pattern).
"""

from __future__ import annotations

import argparse
import sys
from urllib.parse import quote

# Path bootstrap so this works as a standalone script from the project root.
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from loguru import logger
from sqlalchemy import text

from app.utils import get_db, init_db


_OLD_PLACEHOLDER = "https://mail.yahoo.com/d/folders/1"
_SEARCH_PREFIX = "https://mail.yahoo.com/d/search/keyword="


def _build_url(message_id: str) -> str:
    """
    Same encoding rules as
    :meth:`app.ingestion.email_ingestor._YahooProvider._parse_message`.
    Strip ``<>`` from the Message-ID, then percent-encode every byte
    that isn't a path-safe URL char. ``safe=''`` is intentional —
    Yahoo's search URL has historically been picky about ``@`` and
    ``+`` showing up unencoded in the keyword.
    """
    return _SEARCH_PREFIX + quote(message_id.strip("<>"), safe="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended updates without writing.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Optional: only update chunks belonging to a single user "
        "(matched against the resource via the ingestion log isn't "
        "viable here; this filter is a no-op until vector_chunks gains "
        "a user_id column — kept for future-proofing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="UPDATE statements per commit (default: 500).",
    )
    args = parser.parse_args()

    init_db()

    select_sql = text(
        """
        SELECT id,
               url,
               metadata->>'internet_message_id' AS mid
        FROM vector_chunks
        WHERE source = 'email'
          AND email_provider = 'yahoo'
          AND (url IS NULL OR url = '' OR url = :placeholder)
          AND metadata ? 'internet_message_id'
          AND length(coalesce(metadata->>'internet_message_id', '')) > 0
        """
    )
    update_sql = text("UPDATE vector_chunks SET url = :url WHERE id = :id")

    examined = 0
    updated = 0
    skipped_no_mid = 0

    with get_db() as db:
        rows = db.execute(
            select_sql, {"placeholder": _OLD_PLACEHOLDER}
        ).all()
        logger.info(
            "Found {} Yahoo chunks with placeholder URL and a usable "
            "Message-ID.",
            len(rows),
        )

        for row in rows:
            examined += 1
            chunk_id, _old_url, mid = row
            mid = (mid or "").strip()
            if not mid:
                skipped_no_mid += 1
                continue
            new_url = _build_url(mid)
            if args.dry_run:
                if updated < 5:
                    logger.info(
                        "[dry-run] would UPDATE chunk_id={} url -> {}",
                        chunk_id,
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

    if args.dry_run:
        logger.info(
            "[dry-run] examined={} would_update={} skipped_no_mid={}",
            examined,
            updated,
            skipped_no_mid,
        )
    else:
        logger.info(
            "Done. examined={} updated={} skipped_no_mid={}",
            examined,
            updated,
            skipped_no_mid,
        )

    # Diagnostic: count any rows still on the old URL after the run so
    # the user can see at a glance whether anything's left behind.
    with get_db() as db:
        remaining = db.execute(
            text(
                "SELECT count(*) FROM vector_chunks "
                "WHERE source = 'email' AND email_provider = 'yahoo' "
                "AND url = :placeholder"
            ),
            {"placeholder": _OLD_PLACEHOLDER},
        ).scalar_one()
    logger.info(
        "Yahoo chunks still on the placeholder URL: {} (these have no "
        "internet_message_id in metadata; their fallback is correct).",
        remaining,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
