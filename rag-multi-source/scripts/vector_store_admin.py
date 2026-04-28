"""
Vector store maintenance utility.

  --strategy report
      Print row counts per source and overall index health (vector count,
      table size, HNSW index size).

  --strategy purge-source --source <jira|confluence|sql|git>
      Delete every chunk for a single source. Useful when you need to
      reset a connector during testing.

  --strategy nuke --confirm-i-know-what-im-doing
      ``TRUNCATE`` the entire ``vector_chunks`` table. After running this
      every user must re-ingest before the chat can answer anything.

Examples:
    python -m scripts.cleanup_legacy_vectors --strategy report
    python -m scripts.cleanup_legacy_vectors --strategy purge-source --source jira
    python -m scripts.cleanup_legacy_vectors --strategy nuke \
        --confirm-i-know-what-im-doing
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger
from sqlalchemy import text

from app.utils import engine, get_db
from core.vector_store import index_stats


def _strategy_report() -> None:
    stats = index_stats()
    logger.info("vector_chunks per-source counts: {}", stats)

    with engine.begin() as conn:
        size = conn.execute(
            text(
                "SELECT pg_size_pretty(pg_total_relation_size('vector_chunks')) "
                "AS table_size, "
                "pg_size_pretty(pg_relation_size('ix_vector_chunks_hnsw')) "
                "AS hnsw_size"
            )
        ).first()
        logger.info(
            "Storage: table={} HNSW index={}",
            size.table_size if size else "?",
            size.hnsw_size if size else "?",
        )


def _strategy_purge_source(source: str) -> None:
    if source not in {"jira", "confluence", "sql", "git"}:
        logger.error("Unknown source: {}", source)
        sys.exit(2)
    with get_db() as db:
        result = db.execute(
            text("DELETE FROM vector_chunks WHERE source = :s"), {"s": source}
        )
        logger.info("Deleted {} rows for source={}", result.rowcount, source)


def _strategy_nuke(confirmed: bool) -> None:
    if not confirmed:
        logger.error(
            "Refusing to nuke without --confirm-i-know-what-im-doing. "
            "This TRUNCATEs vector_chunks (every embedding lost)."
        )
        sys.exit(2)
    with engine.begin() as conn:
        # TRUNCATE is faster than DELETE and reclaims space immediately.
        conn.execute(text("TRUNCATE TABLE vector_chunks RESTART IDENTITY"))
    logger.warning("vector_chunks truncated. All users must re-ingest.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["report", "purge-source", "nuke"],
    )
    parser.add_argument(
        "--source",
        choices=["jira", "confluence", "sql", "git"],
        help="Required for --strategy purge-source",
    )
    parser.add_argument(
        "--confirm-i-know-what-im-doing",
        action="store_true",
        help="Required to actually run --strategy nuke.",
    )
    args = parser.parse_args()

    if args.strategy == "report":
        _strategy_report()
    elif args.strategy == "purge-source":
        if not args.source:
            logger.error("--source is required for --strategy purge-source")
            sys.exit(2)
        _strategy_purge_source(args.source)
    elif args.strategy == "nuke":
        _strategy_nuke(args.confirm_i_know_what_im_doing)


if __name__ == "__main__":
    main()
