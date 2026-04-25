"""
One-shot migration script: clean up the old per-user-duplicated vectors.

The previous architecture stored a separate copy of every chunk per user, with
``user_id`` as a metadata field. The new architecture stores each resource
exactly once and enforces tenancy at query time. This script gives you three
options for migrating an existing Pinecone index:

  --strategy nuke
      Delete EVERYTHING in the index, then re-ingest from scratch under each
      user. Safest, simplest, and the recommended path for small indexes
      (< ~100k vectors).

  --strategy strip-user-id  (DRY-RUN by default)
      Walk the index, find vectors whose metadata still contains ``user_id``,
      group them by ``resource_id`` (if present) and keep only ONE copy.
      The kept copy has its ``user_id`` field removed via a metadata update.
      All other duplicates are deleted. Use ``--apply`` to actually mutate.

  --strategy report
      Print Pinecone index stats and a sample of vectors. No mutations.

Examples:
    python -m scripts.cleanup_legacy_vectors --strategy report
    python -m scripts.cleanup_legacy_vectors --strategy strip-user-id        # dry run
    python -m scripts.cleanup_legacy_vectors --strategy strip-user-id --apply
    python -m scripts.cleanup_legacy_vectors --strategy nuke --confirm-i-know-what-im-doing
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Any

from loguru import logger

from app.config import settings
from core.vector_store import _pinecone_client, get_index


def _strategy_report() -> None:
    index = get_index()
    stats = index.describe_index_stats()
    logger.info("Index: {}", settings.PINECONE_INDEX_NAME)
    logger.info("Stats: {}", stats)


def _strategy_nuke(confirmed: bool) -> None:
    if not confirmed:
        logger.error(
            "Refusing to nuke without --confirm-i-know-what-im-doing. "
            "This DELETES every vector in {!r}.",
            settings.PINECONE_INDEX_NAME,
        )
        sys.exit(2)
    index = get_index()
    logger.warning("Deleting ALL vectors in {!r}…", settings.PINECONE_INDEX_NAME)
    index.delete(delete_all=True)
    logger.info("Done. You can now re-ingest from the UI / CLI.")


def _strategy_strip_user_id(apply: bool, page_size: int = 100) -> None:
    """
    Walk the index, dedup by resource_id, and remove ``user_id`` from metadata.

    NOTE: Pinecone's ``index.list()`` paginates over IDs only — to inspect
    metadata we must call ``fetch()`` in batches. This is therefore O(N) in
    vector count; for very large indexes prefer ``--strategy nuke`` and
    re-ingest.
    """
    index = get_index()

    seen_resource: dict[str, str] = {}    # resource_id → kept vector id
    to_delete: list[str] = []
    to_strip: list[str] = []              # vector ids whose metadata we'll rewrite

    total_examined = 0
    for page in index.list():
        ids = list(page) if not isinstance(page, list) else page
        for i in range(0, len(ids), page_size):
            batch_ids = ids[i : i + page_size]
            fetched = index.fetch(ids=batch_ids)
            vectors = (
                fetched.get("vectors")
                if isinstance(fetched, dict)
                else getattr(fetched, "vectors", {})
            ) or {}

            for vid, vec in vectors.items():
                total_examined += 1
                md = (
                    vec.get("metadata")
                    if isinstance(vec, dict)
                    else getattr(vec, "metadata", {})
                ) or {}
                resource_id = md.get("resource_id")
                has_user_id = "user_id" in md

                if resource_id is None:
                    # Pre-resource_id vectors: best we can do is delete them.
                    to_delete.append(vid)
                    continue

                if resource_id in seen_resource:
                    to_delete.append(vid)
                else:
                    seen_resource[resource_id] = vid
                    if has_user_id:
                        to_strip.append(vid)

    logger.info(
        "Examined {} vectors → {} to delete, {} to strip user_id, {} unique resources",
        total_examined,
        len(to_delete),
        len(to_strip),
        len(seen_resource),
    )

    if not apply:
        logger.warning("Dry run — pass --apply to actually mutate the index.")
        return

    # Delete dupes / orphans
    for i in range(0, len(to_delete), settings.PINECONE_BATCH_SIZE):
        batch = to_delete[i : i + settings.PINECONE_BATCH_SIZE]
        index.delete(ids=batch)
    logger.info("Deleted {} duplicate vectors.", len(to_delete))

    # Strip user_id from kept copies
    for vid in to_strip:
        try:
            index.update(id=vid, set_metadata={"user_id": None})
        except Exception as exc:
            logger.warning("Could not strip user_id from {}: {}", vid, exc)
    logger.info("Stripped user_id from {} vectors.", len(to_strip))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["report", "nuke", "strip-user-id"],
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="For strip-user-id: actually mutate the index (default is dry-run).",
    )
    parser.add_argument(
        "--confirm-i-know-what-im-doing",
        action="store_true",
        help="Required to actually run --strategy nuke.",
    )
    args = parser.parse_args()

    # Touch the client once so we fail fast on missing creds.
    _pinecone_client()

    if args.strategy == "report":
        _strategy_report()
    elif args.strategy == "nuke":
        _strategy_nuke(args.confirm_i_know_what_im_doing)
    elif args.strategy == "strip-user-id":
        _strategy_strip_user_id(apply=args.apply)


if __name__ == "__main__":
    main()
