"""
CLI entry point for triggering ingestion.

Usage:
    python -m app.ingestion.cli --help
    python -m app.ingestion.cli --source jira       --mode full        --scope PROJ
    python -m app.ingestion.cli --source confluence --mode incremental --scope MYSPACE
    python -m app.ingestion.cli --source sql        --mode full        --scope mydb
    python -m app.ingestion.cli --source git        --mode incremental --scope main
    python -m app.ingestion.cli --source email      --mode incremental
    python -m app.ingestion.cli --source email      --mode month --month 2025-03
    python -m app.ingestion.cli --source email      --mode full   --scope outlook
    python -m app.ingestion.cli --source all        --mode incremental
"""

from __future__ import annotations

import argparse
import sys
from getpass import getpass
from typing import Optional

from loguru import logger

from app.utils import get_db, init_db, load_all_credentials


def _authenticate(email: str, password: Optional[str] = None) -> dict:
    """Verify credentials and return a {"id", "email"} dict."""
    from app.auth import get_user_by_email, verify_password

    if password is None:
        password = getpass(f"Password for {email}: ")

    with get_db() as db:
        user = get_user_by_email(db, email)
        if user is None or not verify_password(password, user.password_hash):
            logger.error("Authentication failed.")
            sys.exit(1)
        return {"id": user.id, "email": user.email}


def _make_ingestor(
    db,
    source: str,
    user_id: str,
    scope: str,
    mode: str,
    month: Optional[str] = None,
):
    creds = load_all_credentials(db, user_id, source)
    if not creds:
        raise RuntimeError(
            f"No credentials found for source '{source}'. "
            "Please configure them in the Streamlit UI under Settings."
        )

    # The Email ingestor is the only one that supports a non-{full,
    # incremental} mode ("month"). Every other ingestor only sees the
    # baseline two values, so coerce the mode here for them.
    base_mode = mode if mode in {"full", "incremental"} else "incremental"

    if source == "jira":
        from app.ingestion.jira_ingestor import JiraIngestor
        return JiraIngestor(
            db=db, user_id=user_id, credentials=creds, scope=scope, mode=base_mode
        )
    if source == "confluence":
        from app.ingestion.confluence_ingestor import ConfluenceIngestor
        return ConfluenceIngestor(
            db=db, user_id=user_id, credentials=creds, scope=scope, mode=base_mode
        )
    if source == "sql":
        from app.ingestion.sql_ingestor import SQLIngestor
        return SQLIngestor(
            db=db, user_id=user_id, credentials=creds, scope=scope, mode=base_mode
        )
    if source == "git":
        from app.ingestion.git_ingestor import GitIngestor
        return GitIngestor(
            db=db, user_id=user_id, credentials=creds, scope=scope, mode=base_mode
        )
    if source == "email":
        from app.ingestion.email_ingestor import EmailIngestor
        return EmailIngestor(
            db=db,
            user_id=user_id,
            credentials=creds,
            scope=scope,
            mode=mode,
            month=month,
        )

    raise ValueError(f"Unknown source: {source}")


ALL_SOURCES = ["jira", "confluence", "sql", "git", "email"]


def run_ingestion(
    user_id: str,
    source: str,
    mode: str,
    scope: str,
    month: Optional[str] = None,
) -> None:
    sources_to_run = ALL_SOURCES if source == "all" else [source]

    for src in sources_to_run:
        logger.info("Starting {} / {} / scope={}", src, mode, scope)
        try:
            with get_db() as db:
                ingestor = _make_ingestor(db, src, user_id, scope, mode, month=month)
                result = ingestor.run()
            logger.info(
                "Finished {}: items={} vectors={} last_updated={}",
                src,
                result.items_processed,
                result.vectors_upserted,
                result.last_item_updated_at,
            )
        except Exception as exc:
            logger.exception("Failed to run ingestor for {}: {}", src, exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion.cli",
        description="CorporateRAG ingestion pipeline",
    )
    parser.add_argument(
        "--source",
        choices=ALL_SOURCES + ["all"],
        required=True,
        help="Which source to ingest (use 'all' for every source).",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "month"],
        default="incremental",
        help=(
            "full: wipe-then-rebuild for the scope (delete by metadata filter "
            "first). incremental: only new/changed items since the last "
            "successful run. month: (email source only) ingest exactly the "
            "calendar month given by --month YYYY-MM."
        ),
    )
    parser.add_argument(
        "--month",
        default=None,
        help=(
            "Used with --mode month (email source). Format: YYYY-MM. "
            "Example: --mode month --month 2025-03"
        ),
    )
    parser.add_argument(
        "--scope",
        default="all",
        help=(
            "Scope within the source: Jira project key, Confluence space key, "
            "SQL database name, Git branch, email provider "
            "(outlook | gmail | yahoo) or folder/label/IMAP-folder name, "
            "or 'all'."
        ),
    )
    parser.add_argument(
        "--email", required=True, help="Your CorporateRAG account email."
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Your CorporateRAG account password (prompted if omitted).",
    )

    args = parser.parse_args()

    init_db()

    user = _authenticate(args.email, args.password)
    logger.info("Authenticated as {} ({})", user["email"], user["id"])

    if args.mode == "month" and not args.month:
        logger.error("--mode month requires --month YYYY-MM")
        sys.exit(2)

    run_ingestion(
        user_id=user["id"],
        source=args.source,
        mode=args.mode,
        scope=args.scope,
        month=args.month,
    )


if __name__ == "__main__":
    main()
