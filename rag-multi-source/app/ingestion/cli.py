"""
CLI entry point for triggering ingestion.

Usage:
    python -m app.ingestion.cli --help
    python -m app.ingestion.cli --source jira --mode full --scope PROJ
    python -m app.ingestion.cli --source confluence --mode incremental --scope MYSPACE
    python -m app.ingestion.cli --source sql --mode full --scope mydb
    python -m app.ingestion.cli --source all --mode incremental
"""

from __future__ import annotations

import argparse
import sys
from getpass import getpass
from typing import Optional

from loguru import logger

from app.utils import get_db, init_db


def _authenticate(email: str, password: Optional[str] = None) -> dict:
    """Verify credentials and return user dict."""
    from app.auth import get_user_by_email, verify_password

    if password is None:
        password = getpass(f"Password for {email}: ")

    with get_db() as db:
        user = get_user_by_email(db, email)
        if user is None or not verify_password(password, user.password_hash):
            logger.error("Authentication failed.")
            sys.exit(1)
        return {"id": user.id, "email": user.email}


def _get_ingestor(source: str, user_id: str):
    from app.ingestion.jira_ingestor import JiraIngestor
    from app.ingestion.confluence_ingestor import ConfluenceIngestor
    from app.ingestion.sql_ingestor import SQLIngestor
    from app.utils import load_all_credentials

    with get_db() as db:
        creds = load_all_credentials(db, user_id, source)

    if not creds:
        logger.error(
            "No credentials found for source '{}'. "
            "Please configure them in the Streamlit UI under Settings.",
            source,
        )
        sys.exit(1)

    if source == "jira":
        return JiraIngestor(user_id=user_id, credentials=creds)
    if source == "confluence":
        return ConfluenceIngestor(user_id=user_id, credentials=creds)
    if source == "sql":
        return SQLIngestor(user_id=user_id, credentials=creds)
    if source == "git":
        from app.ingestion.git_ingestor import GitIngestor
        return GitIngestor(user_id=user_id, credentials=creds)

    raise ValueError(f"Unknown source: {source}")


def run_ingestion(
    user_id: str,
    source: str,
    mode: str,
    scope: str,
) -> None:
    ALL_SOURCES = ["jira", "confluence", "sql", "git"]
    sources_to_run = ALL_SOURCES if source == "all" else [source]

    for src in sources_to_run:
        logger.info("── Starting {} / {} / scope={} ──", src, mode, scope)
        try:
            ingestor = _get_ingestor(src, user_id)
            log = ingestor.run(scope_key=scope, mode=mode)
            logger.info(
                "── Finished: status={} items={} vectors={}",
                log.status,
                log.items_processed,
                log.vectors_upserted,
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
        choices=["jira", "confluence", "sql", "git", "all"],
        required=True,
        help="Which source to ingest (use 'all' for every source).",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help="full: clear + re-ingest everything. incremental: only new/changed items.",
    )
    parser.add_argument(
        "--scope",
        default="all",
        help=(
            "Scope within the source: Jira project key, Confluence space key, "
            "SQL database name, or 'all'."
        ),
    )
    parser.add_argument("--email", required=True, help="Your CorporateRAG account email.")
    parser.add_argument(
        "--password",
        default=None,
        help="Your CorporateRAG account password (prompted if omitted).",
    )

    args = parser.parse_args()

    # Bootstrap DB
    init_db()

    # Auth
    user = _authenticate(args.email, args.password)
    logger.info("Authenticated as {} ({})", user["email"], user["id"])

    # Run
    run_ingestion(
        user_id=user["id"],
        source=args.source,
        mode=args.mode,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
