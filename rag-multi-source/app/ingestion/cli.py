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

Recipe-driven ingestion (additive — doesn't disturb the legacy --source flag):
    python -m app.ingestion.cli --list-recipes
    python -m app.ingestion.cli --recipe example_jira --scope PROJ --mode incremental
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


def run_recipe_ingestion(
    user_id: str,
    recipe_name: str,
    mode: str,
    scope: str,
) -> None:
    """Run a single recipe by name. Mirrors :func:`run_ingestion` semantics."""
    # Imported lazily so a broken recipes/ directory can't kill the
    # legacy --source path.
    from app.ingestion.recipe_runner import run_recipe

    base_mode = mode if mode in {"full", "incremental"} else "incremental"
    logger.info(
        "Starting recipe '{}' / {} / scope={}", recipe_name, base_mode, scope
    )
    try:
        with get_db() as db:
            result = run_recipe(
                db,
                user_id,
                recipe_name,
                scope=scope,
                mode=base_mode,
            )
        logger.info(
            "Finished recipe '{}': items={} vectors={} last_updated={}",
            recipe_name,
            result.items_processed,
            result.vectors_upserted,
            result.last_item_updated_at,
        )
    except Exception as exc:
        logger.exception(
            "Failed to run recipe '{}': {}", recipe_name, exc
        )


def _print_recipe_list() -> None:
    """Pretty-print every discovered recipe + exit."""
    from recipes import list_recipes

    recipes = list_recipes()
    if not recipes:
        print(
            "No recipes found. Drop a .yaml file into the recipes/ "
            "directory to register one."
        )
        return
    print(f"Available recipes ({len(recipes)}):\n")
    name_w = max(len(r.name) for r in recipes)
    for r in recipes:
        parser_label = "builtin" if r.is_builtin else r.parser
        print(
            f"  {r.name.ljust(name_w)}  "
            f"source={r.source:<14}  parser={parser_label}"
        )
        if r.description:
            print(f"  {' ' * name_w}    {r.description}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion.cli",
        description="CorporateRAG ingestion pipeline",
    )
    # ── Discovery (no auth needed; can run before --email/--password). ──
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help=(
            "Print every discovered recipe (declarative YAML + Python "
            "files under recipes/) and exit. No authentication required."
        ),
    )

    # ── Recipe / source dispatch (mutually exclusive but optional —
    #    --list-recipes may be the only flag). ──
    parser.add_argument(
        "--source",
        choices=ALL_SOURCES + ["all"],
        default=None,
        help="Which legacy source to ingest (use 'all' for every source).",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help=(
            "Name of a recipe (see --list-recipes). Mutually exclusive "
            "with --source. Recipes are discovered automatically from the "
            "recipes/ directory."
        ),
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
            "Scope within the source/recipe: Jira project key, Confluence "
            "space key, SQL database name, Git branch, email provider "
            "(outlook | gmail | yahoo) or folder/label/IMAP-folder name, "
            "or 'all'. For recipes, the meaning is defined by the recipe's "
            "scope_field."
        ),
    )
    parser.add_argument(
        "--email",
        default=None,
        help=(
            "Your CorporateRAG account email. Required for --source / "
            "--recipe runs (not needed for --list-recipes)."
        ),
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Your CorporateRAG account password (prompted if omitted).",
    )

    args = parser.parse_args()

    # ── Discovery shortcut ──
    if args.list_recipes:
        _print_recipe_list()
        return

    if args.source and args.recipe:
        logger.error("--source and --recipe are mutually exclusive.")
        sys.exit(2)
    if not args.source and not args.recipe:
        logger.error(
            "One of --source, --recipe or --list-recipes is required. "
            "Run with --help for full usage."
        )
        sys.exit(2)
    if not args.email:
        logger.error("--email is required for ingestion runs.")
        sys.exit(2)

    init_db()

    user = _authenticate(args.email, args.password)
    logger.info("Authenticated as {} ({})", user["email"], user["id"])

    if args.mode == "month" and not args.month:
        logger.error("--mode month requires --month YYYY-MM")
        sys.exit(2)

    if args.recipe:
        run_recipe_ingestion(
            user_id=user["id"],
            recipe_name=args.recipe,
            mode=args.mode,
            scope=args.scope,
        )
    else:
        run_ingestion(
            user_id=user["id"],
            source=args.source,
            mode=args.mode,
            scope=args.scope,
            month=args.month,
        )


if __name__ == "__main__":
    main()
