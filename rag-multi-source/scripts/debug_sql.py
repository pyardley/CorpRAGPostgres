"""
Diagnose why SQL Server ingestion isn't picking up user tables.

Runs three checks against the same connection string the app uses:

    1. List every chunk we've actually indexed for source='sql' so we can
       see what made it into vector_chunks.
    2. Re-open the connection and report CURRENT_USER, the database we
       landed in, and the SQL login name — proves we're actually inside
       CustomerDemo (or wherever).
    3. List tables visible via INFORMATION_SCHEMA.TABLES *and* via
       sys.tables. The two views can return different rows on locked-down
       SQL Server installs — comparing them tells us whether it's a
       permissions problem or an ingest-query problem.

Usage:
    .venv\\Scripts\\python.exe scripts/debug_sql.py
    .venv\\Scripts\\python.exe scripts/debug_sql.py --db CustomerDemo --email me@org.com
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# When invoked as `python scripts/debug_sql.py`, only the scripts/ folder
# is on sys.path — `app` and `models` aren't reachable. Add the project
# root so direct invocation works the same as `python -m scripts.debug_sql`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import URL  # noqa: E402

from app.utils import get_db, load_all_credentials  # noqa: E402
from models.user import User  # noqa: E402
from models.vector_chunk import VectorChunk  # noqa: E402


def _pick_user(db, email: str | None):
    if email:
        u = db.query(User).filter_by(email=email.lower().strip()).first()
        if not u:
            print(f"!! No user with email {email!r}", file=sys.stderr)
            sys.exit(2)
        return u
    users = db.query(User).all()
    if not users:
        print("!! No users in the database.", file=sys.stderr)
        sys.exit(2)
    if len(users) > 1:
        print(
            "!! Multiple users found — pass --email to pick one. Users:",
            file=sys.stderr,
        )
        for u in users:
            print(f"     {u.email}", file=sys.stderr)
        sys.exit(2)
    return users[0]


def _swap_database(conn_str: str, db_name: str) -> str:
    if "DATABASE=" in conn_str.upper():
        return re.sub(
            r"DATABASE=[^;]+",
            f"DATABASE={db_name}",
            conn_str,
            flags=re.IGNORECASE,
        )
    return conn_str + f";DATABASE={db_name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="CustomerDemo", help="Database to probe")
    parser.add_argument("--email", default=None, help="User whose creds to use")
    args = parser.parse_args()

    # 1. What's actually indexed for source='sql'?
    print("=" * 70)
    print(" 1. Chunks already indexed for source='sql'")
    print("=" * 70)
    with get_db() as db:
        rows = (
            db.query(
                VectorChunk.resource_id,
                VectorChunk.object_name,
                VectorChunk.db_name,
                VectorChunk.extra,
            )
            .filter(VectorChunk.source == "sql")
            .order_by(VectorChunk.db_name, VectorChunk.object_name)
            .all()
        )
    print(f"\nTotal SQL chunks: {len(rows)}\n")
    for r in rows:
        obj_type = (r.extra or {}).get("object_type", "?")
        print(f"  {obj_type:10s}  {r.db_name}.{r.object_name}    ({r.resource_id})")
    print()

    # 2. Live probe of SQL Server.
    print("=" * 70)
    print(f" 2. Live probe of SQL Server (DATABASE={args.db})")
    print("=" * 70)
    with get_db() as db:
        user = _pick_user(db, args.email)
        creds = load_all_credentials(db, user.id, "sql")
    if "conn_str" not in creds:
        print("!! No SQL conn_str saved for", user.email, file=sys.stderr)
        sys.exit(2)
    conn_str = _swap_database(creds["conn_str"], args.db)

    engine = create_engine(URL.create("mssql+pyodbc", query={"odbc_connect": conn_str}))
    with engine.connect() as conn:
        cu, dbn, sqluser = conn.execute(
            text("SELECT CURRENT_USER, DB_NAME(), SUSER_SNAME()")
        ).first()
        print(f"\n  CURRENT_USER  = {cu}")
        print(f"  DB_NAME()     = {dbn}")
        print(f"  SUSER_SNAME() = {sqluser}")

        print("\n  INFORMATION_SCHEMA.TABLES (this is what the ingestor uses):")
        rows = conn.execute(
            text(
                "SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE "
                "FROM INFORMATION_SCHEMA.TABLES "
                "ORDER BY TABLE_TYPE, TABLE_SCHEMA, TABLE_NAME"
            )
        ).all()
        if not rows:
            print("    (no rows — that's the problem)")
        for r in rows:
            print(f"    {r.TABLE_TYPE:12s} {r.TABLE_SCHEMA}.{r.TABLE_NAME}")

        print("\n  sys.tables (the more permissive system view):")
        rows = conn.execute(
            text(
                "SELECT s.name AS schema_name, t.name AS table_name "
                "FROM sys.tables t "
                "JOIN sys.schemas s ON s.schema_id = t.schema_id "
                "ORDER BY s.name, t.name"
            )
        ).all()
        if not rows:
            print("    (no rows)")
        for r in rows:
            print(f"    {r.schema_name}.{r.table_name}")

        print("\n  sys.views:")
        rows = conn.execute(
            text(
                "SELECT s.name AS schema_name, v.name AS view_name "
                "FROM sys.views v "
                "JOIN sys.schemas s ON s.schema_id = v.schema_id "
                "ORDER BY s.name, v.name"
            )
        ).all()
        if not rows:
            print("    (no rows)")
        for r in rows:
            print(f"    {r.schema_name}.{r.view_name}")

    engine.dispose()
    print()


if __name__ == "__main__":
    main()
