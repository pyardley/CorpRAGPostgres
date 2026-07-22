"""
Shared T-SQL table DDL rendering.

Both `app.ingestion.sql_ingestor` (reconstructing a table's definition
at ingestion time from `INFORMATION_SCHEMA.COLUMNS`) and
`mcp_server.tools.sql_schema_tools` (the live `sql_object_definition`
tool's table case) need to turn a set of column rows into a
`CREATE TABLE` statement. This is the one place that does it, so the
two representations never drift apart.
"""

from __future__ import annotations

from typing import Any


def render_table_ddl(full_name: str, columns: list[Any]) -> str:
    """
    Build a `CREATE TABLE ... (...)` string from column rows shaped like
    `(schema, table, column_name, data_type, max_length, is_nullable,
    column_default)` — the exact row shape an `INFORMATION_SCHEMA.COLUMNS`
    query returns here. Only column name/type/nullability/default are
    reconstructed; FK/PK/CHECK constraints aren't in that view, so they
    never appear in the rendered DDL.
    """
    ddl_lines: list[str] = []
    for col in columns:
        col_name = col[2]
        data_type = col[3]
        max_len = col[4]
        nullable = col[5]
        default = col[6]
        type_str = data_type + (f"({max_len})" if max_len else "")
        null_str = "NULL" if nullable == "YES" else "NOT NULL"
        default_str = f" DEFAULT {default}" if default else ""
        ddl_lines.append(f"  {col_name} {type_str} {null_str}{default_str}")

    return "CREATE TABLE " + full_name + " (\n" + ",\n".join(ddl_lines) + "\n)"
