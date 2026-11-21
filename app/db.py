from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from app.config import get_settings

if TYPE_CHECKING:
    import psycopg

_FORBIDDEN_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "copy",
    "comment",
    "vacuum",
    "lock",
    "reindex",
    "cluster",
    "refresh",
    "do",
    "call",
    "execute",
    "set",
    "reset",
    "load",
    "listen",
    "notify",
    "into",
}


@contextmanager
def get_connection() -> Iterator["psycopg.Connection"]:
    import psycopg
    from psycopg.rows import dict_row

    settings = get_settings()
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        yield conn


def list_tables(conn: "psycopg.Connection") -> list[str]:
    rows = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    ).fetchall()
    return [row["table_name"] for row in rows]


def table_schema(conn: "psycopg.Connection", table: str) -> dict[str, Any]:
    columns = conn.execute(
        """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table,),
    ).fetchall()

    primary_keys = conn.execute(
        """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = 'public'
          AND tc.table_name = %s
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position
        """,
        (table,),
    ).fetchall()

    foreign_keys = conn.execute(
        """
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table,
            ccu.column_name AS foreign_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.table_schema = 'public'
          AND tc.table_name = %s
          AND tc.constraint_type = 'FOREIGN KEY'
        """,
        (table,),
    ).fetchall()

    return {
        "table": table,
        "columns": columns,
        "primary_keys": [row["column_name"] for row in primary_keys],
        "foreign_keys": foreign_keys,
    }


def sample_rows(conn: "psycopg.Connection", table: str, limit: int) -> list[dict[str, Any]]:
    from psycopg.sql import SQL, Identifier

    query = SQL("SELECT * FROM {} LIMIT %s").format(Identifier(table))
    return conn.execute(query, (limit,)).fetchall()


def is_read_only_select(sql: str) -> tuple[bool, str]:
    import sqlparse
    from sqlparse.sql import Token

    statements = [stmt for stmt in sqlparse.parse(sql) if stmt.value.strip()]
    if not statements:
        return False, "Empty SQL query."
    if len(statements) != 1:
        return False, "Only a single SQL statement is allowed."

    statement = statements[0]
    first_keyword: str | None = None
    for token in statement.flatten():
        if not isinstance(token, Token) or token.is_whitespace:
            continue
        ttype = str(token.ttype) if token.ttype else ""
        if ttype.startswith("Token.Keyword") or ttype.startswith("Token.DML"):
            first_keyword = token.value.lower()
            break

    if first_keyword not in {"select", "with"}:
        return False, "Only SELECT (or WITH ... SELECT) queries are allowed."

    for token in statement.flatten():
        ttype = str(token.ttype) if token.ttype else ""
        if not (ttype.startswith("Token.Keyword") or ttype.startswith("Token.DML")):
            continue
        value = token.value.lower().strip()
        if value in _FORBIDDEN_KEYWORDS:
            return False, f"Forbidden SQL keyword: {value}"

    normalized = " ".join(sql.lower().split())
    if " for update" in normalized or " for share" in normalized:
        return False, "Locking clauses (FOR UPDATE / FOR SHARE) are not allowed."

    return True, ""


def run_select(conn: "psycopg.Connection", sql: str) -> list[dict[str, Any]]:
    allowed, reason = is_read_only_select(sql)
    if not allowed:
        raise ValueError(reason)
    return conn.execute(sql).fetchall()


def format_rows(rows: Sequence[dict[str, Any]], max_rows: int = 50) -> str:
    if not rows:
        return "Query returned 0 rows."
    shown = list(rows[:max_rows])
    headers = list(shown[0].keys())
    lines = [" | ".join(headers)]
    lines.append(" | ".join("---" for _ in headers))
    for row in shown:
        lines.append(" | ".join(_stringify(row[h]) for h in headers))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows omitted)")
    return "\n".join(lines)


def _stringify(value: Any) -> str:
    if value is None:
        return "NULL"
    return str(value).replace("\n", " ")
