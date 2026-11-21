from __future__ import annotations

from typing import Any, Callable

from app.config import get_settings
from app.db import (
    format_rows,
    get_connection,
    list_tables,
    run_select,
    sample_rows,
    table_schema,
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "sql_db_list_tables",
            "description": (
                "Input is an empty string, output is a comma-separated list of "
                "tables in the database."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_input": {
                        "type": "string",
                        "description": "An empty string.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_schema",
            "description": (
                "Input to this tool is a comma-separated list of tables, output is "
                "the schema and sample rows for those tables. Be sure that the "
                "tables actually exist by calling sql_db_list_tables first! "
                "Example Input: table1, table2, table3"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "string",
                        "description": "Comma-separated list of table names.",
                    }
                },
                "required": ["table_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_query",
            "description": (
                "Input to this tool is a detailed and correct SQL query, output is "
                "a result from the database. If the query is not correct, an error "
                "message will be returned. If an error is returned, rewrite the "
                "query, check the query, and try again. If you encounter an issue "
                "with Unknown column 'xxxx' in 'field list', use sql_db_schema to "
                "query the correct table fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A detailed and correct PostgreSQL query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_query_checker",
            "description": (
                "Use this tool to double check if your query is correct before "
                "executing it. Always use this tool before executing a query with "
                "sql_db_query!"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to double-check.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def sql_db_list_tables(tool_input: str = "") -> str:
    del tool_input
    with get_connection() as conn:
        tables = list_tables(conn)
    return ", ".join(tables) if tables else "(no tables found)"


def sql_db_schema(table_names: str) -> str:
    requested = [name.strip() for name in table_names.split(",") if name.strip()]
    if not requested:
        return "Error: provide a comma-separated list of table names."

    settings = get_settings()
    parts: list[str] = []
    with get_connection() as conn:
        existing = set(list_tables(conn))
        missing = [name for name in requested if name not in existing]
        if missing:
            parts.append(
                "Error: unknown table(s): "
                + ", ".join(missing)
                + f". Available tables: {', '.join(sorted(existing)) or '(none)'}"
            )
            requested = [name for name in requested if name in existing]

        for table in requested:
            schema = table_schema(conn, table)
            create_lines = [f'Table "{table}"']
            create_lines.append("Columns:")
            for col in schema["columns"]:
                nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col["column_default"] else ""
                create_lines.append(
                    f"  - {col['column_name']}: {col['data_type']} {nullable}{default}"
                )
            if schema["primary_keys"]:
                create_lines.append(
                    "Primary key: " + ", ".join(schema["primary_keys"])
                )
            if schema["foreign_keys"]:
                create_lines.append("Foreign keys:")
                for fk in schema["foreign_keys"]:
                    create_lines.append(
                        f"  - {fk['column_name']} -> {fk['foreign_table']}.{fk['foreign_column']}"
                    )
            samples = sample_rows(conn, table, settings.sample_rows)
            create_lines.append("Sample rows:")
            create_lines.append(format_rows(samples))
            parts.append("\n".join(create_lines))

    return "\n\n".join(parts) if parts else "Error: no valid tables requested."


def sql_db_query(query: str) -> str:
    try:
        with get_connection() as conn:
            rows = run_select(conn, query)
        return format_rows(rows)
    except Exception as exc:
        return f"Error: {exc}"


def first_string_arg(arguments: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = arguments.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        return str(value)
    if len(arguments) == 1:
        value = next(iter(arguments.values()))
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        return str(value)
    return ""


ToolFn = Callable[..., str]


def build_tool_registry(query_checker: ToolFn) -> dict[str, ToolFn]:
    return {
        "sql_db_list_tables": sql_db_list_tables,
        "sql_db_schema": sql_db_schema,
        "sql_db_query": sql_db_query,
        "sql_db_query_checker": query_checker,
    }


def dispatch_tool(name: str, arguments: dict[str, Any], registry: dict[str, ToolFn]) -> str:
    if name not in registry:
        return f"Error: unknown tool '{name}'."

    if name == "sql_db_list_tables":
        return registry[name](first_string_arg(arguments, "tool_input", "input", "query"))
    if name == "sql_db_schema":
        tables = first_string_arg(arguments, "table_names", "tables", "tool_input", "input")
        return registry[name](tables)
    if name in {"sql_db_query", "sql_db_query_checker"}:
        query = first_string_arg(arguments, "query", "sql", "tool_input", "input")
        if not query.strip():
            return "Error: query argument is required."
        return registry[name](query)
    return registry[name](**arguments)
