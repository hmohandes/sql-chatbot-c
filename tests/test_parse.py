from app.agent import _extract_sql, _fallback_parse, _normalize_tool_call, _version_tuple


def test_fallback_parse_tool_call():
    raw = (
        '<|tool_call>call:sql_db_list_tables{}<tool_call|>'
        '<|tool_response>'
    )
    parsed = _fallback_parse(raw)
    assert parsed["tool_calls"]
    assert parsed["tool_calls"][0]["function"]["name"] == "sql_db_list_tables"


def test_fallback_parse_query_args():
    raw = (
        '<|tool_call>call:sql_db_query{query:<|"|>SELECT id FROM customers LIMIT 5<|"|>}'
        "<tool_call|>"
    )
    parsed = _fallback_parse(raw)
    args = parsed["tool_calls"][0]["function"]["arguments"]
    assert "SELECT id FROM customers LIMIT 5" in args["query"]


def test_normalize_openai_style_call():
    name, arguments = _normalize_tool_call(
        {
            "type": "function",
            "function": {"name": "sql_db_schema", "arguments": {"table_names": "a, b"}},
        }
    )
    assert name == "sql_db_schema"
    assert arguments["table_names"] == "a, b"


def test_version_tuple_ignores_local_suffix():
    assert _version_tuple("5.24.0") >= (5, 10, 1)
    assert _version_tuple("4.49.0") < (5, 10, 1)
    assert _version_tuple("2.15.0+cpu") == (2, 15, 0)


def test_extract_sql_from_fence():
    text = "Here is the query:\n```sql\nSELECT 1\n```\n"
    assert _extract_sql(text) == "SELECT 1"
