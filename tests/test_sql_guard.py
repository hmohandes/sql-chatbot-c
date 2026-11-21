from app.db import is_read_only_select


def test_allows_select():
    ok, reason = is_read_only_select("SELECT id, name FROM customers LIMIT 5")
    assert ok, reason


def test_allows_cte():
    ok, reason = is_read_only_select(
        "WITH top_customers AS (SELECT id FROM customers) SELECT * FROM top_customers"
    )
    assert ok, reason


def test_rejects_insert():
    ok, reason = is_read_only_select("INSERT INTO customers (name) VALUES ('x')")
    assert not ok
    assert "SELECT" in reason


def test_rejects_drop():
    ok, _ = is_read_only_select("DROP TABLE customers")
    assert not ok


def test_rejects_multiple_statements():
    ok, _ = is_read_only_select("SELECT 1; SELECT 2")
    assert not ok


def test_rejects_for_update():
    ok, _ = is_read_only_select("SELECT id FROM customers FOR UPDATE")
    assert not ok
