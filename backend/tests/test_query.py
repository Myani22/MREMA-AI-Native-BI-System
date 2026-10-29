from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.query_system import QueryEngine, Table


@pytest.fixture
def engine() -> QueryEngine:
    return QueryEngine()


@pytest.fixture
def schema() -> dict[str, dict[str, str]]:
    return {
        "orders": {
            "id": "int",
            "user_id": "int",
            "status": "str",
            "amount": "float",
            "created_at": "date",
        },
        "users": {
            "id": "int",
            "name": "str",
            "country": "str",
            "active": "bool",
        },
    }

@pytest.fixture
def sources() -> dict[str, list[dict[str, Any]]]:
    return {
        "orders": [
            {"id": 1, "user_id": 10, "status": "paid",
             "amount": 100.0, "created_at": date(2026, 1, 1)},
            {"id": 2, "user_id": 10, "status": "paid",
             "amount": 50.0,  "created_at": date(2026, 1, 5)},
            {"id": 3, "user_id": 20, "status": "pending",
             "amount": 15.0,  "created_at": date(2026, 1, 3)},
            {"id": 4, "user_id": 30, "status": "cancelled",
             "amount": None,  "created_at": date(2026, 1, 2)},
        ],
        "users": [
            {"id": 10, "name": "Alice",   "country": "SE", "active": True},
            {"id": 20, "name": "Bob",     "country": "DE", "active": False},
            {"id": 40, "name": "Charlie", "country": "SE", "active": True},
        ],
    }


def test_linear_pipeline_roundtrip(engine, schema):
    raw = {
        "steps": [
            {"var": "T1", "op": "source", "table": "orders"},
            {"var": "C1", "op": "select", "input": "T1", "columns": ["status", "amount"]},
            {"var": "F1", "op": "filter", "input": "C1",
             "condition": {"column": "amount", "cmp": "gt", "value": 20}},
            {"var": "S1", "op": "sort", "input": "F1",
             "by": [{"column": "amount", "dir": "asc"}]},
            {"var": "L1", "op": "limit", "input": "S1", "n": 1},
        ],
        "output": "L1",
    }
    query = engine.from_dict(raw)
    assert engine.to_dict(query) == raw

    val = engine.validate(query, schema)
    assert val.ok
    assert val.output_schema == {"status": "str", "amount": "float"}


def test_linear_pipeline_execute(engine, schema, sources):
    raw = {
        "steps": [
            {"var": "T1", "op": "source", "table": "orders"},
            {"var": "C1", "op": "select", "input": "T1", "columns": ["status", "amount"]},
            {"var": "F1", "op": "filter", "input": "C1",
             "condition": {"column": "amount", "cmp": "gt", "value": 20}},
            {"var": "S1", "op": "sort", "input": "F1",
             "by": [{"column": "amount", "dir": "asc"}]},
            {"var": "L1", "op": "limit", "input": "S1", "n": 1},
        ],
        "output": "L1",
    }
    query = engine.from_dict(raw)
    assert engine.execute(query, sources).to_dicts() == [{"status": "paid", "amount": 50.0}]


def test_leaf_conditions(engine, schema, sources):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "F1", "op": "filter", "input": "O", "condition": {"and": [
                {"column": "amount", "cmp": "is_not_null"},
                {"column": "amount", "cmp": "between", "min": 15.0, "max": 100.0},
                {"column": "status", "cmp": "in", "values": ["paid", "pending"]},
                {"column": "status", "cmp": "not_in", "values": ["cancelled"]},
                {"column": "status", "cmp": "like", "pattern": "p%"},
                {"column": "created_at", "cmp": "ge", "value": "2026-01-03"},
            ]}},
            {"var": "S", "op": "select", "input": "F1", "columns": ["id"]},
            {"var": "O_SORT", "op": "sort", "input": "S", "by": [{"column": "id", "dir": "asc"}]},
        ],
        "output": "O_SORT",
    }
    query = engine.from_dict(raw)
    assert engine.to_dict(query) == raw
    assert engine.validate(query, schema).ok
    assert engine.execute(query, sources).to_dicts() == [{"id": 2}, {"id": 3}]


def test_logical_combinators_not_or_is_null(engine, schema, sources):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "F", "op": "filter", "input": "O", "condition": {"or": [
                {"column": "amount", "cmp": "is_null"},
                {"not": {"and": [
                    {"column": "status", "cmp": "eq", "value": "paid"},
                    {"column": "amount", "cmp": "le", "value": 100.0},
                ]}},
            ]}},
            {"var": "S", "op": "select", "input": "F", "columns": ["id"]},
            {"var": "SRT", "op": "sort", "input": "S", "by": [{"column": "id", "dir": "asc"}]},
        ],
        "output": "SRT",
    }
    query = engine.from_dict(raw)
    assert engine.validate(query, schema).ok
    assert engine.execute(query, sources).to_dicts() == [{"id": 3}, {"id": 4}]


def test_like_underscore_matches_single_char(engine, schema, sources):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "F", "op": "filter", "input": "O",
             "condition": {"column": "status", "cmp": "like", "pattern": "p__d"}},
            {"var": "S", "op": "select", "input": "F", "columns": ["id"]},
            {"var": "SR", "op": "sort", "input": "S", "by": [{"column": "id", "dir": "asc"}]},
        ],
        "output": "SR",
    }
    query = engine.from_dict(raw)
    assert engine.validate(query, schema).ok
    assert engine.execute(query, sources).to_dicts() == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize(
    "condition, expected_ids",
    [
        ({"column": "status", "cmp": "eq", "value": "paid"}, [1, 2]),
        ({"column": "status", "cmp": "ne", "value": "paid"}, [3, 4]),
        ({"column": "amount", "cmp": "gt", "value": 50}, [1]),
        ({"column": "amount", "cmp": "ge", "value": 50}, [1, 2]),
        ({"column": "amount", "cmp": "lt", "value": 50}, [3]),
        ({"column": "amount", "cmp": "le", "value": 15}, [3]),
        ({"column": "amount", "cmp": "is_null"}, [4]),
        ({"column": "amount", "cmp": "is_not_null"}, [1, 2, 3]),
    ],
)
def test_leaf_comparisons_matrix(engine, schema, sources, condition, expected_ids):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "F", "op": "filter", "input": "O", "condition": condition},
            {"var": "S", "op": "select", "input": "F", "columns": ["id"]},
            {"var": "SR", "op": "sort", "input": "S", "by": [{"column": "id", "dir": "asc"}]},
        ],
        "output": "SR",
    }
    query = engine.from_dict(raw)
    assert engine.validate(query, schema).ok
    assert [r["id"] for r in engine.execute(query, sources).to_dicts()] == expected_ids


def test_distinct_and_null_sorting(engine, schema, sources):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "S", "op": "select", "input": "O", "columns": ["status", "amount"]},
            {"var": "SRT", "op": "sort", "input": "S", "by": [{"column": "amount", "dir": "desc"}]},
            {"var": "ONLY_STATUS", "op": "select", "input": "SRT", "columns": ["status"]},
            {"var": "D", "op": "distinct", "input": "ONLY_STATUS"},
        ],
        "output": "D",
    }
    query = engine.from_dict(raw)
    assert engine.validate(query, schema).ok

    sorted_rows = engine.execute(engine.from_dict({**raw, "output": "SRT"}), sources).to_dicts()
    assert sorted_rows[-1]["amount"] is None

    assert engine.execute(query, sources).to_dicts() == [
        {"status": "paid"},
        {"status": "pending"},
        {"status": "cancelled"},
    ]


def test_aggregate_and_having(engine, schema, sources):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "A", "op": "aggregate", "input": "O", "group_by": ["status"],
             "aggregates": [
                 {"func": "count",     "column": "*",          "alias": "row_cnt"},
                 {"func": "count",     "column": "amount",     "alias": "non_null_cnt"},
                 {"func": "sum",       "column": "amount",     "alias": "total_amount"},
                 {"func": "avg",       "column": "amount",     "alias": "avg_amount"},
                 {"func": "min",       "column": "created_at", "alias": "first_date"},
                 {"func": "max",       "column": "created_at", "alias": "last_date"},
             ],
             "having": {"column": "row_cnt", "cmp": "ge", "value": 1}},
            {"var": "S", "op": "sort", "input": "A",
             "by": [{"column": "row_cnt", "dir": "desc"}, {"column": "status", "dir": "asc"}]},
        ],
        "output": "S",
    }
    query = engine.from_dict(raw)
    assert engine.to_dict(query) == raw

    val = engine.validate(query, schema)
    assert val.ok
    assert val.output_schema == {
        "status": "str",
        "row_cnt": "int",
        "non_null_cnt": "int",
        "total_amount": "float",
        "avg_amount": "float",
        "first_date": "date",
        "last_date": "date",
    }

    assert engine.execute(query, sources).to_dicts() == [
        {"status": "paid", "row_cnt": 2, "non_null_cnt": 2, "total_amount": 150.0,
         "avg_amount": 75.0, "first_date": date(2026, 1, 1), "last_date": date(2026, 1, 5)},
        {"status": "cancelled", "row_cnt": 1, "non_null_cnt": 0, "total_amount": 0,
         "avg_amount": None, "first_date": date(2026, 1, 2), "last_date": date(2026, 1, 2)},
        {"status": "pending", "row_cnt": 1, "non_null_cnt": 1, "total_amount": 15.0,
         "avg_amount": 15.0, "first_date": date(2026, 1, 3), "last_date": date(2026, 1, 3)},
    ]


def test_aggregate_distinct_minmax_without_group_by(engine, schema, sources):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "A", "op": "aggregate", "input": "O", "group_by": [],
             "aggregates": [
                 {"func": "count_distinct", "column": "user_id", "alias": "uniq_users"},
                 {"func": "min",            "column": "amount",  "alias": "min_amount"},
                 {"func": "max",            "column": "amount",  "alias": "max_amount"},
             ]},
        ],
        "output": "A",
    }
    query = engine.from_dict(raw)
    val = engine.validate(query, schema)
    assert val.ok
    assert val.output_schema == {"uniq_users": "int", "min_amount": "float", "max_amount": "float"}
    assert engine.execute(query, sources).to_dicts() == [
        {"uniq_users": 3, "min_amount": 15.0, "max_amount": 100.0}
    ]


@pytest.mark.parametrize("how, expected_len", [("inner", 3), ("left", 4), ("right", 4)])
def test_joins(engine, schema, sources, how, expected_len):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "U", "op": "source", "table": "users"},
            {"var": "J", "op": "join", "left": "O", "right": "U",
             "on": [{"left_col": "user_id", "right_col": "id"}], "how": how},
        ],
        "output": "J",
    }
    query = engine.from_dict(raw)
    val = engine.validate(query, schema)
    assert val.ok
    assert val.output_schema is not None
    assert "id_left" in val.output_schema and "id_right" in val.output_schema
    assert len(engine.execute(query, sources).to_dicts()) == expected_len


def test_multi_key_join():
    engine = QueryEngine()
    schema = {
        "orders": {"id": "int", "user_id": "int", "status": "str"},
        "users":  {"id": "int", "country": "str", "status": "str"},
    }
    sources = {
        "orders": [{"id": 1, "user_id": 10, "status": "paid"}],
        "users":  [{"id": 10, "country": "SE", "status": "paid"}],
    }
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "U", "op": "source", "table": "users"},
            {"var": "J", "op": "join", "left": "O", "right": "U", "how": "inner",
             "on": [{"left_col": "user_id", "right_col": "id"},
                    {"left_col": "status", "right_col": "status"}]},
        ],
        "output": "J",
    }
    query = engine.from_dict(raw)
    assert engine.validate(query, schema).ok
    rows = engine.execute(query, sources).to_dicts()
    assert len(rows) == 1
    assert rows[0]["country"] == "SE"


def test_dataframe_and_table_sources():
    class DummyDataFrame:
        def __init__(self, columns: list[str], data: list[dict[str, Any]]) -> None:
            self.columns = columns
            self._data = data
        def to_dict(self, orient: str = "records") -> list[dict[str, Any]]:
            if orient != "records":
                raise ValueError(orient)
            return list(self._data)

    df_like = DummyDataFrame(
        columns=["id", "name", "country", "active"],
        data=[
            {"id": 10, "name": "Alice",   "country": "SE", "active": True},
            {"id": 20, "name": "Bob",     "country": "DE", "active": False},
            {"id": 40, "name": "Charlie", "country": "SE", "active": True},
        ],
    )
    table_obj = Table.from_source([
        {"id": 1, "user_id": 10, "status": "paid",      "amount": 100.0, "created_at": "2026-01-01"},
        {"id": 2, "user_id": 10, "status": "paid",      "amount": 50.0,  "created_at": "2026-01-05"},
        {"id": 3, "user_id": 20, "status": "pending",   "amount": 15.0,  "created_at": "2026-01-03"},
        {"id": 4, "user_id": 30, "status": "cancelled", "amount": None,  "created_at": "2026-01-02"},
    ])

    engine = QueryEngine()
    schema = {
        "users": {"id": "int", "name": "str", "country": "str", "active": "bool"},
        "orders": {"id": "int", "user_id": "int", "status": "str", "amount": "float", "created_at": "date"},
    }
    raw = {
        "steps": [
            {"var": "U", "op": "source", "table": "users"},
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "F", "op": "filter", "input": "U",
             "condition": {"column": "active", "cmp": "eq", "value": True}},
            {"var": "S", "op": "select", "input": "F", "columns": ["name"]},
        ],
        "output": "S",
    }
    query = engine.from_dict(raw)
    assert engine.validate(query, schema).ok
    assert engine.execute(query, {"users": df_like, "orders": table_obj}).to_dicts() == [
        {"name": "Alice"},
        {"name": "Charlie"},
    ]


def test_limit_zero_returns_empty_table(engine, schema, sources):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "L", "op": "limit", "input": "O", "n": 0},
        ],
        "output": "L",
    }
    query = engine.from_dict(raw)
    assert engine.validate(query, schema).ok
    assert engine.execute(query, sources).to_dicts() == []


def test_validation_collects_multiple_errors(engine, schema):
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "O", "op": "select", "input": "O", "columns": ["non_existent"]},
            {"var": "BAD_REF", "op": "distinct", "input": "MISSING_VAR"},
            {"var": "F", "op": "filter", "input": "O", "condition": {"and": [
                {"column": "id", "cmp": "eq", "value": 1},
                {"column": "amount", "cmp": "gt", "value": "wrong_type"},
                {"column": "status", "cmp": "eq", "value": "paid"},
            ]}},
            {"var": "U_OP", "op": "pivot", "input": "O"},
        ],
        "output": "UNKNOWN_OUT",
    }
    query = engine.from_dict(raw)
    val = engine.validate(query, schema)
    assert not val.ok
    assert val.output_schema is None

    reasons = {e.reason for e in val.errors}
    assert {
        "duplicate_var",
        "column_not_found",
        "undefined_reference",
        "type_mismatch",
        "unsupported_operation",
        "output_not_found",
    }.issubset(reasons)

    assert any(
        e.reason == "type_mismatch" and e.field == "condition.and[1].value"
        for e in val.errors
    )


def test_source_table_missing(engine, schema):
    query = engine.from_dict({"steps": [{"var": "X", "op": "source", "table": "nope"}], "output": "X"})
    val = engine.validate(query, schema)
    assert not val.ok
    assert any(e.reason == "table_not_found" and e.field == "table" for e in val.errors)