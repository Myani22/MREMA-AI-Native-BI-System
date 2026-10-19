from __future__ import annotations

from datetime import date
from typing import Any

from app.query_system import QueryEngine, Table


SCHEMA: dict[str, dict[str, str]] = {
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


SOURCES: dict[str, list[dict[str, Any]]] = {
    "orders": [
        {
            "id": 1,
            "user_id": 10,
            "status": "paid",
            "amount": 100.0,
            "created_at": "2026-01-01",
        },
        {
            "id": 2,
            "user_id": 10,
            "status": "paid",
            "amount": 50.0,
            "created_at": "2026-01-05",
        },
        {
            "id": 3,
            "user_id": 20,
            "status": "pending",
            "amount": 15.0,
            "created_at": "2026-01-03",
        },
        {
            "id": 4,
            "user_id": 30,
            "status": "cancelled",
            "amount": None,
            "created_at": "2026-01-02",
        },
    ],
    "users": [
        {"id": 10, "name": "Alice", "country": "SE", "active": True},
        {"id": 20, "name": "Bob", "country": "DE", "active": False},
        {"id": 40, "name": "Charlie", "country": "SE", "active": True},
    ],
}


def test_linear_pipeline_and_roundtrip() -> bool:
    engine = QueryEngine()
    raw = {
        "steps": [
            {"var": "T1", "op": "source", "table": "orders"},
            {"var": "C1", "op": "select", "input": "T1", "columns": ["status", "amount"]},
            {
                "var": "F1",
                "op": "filter",
                "input": "C1",
                "condition": {"column": "amount", "cmp": "gt", "value": 20},
            },
            {
                "var": "S1",
                "op": "sort",
                "input": "F1",
                "by": [{"column": "amount", "dir": "asc"}],
            },
            {"var": "L1", "op": "limit", "input": "S1", "n": 1},
        ],
        "output": "L1",
    }
    query = engine.from_dict(raw)
    if engine.to_dict(query) != raw:
        return False

    val = engine.validate(query, SCHEMA)
    if not val.ok or val.output_schema != {"status": "str", "amount": "float"}:
        return False

    res = engine.execute(query, SOURCES).to_dicts()
    return res == [{"status": "paid", "amount": 50.0}]


def test_leaf_conditions() -> bool:
    engine = QueryEngine()
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {
                "var": "F1",
                "op": "filter",
                "input": "O",
                "condition": {
                    "and": [
                        {"column": "amount", "cmp": "is_not_null"},
                        {"column": "amount", "cmp": "between", "min": 15.0, "max": 100.0},
                        {"column": "status", "cmp": "in", "values": ["paid", "pending"]},
                        {"column": "status", "cmp": "not_in", "values": ["cancelled"]},
                        {"column": "status", "cmp": "like", "pattern": "p%"},
                        {"column": "created_at", "cmp": "ge", "value": "2026-01-03"},
                    ]
                },
            },
            {"var": "S", "op": "select", "input": "F1", "columns": ["id"]},
            {"var": "O_SORT", "op": "sort", "input": "S", "by": [{"column": "id", "dir": "asc"}]},
        ],
        "output": "O_SORT",
    }
    query = engine.from_dict(raw)
    if engine.to_dict(query) != raw:
        return False

    val = engine.validate(query, SCHEMA)
    if not val.ok:
        return False

    res = engine.execute(query, SOURCES).to_dicts()
    return res == [{"id": 2}, {"id": 3}]


def test_logical_combinators_not_or_is_null() -> bool:
    engine = QueryEngine()
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {
                "var": "F",
                "op": "filter",
                "input": "O",
                "condition": {
                    "or": [
                        {"column": "amount", "cmp": "is_null"},
                        {
                            "not": {
                                "and": [
                                    {"column": "status", "cmp": "eq", "value": "paid"},
                                    {"column": "amount", "cmp": "le", "value": 100.0},
                                ]
                            }
                        },
                    ]
                },
            },
            {"var": "S", "op": "select", "input": "F", "columns": ["id"]},
            {"var": "SRT", "op": "sort", "input": "S", "by": [{"column": "id", "dir": "asc"}]},
        ],
        "output": "SRT",
    }
    query = engine.from_dict(raw)
    val = engine.validate(query, SCHEMA)
    if not val.ok:
        return False

    res = engine.execute(query, SOURCES).to_dicts()
    return res == [{"id": 3}, {"id": 4}]


def test_distinct_and_null_sorting() -> bool:
    engine = QueryEngine()
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "S", "op": "select", "input": "O", "columns": ["status", "amount"]},
            {
                "var": "SRT",
                "op": "sort",
                "input": "S",
                "by": [{"column": "amount", "dir": "desc"}],
            },
            {"var": "ONLY_STATUS", "op": "select", "input": "SRT", "columns": ["status"]},
            {"var": "D", "op": "distinct", "input": "ONLY_STATUS"},
        ],
        "output": "D",
    }
    query = engine.from_dict(raw)
    val = engine.validate(query, SCHEMA)
    if not val.ok:
        return False

    sorted_with_null = engine.execute(
        engine.from_dict({**raw, "output": "SRT"}), SOURCES
    ).to_dicts()
    if sorted_with_null[-1]["amount"] is not None:
        return False

    distinct_res = engine.execute(query, SOURCES).to_dicts()
    return distinct_res == [
        {"status": "paid"},
        {"status": "pending"},
        {"status": "cancelled"},
    ]


def test_aggregate_and_having() -> bool:
    engine = QueryEngine()
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {
                "var": "A",
                "op": "aggregate",
                "input": "O",
                "group_by": ["status"],
                "aggregates": [
                    {"func": "count", "column": "*", "alias": "row_cnt"},
                    {"func": "count", "column": "amount", "alias": "non_null_cnt"},
                    {"func": "sum", "column": "amount", "alias": "total_amount"},
                    {"func": "avg", "column": "amount", "alias": "avg_amount"},
                    {"func": "min", "column": "created_at", "alias": "first_date"},
                    {"func": "max", "column": "created_at", "alias": "last_date"},
                ],
                "having": {"column": "row_cnt", "cmp": "ge", "value": 1},
            },
            {
                "var": "S",
                "op": "sort",
                "input": "A",
                "by": [{"column": "row_cnt", "dir": "desc"}, {"column": "status", "dir": "asc"}],
            },
        ],
        "output": "S",
    }
    query = engine.from_dict(raw)
    if engine.to_dict(query) != raw:
        return False

    val = engine.validate(query, SCHEMA)
    expected_schema = {
        "status": "str",
        "row_cnt": "int",
        "non_null_cnt": "int",
        "total_amount": "float",
        "avg_amount": "float",
        "first_date": "date",
        "last_date": "date",
    }
    if not val.ok or val.output_schema != expected_schema:
        return False

    res = engine.execute(query, SOURCES).to_dicts()
    return res == [
        {
            "status": "paid",
            "row_cnt": 2,
            "non_null_cnt": 2,
            "total_amount": 150.0,
            "avg_amount": 75.0,
            "first_date": date(2026, 1, 1),
            "last_date": date(2026, 1, 5),
        },
        {
            "status": "cancelled",
            "row_cnt": 1,
            "non_null_cnt": 0,
            "total_amount": None,
            "avg_amount": None,
            "first_date": date(2026, 1, 2),
            "last_date": date(2026, 1, 2),
        },
        {
            "status": "pending",
            "row_cnt": 1,
            "non_null_cnt": 1,
            "total_amount": 15.0,
            "avg_amount": 15.0,
            "first_date": date(2026, 1, 3),
            "last_date": date(2026, 1, 3),
        },
    ]


def test_joins_inner_left_right_and_collisions() -> bool:
    engine = QueryEngine()
    for how, expected_len in (("inner", 3), ("left", 4), ("right", 4)):
        raw = {
            "steps": [
                {"var": "O", "op": "source", "table": "orders"},
                {"var": "U", "op": "source", "table": "users"},
                {
                    "var": "J",
                    "op": "join",
                    "left": "O",
                    "right": "U",
                    "on": [{"left_col": "user_id", "right_col": "id"}],
                    "how": how,
                },
            ],
            "output": "J",
        }
        query = engine.from_dict(raw)
        val = engine.validate(query, SCHEMA)
        if not val.ok or val.output_schema is None:
            return False
        if "id_left" not in val.output_schema or "id_right" not in val.output_schema:
            return False

        rows = engine.execute(query, SOURCES).to_dicts()
        if len(rows) != expected_len:
            return False

    return True


def test_dataframe_and_table_sources() -> bool:
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
        data=SOURCES["users"],
    )
    table_obj = Table.from_source(SOURCES["orders"])

    engine = QueryEngine()
    raw = {
        "steps": [
            {"var": "U", "op": "source", "table": "users"},
            {"var": "O", "op": "source", "table": "orders"},
            {
                "var": "F",
                "op": "filter",
                "input": "U",
                "condition": {"column": "active", "cmp": "eq", "value": True},
            },
            {"var": "S", "op": "select", "input": "F", "columns": ["name"]},
        ],
        "output": "S",
    }
    query = engine.from_dict(raw)
    if not engine.validate(query, SCHEMA).ok:
        return False

    res = engine.execute(query, {"users": df_like, "orders": table_obj}).to_dicts()
    return res == [{"name": "Alice"}, {"name": "Charlie"}]


def test_validation_error_collection_and_condition_index() -> bool:
    engine = QueryEngine()
    raw = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {"var": "O", "op": "select", "input": "O", "columns": ["non_existent"]},
            {"var": "BAD_REF", "op": "distinct", "input": "MISSING_VAR"},
            {
                "var": "F",
                "op": "filter",
                "input": "O",
                "condition": {
                    "and": [
                        {"column": "id", "cmp": "eq", "value": 1},
                        {"column": "amount", "cmp": "gt", "value": "wrong_type"},
                        {"column": "status", "cmp": "eq", "value": "paid"},
                    ]
                },
            },
            {"var": "U_OP", "op": "pivot", "input": "O"},
        ],
        "output": "UNKNOWN_OUT",
    }
    query = engine.from_dict(raw)
    val = engine.validate(query, SCHEMA)
    if val.ok or val.output_schema is not None:
        return False

    reasons = {err.reason for err in val.errors}
    expected_reasons = {
        "duplicate_var",
        "column_not_found",
        "undefined_reference",
        "type_mismatch",
        "unsupported_operation",
        "output_not_found",
    }
    if not expected_reasons.issubset(reasons):
        return False

    return any(
        err.reason == "type_mismatch" and err.field == "condition.and[1].value"
        for err in val.errors
    )


def main() -> None:
    cnt = 0
    tests = [
        [test_linear_pipeline_and_roundtrip, True],
        [test_leaf_conditions, True],
        [test_logical_combinators_not_or_is_null, True],
        [test_distinct_and_null_sorting, True],
        [test_aggregate_and_having, True],
        [test_joins_inner_left_right_and_collisions, True],
        [test_dataframe_and_table_sources, True],
        [test_validation_error_collection_and_condition_index, True],
    ]

    for idx, (test_fn, expected) in enumerate(tests, start=1):
        actual = test_fn()
        passed = actual == expected
        if passed:
            cnt += 1
        status = "OK" if passed else "FAIL"
        print(f"[{idx}/{len(tests)}] {test_fn.__name__}: {status} (got {actual}, expected {expected})")

    print(f"\nPassed: {cnt}/{len(tests)}")


if __name__ == "__main__":
    main()