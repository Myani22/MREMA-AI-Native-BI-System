from __future__ import annotations

import pytest
from app.agent.integrations.query_dsl import QueryArgs
from app.agent.pipeline_validator import validate_chart_pipeline


def _parse(payload: dict) -> QueryArgs:
    return QueryArgs.model_validate(payload)


def test_valid_bar_pipeline():
    args = _parse(
        {
            "steps": [
                {"var": "T", "op": "source", "table": "orders"},
                {"var": "F", "op": "filter", "input": "T", "condition": {"column": "status", "cmp": "eq", "value": "paid"}},
                {
                    "var": "A",
                    "op": "aggregate",
                    "input": "F",
                    "group_by": ["city"],
                    "aggregates": [{"func": "sum", "column": "amount", "alias": "Выручка"}],
                },
                {"var": "L", "op": "limit", "input": "A", "n": 10},
            ],
            "output": "L",
        }
    )
    assert validate_chart_pipeline(args, chart_type="bar") == []


def test_valid_join_pipeline():
    args = _parse(
        {
            "steps": [
                {"var": "T1", "op": "source", "table": "orders"},
                {"var": "T2", "op": "source", "table": "users"},
                {"var": "J", "op": "join", "left": "T1", "right": "T2", "on": [{"left_col": "user_id", "right_col": "id"}], "how": "inner"},
                {
                    "var": "A",
                    "op": "aggregate",
                    "input": "J",
                    "group_by": ["city"],
                    "aggregates": [{"func": "count_distinct", "column": "user_id", "alias": "Клиенты"}],
                },
            ],
            "output": "A",
        }
    )
    assert validate_chart_pipeline(args, chart_type="bar") == []


def test_forbidden_ops_rejected():
    args = _parse(
        {
            "steps": [
                {"var": "T", "op": "source", "table": "orders"},
                {"var": "S", "op": "select", "input": "T", "columns": ["city"]},
                {"var": "D", "op": "distinct", "input": "S"},
            ],
            "output": "D",
        }
    )
    errors = validate_chart_pipeline(args, chart_type="table")
    assert any("select запрещён" in e for e in errors)
    assert any("distinct запрещён" in e for e in errors)


def test_sort_allowed_for_all_charts():
    args = _parse(
        {
            "steps": [
                {"var": "T", "op": "source", "table": "orders"},
                {
                    "var": "A",
                    "op": "aggregate",
                    "input": "T",
                    "group_by": ["city"],
                    "aggregates": [{"func": "sum", "column": "amount", "alias": "Выручка"}],
                },
                {"var": "S", "op": "sort", "input": "A", "by": [{"column": "Выручка", "dir": "desc"}]},
            ],
            "output": "S",
        }
    )
    assert validate_chart_pipeline(args, chart_type="table") == []
    bar_errors = validate_chart_pipeline(args, chart_type="bar")
    assert bar_errors == []


def test_filter_after_aggregate_rejected():
    args = _parse(
        {
            "steps": [
                {"var": "T", "op": "source", "table": "orders"},
                {
                    "var": "A",
                    "op": "aggregate",
                    "input": "T",
                    "group_by": ["city"],
                    "aggregates": [{"func": "sum", "column": "amount", "alias": "Выручка"}],
                },
                {"var": "F", "op": "filter", "input": "A", "condition": {"column": "Выручка", "cmp": "gt", "value": 100}},
            ],
            "output": "F",
        }
    )
    errors = validate_chart_pipeline(args, chart_type="bar")
    assert any("нарушен порядок" in e for e in errors)