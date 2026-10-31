from datetime import date, datetime

import pytest

from app.agent.calculation import describe_calculation, is_complex
from app.agent.catalog_refs import fix_query_args_columns
from app.agent.contracts.catalog import Catalog, ColumnInfo, SemanticType, TableInfo
from app.agent.integrations.csv_store import parse_csv
from app.agent.integrations.dsl_to_sql import compile_dsl_to_sql
from app.agent.integrations.query_dsl import QueryArgs
from app.agent.pipeline_validator import validate_chart_pipeline
from app.query_system import QueryEngine
from app.query_system.dates import infer_date_parser, parse_one
from app.query_system.expressions import ExpressionError, evaluate, infer_type, parse_expression


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2025-03-05", datetime(2025, 3, 5)),
        ("2025-03-05T14:23:00Z", datetime(2025, 3, 5, 14, 23)),
        ("2025-03-05 14:23:11.123+03:00", datetime(2025, 3, 5, 14, 23, 11, 123000)),
        ("05.03.2025", datetime(2025, 3, 5)),
        ("05.03.2025 18:30", datetime(2025, 3, 5, 18, 30)),
        ("2025/3/5", datetime(2025, 3, 5)),
        ("5 марта 2025", datetime(2025, 3, 5)),
        ("5 марта 2025 г.", datetime(2025, 3, 5)),
        ("март 2025", datetime(2025, 3, 1)),
        ("Mar 5, 2025", datetime(2025, 3, 5)),
        ("2025-03", datetime(2025, 3, 1)),
    ],
)
def test_parse_one_formats(raw, expected):
    parsed = parse_one(raw)
    assert parsed is not None and parsed[0] == expected


def test_day_month_order_is_inferred_from_column():
    us = infer_date_parser(["03/05/2025", "03/25/2025"])
    ru = infer_date_parser(["05/03/2025", "25/03/2025"])
    assert us.parse("03/05/2025") == date(2025, 3, 5)
    assert ru.parse("05/03/2025") == date(2025, 3, 5)
    assert infer_date_parser(["12", "abc"]) is None


def test_csv_keeps_time_and_derives_periods():
    raw = "Создано,Сумма\n2025-03-05T14:23:00Z,1\n2025-03-31T23:59:00Z,2\n".encode()
    t = parse_csv(raw, "x.csv")
    assert t.schema["Создано"] == "datetime"
    assert t.rows[1]["Создано"] == datetime(2025, 3, 31, 23, 59)
    assert t.rows[1]["Создано (месяц)"] == date(2025, 3, 1)


def test_expression_types_and_null_semantics():
    schema = {"a": "int", "b": "float", "d1": "datetime", "d2": "date", "s": "str"}
    assert infer_type(parse_expression("[a] / [b]"), schema) == "float"
    assert infer_type(parse_expression("date_diff('day', [d2], [d1])"), schema) == "float"
    assert infer_type(parse_expression("if([s] = 'x' and [a] > 1, [a], null)"), schema) == "int"
    with pytest.raises(ExpressionError):
        infer_type(parse_expression("[d1] - [d2]"), schema)
    with pytest.raises(ExpressionError):
        infer_type(parse_expression("[нет] + 1"), schema)
    with pytest.raises(ExpressionError):
        parse_expression("[a] +")
    row = {"a": 4, "b": 0, "d1": datetime(2025, 3, 2, 12), "d2": date(2025, 3, 1), "s": None}
    assert evaluate(parse_expression("[a] / [b]"), row) is None
    assert evaluate(parse_expression("date_diff('hour', [d2], [d1])"), row) == 36
    assert evaluate(parse_expression("greatest(date_diff(day, [d1], [d2]), 0)"), row) == 0
    assert evaluate(parse_expression("days_in_month([d2])"), row) == 31
    assert evaluate(parse_expression("[d1] >= '2025-03-02' and [d1] < '2025-04-01'"), row) is True
    assert evaluate(parse_expression("if([s] = 'x', 1, 0)"), row) == 0
    assert evaluate(parse_expression("round(2.345, 2)"), row) == 2.35


def test_filter_compares_datetime_with_date_literal():
    engine = QueryEngine()
    rows = [{"t": datetime(2025, 3, 31, 23, 0)}, {"t": datetime(2025, 4, 1, 0, 0)}]
    query = engine.from_dict({"steps": [
        {"var": "T", "op": "source", "table": "x"},
        {"var": "F", "op": "filter", "input": "T", "condition": {"column": "t", "cmp": "lt", "value": "2025-04-01"}},
    ], "output": "F"})
    assert engine.validate(query, {"x": {"t": "datetime"}}).ok
    assert len(engine.execute(query, {"x": rows})) == 1


def test_nested_conditions_and_derive_compile_to_sql():
    args = QueryArgs.model_validate({
        "steps": [
            {"var": "T", "op": "source", "table": "t"},
            {"var": "D", "op": "derive", "input": "T", "columns": [
                {"name": "h", "expr": "date_diff('hour', [a], [b])"},
                {"name": "late", "expr": "greatest([h] - 24, 0)"},
            ]},
            {"var": "F", "op": "filter", "input": "D", "condition": {"and": [
                {"column": "b", "cmp": "lt", "value": "2025-04-01"},
                {"or": [{"column": "c", "cmp": "is_null"}, {"column": "c", "cmp": "ge", "value": "2025-04-01"}]},
            ]}},
            {"var": "A", "op": "aggregate", "input": "F", "group_by": ["g"], "aggregates": [
                {"func": "percentile", "column": "late", "alias": "p90", "p": 0.9},
            ]},
        ],
        "output": "A",
    })
    sql = compile_dsl_to_sql(args, columns={"t": ["a", "b", "c", "g"]})
    assert "EXTRACT(EPOCH FROM (CAST(\"b\" AS timestamp) - CAST(\"a\" AS timestamp))) / 3600.0" in sql
    assert "GREATEST(((EXTRACT(EPOCH" in sql
    assert '("c" IS NULL OR "c" >= \'2025-04-01\')' in sql
    assert 'percentile_cont(0.9) WITHIN GROUP (ORDER BY "late") AS "p90"' in sql


def test_pipeline_allows_two_level_aggregation_but_not_late_filter():
    two_level = QueryArgs.model_validate({
        "steps": [
            {"var": "T", "op": "source", "table": "t"},
            {"var": "A", "op": "aggregate", "input": "T", "group_by": ["job", "chain"],
             "aggregates": [{"func": "count", "column": "*", "alias": "n"}]},
            {"var": "D", "op": "derive", "input": "A", "columns": [{"name": "retry", "expr": "if([n] > 1, 1, 0)"}]},
            {"var": "B", "op": "aggregate", "input": "D", "group_by": ["job"],
             "aggregates": [{"func": "avg", "column": "retry", "alias": "share"}]},
            {"var": "P", "op": "derive", "input": "B", "columns": [{"name": "pct", "expr": "[share] * 100"}]},
            {"var": "S", "op": "sort", "input": "P", "by": [{"column": "pct", "dir": "desc"}]},
        ],
        "output": "S",
    })
    assert validate_chart_pipeline(two_level, chart_type="table") == []
    late = two_level.model_copy(update={"steps": [*two_level.steps[:2], QueryArgs.model_validate({
        "steps": [{"var": "F", "op": "filter", "input": "A", "condition": {"column": "n", "cmp": "gt", "value": 1}}],
        "output": "F",
    }).steps[0]], "output": "F"})
    assert any("having" in e for e in validate_chart_pipeline(late, chart_type="table"))


def test_formula_columns_are_bracketed_and_fixed():
    catalog = Catalog(tables=[TableInfo(name="t", row_count=1, columns=[
        ColumnInfo(name="Цена, руб.", db_type="float", semantic_type=SemanticType.NUMERIC),
        ColumnInfo(name="Количество", db_type="int", semantic_type=SemanticType.NUMERIC),
    ])])
    data = {"steps": [
        {"var": "T", "op": "source", "table": "t"},
        {"var": "D", "op": "derive", "input": "T", "columns": [{"name": "Сумма", "expr": "Количество * Цена, руб. + [количество]"}]},
        {"var": "A", "op": "aggregate", "input": "D", "group_by": [], "aggregates": [{"func": "sum", "column": "сумма", "alias": "Итого"}]},
    ], "output": "A"}
    fixed = fix_query_args_columns(data, catalog)["steps"]
    assert fixed[1]["columns"][0]["expr"] == "Количество * [Цена, руб.] + [Количество]"
    assert fixed[2]["aggregates"][0]["column"] == "Сумма"


def test_calculation_text_is_readable():
    args = {"steps": [
        {"var": "T", "op": "source", "table": "Обращения"},
        {"var": "F", "op": "filter", "input": "T", "condition": {"and": [
            {"column": "Закрыто", "cmp": "ge", "value": "2025-03-01"},
            {"column": "Закрыто", "cmp": "lt", "value": "2025-04-01"},
        ]}},
        {"var": "A", "op": "aggregate", "input": "F", "group_by": ["Группа"], "aggregates": [
            {"func": "percentile", "column": "Часы", "alias": "P90", "p": 0.9},
            {"func": "count", "column": "*", "alias": "N"},
        ], "having": {"column": "N", "cmp": "ge", "value": 100}},
        {"var": "L", "op": "limit", "input": "A", "n": 1},
    ], "output": "L"}
    assert is_complex(args)
    lines = describe_calculation(args)
    assert lines[1] == "Отбираем строки: Закрыто ≥ 2025-03-01 и Закрыто < 2025-04-01."
    assert "90-й перцентиль «Часы» → «P90»" in lines[2] and "N ≥ 100" in lines[2]
    assert lines[-1] == "Берём первую строку."
