from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.chart_system import (
    ChartEngine,
    get_capabilities,
    get_chart_type,
    get_chart_types,
    get_engine,
    get_full,
)
from app.chart_system.base import SortSpec, apply_multi_level_sort


@pytest.fixture
def engine() -> ChartEngine:
    return ChartEngine()


@pytest.fixture
def schema() -> dict[str, str]:
    return {
        "id": "int",
        "status": "str",
        "orders_count": "int",
        "revenue": "float",
        "created_at": "date",
        "active": "bool",
    }


ROUNDTRIP_CASES: list[dict[str, Any]] = [
    {
        "type": "bar",
        "title": "Заказы по статусам",
        "x": {"name": "status", "description": "статус"},
        "y": {"name": "orders_count", "description": "количество"},
        "sort": [{"by": "y", "dir": "desc"}, {"by": "x", "dir": "asc"}],
        "limit": 50,
    },
    {
        "type": "bar",
        "title": "Заказы по активности (bool по оси X, без sort/limit)",
        "x": {"name": "active", "description": "флаг активности"},
        "y": {"name": "revenue", "description": "выручка"},
    },
    {
        "type": "line",
        "title": "Динамика по датам",
        "x": {"name": "created_at", "description": "дата"},
        "y": {"name": "revenue", "description": "выручка"},
        "limit": 1000,
    },
    {
        "type": "line",
        "title": "Динамика по числовой оси X",
        "x": {"name": "id", "description": "номер шага"},
        "y": {"name": "orders_count", "description": "заказы"},
    },
    {
        "type": "scatter",
        "title": "Связь заказов и выручки",
        "x": {"name": "orders_count", "description": "заказы"},
        "y": {"name": "revenue", "description": "выручка"},
        "limit": 500,
    },
    {
        "type": "card",
        "title": "Общая выручка",
        "value": {"name": "revenue", "description": "итого"},
    },
    {
        "type": "table",
        "title": "Полная выгрузка",
        "columns": [
            {"name": "id", "description": "ID"},
            {"name": "status", "description": "Статус"},
            {"name": "revenue", "description": "Выручка"},
            {"name": "created_at", "description": "Дата"},
            {"name": "active", "description": "Активен"},
        ],
        "sort": [{"by": "revenue", "dir": "desc"}, {"by": "id", "dir": "asc"}],
        "limit": 100,
    },
]


@pytest.mark.parametrize("raw", ROUNDTRIP_CASES, ids=lambda r: r["title"])
def test_roundtrip_and_validation_ok(engine, schema, raw):
    spec = engine.from_dict(raw)
    assert engine.to_dict(spec) == raw
    val = engine.validate(spec, schema, row_count=40)
    assert val.ok
    assert val.errors == []
    assert val.output_schema is None


def test_multi_level_sort_stability_dates_and_nulls():
    rows: list[dict[str, Any]] = [
        {"group": "B", "score": 10,   "dt": "2026-01-05"},
        {"group": "A", "score": None, "dt": "2026-01-01"},
        {"group": "A", "score": 20,   "dt": date(2026, 1, 3)},
        {"group": "A", "score": 20,   "dt": "2026-01-02"},
        {"group": "B", "score": 20,   "dt": "2026-01-01"},
    ]
    sort_specs = (
        SortSpec(by="score", dir="desc"),
        SortSpec(by="group", dir="asc"),
        SortSpec(by="dt",    dir="asc"),
    )
    assert apply_multi_level_sort(rows, sort_specs) == [
        {"group": "A", "score": 20,   "dt": "2026-01-02"},
        {"group": "A", "score": 20,   "dt": date(2026, 1, 3)},
        {"group": "B", "score": 20,   "dt": "2026-01-01"},
        {"group": "B", "score": 10,   "dt": "2026-01-05"},
        {"group": "A", "score": None, "dt": "2026-01-01"},
    ]


def test_multi_level_sort_key_mapping():
    rows = [
        {"group": "B", "score": 20},
        {"group": "A", "score": 20},
        {"group": "A", "score": None},
        {"group": "B", "score": 10},
    ]
    mapped = apply_multi_level_sort(
        rows,
        (SortSpec(by="y", dir="asc"), SortSpec(by="x", dir="desc")),
        key_mapping={"x": "group", "y": "score"},
    )
    assert mapped[0]["score"] == 10
    assert mapped[1]["group"] == "B"
    assert mapped[-1]["score"] is None


def test_bar_multi_level_sort_priority():
    rows = [
        {"status": "b", "orders_count": 10},
        {"status": "a", "orders_count": 10},
        {"status": "c", "orders_count": 20},
    ]
    sorted_rows = apply_multi_level_sort(
        rows,
        (SortSpec(by="y", dir="desc"), SortSpec(by="x", dir="asc")),
        key_mapping={"x": "status", "y": "orders_count"},
    )
    assert [r["status"] for r in sorted_rows] == ["c", "a", "b"]


def test_catalog_engine_and_capabilities():
    eng = get_engine()
    caps = get_capabilities()
    assert eng.description
    assert len(eng.rules) >= 5
    assert caps.allowed and caps.unsupported


def test_catalog_chart_types_registered():
    assert [c.name for c in get_chart_types()] == ["bar", "line", "scatter", "card", "table"]
    assert get_chart_type("pie") is None
    for name in ["bar", "line", "scatter", "card", "table"]:
        spec = get_chart_type(name)
        assert spec is not None
        assert spec.name == name
        assert spec.fields
        assert spec.when_to_use


def test_catalog_prompt_formats():
    pseudo = get_full(json_syntax=False)
    as_json = get_full(json_syntax=True)
    assert "bar: x=status, y=orders_count, sort=y desc, limit=50" in pseudo
    assert "card: value=total_revenue" in pseudo
    assert "```json" in as_json
    assert '"type": "scatter"' in as_json


def test_bar_accepts_any_row_count(engine, schema):
    payload = {
        "type": "bar",
        "title": "Тест границ bar",
        "x": {"name": "status", "description": ""},
        "y": {"name": "orders_count", "description": ""},
        "limit": 200,
    }
    spec = engine.from_dict(payload)
    for rc in (200, 201, 5000):
        assert engine.validate(spec, schema, row_count=rc).ok


def test_bar_rejects_negative_row_count(engine, schema):
    payload = {
        "type": "bar",
        "title": "Тест границ bar",
        "x": {"name": "status", "description": ""},
        "y": {"name": "orders_count", "description": ""},
    }
    val = engine.validate(engine.from_dict(payload), schema, row_count=-5)
    assert not val.ok
    assert any(e.field == "row_count" and e.reason == "invalid_parameter" for e in val.errors)


def test_bar_validation_errors(engine, schema):
    bad = engine.from_dict({
        "type": "bar",
        "title": "Ошибки параметров bar",
        "x": {"name": "created_at", "description": "date запрещён"},
        "y": {"name": "status", "description": "str запрещён"},
        "sort": [{"by": "status", "dir": "asc"}, {"by": "y", "dir": "sideways"}],
        "limit": 201,
    })
    val = engine.validate(bad, schema)
    assert not val.ok
    err_set = {(e.field, e.reason) for e in val.errors}
    assert {
        ("x.name", "type_mismatch"),
        ("y.name", "type_mismatch"),
        ("sort[0].by", "invalid_parameter"),
        ("sort[1].dir", "invalid_parameter"),
        ("limit", "invalid_parameter"),
    }.issubset(err_set)


def test_line_constraints(engine, schema):
    bad = engine.from_dict({
        "type": "line",
        "title": "Невалидный line",
        "x": {"name": "status", "description": "str запрещён для line.x"},
        "y": {"name": "active", "description": "bool запрещён для line.y"},
        "sort": [{"by": "x", "dir": "asc"}],
        "limit": 5001,
    })
    val = engine.validate(bad, schema)
    err_set = {(e.field, e.reason) for e in val.errors}
    assert {
        ("x.name", "type_mismatch"),
        ("y.name", "type_mismatch"),
        ("sort", "unsupported_operation"),
        ("limit", "invalid_parameter"),
    }.issubset(err_set)


def test_scatter_constraints(engine, schema):
    bad = engine.from_dict({
        "type": "scatter",
        "title": "Невалидный scatter",
        "x": {"name": "created_at", "description": "date запрещён для scatter.x"},
        "y": {"name": "revenue", "description": ""},
        "sort": [],
        "limit": 0,
    })
    val = engine.validate(bad, schema)
    err_set = {(e.field, e.reason) for e in val.errors}
    assert {
        ("x.name", "type_mismatch"),
        ("sort", "unsupported_operation"),
        ("limit", "invalid_parameter"),
    }.issubset(err_set)


def test_card_constraints(engine, schema):
    bad = engine.from_dict({
        "type": "card",
        "title": "Невалидный card",
        "value": {"name": "created_at", "description": "date запрещён для card"},
        "sort": [{"by": "value", "dir": "desc"}],
        "limit": 1,
    })
    val = engine.validate(bad, schema)
    err_set = {(e.field, e.reason) for e in val.errors}
    assert {
        ("value.name", "type_mismatch"),
        ("sort", "unsupported_operation"),
        ("limit", "unsupported_operation"),
    }.issubset(err_set)


def test_table_rejects_empty_columns(engine, schema):
    spec = engine.from_dict({"type": "table", "title": "Пустые колонки", "columns": []})
    assert not engine.validate(spec, schema).ok


def test_table_sort_scope_and_limit(engine, schema):
    spec = engine.from_dict({
        "type": "table",
        "title": "Сортировка по колонке вне списка columns и по оси x",
        "columns": [
            {"name": "id", "description": ""},
            {"name": "missing_col", "description": ""},
        ],
        "sort": [{"by": "revenue", "dir": "desc"}, {"by": "x", "dir": "asc"}],
        "limit": 10001,
    })
    val = engine.validate(spec, schema)
    assert not val.ok
    err_set = {(e.field, e.reason) for e in val.errors}
    assert {
        ("columns[1].name", "column_not_found"),
        ("sort[0].by", "invalid_parameter"),
        ("sort[1].by", "invalid_parameter"),
        ("limit", "invalid_parameter"),
    }.issubset(err_set)


def test_non_dict_payload(engine, schema):
    spec = engine.from_dict(["not", "a", "dict"])
    assert not engine.validate(spec, schema).ok


def test_unknown_chart_type(engine, schema):
    spec = engine.from_dict({"type": "pie", "title": "Круговая диаграмма"})
    val = engine.validate(spec, schema)
    assert not val.ok
    assert any(
        e.field == "type"
        and e.reason == "unsupported_operation"
        and e.valid_options == ["bar", "card", "line", "scatter", "table"]
        for e in val.errors
    )


def test_malformed_bar_and_bad_schema(engine):
    malformed = engine.from_dict({
        "type": "bar",
        "title": "   ",
        "x": "status_as_string_instead_of_object",
        "y": {"name": "   ", "description": ""},
        "sort": "not_a_list",
        "limit": True,
    })
    val = engine.validate(malformed, {"status": "str", "bad_col": "decimal"}, row_count=-5)
    assert not val.ok
    err_set = {(e.field, e.reason) for e in val.errors}
    assert {
        ("schema.bad_col", "invalid_parameter"),
        ("row_count", "invalid_parameter"),
        ("title", "invalid_parameter"),
        ("x", "invalid_parameter"),
        ("y.name", "invalid_parameter"),
        ("sort", "invalid_parameter"),
        ("limit", "invalid_parameter"),
    }.issubset(err_set)


def test_unsupported_schema_type(engine):
    spec = engine.from_dict({
        "type": "card", "title": "X",
        "value": {"name": "revenue", "description": ""},
    })
    val = engine.validate(spec, {"revenue": "decimal"})
    assert not val.ok
    assert any(e.reason == "invalid_parameter" and e.field == "schema.revenue" for e in val.errors)


def test_empty_data_renders_placeholder(engine, schema):
    spec = engine.from_dict({
        "type": "bar", "title": "Пусто",
        "x": {"name": "status", "description": ""},
        "y": {"name": "orders_count", "description": ""},
    })
    assert engine.validate(spec, schema).ok
    png = engine.matplotlib_out(spec, [])
    assert png.startswith(b"\x89PNG")
    assert len(png) > 100


def test_line_sorts_x_automatically_and_renders(engine, schema):
    spec = engine.from_dict({
        "type": "line", "title": "По датам",
        "x": {"name": "created_at", "description": ""},
        "y": {"name": "revenue", "description": ""},
    })
    assert engine.validate(spec, schema).ok
    rows = [
        {"created_at": "2026-01-03", "revenue": 30.0},
        {"created_at": "2026-01-01", "revenue": 10.0},
        {"created_at": "2026-01-02", "revenue": 20.0},
    ]
    assert engine.matplotlib_out(spec, rows).startswith(b"\x89PNG")


def test_bar_renders(engine, schema):
    spec = engine.from_dict({
        "type": "bar", "title": "Бар",
        "x": {"name": "status", "description": ""},
        "y": {"name": "orders_count", "description": ""},
        "sort": [{"by": "y", "dir": "desc"}],
    })
    assert engine.validate(spec, schema).ok
    rows = [
        {"status": "a", "orders_count": 10},
        {"status": "b", "orders_count": 20},
    ]
    assert engine.matplotlib_out(spec, rows).startswith(b"\x89PNG")