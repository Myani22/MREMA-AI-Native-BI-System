from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.chart_system import ChartEngine
from app.chart_system.base import apply_multi_level_sort, parse_sort_specs


OUT_DIR = Path("chart_plots")

SCHEMA: dict[str, str] = {
    "country": "str",
    "revenue": "float",
    "orders_cnt": "int",
    "order_date": "date",
    "customer_id": "int",
    "total_spent": "float",
    "orders_count": "int",
    "order_id": "int",
    "status": "str",
    "amount": "float",
}


BAR_DATA: list[dict[str, Any]] = [
    {"country": "SE", "revenue": 250.0, "orders_cnt": 3},
    {"country": "DE", "revenue": 180.0, "orders_cnt": 2},
    {"country": "FI", "revenue": 95.0, "orders_cnt": 1},
    {"country": "NO", "revenue": 310.0, "orders_cnt": 4},
    {"country": "DK", "revenue": 140.0, "orders_cnt": 2},
    {"country": "EE", "revenue": 60.0, "orders_cnt": 1},
]

LINE_DATA: list[dict[str, Any]] = [
    {"order_date": "2026-01-03", "revenue": 120.0},
    {"order_date": "2026-01-01", "revenue": 80.0},
    {"order_date": "2026-01-05", "revenue": 260.0},
    {"order_date": "2026-01-02", "revenue": 95.0},
    {"order_date": "2026-01-04", "revenue": 210.0},
]

SCATTER_DATA: list[dict[str, Any]] = [
    {"customer_id": 1, "orders_count": 2, "total_spent": 200.0},
    {"customer_id": 2, "orders_count": 5, "total_spent": 640.0},
    {"customer_id": 3, "orders_count": 1, "total_spent": 50.0},
    {"customer_id": 4, "orders_count": 8, "total_spent": 1120.0},
    {"customer_id": 5, "orders_count": 3, "total_spent": 310.0},
]

CARD_DATA: list[dict[str, Any]] = [
    {"revenue": 1035.0},
]

TABLE_DATA: list[dict[str, Any]] = [
    {"order_id": 101, "status": "paid", "amount": 120.0, "country": "SE"},
    {"order_id": 103, "status": "cancelled", "amount": 300.0, "country": "DE"},
    {"order_id": 102, "status": "paid", "amount": 80.0, "country": "SE"},
    {"order_id": 105, "status": "pending", "amount": 220.0, "country": "NO"},
    {"order_id": 104, "status": "paid", "amount": 50.0, "country": "FI"},
    {"order_id": 106, "status": "paid", "amount": 410.0, "country": "NO"},
]


def _print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"  {title}:")
    if not rows:
        print("    <no rows>")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    
    header = "    " + " | ".join(str(c).ljust(widths[c]) for c in cols)
    print(header)
    print("    " + "-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        line = "    " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)
    print()


def _run_case(
    engine: ChartEngine,
    index: int,
    total: int,
    label: str,
    payload: dict[str, Any],
    data: list[dict[str, Any]],
    filename: str,
    expect_ok: bool = True,
) -> bool:
    print("\n")
    print(f"[{index}/{total}] {label.upper()}")
    print()
    print(f"  Spec: {payload}\n")

    _print_table(data, "Input Data (rows)")

    spec = engine.from_dict(payload)
    
    if spec.name in ("bar", "table") and hasattr(spec, "sort") and spec.sort:
        key_mapping = None
        if spec.name == "bar":
            key_mapping = {"x": spec.x.name, "y": spec.y.name}
        
        sorted_data = apply_multi_level_sort(data, spec.sort, key_mapping=key_mapping)

        eff_limit = getattr(spec, "limit", None) or (50 if spec.name == "bar" else 100)
        sorted_data = sorted_data[:eff_limit]
        
        if sorted_data != data:
            _print_table(sorted_data, "Data AFTER Engine Sort (what will be plotted)")

    validation = engine.validate(spec, SCHEMA, row_count=len(data))
    
    if not validation.ok:
        print(f"  Validation status: FAILED ({len(validation.errors)} errors)")
        for err in validation.errors:
            print(
                f"    [step={err.step_index}, var='{err.var}'] reason={err.reason}, "
                f"field={err.field}, value={err.value}, valid_options={err.valid_options}"
            )
        if expect_ok:
            print("\n  -> UNEXPECTED validation failure")
            return False
        print("\n  -> Expected failure. Render skipped.")
        return True

    print("  Validation status: OK")
    png = engine.matplotlib_out(spec, data)
    out_path = OUT_DIR / filename
    out_path.write_bytes(png)
    print(f"\n  -> PNG saved: {out_path.resolve()} ({len(png)} bytes)\n")
    return True


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    engine = ChartEngine()

    cases = [
        (
            "Bar / No sort (Original order)", 
            {
                "type": "bar",
                "title": "Выручка по странам (без сортировки)",
                "x": {"name": "country", "description": ""},
                "y": {"name": "revenue", "description": ""},
                "limit": 50,
            },
            BAR_DATA,
            "01_bar_no_sort.png",
            True,
        ),
        (
            "Bar / Sort y Ascending",
            {
                "type": "bar",
                "title": "Выручка по странам (возрастание)",
                "x": {"name": "country", "description": ""},
                "y": {"name": "revenue", "description": ""},
                "sort": [{"by": "y", "dir": "asc"}],
                "limit": 50,
            },
            BAR_DATA,
            "02_bar_sort_asc.png",
            True,
        ),
        (
            "Bar / Sort y Descending (Multi-level: y desc, then x asc)",
            {
                "type": "bar",
                "title": "Выручка по странам (убывание + доп. сортировка)",
                "x": {"name": "country", "description": ""},
                "y": {"name": "orders_cnt", "description": ""},
                "sort": [{"by": "y", "dir": "desc"}, {"by": "x", "dir": "asc"}],
                "limit": 50,
            },
            BAR_DATA,
            "03_bar_sort_desc_multi.png",
            True,
        ),
        (
            "Line / Unsorted input (Engine must sort X automatically)",
            {
                "type": "line",
                "title": "Выручка по дням (сортировка по X)",
                "x": {"name": "order_date", "description": "дата"},
                "y": {"name": "revenue", "description": "выручка"},
                "limit": 1000,
            },
            LINE_DATA,
            "04_line_unsorted_input.png",
            True,
        ),
        (
            "Scatter / Orders vs Spend",
            {
                "type": "scatter",
                "title": "Число заказов vs суммарные траты",
                "x": {"name": "orders_count", "description": ""},
                "y": {"name": "total_spent", "description": ""},
                "limit": 1000,
            },
            SCATTER_DATA,
            "05_scatter.png",
            True,
        ),
        (
            "Card / Single KPI",
            {
                "type": "card",
                "title": "Общая выручка",
                "value": {"name": "revenue", "description": "сумма"},
            },
            CARD_DATA,
            "06_card.png",
            True,
        ),
        (
            "Table / No sort",
            {
                "type": "table",
                "title": "Заказы (без сортировки)",
                "columns": [
                    {"name": "order_id", "description": "ID"},
                    {"name": "status", "description": "статус"},
                    {"name": "amount", "description": "сумма"},
                    {"name": "country", "description": "страна"},
                ],
                "limit": 100,
            },
            TABLE_DATA,
            "07_table_no_sort.png",
            True,
        ),
        (
            "Table / Sort amount Ascending",
            {
                "type": "table",
                "title": "Заказы (сумма по возрастанию)",
                "columns": [
                    {"name": "order_id", "description": "ID"},
                    {"name": "status", "description": "статус"},
                    {"name": "amount", "description": "сумма"},
                    {"name": "country", "description": "страна"},
                ],
                "sort": [{"by": "amount", "dir": "asc"}],
                "limit": 100,
            },
            TABLE_DATA,
            "08_table_sort_asc.png",
            True,
        ),
        (
            "Table / Sort amount Descending",
            {
                "type": "table",
                "title": "Заказы (сумма по убыванию)",
                "columns": [
                    {"name": "order_id", "description": "ID"},
                    {"name": "status", "description": "статус"},
                    {"name": "amount", "description": "сумма"},
                    {"name": "country", "description": "страна"},
                ],
                "sort": [{"by": "amount", "dir": "desc"}],
                "limit": 100,
            },
            TABLE_DATA,
            "09_table_sort_desc.png",
            True,
        ),
        (
            "Bar / Empty data -> placeholder",
            {
                "type": "bar",
                "title": "Пустой bar",
                "x": {"name": "country", "description": ""},
                "y": {"name": "revenue", "description": ""},
            },
            [],
            "10_bar_empty.png",
            True,
        ),
        (
            "Bar / Invalid spec (str in y) -> validation error, no PNG",
            {
                "type": "bar",
                "title": "Некорректный bar",
                "x": {"name": "country", "description": ""},
                "y": {"name": "status", "description": "str вместо числа"},
                "sort": [{"by": "country", "dir": "asc"}],
            },
            BAR_DATA,
            "11_bar_invalid.png",
            False,
        ),
    ]

    passed = 0
    for idx, (label, payload, data, filename, expect_ok) in enumerate(cases, start=1):
        ok = _run_case(
            engine=engine,
            index=idx,
            total=len(cases),
            label=label,
            payload=payload,
            data=data,
            filename=filename,
            expect_ok=expect_ok,
        )
        if ok:
            passed += 1

    print("\n\n")
    print(f"SUMMARY: Passed {passed}/{len(cases)} tests")
    print(f"PNGs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()