from __future__ import annotations
from typing import Any


def main() -> None:
    # БД + данные (например, полученные после выполнения QueryEngine)
    schema: dict[str, str] = {
        "country": "str",
        "revenue": "float",
        "orders_cnt": "int",
    }

    data: list[dict[str, Any]] = [
        {"country": "SE", "revenue": 250.0, "orders_cnt": 3},
        {"country": "DE", "revenue": 180.0, "orders_cnt": 2},
        {"country": "FI", "revenue": 95.0, "orders_cnt": 1},
    ]

    # Каталог
    from app.chart_system import (
        get_capabilities,
        get_chart_type,
        get_chart_types,
        get_engine,
        get_full,
    )

    print("1. Работа с каталогом")
    bar_spec = get_chart_type("bar")
    table_spec = get_chart_type("table")
    caps = get_capabilities()

    if bar_spec and table_spec:
        print(f"График '{bar_spec.name}': {bar_spec.description}")
        print(f"График '{table_spec.name}': {table_spec.description}")
    print(
        f"Всего типов графиков: {len(get_chart_types())}, "
        f"поддерживаемых фич: {len(caps.allowed)}, ограничений: {len(caps.unsupported)}"
    )

    prompt_doc = get_full(json_syntax=False)
    print("Фрагмент документа для LLM (первые 12 строк):")
    print("\n".join(prompt_doc.splitlines()[:12]))

    # Движок графиков
    from app.chart_system import ChartEngine

    engine = ChartEngine()

    print("\n2. Рабочая спецификация графика и её валидация.")
    valid_payload: dict[str, Any] = {
        "type": "bar",
        "title": "Выручка по странам",
        "x": {"name": "country", "description": "код страны клиента"},
        "y": {"name": "revenue", "description": "суммарная выручка по стране"},
        "sort": [{"by": "y", "dir": "desc"}, {"by": "x", "dir": "asc"}],
        "limit": 50,
    }

    chart = engine.from_dict(valid_payload)
    validation = engine.validate(chart, schema, row_count=len(data))
    print(f"Валидация успешна: {validation.ok}")
    print(f"Сериализация совпадает с исходником: {engine.to_dict(chart) == valid_payload}")

    if validation.ok:
        png_bytes = engine.matplotlib_out(chart, data)
        print(f"Сгенерирован PNG-файл, размер: {len(png_bytes)} байт")

    print("\n3. Спецификация с ошибками. Валидация.")
    invalid_payload: dict[str, Any] = {
        "type": "bar",
        "title": "   ",
        "x": {"name": "unknown_col", "description": "несуществующая колонка"},
        "y": {"name": "country", "description": "строковая колонка вместо числовой"},
        "sort": [{"by": "country", "dir": "up"}],
        "limit": 500,
    }

    bad_chart = engine.from_dict(invalid_payload)
    bad_validation = engine.validate(bad_chart, schema, row_count=300)
    print(
        f"Валидация успешна: {bad_validation.ok} "
        f"(найдено ошибок: {len(bad_validation.errors)})"
    )
    for err in bad_validation.errors:
        print(
            f"  [step={err.step_index}, var='{err.var}'] "
            f"reason={err.reason}, field={err.field}, value={err.value}, "
            f"valid_options={err.valid_options}"
        )


if __name__ == "__main__":
    main()