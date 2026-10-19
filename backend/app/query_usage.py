from __future__ import annotations
from typing import Any


def main() -> None:
    # БД
    schema: dict[str, dict[str, str]] = {
        "users": {"id": "int", "name": "str", "country": "str"},
        "orders": {"id": "int", "user_id": "int", "status": "str", "amount": "float"},
    }

    sources: dict[str, list[dict[str, Any]]] = {
        "users": [
            {"id": 1, "name": "Alice", "country": "SE"},
            {"id": 2, "name": "Bob", "country": "DE"},
            {"id": 3, "name": "Clara", "country": "SE"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "status": "paid", "amount": 120.0},
            {"id": 102, "user_id": 1, "status": "paid", "amount": 80.0},
            {"id": 103, "user_id": 2, "status": "cancelled", "amount": 300.0},
            {"id": 104, "user_id": 3, "status": "paid", "amount": 50.0},
        ],
    }

    #КАТАЛОГ
    from app.query_system import (
        get_capabilities,
        get_condition,
        get_full,
        get_step,
        get_conditions,
        get_steps,
        get_engine
    )
    print("1. Работа с каталогом")
    agg_spec = get_step("aggregate")
    between_spec = get_condition("between")
    caps = get_capabilities()

    if agg_spec and between_spec:
        print(f"Шаг '{agg_spec.name}': {agg_spec.description}")
        print(f"Условие '{between_spec.name}': {between_spec.description}")
    print(f"Поддерживаемых фич: {len(caps.allowed)}, ограничений: {len(caps.unsupported)}")

    prompt_doc = get_full(json_syntax=False)
    print("Фрагмент документа для LLM (первые 12 строк):")
    print("\n".join(prompt_doc.splitlines()[:12]))

    # Движок запросов
    from app.query_system import QueryEngine

    engine = QueryEngine()

    print("2. Рабочий запрос и его валидация.")
    valid_payload: dict[str, Any] = {
        "steps": [
            {"var": "U", "op": "source", "table": "users"},
            {"var": "O", "op": "source", "table": "orders"},
            {
                "var": "F_ORD",
                "op": "filter",
                "input": "O",
                "condition": {
                    "and": [
                        {"column": "status", "cmp": "eq", "value": "paid"},
                        {"column": "amount", "cmp": "ge", "value": 50.0},
                    ]
                },
            },
            {
                "var": "J",
                "op": "join",
                "left": "U",
                "right": "F_ORD",
                "on": [{"left_col": "id", "right_col": "user_id"}],
                "how": "inner",
            },
            {
                "var": "AGG",
                "op": "aggregate",
                "input": "J",
                "group_by": ["country"],
                "aggregates": [
                    {"func": "sum", "column": "amount", "alias": "revenue"},
                    {"func": "count", "column": "*", "alias": "orders_cnt"},
                ],
                "having": {"column": "revenue", "cmp": "gt", "value": 100.0},
            },
            {
                "var": "RES",
                "op": "sort",
                "input": "AGG",
                "by": [{"column": "revenue", "dir": "desc"}],
            },
        ],
        "output": "RES",
    }

    query = engine.from_dict(valid_payload)
    validation = engine.validate(query, schema)
    print(f"Валидация успешна: {validation.ok}")
    print(f"Выходная схема: {validation.output_schema}")

    if validation.ok:
        table = engine.execute(query, sources)
        print(f"Результат выполнения: {table.to_dicts()}")

    print("3. Запрос с ошибкой. Валидация.")
    invalid_payload: dict[str, Any] = {
        "steps": [
            {"var": "O", "op": "source", "table": "orders"},
            {
                "var": "F_BAD",
                "op": "filter",
                "input": "O",
                "condition": {
                    "and": [
                        {"column": "unknown_col", "cmp": "eq", "value": 1},
                        {"column": "amount", "cmp": "gt", "value": "not_a_float"},
                    ]
                },
            },
            {"var": "S_BAD", "op": "select", "input": "MISSING_VAR", "columns": ["id"]},
        ],
        "output": "UNKNOWN_OUT",
    }

    bad_query = engine.from_dict(invalid_payload)
    bad_validation = engine.validate(bad_query, schema)
    print(f"Валидация успешна: {bad_validation.ok} (найдено ошибок: {len(bad_validation.errors)})")
    for err in bad_validation.errors:
        print(
            f"  [step={err.step_index}, var='{err.var}'] "
            f"reason={err.reason}, field={err.field}, value={err.value}, "
            f"valid_options={err.valid_options}"
        )


if __name__ == "__main__":
    main()