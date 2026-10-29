from __future__ import annotations

import json
from typing import Any

from app.query_system import QueryEngine

from .catalog_refs import fix_query_args_columns
from .contracts.catalog import Catalog
from .contracts.response import ChartType
from .error_budget import ErrorBudget
from .integrations.dsl_exec import catalog_schema
from .integrations.query_dsl import QueryArgs, format_errors
from .llm.client import ChatMessage, LLMClient, system_msg, user_msg
from .llm.errors import StructuredOutputError
from .llm.structured import generate_structured
from .pipeline_validator import validate_chart_pipeline
from .prompts import DSL_EXAMPLES_BY_TYPE, DSL_FEEDBACK, DSL_SYSTEM
from .schema_linking import render_schema

_engine = QueryEngine()


def _validate_against_engine(args: QueryArgs, catalog: Catalog, tables: list[str]) -> list[str]:
    full_schema = catalog_schema(catalog)
    if tables:
        scoped = {t: full_schema[t] for t in tables if t in full_schema}
        schema = scoped or full_schema
    else:
        schema = full_schema
    query = _engine.from_dict(args.to_engine())
    validation = _engine.validate(query, schema)
    if validation.ok:
        return []
    return [format_errors(validation.errors)]


def _messages(
    question: str,
    chart_type: ChartType,
    title: str,
    purpose: str,
    catalog: Catalog,
    tables: list[str],
    feedback: tuple[QueryArgs | None, str] | None,
    *,
    original_dsl: dict[str, Any] | None = None,
) -> list[ChatMessage]:
    type_key = "card" if chart_type is ChartType.KPI else chart_type.value
    example_dict = DSL_EXAMPLES_BY_TYPE.get(type_key, DSL_EXAMPLES_BY_TYPE["bar"])
    schema_text = render_schema(catalog, tables)
    prompt = DSL_SYSTEM.format(
        chart_type=type_key,
        question=question,
        purpose=purpose or title,
        schema=schema_text,
        example_json=json.dumps(example_dict, ensure_ascii=False),
    )
    user_content = f"Заголовок графика: «{title}».\nВопрос: {question}\nНазначение: {purpose or title}"
    if original_dsl is not None:
        user_content += (
            "\n\nПредыдущая версия DSL (её нужно изменить, а НЕ переписать заново):\n"
            f"```json\n{json.dumps(original_dsl, ensure_ascii=False, indent=2)}\n```\n\n"
            "ЖЁСТКИЕ ПРАВИЛА ПРИ РЕДАКТИРОВАНИИ:\n"
            "1. СОХРАНИ все шаги source, join, filter, aggregate из предыдущей версии, "
            "если пользователь явно не попросил их снять или изменить.\n"
            "2. Если меняется сортировка/лимит/тип графика — это НЕ повод удалять filter.\n"
            "3. Если меняется условие фильтрации — замени существующий filter, "
            "не удаляй его целиком.\n"
            "4. Если пользователь просит удалить фильтр — только тогда не включай filter.\n"
        )
    if feedback is not None:
        prev_args, err_text = feedback
        prev_json = (
            json.dumps(prev_args.to_engine(), ensure_ascii=False)
            if prev_args is not None
            else "(невалидный JSON)"
        )
        user_content += DSL_FEEDBACK.format(prev_dsl=prev_json, error=err_text)
    return [system_msg(prompt), user_msg(user_content)]


async def build_dsl(
    llm: LLMClient,
    *,
    question: str,
    chart_type: ChartType,
    title: str,
    purpose: str = "",
    catalog: Catalog,
    tables: list[str],
    budget: ErrorBudget,
    max_attempts_per_step: int = 2,
    original_dsl: dict[str, Any] | None = None, 
) -> tuple[QueryArgs | None, list[str]]:
    """LLM #3 + repair-loop. Возвращает (QueryArgs | None, warnings)."""
    warnings: list[str] = []
    feedback: tuple[QueryArgs | None, str] | None = None
    total_tries = max_attempts_per_step + 1
    seen_hashes = set()

    for attempt in range(total_tries):
        if attempt > 0 and not budget.consume(f"dsl:{title}"):
            warnings.append(f"{title}: бюджет исчерпан")
            return None, warnings

        messages = _messages(question, chart_type, title, purpose, catalog, tables, feedback, original_dsl=original_dsl)
        prompt_text = "\n".join(m.content for m in messages)

        try:
            args = await generate_structured(
                llm,
                messages,
                QueryArgs,
                schema_name="dsl_query",
                max_attempts=1,
                preprocess=lambda d: fix_query_args_columns(d, catalog),
                validate=lambda a: validate_chart_pipeline(a, chart_type=chart_type.value),
            )
            completion_text = args.model_dump_json()
        except StructuredOutputError as e:
            completion_text = e.raw
            
            h = hash((prompt_text, completion_text))
            if h in seen_hashes:
                warnings.append(f"{title}: модель повторила тот же ответ")
                return None, warnings
            seen_hashes.add(h)

            err_msg = e.errors[0] if e.errors else "невалидный JSON"
            warnings.append(f"{title}: невалидный DSL ({err_msg})")
            feedback = (None, err_msg)
            continue

        h = hash((prompt_text, completion_text))
        if h in seen_hashes:
            warnings.append(f"{title}: модель повторила тот же ответ")
            return None, warnings
        seen_hashes.add(h)

        dsl_errors = _validate_against_engine(args, catalog, tables)
        if not dsl_errors:
            return args, warnings

        warnings.append(f"{title}: {dsl_errors[0]}")
        feedback = (args, dsl_errors[0])

    return None, warnings