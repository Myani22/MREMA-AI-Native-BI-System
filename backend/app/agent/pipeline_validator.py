from __future__ import annotations

from .integrations.query_dsl import (
    AggregateStepArg,
    DeriveArg,
    FilterArg,
    JoinArg,
    LimitArg,
    QueryArgs,
    SortArg,
    SourceArg,
    StepArg,
)

ALLOWED_OPS = {"source", "filter", "derive", "join", "aggregate", "sort", "limit"}
FORBIDDEN_OPS = {"select", "distinct"}

_STAGE_ORDER = {
    "source": 0,
    "join": 1,
    "filter": 2,
    "derive": 2,
    "aggregate": 2,
    "sort": 3,
    "limit": 4,
}
_MAX_COUNT = {"source": 2, "join": 1, "filter": 3, "derive": 4, "aggregate": 3, "sort": 1, "limit": 1}


def _step_op(step: StepArg) -> str:
    op = getattr(step, "op", None)
    if isinstance(op, str):
        return op
    return step.__class__.__name__.replace("Step", "").replace("Arg", "").lower()


def validate_chart_pipeline(args: QueryArgs, *, chart_type: str) -> list[str]:
    """Возвращает список ошибок порядка и структуры шагов DSL. Пустой список = валидно."""
    errors: list[str] = []
    steps = args.steps
    if not steps:
        return ["пустой steps"]

    if not isinstance(steps[0], SourceArg):
        errors.append("первый шаг должен быть source")
        return errors

    op_counts = {op: 0 for op in ALLOWED_OPS}
    current_stage = 0
    source_vars: set[str] = set()
    agg_step: AggregateStepArg | None = None
    after_agg: set[str] = set()

    for i, step in enumerate(steps):
        op = _step_op(step)
        if op not in ALLOWED_OPS:
            errors.append(
                f"шаг {i}: {op} запрещён в chart-контексте (разрешено: {sorted(ALLOWED_OPS)})"
            )
            continue

        stage = _STAGE_ORDER[op]
        if stage < current_stage or (isinstance(step, FilterArg) and agg_step is not None):
            hint = " Для отбора групп после aggregate используй having." if isinstance(step, FilterArg) else ""
            errors.append(
                f"шаг {i}: нарушен порядок шагов ({op} не может идти после более поздней стадии).{hint}"
            )
        current_stage = max(current_stage, stage)

        op_counts[op] += 1
        if op_counts[op] > _MAX_COUNT[op]:
            errors.append(
                f"шаг {i}: слишком много шагов {op} (максимум {_MAX_COUNT[op]}; join требует ровно 2 source)"
            )

        if isinstance(step, SourceArg):
            source_vars.add(step.var)
        elif isinstance(step, JoinArg):
            if step.left not in source_vars or step.right not in source_vars:
                non_source = sorted(
                    {step.left, step.right} - source_vars
                )
                errors.append(
                    f"шаг {i}: join ссылается на {non_source}, но это не source. "
                    f"Порядок должен быть: source, source, join, затем filter и aggregate. "
                    f"ПЕРЕСТАВЬ join сразу после второго source. "
                    f"НЕ УДАЛЯЙ существующие filter/aggregate — они должны остаться ниже join."
                )
        elif isinstance(step, AggregateStepArg):
            agg_step = step
            after_agg = set()
        elif isinstance(step, DeriveArg) and agg_step is not None:
            after_agg |= {c.name for c in step.columns}

    if op_counts["join"] == 0 and op_counts["source"] > 1:
        errors.append("без join разрешён только один шаг source")
    if op_counts["join"] > 0 and op_counts["source"] != 2:
        errors.append("join требует ровно 2 source")

    if op_counts["sort"] > 0 and op_counts["aggregate"] == 0:
        errors.append("sort требует aggregate перед ним (или уберите sort)")

    if op_counts["sort"] > 0 and agg_step is not None:
        allowed_sort_cols = {*agg_step.group_by, *(a.alias for a in agg_step.aggregates), *after_agg}
        for step in steps:
            if isinstance(step, SortArg):
                for key in step.by:
                    if key.column not in allowed_sort_cols:
                        errors.append(
                            f"sort по колонке «{key.column}» запрещён: допустимы только выходы последнего aggregate и derive после него {sorted(allowed_sort_cols)}"
                        )

    if op_counts["limit"] > 0:
        if not isinstance(steps[-1], LimitArg):
            errors.append("limit должен быть последним шагом")
        elif steps[-1].n <= 0:
            errors.append("limit.n должен быть > 0")

    if args.output != steps[-1].var:
        errors.append(f"output={args.output} должен указывать на последний var={steps[-1].var}")

    return errors
