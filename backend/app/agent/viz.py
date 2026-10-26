from __future__ import annotations

from collections.abc import Collection

from .contracts.catalog import TIME_TYPES, SemanticType
from .contracts.response import CHART_LABELS, ChartSpec, ChartType, ResultTable, TableColumn
from .contracts.tools import ChartHint

MAX_PIE_SLICES = 8


def _distinct(table: ResultTable, column: str) -> int:
    idx = [c.name for c in table.columns].index(column)
    return len({row[idx] for row in table.rows})


def _non_negative(table: ResultTable, column: str) -> bool:
    idx = [c.name for c in table.columns].index(column)
    return all(isinstance(r[idx], int | float) and r[idx] >= 0 for r in table.rows)


def _label(c: TableColumn | None) -> str | None:
    return (c.label or c.name) if c else None


def candidates(
    table: ResultTable, hint: ChartHint | None = None, allowed: Collection[ChartType] | None = None
) -> list[ChartSpec]:
    names = {c.name for c in table.columns}
    num = [c.name for c in table.columns if c.semantic_type is SemanticType.NUMERIC]
    time = [c.name for c in table.columns if c.semantic_type in TIME_TYPES]
    cat = [c.name for c in table.columns if c.semantic_type not in TIME_TYPES and c.semantic_type is not SemanticType.NUMERIC]
    if hint:
        hint_y = [y for y in hint.y if y in names]
        num = hint_y + [n for n in num if n not in hint_y] if hint_y else num
    rows = len(table.rows)
    if rows == 0:
        return []
    specs: list[ChartSpec] = []

    def add(chart_type: ChartType, x: str | None, y: list[str], series: str | None = None) -> None:
        specs.append(ChartSpec(chart_type=chart_type, x=x, y=y, series=series))

    x_hint = hint.x if hint and hint.x in names else None
    if rows == 1 and num and not time:
        add(ChartType.KPI, None, num[:1])
    elif time and num:
        x = x_hint if x_hint in time else time[0]
        series = cat[0] if cat and _distinct(table, cat[0]) <= 10 else None
        add(ChartType.LINE, x, num[:3], series)
        add(ChartType.AREA, x, num[:1], series)
        add(ChartType.BAR, x, num[:1], series)
    elif len(cat) >= 2 and num:
        add(ChartType.HEATMAP, cat[0], num[:1], cat[1])
        add(ChartType.BAR, cat[0], num[:1], cat[1])
    elif cat and num:
        x = x_hint if x_hint in cat else cat[0]
        add(ChartType.BAR, x, num[:3])
        if _distinct(table, x) <= MAX_PIE_SLICES and _non_negative(table, num[0]):
            add(ChartType.PIE, x, num[:1])
        add(ChartType.LINE, x, num[:1])
    elif len(num) >= 2:
        add(ChartType.SCATTER, num[0], [num[1]])
        add(ChartType.LINE, num[0], [num[1]])
    elif len(num) == 1 and rows > 1:
        add(ChartType.HISTOGRAM, num[0], [])
    add(ChartType.TABLE, None, [])

    if allowed is not None:
        specs = [s for s in specs if s.chart_type in allowed or s.chart_type is ChartType.TABLE]
    if hint and hint.preferred:
        specs.sort(key=lambda s: s.chart_type is not hint.preferred)
    for s in specs:
        s.x_label = _label(table.column(s.x)) if s.x else None
        s.y_label = _label(table.column(s.y[0])) if s.y else None
    return specs


def choose_chart(
    table: ResultTable,
    *,
    hint: ChartHint | None = None,
    requested: ChartType | None = None,
    title: str = "",
    allowed: Collection[ChartType] | None = None,
) -> tuple[ChartSpec | None, str | None]:
    options = candidates(table, hint, allowed)
    if not options:
        return None, None
    note = None
    if requested is not None:
        match = next((s for s in options if s.chart_type is requested), None)
        if match is None:
            match = options[0]
            unsupported = allowed is not None and requested not in allowed and requested is not ChartType.TABLE
            reason = "пока не поддерживается" if unsupported else "не подходит для этих данных"
            note = (
                f"{CHART_LABELS[requested].capitalize()} {reason}, "
                f"поэтому показываю: {CHART_LABELS[match.chart_type]}."
            )
        chosen = match
    else:
        chosen = options[0]
        if chosen.chart_type is ChartType.TABLE:
            return None, None
    return chosen.model_copy(update={"title": title}), note


def alternative_chart(
    table: ResultTable,
    current: ChartType | None,
    hint: ChartHint | None = None,
    allowed: Collection[ChartType] | None = None,
) -> ChartSpec | None:
    options = candidates(table, hint, allowed)
    if not options:
        return None
    types = [s.chart_type for s in options]
    if current in types:
        return options[(types.index(current) + 1) % len(options)]
    return options[0]
