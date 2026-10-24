from __future__ import annotations

import re

from .contracts.catalog import TIME_TYPES, Catalog, ColumnInfo, SemanticType, TableInfo

_WORD_RE = re.compile(r"\w+", re.U)


def _stems(text: str | None) -> set[str]:
    if not text:
        return set()
    return {w[:5] for w in _WORD_RE.findall(text.lower()) if len(w) > 2}


def _table_score(question: set[str], table: TableInfo) -> float:
    score = 3 * len(question & (_stems(table.name) | _stems(table.display_name) | _stems(table.description)))
    for c in table.columns:
        score += 2 * len(question & (_stems(c.label) | _stems(c.description)))
        values = " ".join(v.value for v in c.top_values) + " " + " ".join(c.sample_values)
        score += len(question & _stems(values))
    return score


def select_tables(question: str, catalog: Catalog, *, focus: list[str] | None = None, max_tables: int = 3) -> list[str]:
    known = {t.name for t in catalog.tables}
    focus = [t for t in (focus or []) if t in known]
    if len(catalog.tables) <= max_tables:
        return [t.name for t in catalog.tables]
    q = _stems(question)
    scored = sorted(((_table_score(q, t), t.name) for t in catalog.tables), reverse=True)
    picked = [name for score, name in scored if score > 0][:max_tables]
    for name in focus:
        if name not in picked:
            picked.append(name)
    if not picked:
        picked = [t.name for t in sorted(catalog.tables, key=lambda t: t.row_count, reverse=True)[:max_tables]]
    return picked


def _column_line(c: ColumnInfo) -> str:
    parts = [c.semantic_type.value]
    if c.unit:
        parts.append(c.unit)
    if c.semantic_type in TIME_TYPES or c.semantic_type is SemanticType.NUMERIC:
        if c.min_value is not None and c.max_value is not None:
            parts.append(f"от {c.min_value} до {c.max_value}")
    elif c.top_values:
        parts.append("напр.: " + ", ".join(v.value for v in c.top_values[:4]))
    elif c.sample_values:
        parts.append("напр.: " + ", ".join(c.sample_values[:3]))
    line = f'  - «{c.name}» ({"; ".join(parts)})'
    if c.display_name and c.display_name != c.name:
        line += f" — в файле «{c.display_name}»"
    if c.description:
        line += f" — {c.description}"
    return line


def render_schema(catalog: Catalog, tables: list[str], *, max_columns: int = 60) -> str:
    lines: list[str] = []
    budget = max_columns
    for name in tables:
        t = catalog.table(name)
        if t is None:
            continue
        header = f'Таблица «{t.name}»'
        if t.display_name and t.display_name != t.name:
            header += f" («{t.display_name}»)"
        header += f", {t.row_count} строк"
        if t.period_start and t.period_end:
            header += f", период {t.period_start} — {t.period_end}"
        lines.append(header + (f". {t.description}" if t.description else ""))
        shown = t.columns[: max(budget, 0)]
        lines.extend(_column_line(c) for c in shown)
        if len(shown) < len(t.columns):
            lines.append(f"  ... ещё {len(t.columns) - len(shown)} колонок")
        budget -= len(shown)
    rels = [r for r in catalog.relations if r.left_table in tables and r.right_table in tables]
    if rels:
        lines.append("Связи:")
        lines.extend(
            f'  - «{r.left_table}».«{r.left_column}» = «{r.right_table}».«{r.right_column}» ({r.cardinality})'
            for r in rels
        )
    return "\n".join(lines)


def render_tables_brief(catalog: Catalog) -> str:
    if not catalog.tables:
        return "нет загруженных таблиц"
    return "; ".join(f"{t.name} ({t.label}, {t.row_count} строк)" for t in catalog.tables)
