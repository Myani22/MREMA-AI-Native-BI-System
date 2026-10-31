from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, date
from typing import Any

from .contracts.catalog import SemanticType
from .contracts.response import ChartSpec, ChartType, ResultTable

MAX_LINE_POINTS = 1000
MAX_SCATTER_POINTS = 3000
MAX_BAR_CATEGORIES = 100
MAX_TABLE_ROWS = 500


def _safe_sort_key(row: list[Any], idx: int) -> tuple[int, Any]:
    val = row[idx]
    if val is None:
        return (1, "")
    if isinstance(val, (int, float)):
        return (0, val)
    if hasattr(val, "isoformat"):
        return (0, val.isoformat())
    return (0, str(val))


def _average_datetime(dates: list[Any]) -> datetime | date | None:
    valid_dates = [d for d in dates if d is not None]
    if not valid_dates:
        return None
    
    timestamps = []
    is_date_only = True
    for d in valid_dates:
        if isinstance(d, str):
            try:
                d = datetime.fromisoformat(d.replace('Z', '+00:00'))
            except ValueError:
                continue
        if isinstance(d, datetime):
            is_date_only = False
            timestamps.append(d.timestamp())
        elif isinstance(d, date):
            timestamps.append(datetime(d.year, d.month, d.day).timestamp())
            
    if not timestamps:
        return valid_dates[0]
        
    avg_ts = sum(timestamps) / len(timestamps)
    avg_dt = datetime.fromtimestamp(avg_ts)
    
    if is_date_only:
        return avg_dt.date()
    return avg_dt


def _aggregate_bucket(bucket: list[list[Any]], table: ResultTable) -> list[Any]:
    if not bucket:
        return []
    
    agg_row = []
    for col_idx, col in enumerate(table.columns):
        vals = [row[col_idx] for row in bucket if row[col_idx] is not None]
        if not vals:
            agg_row.append(None)
            continue
            
        if col.semantic_type == SemanticType.NUMERIC:
            try:
                agg_row.append(sum(vals) / len(vals))
            except TypeError:
                agg_row.append(Counter(vals).most_common(1)[0][0])
                
        elif col.semantic_type in (SemanticType.DATETIME, SemanticType.DATE):
            agg_row.append(_average_datetime(vals))
            
        else:
            agg_row.append(Counter(vals).most_common(1)[0][0])
            
    return agg_row


def downsample_for_chart(table: ResultTable, spec: ChartSpec) -> tuple[ResultTable, ChartSpec, bool]:
    rows = table.rows
    total = len(rows)
    if total == 0:
        return table, spec, False

    sampled_rows = rows
    is_sampled = False
    chart_type = spec.chart_type

    if chart_type in (ChartType.LINE, ChartType.AREA):
        if total > MAX_LINE_POINTS:
            if spec.x:
                try:
                    idx = [c.name for c in table.columns].index(spec.x)
                    sorted_rows = sorted(rows, key=lambda r: _safe_sort_key(r, idx))
                except ValueError:
                    sorted_rows = rows
            else:
                sorted_rows = rows
            
            bucket_size = total / MAX_LINE_POINTS
            sampled_rows = []
            for i in range(MAX_LINE_POINTS):
                start_idx = int(i * bucket_size)
                end_idx = int((i + 1) * bucket_size)
                bucket = sorted_rows[start_idx:end_idx]
                if bucket:
                    sampled_rows.append(_aggregate_bucket(bucket, table))
                    
            is_sampled = True

    elif chart_type == ChartType.SCATTER:
        if total > MAX_SCATTER_POINTS:
            sampled_rows = random.sample(rows, MAX_SCATTER_POINTS)
            is_sampled = True

    elif chart_type == ChartType.BAR:
        if total > MAX_BAR_CATEGORIES:
            sampled_rows = rows[:MAX_BAR_CATEGORIES]
            is_sampled = True

    elif chart_type == ChartType.TABLE:
        if total > MAX_TABLE_ROWS:
            sampled_rows = rows[:MAX_TABLE_ROWS]
            is_sampled = True

    if not is_sampled:
        return table, spec, False

    new_table = table.model_copy(update={
        "rows": sampled_rows,
        "total_rows": total,
        "truncated": True
    })
    
    if chart_type in (ChartType.LINE, ChartType.AREA):
        suffix = " (усреднение)"
    elif chart_type == ChartType.SCATTER:
        suffix = " (выборка)"
    else:
        suffix = " (топ)"
        
    new_spec = spec.model_copy(update={
        "title": f"{spec.title}{suffix}" if spec.title else suffix.strip()
    })
    
    return new_table, new_spec, True