import io
import re
import uuid
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db import SessionLocal
from app.models import Dataset


def safe_column_name(value: Any, used: set[str]) -> str:
    # Normalize headers before creating SQL columns. Original names remain in profile.
    name = str(value).strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"col_{name}"
    name = name[:50]
    candidate = name
    suffix = 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = f"{name[:63-len(tail)]}{tail}"
        suffix += 1
    used.add(candidate)
    return candidate


def _profile_frame(df: pd.DataFrame, original_headers: list[str]) -> list[dict]:
    profile = []
    for i, column in enumerate(df.columns):
        series = df[column]
        dtype = str(series.dtype)
        if pd.api.types.is_bool_dtype(series):
            logical_type = "bool"
        elif pd.api.types.is_integer_dtype(series):
            logical_type = "int"
        elif pd.api.types.is_float_dtype(series):
            logical_type = "float"
        elif pd.api.types.is_datetime64_any_dtype(series):
            logical_type = "date"
        else:
            logical_type = "str"

        non_null = series.dropna()
        examples = [None if pd.isna(v) else str(v)[:160] for v in non_null.head(5).tolist()]
        item = {
            "name": str(column),
            "original_name": str(original_headers[i]) if i < len(original_headers) else str(column),
            "type": logical_type,
            "pandas_dtype": dtype,
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "examples": examples,
        }
        if logical_type in ("int", "float") and len(non_null):
            item["min"] = _json_scalar(non_null.min())
            item["max"] = _json_scalar(non_null.max())
        profile.append(item)
    return profile


def _json_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def import_csv_bytes(content: bytes, filename: str, engine: Engine) -> dict:
    # Keep a stable original-header mapping; pandas mangles duplicate headers, which is safer
    # than silently creating duplicate SQL column names.
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="cp1251", low_memory=False)
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать CSV: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать CSV: {exc}") from exc

    if len(df.columns) == 0:
        raise ValueError("CSV не содержит столбцов.")
    if len(df.columns) > 500:
        raise ValueError("Слишком много столбцов: максимум 500.")

    original_headers = [str(c) for c in df.columns]
    used: set[str] = set()
    new_headers = [safe_column_name(c, used) for c in original_headers]
    df.columns = new_headers

    table_name = f"dataset_{uuid.uuid4().hex[:20]}"
    try:
        # Explicit schema and generated table name; column names are normalized.
        df.to_sql(
            name=table_name,
            con=engine,
            schema="data",
            if_exists="fail",
            index=False,
            chunksize=2000,
            method="multi",
        )
    except Exception as exc:
        raise RuntimeError(f"Не удалось записать данные в PostgreSQL: {exc}") from exc

    profile = _profile_frame(df, original_headers)
    try:
        with SessionLocal() as session:
            dataset = Dataset(
                original_filename=filename[:255],
                table_name=table_name,
                row_count=int(len(df)),
                columns_profile=profile,
            )
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            return {
                "id": dataset.id,
                "original_filename": dataset.original_filename,
                "table_name": dataset.table_name,
                "row_count": dataset.row_count,
                "columns": profile,
            }
    except Exception:
        # Avoid leaving an orphaned imported table if metadata persistence fails.
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS data."{table_name}"'))
        raise
