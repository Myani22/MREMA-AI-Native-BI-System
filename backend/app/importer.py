import io
import re
import uuid
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db import SessionLocal
from app.models import Dataset


def _fit_identifier(name: str, limit: int = 63) -> str:
    while len(name.encode("utf-8")) > limit:
        name = name[:-1]
    return name


def safe_column_name(value: Any, used: set[str]) -> str:
    name = str(value).strip().lower()
    name = re.sub(r"[^\w]+", "_", name).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"col_{name}"
    name = _fit_identifier(name, 55)
    candidate = name
    suffix = 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = _fit_identifier(name, 63 - len(tail)) + tail
        suffix += 1
    used.add(candidate)
    return candidate


_DATE_FORMATS = (
    (re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$"), None),
    (re.compile(r"^\d{2}\.\d{2}\.\d{4}$"), "%d.%m.%Y"),
    (re.compile(r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}(:\d{2})?$"), "mixed"),
)


def _parse_dates(df: pd.DataFrame, min_share: float = 0.95) -> pd.DataFrame:
    for column in df.columns:
        series = df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        values = series.dropna().astype(str).str.strip()
        if values.empty:
            continue
        sample = values.head(200)
        for pattern, fmt in _DATE_FORMATS:
            if sample.str.match(pattern).mean() < min_share:
                continue
            if fmt == "mixed":
                parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
            else:
                parsed = pd.to_datetime(series, errors="coerce", format=fmt)
            if parsed.notna().sum() >= min_share * len(values):
                df[column] = parsed
            break
    return df


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
    df = _parse_dates(df)

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