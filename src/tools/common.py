"""Helpers shared by every tool."""

import pandas as pd

from src.loader import DatasetNotFound, DatasetStore
from src.schemas import ToolResult, error_result

MAX_HINT_COLUMNS = 50


def fetch(store: DatasetStore, dataset_id: str) -> tuple[pd.DataFrame | None, ToolResult | None]:
    """Return (dataframe, None) or (None, error result). Never raises."""
    try:
        return store.get(dataset_id), None
    except DatasetNotFound as exc:
        return None, error_result(str(exc), hint="Load a dataset first.")


def column_kind(series: pd.Series) -> str:
    """One of: boolean, numeric, datetime, text.

    Uses pandas type helpers instead of comparing dtype to object, because newer
    pandas versions store text in a dedicated string dtype.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def check_columns(df: pd.DataFrame, columns: list[str]) -> ToolResult | None:
    """Return an error result if any column is unknown, else None."""
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return None
    valid = list(df.columns)[:MAX_HINT_COLUMNS]
    return error_result(
        f"Unknown column(s): {', '.join(map(str, missing))}.",
        hint=f"Valid columns: {', '.join(map(str, valid))}",
    )
