"""Tools 1 and 2: profile_dataset and validate_data.

Both are read-only. They return compact summaries, never whole dataframes.
"""

import json
import re
import warnings

import pandas as pd

from src.loader import DatasetStore
from src.schemas import ToolResult, ok_result
from src.tools.common import column_kind, fetch

MAX_PROFILE_COLUMNS = 100
SAMPLE_ROWS = 5
DATE_SAMPLE_SIZE = 500

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}

# Columns whose name suggests a count or amount that should not be negative.
NON_NEGATIVE_NAME = re.compile(
    r"(^|_|\s)(age|units?|qty|quantity|count|price|amount|revenue|salary|sales|cost)($|_|\s)",
    re.IGNORECASE,
)


def _num(value) -> float | None:
    return None if pd.isna(value) else round(float(value), 4)


# --------------------------------------------------------------------------
# Tool 1: profile_dataset
# --------------------------------------------------------------------------

def profile_dataset(store: DatasetStore, dataset_id: str) -> ToolResult:
    """Shape, types, missing values, duplicates, unique counts and sample rows."""
    df, error = fetch(store, dataset_id)
    if error:
        return error

    n_rows, n_cols = df.shape
    duplicates = int(df.duplicated().sum())

    column_info = []
    columns_with_missing = 0
    for name in list(df.columns)[:MAX_PROFILE_COLUMNS]:
        series = df[name]
        missing = int(series.isna().sum())
        columns_with_missing += missing > 0
        info = {
            "name": str(name),
            "kind": column_kind(series),
            "dtype": str(series.dtype),
            "missing": missing,
            "missing_pct": round(100 * missing / n_rows, 1),
            "unique": int(series.nunique(dropna=True)),
        }
        if info["kind"] == "numeric" and missing < n_rows:
            info["min"] = _num(series.min())
            info["max"] = _num(series.max())
        column_info.append(info)

    sample_rows = json.loads(df.head(SAMPLE_ROWS).to_json(orient="records", date_format="iso"))

    return ok_result(
        f"{n_rows} rows x {n_cols} columns; {duplicates} duplicate rows; "
        f"{columns_with_missing} columns have missing values.",
        rows=n_rows,
        columns=n_cols,
        duplicate_rows=duplicates,
        column_info=column_info,
        sample_rows=sample_rows,
        columns_truncated=n_cols > MAX_PROFILE_COLUMNS,
    )


# --------------------------------------------------------------------------
# Tool 2: validate_data
# --------------------------------------------------------------------------

def _issue(type_: str, column: str | None, severity: str, detail: str) -> dict:
    return {"type": type_, "column": column, "severity": severity, "detail": detail}


def _numeric_share(text: pd.Series) -> float:
    """Share of values in a text column that can be read as numbers."""
    return float(pd.to_numeric(text, errors="coerce").notna().mean())


def _date_like(text: pd.Series) -> tuple[bool, int]:
    """Does this text column hold dates? Returns (is_date_like, unparseable_count).

    Only values that contain a digit and are not plain numbers are tried, so
    names and numeric strings are not mistaken for dates.
    """
    numeric = pd.to_numeric(text, errors="coerce").notna()
    has_digit = text.str.contains(r"\d", regex=True)
    candidates = text[~numeric & has_digit]
    if len(candidates) < 0.5 * len(text):
        return False, 0
    sample = candidates.head(DATE_SAMPLE_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    overall_success = float(parsed.notna().mean()) * len(candidates) / len(text)
    if overall_success < 0.8:
        return False, 0
    unparseable = len(text) - round(overall_success * len(text))
    return True, unparseable


def validate_data(store: DatasetStore, dataset_id: str) -> ToolResult:
    """Find data quality problems. Reports issues, changes nothing."""
    df, error = fetch(store, dataset_id)
    if error:
        return error

    n_rows = len(df)
    issues: list[dict] = []

    duplicates = int(df.duplicated().sum())
    if duplicates:
        pct = 100 * duplicates / n_rows
        issues.append(
            _issue(
                "duplicate_rows",
                None,
                "medium" if pct >= 1 else "low",
                f"{duplicates} duplicate rows ({pct:.1f}% of rows).",
            )
        )

    for name in df.columns:
        series = df[name]
        col = str(name)
        non_null = series.dropna()
        kind = column_kind(series)

        if non_null.empty:
            issues.append(_issue("empty_column", col, "high", "Every value is missing."))
            continue

        missing = int(series.isna().sum())
        if missing:
            pct = 100 * missing / n_rows
            severity = "high" if pct >= 40 else "medium" if pct >= 5 else "low"
            issues.append(
                _issue("missing_values", col, severity, f"{missing} missing values ({pct:.1f}%).")
            )

        if n_rows >= 2 and non_null.nunique() == 1:
            issues.append(
                _issue(
                    "constant_column",
                    col,
                    "low",
                    f"Every non-missing value is the same ({non_null.iloc[0]!r}).",
                )
            )

        if kind == "numeric" and NON_NEGATIVE_NAME.search(col):
            negatives = int((non_null < 0).sum())
            if negatives:
                issues.append(
                    _issue(
                        "negative_values",
                        col,
                        "medium",
                        f"{negatives} negative values in a column that should not be negative.",
                    )
                )

        if kind == "text":
            text = non_null.astype(str).str.strip()

            share = _numeric_share(text)
            if share == 1.0:
                issues.append(
                    _issue("numbers_as_text", col, "low", "All values are numbers stored as text.")
                )
            elif share >= 0.5:
                bad = int(round((1 - share) * len(text)))
                issues.append(
                    _issue(
                        "mixed_types",
                        col,
                        "high",
                        f"Mostly numbers but {bad} value(s) are not numeric.",
                    )
                )
            else:
                is_date, unparseable = _date_like(text)
                if is_date:
                    extra = f" {unparseable} value(s) could not be parsed." if unparseable else ""
                    issues.append(
                        _issue("date_like_text", col, "medium", "Dates stored as text." + extra)
                    )
                elif n_rows >= 20 and text.nunique() / len(text) > 0.9:
                    issues.append(
                        _issue(
                            "high_cardinality_text",
                            col,
                            "info",
                            f"{text.nunique()} distinct values in {len(text)} rows; "
                            "likely an identifier or free text, not a category.",
                        )
                    )

    issues.sort(key=lambda i: (SEVERITY_ORDER[i["severity"]], i["column"] or ""))

    counts = {sev: sum(1 for i in issues if i["severity"] == sev) for sev in SEVERITY_ORDER}
    if not issues:
        return ok_result("No data quality issues found.", issues=[], counts=counts)
    return ok_result(
        f"{len(issues)} issue(s) found: {counts['high']} high, {counts['medium']} medium, "
        f"{counts['low']} low, {counts['info']} info.",
        issues=issues,
        counts=counts,
    )
