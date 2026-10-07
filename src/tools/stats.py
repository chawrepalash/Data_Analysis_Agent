"""Tools 5, 6 and 7: describe_stats, correlation_analysis, detect_outliers.

All three are read-only and return compact summaries. NaN and +-inf are never
returned as numbers: they become None so every result is valid JSON.
"""

import json
import warnings

import numpy as np
import pandas as pd

from src.loader import DatasetStore
from src.schemas import ToolResult, error_result, ok_result
from src.tools.common import MAX_HINT_COLUMNS, check_columns, column_kind, fetch

MAX_DESCRIBE_COLUMNS = 30
TOP_VALUES = 5
MAX_CORR_COLUMNS = 15
TOP_PAIRS = 5
MAX_EXAMPLES = 5
MAX_EXAMPLE_COLUMNS = 12
MIN_OUTLIER_VALUES = 4

CORR_METHODS = ("pearson", "spearman")
OUTLIER_METHODS = ("iqr", "zscore")
DEFAULT_THRESHOLD = {"iqr": 1.5, "zscore": 3.0}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _num(value) -> float | None:
    """Round to 4 places; NaN, inf and None become None."""
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return round(value, 4) if np.isfinite(value) else None


def _as_float(series: pd.Series) -> pd.Series:
    """Numeric series as float64 with +-inf turned into NaN."""
    return series.astype("float64").replace([np.inf, -np.inf], np.nan)


def _duplicate_names_error(df: pd.DataFrame) -> ToolResult | None:
    if df.columns.is_unique:
        return None
    return error_result(
        "Duplicate column names make this analysis ambiguous.",
        hint="Rename the duplicated columns in the source file and reload it.",
    )


def _numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if column_kind(df[c]) == "numeric"]


def _numeric_hint(df: pd.DataFrame) -> str:
    names = [str(c) for c in _numeric_columns(df)[:MAX_HINT_COLUMNS]]
    return f"Numeric columns: {', '.join(names) if names else '(none)'}"


# --------------------------------------------------------------------------
# Tool 5: describe_stats
# --------------------------------------------------------------------------

def _describe_column(series: pd.Series, name: str) -> dict:
    kind = column_kind(series)
    info = {
        "column": name,
        "kind": kind,
        "count": int(series.notna().sum()),
        "missing": int(series.isna().sum()),
    }
    if kind == "numeric":
        values = _as_float(series).dropna()
        keys = ("mean", "median", "std", "min", "q1", "q3", "max")
        info.update({k: None for k in keys})
        if not values.empty:
            info.update(
                mean=_num(values.mean()),
                median=_num(values.median()),
                std=_num(values.std()),
                min=_num(values.min()),
                q1=_num(values.quantile(0.25)),
                q3=_num(values.quantile(0.75)),
                max=_num(values.max()),
            )
    elif kind == "datetime":
        non_null = series.dropna()
        info["min"] = non_null.min().isoformat() if not non_null.empty else None
        info["max"] = non_null.max().isoformat() if not non_null.empty else None
    else:  # text and boolean: top values
        counts = series.dropna().value_counts().head(TOP_VALUES)
        info["unique"] = int(series.nunique(dropna=True))
        info["top_values"] = [
            {"value": str(k), "count": int(c)} for k, c in counts.items()]
    return info


def describe_stats(store: DatasetStore, dataset_id: str, columns: list[str] | None = None) -> ToolResult:
    """Numeric: mean, median, std, min, quartiles, max. Others: top values."""
    df, error = fetch(store, dataset_id)
    if error:
        return error
    if (error := _duplicate_names_error(df)) is not None:
        return error

    if columns:
        if (error := check_columns(df, list(columns))) is not None:
            return error
        selected = list(dict.fromkeys(columns))
    else:
        selected = list(df.columns)

    truncated = len(selected) > MAX_DESCRIBE_COLUMNS
    selected = selected[:MAX_DESCRIBE_COLUMNS]
    stats = [_describe_column(df[c], str(c)) for c in selected]

    n_numeric = sum(s["kind"] == "numeric" for s in stats)
    summary = f"Described {len(stats)} column(s): {n_numeric} numeric, {len(stats) - n_numeric} other."
    if truncated:
        summary += f" Only the first {MAX_DESCRIBE_COLUMNS} columns were described."
    return ok_result(summary, rows=len(df), columns=stats, truncated=truncated)


# --------------------------------------------------------------------------
# Tool 6: correlation_analysis
# --------------------------------------------------------------------------

def correlation_analysis(
    store: DatasetStore,
    dataset_id: str,
    method: str = "pearson",
    columns: list[str] | None = None,
) -> ToolResult:
    """Correlation matrix plus the top 5 pairs by absolute value."""
    df, error = fetch(store, dataset_id)
    if error:
        return error
    if (error := _duplicate_names_error(df)) is not None:
        return error
    if method not in CORR_METHODS:
        return error_result(f"Unknown method '{method}'.", hint=f"Use one of: {', '.join(CORR_METHODS)}.")

    numeric = _numeric_columns(df)
    if columns:
        if (error := check_columns(df, list(columns))) is not None:
            return error
        not_numeric = [str(c) for c in columns if c not in numeric]
        if not_numeric:
            return error_result(
                f"Not numeric: {', '.join(not_numeric)}.", hint=_numeric_hint(df)
            )
        selected = list(dict.fromkeys(columns))
    else:
        selected = numeric

    if len(selected) < 2:
        return error_result(
            f"Need at least 2 numeric columns, found {len(selected)}.", hint=_numeric_hint(df)
        )

    truncated = len(selected) > MAX_CORR_COLUMNS
    selected = selected[:MAX_CORR_COLUMNS]
    data = df[selected].astype("float64").replace([np.inf, -np.inf], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = data.corr(method=method)

    names = [str(c) for c in selected]
    matrix = {
        a: {b: _num(corr.iloc[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)
    }
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = _num(corr.iloc[i, j])
            if value is not None:
                pairs.append(
                    {"a": names[i], "b": names[j], "correlation": value})
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    pairs = pairs[:TOP_PAIRS]

    if pairs:
        top = pairs[0]
        summary = (
            f"Strongest {method} correlation: {top['a']} and {top['b']} "
            f"({top['correlation']:+.2f}) across {len(names)} numeric columns."
        )
    else:
        summary = "No correlations could be computed (columns are constant or have too few overlapping values)."
    if truncated:
        summary += f" Only the first {MAX_CORR_COLUMNS} numeric columns were used."
    return ok_result(summary, method=method, columns=names, matrix=matrix, top_pairs=pairs, truncated=truncated)


# --------------------------------------------------------------------------
# Tool 7: detect_outliers
# --------------------------------------------------------------------------

def detect_outliers(
    store: DatasetStore,
    dataset_id: str,
    column: str,
    method: str = "iqr",
    threshold: float | None = None,
) -> ToolResult:
    """Count values outside the bounds. IQR: Q1/Q3 -/+ threshold x IQR.
    Z-score: mean +/- threshold x std. Default thresholds: 1.5 (iqr), 3.0 (zscore)."""
    df, error = fetch(store, dataset_id)
    if error:
        return error
    if (error := _duplicate_names_error(df)) is not None:
        return error
    if method not in OUTLIER_METHODS:
        return error_result(f"Unknown method '{method}'.", hint=f"Use one of: {', '.join(OUTLIER_METHODS)}.")
    if threshold is None:
        threshold = DEFAULT_THRESHOLD[method]
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        return error_result("Threshold must be a positive number.", hint="For example 1.5 (iqr) or 3 (zscore).")

    if (error := check_columns(df, [column])) is not None:
        return error
    kind = column_kind(df[column])
    if kind != "numeric":
        return error_result(f"Column '{column}' is {kind}, not numeric.", hint=_numeric_hint(df))

    series = _as_float(df[column])
    values = series.dropna()
    if len(values) < MIN_OUTLIER_VALUES:
        return error_result(
            f"Column '{column}' has only {len(values)} usable value(s); need at least {MIN_OUTLIER_VALUES}.",
            hint="Pick a column with more non-missing values.",
        )

    if method == "iqr":
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        lower, upper = q1 - threshold * (q3 - q1), q3 + threshold * (q3 - q1)
        is_outlier = (series < lower) | (series > upper)
    else:
        mean, std = values.mean(), values.std()
        lower, upper = mean - threshold * std, mean + threshold * std
        is_outlier = (series < lower) | (series > upper)
        if not std > 0:  # constant column: nothing can be an outlier
            is_outlier = pd.Series(False, index=series.index)

    mask = is_outlier.to_numpy()
    positions = np.flatnonzero(mask)
    count = int(len(positions))
    n_below = int((series < lower).to_numpy()[mask].sum())
    n_above = count - n_below

    # most extreme examples first (distance from the median)
    deviation = (series.iloc[positions] - values.median()).abs().to_numpy()
    top = positions[np.argsort(-deviation, kind="stable")][:MAX_EXAMPLES]
    other = [c for c in df.columns if c != column][: MAX_EXAMPLE_COLUMNS - 1]
    records = json.loads(
        df.iloc[top][[column] + other].to_json(orient="records", date_format="iso"))
    examples = [{"_row": int(p), **rec} for p, rec in zip(top, records)]

    pct = 100 * count / len(values)
    summary = (
        f"{count} outlier(s) in '{column}' ({pct:.1f}% of {len(values)} values) using {method} "
        f"with threshold {threshold:g}; bounds {lower:.4g} to {upper:.4g}."
    )
    return ok_result(
        summary,
        column=column,
        method=method,
        threshold=threshold,
        n_values=int(len(values)),
        outlier_count=count,
        outlier_pct=round(pct, 2),
        below_lower=n_below,
        above_upper=n_above,
        lower_bound=_num(lower),
        upper_bound=_num(upper),
        examples=examples,
    )
