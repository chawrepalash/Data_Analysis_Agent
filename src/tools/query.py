"""Tool 8: group_aggregate.

Performs grouped aggregations with optional filtering and sorting.
Capped at 50 rows. Returns compact summaries and records.
"""

from typing import Any
import pandas as pd

from src.loader import DatasetStore
from src.schemas import FilterCondition, ToolResult, error_result, ok_result
from src.tools.common import check_columns, fetch

MAX_ROWS = 50


def _apply_filter(df: pd.DataFrame, flt: FilterCondition) -> tuple[pd.DataFrame | None, str | None]:
    col = flt.column
    val = flt.value
    op = flt.op

    try:
        if op == "==":
            return df[df[col] == val], None
        if op == "!=":
            return df[df[col] != val], None
        if op == ">":
            return df[df[col] > val], None
        if op == ">=":
            return df[df[col] >= val], None
        if op == "<":
            return df[df[col] < val], None
        if op == "<=":
            return df[df[col] <= val], None
        if op == "in":
            if not isinstance(val, (list, tuple, set)):
                return None, f"Filter 'in' requires a list or set, received: {type(val).__name__}"
            return df[df[col].isin(val)], None
        if op == "contains":
            return df[df[col].astype(str).str.contains(str(val), case=False, na=False)], None
        return None, f"Unsupported filter operator: '{op}'"
    except Exception as exc:
        return None, f"Filter error on '{col}' with operator '{op}': {exc}"


def group_aggregate(
    store: DatasetStore,
    dataset_id: str,
    group_by: list[str],
    metrics: list[dict[str, Any]],
    filters: list[dict[str, Any]] | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    limit: int = MAX_ROWS,
) -> ToolResult:
    """Group, aggregate, filter, and sort. Returns at most 50 rows."""
    df, error = fetch(store, dataset_id)
    if error:
        return error

    filters = filters or []
    metric_cols = [m.get("column") for m in metrics if m.get("column")]
    filter_cols = [f.get("column") for f in filters if f.get("column")]

    all_cols = list(set(group_by + metric_cols + filter_cols))
    col_err = check_columns(df, all_cols)
    if col_err:
        return col_err

    # Apply filters
    filtered_df = df
    for flt_dict in filters:
        try:
            flt_obj = FilterCondition(**flt_dict)
        except Exception as exc:
            return error_result(f"Invalid filter format: {exc}")

        filtered_df, err = _apply_filter(filtered_df, flt_obj)
        if err:
            return error_result(err, hint="Verify filter column types and operators.")

    if filtered_df.empty:
        return ok_result("Filter matched 0 rows.", rows=[], total_groups=0)

    # Build aggregation mapping
    agg_spec = {}
    rename_cols = {}
    for idx, m in enumerate(metrics):
        col = m["column"]
        func = m.get("agg", "mean")
        out_name = f"{col}_{func}"
        rename_cols[(col, func)] = out_name
        agg_spec.setdefault(col, []).append(func)

    try:
        grouped = filtered_df.groupby(group_by, as_index=False).agg(agg_spec)
        # Flatten MultiIndex if present
        if isinstance(grouped.columns, pd.MultiIndex):
            new_columns = []
            for col, stat in grouped.columns:
                if stat == "":
                    new_columns.append(col)
                else:
                    new_columns.append(f"{col}_{stat}")
            grouped.columns = new_columns

        # Sorting
        if sort_by:
            if sort_by not in grouped.columns:
                valid_sorts = list(grouped.columns)
                return error_result(
                    f"sort_by column '{sort_by}' not in results.",
                    hint=f"Choose one of: {', '.join(valid_sorts)}",
                )
            grouped = grouped.sort_values(by=sort_by, ascending=ascending)

        total_groups = len(grouped)
        effective_limit = min(max(1, limit), MAX_ROWS)
        limited = grouped.head(effective_limit)

        records = limited.to_dict(orient="records")
        # Round floating point values
        for r in records:
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = round(v, 4)

        summary = (
            f"Aggregated {len(filtered_df)} rows into {total_groups} groups by "
            f"{', '.join(group_by)}. Showing top {len(records)} results."
        )
        return ok_result(
            summary,
            rows=records,
            total_groups=total_groups,
            displayed_groups=len(records),
            group_by=group_by,
        )
    except Exception as exc:
        return error_result(f"Aggregation failed: {exc}")
