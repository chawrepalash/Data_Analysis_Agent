"""The tools the LLM can call, and the executor that runs them.

Two jobs:
  1. get_tools()        schemas for llm.bind_tools. They never contain dataset_id or
                        store: the graph supplies the dataset id from its state.
  2. make_executor(store)  runs one tool call safely. It checks the tool name, the
                        argument names and the argument types before calling the
                        real function, and turns every mistake into an error
                        result with a hint the model can read and fix.

Cleaning tools are not here on purpose: cleaning is proposed to the user and applied
only after approval in the UI, so the LLM cannot change the data.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.loader import DatasetStore
from src.schemas import FilterCondition, ToolResult, error_result
from src.tools.charts import create_chart
from src.tools.common import check_columns, column_kind, fetch
from src.schemas import ok_result
from src.tools.query import _apply_filter, group_aggregate
from src.tools.stats import correlation_analysis, describe_stats, detect_outliers


# --------------------------------------------------------------------------
# Argument models: what the LLM is allowed to send
# --------------------------------------------------------------------------

class DescribeStatsArgs(BaseModel):
    columns: list[str] | None = Field(
        default=None,
        description="Column names to describe. Leave out to describe every column.",
    )


class CorrelationArgs(BaseModel):
    method: Literal["pearson", "spearman"] = Field(
        default="pearson",
        description="pearson for straight-line relationships, spearman for rank-based ones.",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Numeric column names to include. Leave out to use all numeric columns.",
    )


class OutlierArgs(BaseModel):
    column: str = Field(description="The numeric column to check.")
    method: Literal["iqr", "zscore"] = Field(
        default="iqr",
        description="iqr uses quartile bounds, zscore uses standard deviations.",
    )
    threshold: float | None = Field(
        default=None,
        description="Bound multiplier. Leave out for the default: 1.5 for iqr, 3.0 for zscore.",
    )


class ChartArgs(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "histogram", "box"] = Field(
        description="The kind of chart to draw."
    )
    x: str = Field(
        description="Column for the x axis. For a histogram, the column to bin.")
    y: str | None = Field(
        default=None,
        description="Numeric column for the y axis (bar, line, scatter, box). Not used by histogram.",
    )
    color: str | None = Field(
        default=None,
        description="Optional category column used to split the data by color.",
    )
    aggregation: Literal["mean", "sum", "count", "min", "max"] | None = Field(
        default=None,
        description="How to combine rows that share the same x value, for bar and line charts. "
        "Defaults to mean when y is given.",
    )


class MetricArgs(BaseModel):
    # a misspelt key is an error, not silently dropped
    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="The column to summarize.")
    agg: Literal["mean", "sum", "median", "min", "max", "count", "nunique", "std"] = Field(
        default="mean",
        description="How to summarize it. mean, sum, median and std need a numeric column. "
        "count counts non-missing values; nunique counts distinct values.",
    )


class FilterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="The column to filter on.")
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"] = Field(
        description="The comparison. Use 'in' with `values`; every other operator uses `value`. "
        "'contains' matches text anywhere in the value, ignoring case."
    )
    value: str | None = Field(
        default=None,
        description="The value to compare with, written as text, for example '1' or 'female'.",
    )
    values: list[str] | None = Field(
        default=None,
        description="The list of allowed values for op 'in', each written as text.",
    )


class GroupAggregateArgs(BaseModel):
    group_by: list[str] = Field(
        default_factory=list,
        description="Columns to group by, for example ['Sex'] or ['Pclass', 'Sex']. "
        "Leave empty to compute one overall result for all rows that pass the filters.",
    )
    metrics: list[MetricArgs] = Field(
        min_length=1,
        description="What to compute for each group. For a rate of a 0/1 column, use agg mean.",
    )
    filters: list[FilterArgs] | None = Field(
        default=None,
        description="Optional conditions that rows must meet before grouping.",
    )
    sort_by: str | None = Field(
        default=None,
        description="Result column to sort by, named column_agg, for example Survived_mean. "
        "Leave out to sort by the first metric.",
    )
    ascending: bool = Field(
        default=False, description="Sort smallest first when true.")
    limit: int = Field(default=20, ge=1, le=50,
                       description="Maximum number of groups to return.")


# --------------------------------------------------------------------------
# group_aggregate adapter
#
# The real group_aggregate takes plain dicts and compares values exactly, so the
# string "1" would never match a numeric column. The adapter checks the request
# against the data and converts filter values to the column's type before calling it.
# --------------------------------------------------------------------------

NUMERIC_AGGS = {"mean", "sum", "median", "std"}


def _convert_value(df: pd.DataFrame, column: str, text: str) -> Any:
    """Turn a text value into the column's type. Raises ValueError if it cannot."""
    kind = column_kind(df[column])
    try:
        if kind == "numeric":
            return float(text)
        if kind == "boolean":
            word = str(text).strip().lower()
            if word in ("true", "1", "yes"):
                return True
            if word in ("false", "0", "no"):
                return False
            raise ValueError
        if kind == "datetime":
            stamp = pd.to_datetime(text)
            if pd.isna(stamp):
                raise ValueError
            return stamp
    except (TypeError, ValueError):
        raise ValueError(
            f"Cannot compare {kind} column '{column}' with {text!r}.") from None
    return str(text)


def _convert_filter(df: pd.DataFrame, flt: dict) -> dict:
    column, op = flt["column"], flt["op"]
    if op == "in":
        values = flt.get("values")
        if not values:
            raise ValueError(
                "Filter operator 'in' needs a non-empty `values` list.")
        return {"column": column, "op": op, "value": [_convert_value(df, column, v) for v in values]}

    value = flt.get("value")
    if value is None:
        raise ValueError(f"Filter operator '{op}' needs a `value`.")
    if op == "contains":
        return {"column": column, "op": op, "value": str(value)}
    return {"column": column, "op": op, "value": _convert_value(df, column, value)}


def _overall_aggregate(df: pd.DataFrame, metrics: list[dict], filters: list[dict]) -> ToolResult:
    """One result row over all rows that pass the filters (no grouping).

    This answers questions such as "what share of children under 10 survived?", which
    the grouped tool cannot, because it always needs a column to group by.
    """
    subset = df
    for flt in filters:
        subset, err = _apply_filter(subset, FilterCondition(**flt))
        if err:
            return error_result(err, hint="Verify filter column types and operators.")

    if subset.empty:
        return ok_result("Filter matched 0 rows.", rows=[], total_groups=0)

    row: dict[str, Any] = {"rows_matched": int(len(subset))}
    for metric in metrics:
        value = getattr(subset[metric["column"]], metric["agg"])()
        if isinstance(value, float):
            value = None if pd.isna(value) else round(value, 4)
        row[f"{metric['column']}_{metric['agg']}"] = value.item(
        ) if hasattr(value, "item") else value
    return ok_result(
        f"Computed {len(metrics)} metric(s) over {len(subset)} of {len(df)} rows (no grouping).",
        rows=[row],
        total_groups=1,
        displayed_groups=1,
        group_by=[],
    )


def _group_aggregate_for_llm(
    store: DatasetStore,
    dataset_id: str,
    group_by: list[str],
    metrics: list[dict],
    filters: list[dict] | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    limit: int = 20,
) -> ToolResult:
    df, error = fetch(store, dataset_id)
    if error:
        return error

    unique: dict[tuple[str, str], dict] = {}
    for metric in metrics:
        key = (metric["column"], metric.get("agg", "mean"))
        unique.setdefault(key, {"column": key[0], "agg": key[1]})
    metric_list = list(unique.values())

    referenced = list(
        dict.fromkeys(
            list(group_by)
            + [m["column"] for m in metric_list]
            + [f["column"] for f in (filters or [])]
        )
    )
    bad_column = check_columns(df, referenced)
    if bad_column:
        return bad_column

    for metric in metric_list:
        kind = column_kind(df[metric["column"]])
        if metric["agg"] in NUMERIC_AGGS and kind not in ("numeric", "boolean"):
            return error_result(
                f"'{metric['agg']}' needs a numeric column, but '{metric['column']}' is {kind}.",
                hint="Use count or nunique for text columns, or pick a numeric column.",
            )

    converted = []
    for flt in filters or []:
        try:
            converted.append(_convert_filter(df, flt))
        except ValueError as exc:
            return error_result(str(exc), hint="Write values the way they appear in the data.")

    if not group_by:
        return _overall_aggregate(df, metric_list, converted)

    if sort_by is None:
        sort_by = f"{metric_list[0]['column']}_{metric_list[0]['agg']}"

    return group_aggregate(
        store,
        dataset_id,
        group_by=list(group_by),
        metrics=metric_list,
        filters=converted or None,
        sort_by=sort_by,
        ascending=ascending,
        limit=limit,
    )


# --------------------------------------------------------------------------
# create_chart adapter
#
# A bar or line chart of y by x with no aggregation draws one stacked segment per row,
# which shows a sum (233 "survivors") where the reader expects an average. Combine rows
# per x value by default instead.
# --------------------------------------------------------------------------

def _create_chart_for_llm(store: DatasetStore, dataset_id: str, **kwargs) -> ToolResult:
    if kwargs.get("chart_type") in ("bar", "line") and kwargs.get("y") and not kwargs.get("aggregation"):
        kwargs["aggregation"] = "mean"
    return create_chart(store, dataset_id, **kwargs)


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., ToolResult]


_SPECS = [
    ToolSpec(
        "describe_stats",
        "Summary statistics. For numeric columns: count, missing, mean, median, std, min, "
        "quartiles and max. For other columns: the most common values. "
        "Use it to understand what a column looks like.",
        DescribeStatsArgs,
        describe_stats,
    ),
    ToolSpec(
        "correlation_analysis",
        "Correlation matrix of numeric columns plus the 5 strongest pairs. "
        "Use it to find which numeric columns move together.",
        CorrelationArgs,
        correlation_analysis,
    ),
    ToolSpec(
        "detect_outliers",
        "Count unusually high or low values in ONE numeric column and give the bounds used. "
        "Use it to check whether a column has extreme values.",
        OutlierArgs,
        detect_outliers,
    ),
    ToolSpec(
        "group_aggregate",
        "Compare groups. Optionally filter rows, group them by one or more columns, and compute "
        "metrics for each group, such as the average or total of a number, or a count. "
        "This is the main tool for questions like 'which group has the highest X' or "
        "'how does X differ by Y'. For one overall number for a subset, such as the survival rate "
        "of children under 10, filter the rows and leave group_by empty. "
        "For a rate of a 0/1 column such as survival, use agg mean.",
        GroupAggregateArgs,
        _group_aggregate_for_llm,
    ),
    ToolSpec(
        "create_chart",
        "Create a Plotly chart (bar, line, scatter, histogram or box). The figure is saved for "
        "the user automatically; you only receive a short confirmation. "
        "Bar and line charts combine rows per x value (mean by default; set aggregation to "
        "sum or count when that is what the question needs).",
        ChartArgs,
        _create_chart_for_llm,
    ),
]

TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in _SPECS}


def _never_called(**kwargs):
    raise RuntimeError("Tools are run by the executor, not by LangChain.")


def get_tools() -> list[StructuredTool]:
    """Tool schemas for llm.bind_tools()."""
    return [
        StructuredTool.from_function(
            func=_never_called,
            name=spec.name,
            description=spec.description,
            args_schema=spec.args_model,
        )
        for spec in TOOLS.values()
    ]


# --------------------------------------------------------------------------
# The executor
# --------------------------------------------------------------------------

def make_executor(store: DatasetStore) -> Callable[[str, dict, str], ToolResult]:
    """Return execute(tool_name, args, dataset_id) -> ToolResult for the graph."""

    def execute(name: str, args: dict, dataset_id: str) -> ToolResult:
        spec = TOOLS.get(name)
        if spec is None:
            return error_result(
                f"Unknown tool '{name}'.",
                hint="Valid tools: " + ", ".join(TOOLS),
            )

        if not isinstance(args, dict):
            return error_result(f"Arguments for '{name}' must be an object.")

        # The dataset id always comes from the graph state. If the model sends one
        # anyway, ignore it instead of failing.
        args = {key: value for key, value in args.items() if key !=
                "dataset_id"}

        allowed = list(spec.args_model.model_fields)
        unknown = sorted(set(args) - set(allowed))
        if unknown:
            return error_result(
                f"Unknown argument(s) for {name}: {', '.join(unknown)}.",
                hint="Valid arguments: " + ", ".join(allowed),
            )

        try:
            parsed = spec.args_model(**args)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "arguments"
            return error_result(
                f"Invalid argument '{where}' for {name}: {first['msg']}.",
                hint="Valid arguments: " + ", ".join(allowed),
            )

        # Only pass what the model actually set, so each tool's own defaults apply.
        return spec.func(store, dataset_id, **parsed.model_dump(exclude_none=True))

    return execute
