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
from typing import Callable, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from src.loader import DatasetStore
from src.schemas import ToolResult, error_result
from src.tools.charts import create_chart
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
    x: str = Field(description="Column for the x axis. For a histogram, the column to bin.")
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
        description="How to combine rows that share the same x value, for bar and line charts.",
    )


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
        "create_chart",
        "Create a Plotly chart (bar, line, scatter, histogram or box). The figure is saved for "
        "the user automatically; you only receive a short confirmation. "
        "Use aggregation with bar or line charts to combine rows per x value, "
        "for example the mean of y for each x.",
        ChartArgs,
        create_chart,
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
        args = {key: value for key, value in args.items() if key != "dataset_id"}

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
