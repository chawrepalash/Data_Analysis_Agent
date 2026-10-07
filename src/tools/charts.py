"""Tool 9: create_chart.

Generates Plotly figures as JSON. Returns chart metadata and serialized figure.
"""

from typing import Any, Literal
import plotly.express as px
import plotly.io as pio
import pandas as pd

from src.loader import DatasetStore
from src.schemas import ToolResult, error_result, ok_result
from src.tools.common import check_columns, column_kind, fetch

MAX_CHART_ROWS = 2000


def create_chart(
    store: DatasetStore,
    dataset_id: str,
    chart_type: Literal["bar", "line", "scatter", "histogram", "box"],
    x: str,
    y: str | None = None,
    color: str | None = None,
    aggregation: Literal["mean", "sum", "count", "min", "max"] | None = None,
) -> ToolResult:
    """Creates a Plotly chart and returns the JSON string representation."""
    df, error = fetch(store, dataset_id)
    if error:
        return error

    needed_cols = [x]
    if y:
        needed_cols.append(y)
    if color:
        needed_cols.append(color)

    col_err = check_columns(df, needed_cols)
    if col_err:
        return col_err

    # Type validation
    x_kind = column_kind(df[x])
    y_kind = column_kind(df[y]) if y else None

    plot_df = df[needed_cols].dropna().copy()
    if plot_df.empty:
        return error_result(f"Selected columns ({', '.join(needed_cols)}) have no valid non-null rows.")

    # Validation per chart type
    if chart_type in ("scatter", "line") and not y:
        return error_result(f"Chart type '{chart_type}' requires both 'x' and 'y' columns.")

    if chart_type == "box" and not y and x_kind != "numeric":
        return error_result("Box plot requires a numeric column for distribution.")

    try:
        # Pre-aggregate if requested
        if aggregation and y and chart_type == "bar":
            group_keys = [x] + ([color] if color else [])
            plot_df = plot_df.groupby(group_keys, as_index=False)[
                y].agg(aggregation)

        # Cap rows for render safety
        if len(plot_df) > MAX_CHART_ROWS:
            plot_df = plot_df.sample(MAX_CHART_ROWS, random_state=42)

        # Plot generation
        if chart_type == "bar":
            fig = px.bar(plot_df, x=x, y=y, color=color,
                         title=f"{chart_type.capitalize()} of {y or x} by {x}")
        elif chart_type == "line":
            plot_df = plot_df.sort_values(by=x)
            fig = px.line(plot_df, x=x, y=y, color=color,
                          title=f"Line Chart of {y} over {x}")
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x, y=y, color=color,
                             title=f"Scatter of {y} vs {x}")
        elif chart_type == "histogram":
            fig = px.histogram(plot_df, x=x, y=y, color=color,
                               title=f"Histogram of {x}")
        elif chart_type == "box":
            fig = px.box(plot_df, x=x if y else None, y=y if y else x,
                         color=color, title=f"Box plot of {y or x}")
        else:
            return error_result(f"Unsupported chart type '{chart_type}'.", hint="Use bar, line, scatter, histogram, or box.")

        fig.update_layout(template="plotly_white",
                          margin=dict(l=40, r=40, t=50, b=40))
        chart_json = pio.to_json(fig)

        summary = f"Created {chart_type} chart with x='{x}'" + \
            (f" and y='{y}'" if y else "") + "."
        return ok_result(
            summary,
            chart_type=chart_type,
            chart_json=chart_json,
            x=x,
            y=y,
        )

    except Exception as exc:
        return error_result(f"Chart creation failed: {exc}")
