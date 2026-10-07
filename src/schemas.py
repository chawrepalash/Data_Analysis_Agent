"""Shared result model. Every tool returns a ToolResult, never a raw dataframe."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    ok: bool
    summary: str = ""  # one sentence the LLM can quote
    # compact numbers and examples
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    hint: str | None = None  # e.g. the list of valid column names


def ok_result(summary: str, **data: Any) -> ToolResult:
    return ToolResult(ok=True, summary=summary, data=data)


def error_result(error: str, hint: str | None = None) -> ToolResult:
    return ToolResult(ok=False, error=error, hint=hint)


class ProposeCleaningArgs(BaseModel):
    dataset_id: str


class ApplyCleaningArgs(BaseModel):
    dataset_id: str
    fix_ids: list[str] = Field(min_length=1, max_length=50)


class AggregateMetric(BaseModel):
    column: str
    agg: Literal["mean", "median", "sum", "min",
                 "max", "count", "nunique"] = "mean"


class FilterCondition(BaseModel):
    column: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: Any


class GroupAggregateArgs(BaseModel):
    dataset_id: str
    group_by: list[str] = Field(min_length=1, max_length=5)
    metrics: list[AggregateMetric] = Field(min_length=1, max_length=10)
    filters: list[FilterCondition] = Field(default_factory=list)
    sort_by: str | None = None
    ascending: bool = False
    limit: int = Field(50, ge=1, le=50)


class CreateChartArgs(BaseModel):
    dataset_id: str
    chart_type: Literal["bar", "line", "scatter", "histogram", "box"]
    x: str
    y: str | None = None
    color: str | None = None
    aggregation: Literal["mean", "sum", "count", "min", "max"] | None = None
