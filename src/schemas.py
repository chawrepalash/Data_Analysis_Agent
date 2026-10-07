"""Shared result model. Every tool returns a ToolResult, never a raw dataframe."""

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    ok: bool
    summary: str = ""  # one sentence the LLM can quote
    data: dict[str, Any] = Field(default_factory=dict)  # compact numbers and examples
    error: str | None = None
    hint: str | None = None  # e.g. the list of valid column names


def ok_result(summary: str, **data: Any) -> ToolResult:
    return ToolResult(ok=True, summary=summary, data=data)


def error_result(error: str, hint: str | None = None) -> ToolResult:
    return ToolResult(ok=False, error=error, hint=hint)
