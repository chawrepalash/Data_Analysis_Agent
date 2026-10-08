"""State definition for LangGraph agents."""

from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state between auto-analysis and chat graphs."""
    dataset_id: str
    profile: dict[str, Any]                          # Compact profile, built once
    messages: Annotated[list[Any], add_messages]     # Chat history and tool messages
    trace: list[dict[str, Any]]                      # step, tool, args, ok, summary
    charts: list[str]                                # Plotly figure JSON strings
    rounds: int                                      # Tool-call rounds for this question
    findings: str                                    # Final text / answer for the UI