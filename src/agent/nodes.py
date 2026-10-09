"""Graph nodes for the auto-analysis and chat graphs.

Auto-analysis flow:   profile_data -> plan -> run_tools -> (plan again | summarize)

Only plan and summarize call the LLM. Every number comes from a tool.
Tools are run by run_tools_node through an injected executor, so the LLM never
chooses the dataset id and never sees the dataframe.
"""

import json
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.llm import invoke_with_retry, message_text
from src.agent.prompts import CHAT_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, SUMMARIZER_SYSTEM_PROMPT
from src.agent.state import AgentState
from src.loader import DatasetStore
from src.schemas import ToolResult, error_result
from src.tools.profile import profile_dataset, validate_data

# (tool_name, args, dataset_id) -> ToolResult. Built by the registry.
ToolExecutor = Callable[[str, dict[str, Any], str], ToolResult]

MAX_PLAN_ATTEMPTS = 2      # the first plan plus one retry
MAX_CALLS_PER_STEP = 6     # tool calls honoured in one step; extras are skipped
MAX_TOOL_CHARS = 8000      # cap on one tool result shown to the LLM
MAX_PROFILE_CHARS = 12000  # cap on the profile shown to the LLM
MAX_ISSUES_IN_PROMPT = 10
CHART_PLACEHOLDER = "[chart created and stored for the UI]"

PLAN_PROMPT = (
    "Dataset profile and data quality issues (JSON):\n{profile}\n\n"
    "Choose 2 to 4 high-value analyses (descriptive statistics, correlations, outliers or "
    "group aggregates) and 1 or 2 charts that together give a good overview of this dataset. "
    "Call the tools now. Use only column names that appear in the profile."
)
RETRY_PROMPT = (
    "Some tool calls failed, or no tools were called. Read each error and hint, fix the "
    "arguments, and call only the tools that still need to run. Do not repeat calls that "
    "already succeeded."
)


def _compact(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------
# Auto-analysis node 1: profile (no LLM)
# --------------------------------------------------------------------------

def profile_node(state: AgentState, store: DatasetStore) -> dict[str, Any]:
    """Auto-analysis Node 1: Runs profile_dataset and validate_data without LLM."""
    dataset_id = state["dataset_id"]
    prof_res = profile_dataset(store, dataset_id)
    val_res = validate_data(store, dataset_id)

    profile_data = {
        "profile": prof_res.data if prof_res.ok else {},
        "validation": val_res.data if val_res.ok else {},
    }

    return {
        "profile": profile_data,
        "trace": [
            {
                "step": "profile",
                "tool": "profile_dataset",
                "args": {"dataset_id": dataset_id},
                "ok": prof_res.ok,
                "summary": prof_res.summary or prof_res.error,
            },
            {
                "step": "validate",
                "tool": "validate_data",
                "args": {"dataset_id": dataset_id},
                "ok": val_res.ok,
                "summary": val_res.summary or val_res.error,
            },
        ],
    }


# --------------------------------------------------------------------------
# Helpers for reading the message history
# --------------------------------------------------------------------------

def _last_ai_message(messages: list[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _latest_batch_start(messages: list[Any]) -> int:
    """Index of the first message after the most recent AI message."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            return index + 1
    return 0


def _latest_tool_messages(messages: list[Any]) -> list[ToolMessage]:
    start = _latest_batch_start(messages)
    return [m for m in messages[start:] if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------
# Auto-analysis node 2: plan (LLM call)
# --------------------------------------------------------------------------

def plan_node(state: AgentState, llm: Any, tools: list[Any]) -> dict[str, Any]:
    """Auto-analysis Node 2: LLM decides what analyses to run based on the profile.

    On a retry it continues the same conversation, so the model sees the errors
    and hints from the calls that failed.
    """
    llm_with_tools = llm.bind_tools(tools)
    history = list(state.get("messages", []))
    new_messages: list[Any] = []

    if not history:
        profile_text = _compact(state.get("profile", {}))
        if len(profile_text) > MAX_PROFILE_CHARS:
            profile_text = profile_text[:MAX_PROFILE_CHARS] + "...[truncated]"
        opener = HumanMessage(content=PLAN_PROMPT.format(profile=profile_text))
        new_messages.append(opener)
        history = [opener]
    else:
        nudge = HumanMessage(content=RETRY_PROMPT)
        new_messages.append(nudge)
        history = history + [nudge]

    response = invoke_with_retry(llm_with_tools, [SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + history)
    new_messages.append(response)
    return {"messages": new_messages}


# --------------------------------------------------------------------------
# Auto-analysis node 3: run the tools the LLM asked for (no LLM)
# --------------------------------------------------------------------------

def _safe_execute(execute_tool: ToolExecutor, name: str, args: dict, dataset_id: str) -> ToolResult:
    try:
        result = execute_tool(name, args, dataset_id)
    except Exception as exc:  # a crashing tool must not crash the graph
        return error_result(
            f"Tool '{name}' failed unexpectedly: {type(exc).__name__}: {exc}",
            hint="Check the arguments and try again.",
        )
    if not isinstance(result, ToolResult):
        return error_result(f"Tool '{name}' returned an unexpected result.")
    return result


def _text_for_llm(result: ToolResult) -> tuple[str, str | None]:
    """The text the LLM sees, and the chart JSON (kept out of that text)."""
    data = dict(result.data)
    chart = data.pop("chart_json", None)
    if chart is not None:
        data["chart_json"] = CHART_PLACEHOLDER
        if not isinstance(chart, str):
            chart = json.dumps(chart, default=str)

    payload: dict[str, Any] = {"ok": result.ok, "summary": result.summary}
    if data:
        payload["data"] = data
    if result.error:
        payload["error"] = result.error
    if result.hint:
        payload["hint"] = result.hint

    text = _compact(payload)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + "...[truncated]"
    return text, (chart if result.ok else None)


def run_tools_node(state: AgentState, execute_tool: ToolExecutor) -> dict[str, Any]:
    """Auto-analysis Node 3: run every tool call in the last AI message.

    Each call gets a ToolMessage, even skipped or failed ones, because the model
    API requires a reply for every tool call it made.
    """
    messages = list(state.get("messages", []))
    last_ai = _last_ai_message(messages)
    calls = list(getattr(last_ai, "tool_calls", None) or [])
    dataset_id = state["dataset_id"]

    tool_messages: list[ToolMessage] = []
    new_trace: list[dict[str, Any]] = []
    new_charts: list[str] = []

    for index, call in enumerate(calls):
        name = call.get("name", "")
        args = dict(call.get("args") or {})
        if index >= MAX_CALLS_PER_STEP:
            result = error_result(
                "Skipped: too many tool calls in one step.",
                hint=f"Make at most {MAX_CALLS_PER_STEP} tool calls per step.",
            )
        else:
            result = _safe_execute(execute_tool, name, args, dataset_id)

        text, chart = _text_for_llm(result)
        if chart:
            new_charts.append(chart)
        tool_messages.append(
            ToolMessage(
                content=text,
                tool_call_id=call.get("id", f"call_{index}"),
                name=name,
                status="success" if result.ok else "error",
            )
        )
        new_trace.append(
            {
                "step": "run_tools",
                "tool": name,
                "args": args,
                "ok": result.ok,
                "summary": result.summary if result.ok else result.error,
            }
        )

    if not calls:
        new_trace.append(
            {"step": "run_tools", "tool": None, "args": {}, "ok": False,
             "summary": "The model made no tool calls."}
        )

    return {
        "messages": tool_messages,
        "trace": list(state.get("trace", [])) + new_trace,
        "charts": list(state.get("charts", [])) + new_charts,
        "rounds": state.get("rounds", 0) + 1,
    }


def route_after_tools(state: AgentState) -> str:
    """Retry planning once if any call failed or none were made; otherwise summarize."""
    if state.get("rounds", 0) >= MAX_PLAN_ATTEMPTS:
        return "summarize"
    batch = _latest_tool_messages(list(state.get("messages", [])))
    if not batch or any(m.status == "error" for m in batch):
        return "plan"
    return "summarize"


# --------------------------------------------------------------------------
# Auto-analysis node 4: summarize (LLM call)
# --------------------------------------------------------------------------

def _data_quality_text(profile: dict[str, Any]) -> str:
    issues = (profile.get("validation") or {}).get("issues", [])[:MAX_ISSUES_IN_PROMPT]
    if not issues:
        return "No data quality issues were found."
    lines = []
    for issue in issues:
        where = f" in '{issue['column']}'" if issue.get("column") else ""
        lines.append(f"- {issue['severity']}: {issue['type']}{where}: {issue['detail']}")
    return "\n".join(lines)


def _shape_text(profile: dict[str, Any]) -> str:
    info = profile.get("profile") or {}
    if "rows" in info and "columns" in info:
        return f"{info['rows']} rows x {info['columns']} columns"
    return "size unknown"


def _fallback_findings(state: AgentState) -> str:
    """Findings built without the LLM, used when no analysis produced a result."""
    profile = state.get("profile") or {}
    return (
        "No analyses could be completed, so there are no analytical findings yet.\n\n"
        f"Dataset: {_shape_text(profile)}.\n\n"
        f"Data quality:\n{_data_quality_text(profile)}\n\n"
        "Try asking a specific question in the chat."
    )


def summarize_node(state: AgentState, llm: Any) -> dict[str, Any]:
    """Auto-analysis Node 4: Turns tool outputs into findings and follow-up questions."""
    messages = list(state.get("messages", []))
    latest_start = _latest_batch_start(messages)

    ok_blocks: list[str] = []
    failed_blocks: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        if message.status == "error":
            if index >= latest_start:  # failures that were fixed on a retry are dropped
                failed_blocks.append(f"Tool: {message.name}\nError: {message.content}")
        else:
            ok_blocks.append(f"Tool: {message.name}\nResult: {message.content}")

    if not ok_blocks:
        return {"findings": _fallback_findings(state)}

    profile = state.get("profile") or {}
    prompt = (
        f"Dataset: {_shape_text(profile)}.\n\n"
        f"Data quality issues:\n{_data_quality_text(profile)}\n\n"
        "Tool results:\n" + "\n\n".join(ok_blocks)
    )
    if failed_blocks:
        prompt += (
            "\n\nThese analyses could not be completed. Say so briefly and do not guess "
            "their results:\n" + "\n\n".join(failed_blocks)
        )
    prompt += "\n\nWrite the findings now."

    response = invoke_with_retry(
        llm, [SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {"findings": message_text(response)}


# --------------------------------------------------------------------------
# Chat graph nodes (unchanged from the previous version)
# --------------------------------------------------------------------------

def chat_agent_node(state: AgentState, llm_with_tools: Any) -> dict[str, Any]:
    """Chat Node: Agent inspects history + profile and decides next tool or answers."""
    messages = list(state.get("messages", []))

    if not messages or not isinstance(messages[0], SystemMessage):
        prof_snippet = json.dumps(state.get("profile", {}), indent=2)
        sys_content = f"{CHAT_SYSTEM_PROMPT}\n\nDataset Profile:\n{prof_snippet}"
        messages = [SystemMessage(content=sys_content)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def chat_count_round_node(state: AgentState) -> dict[str, Any]:
    """Chat Node: Increments round count and updates execution trace."""
    current_rounds = state.get("rounds", 0) + 1
    new_trace = list(state.get("trace", []))
    new_charts = list(state.get("charts", []))

    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage):
            try:
                res_dict = json.loads(msg.content)
                if isinstance(res_dict, dict) and res_dict.get("data", {}).get("chart_json"):
                    chart_json = res_dict["data"]["chart_json"]
                    if chart_json not in new_charts:
                        new_charts.append(chart_json)
            except Exception:
                pass

    return {
        "rounds": current_rounds,
        "trace": new_trace,
        "charts": new_charts,
    }