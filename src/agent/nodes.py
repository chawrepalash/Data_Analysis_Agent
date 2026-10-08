"""Graph nodes for auto-analysis and chat graphs."""

import json
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agent.prompts import CHAT_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, SUMMARIZER_SYSTEM_PROMPT
from src.agent.state import AgentState
from src.loader import DatasetStore
from src.tools.profile import profile_dataset, validate_data


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
                "summary": prof_res.summary,
            },
            {
                "step": "validate",
                "tool": "validate_data",
                "args": {"dataset_id": dataset_id},
                "ok": val_res.ok,
                "summary": val_res.summary,
            },
        ],
    }


def plan_node(state: AgentState, llm: Any, tools: list[Any]) -> dict[str, Any]:
    """Auto-analysis Node 2: LLM decides what analyses to run based on profile."""
    llm_with_tools = llm.bind_tools(tools)
    profile_summary = json.dumps(state["profile"], indent=2)

    prompt = (
        f"Here is the dataset profile and data quality issues:\n{profile_summary}\n\n"
        "Plan and execute 2 to 4 high-value analyses (descriptive statistics, correlations, or group aggregates) "
        "and charts to give a comprehensive overview of this dataset."
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def summarize_node(state: AgentState, llm: Any) -> dict[str, Any]:
    """Auto-analysis Node 4: Turns tool outputs into findings and follow-up questions."""
    history_snippets = []
    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage):
            history_snippets.append(f"Tool Output: {msg.content}")

    prompt = (
        "Based on these executed tool results, summarize the key insights:\n\n"
        + "\n".join(history_snippets)
    )

    messages = [
        SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return {"findings": response.content}


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
