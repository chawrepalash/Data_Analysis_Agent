"""Builds the auto-analysis graph.

    START -> profile_data -> plan -> run_tools --(a call failed, retry left)--> plan
                                         |
                                         +--(otherwise)--> summarize -> END

The graph takes the tools and the executor as arguments, so it can be tested
with a fake LLM and fake tools, and wired to Gemini and the real registry in the app.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    ToolExecutor,
    plan_node,
    profile_node,
    route_after_tools,
    run_tools_node,
    summarize_node,
)
from src.agent.state import AgentState
from src.loader import DatasetStore


def new_state(dataset_id: str) -> AgentState:
    return {
        "dataset_id": dataset_id,
        "profile": {},
        "messages": [],
        "trace": [],
        "charts": [],
        "rounds": 0,
        "findings": "",
    }


def build_auto_graph(store: DatasetStore, llm: Any, tools: list[Any], execute_tool: ToolExecutor):
    """Compile the auto-analysis graph.

    store         where the dataframes live (used by the profile step)
    llm           a chat model; plan binds the tools to it, summarize calls it as is
    tools         tool schemas handed to llm.bind_tools
    execute_tool  runs one tool: (name, args, dataset_id) -> ToolResult
    """

    # Closures, not partials: LangGraph injects arguments named "store" or "config"
    # into node functions, and we do not want that.
    def profile_step(state: AgentState):
        return profile_node(state, store)

    def plan_step(state: AgentState):
        return plan_node(state, llm, tools)

    def run_tools_step(state: AgentState):
        return run_tools_node(state, execute_tool)

    def summarize_step(state: AgentState):
        return summarize_node(state, llm)

    # Node names must differ from state keys, so "profile" is called "profile_data".
    graph = StateGraph(AgentState)
    graph.add_node("profile_data", profile_step)
    graph.add_node("plan", plan_step)
    graph.add_node("run_tools", run_tools_step)
    graph.add_node("summarize", summarize_step)

    graph.add_edge(START, "profile_data")
    graph.add_edge("profile_data", "plan")
    graph.add_edge("plan", "run_tools")
    graph.add_conditional_edges("run_tools", route_after_tools, {
                                "plan": "plan", "summarize": "summarize"})
    graph.add_edge("summarize", END)
    return graph.compile()


def run_auto_analysis(graph, dataset_id: str) -> AgentState:
    """Run the graph on a dataset and return the final state."""
    return graph.invoke(new_state(dataset_id))
