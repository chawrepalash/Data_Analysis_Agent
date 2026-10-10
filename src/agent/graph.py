"""Builds the auto-analysis graph and the chat graph.

Auto-analysis:

    START -> profile_data -> plan -> run_tools --(a call failed, retry left)--> plan
                                         |
                                         +--(otherwise)--> summarize -> END

The graph takes the tools and the executor as arguments, so it can be tested
with a fake LLM and fake tools, and wired to Gemini and the real registry in the app.
"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import END, START, StateGraph

from src.agent.llm import message_text
from src.agent.nodes import (
    EMPTY_ANSWER,
    ToolExecutor,
    chat_agent_node,
    plan_node,
    profile_node,
    route_after_chat_agent,
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
    graph.add_conditional_edges("run_tools", route_after_tools, {"plan": "plan", "summarize": "summarize"})
    graph.add_edge("summarize", END)
    return graph.compile()


def run_auto_analysis(graph, dataset_id: str) -> AgentState:
    """Run the graph on a dataset and return the final state."""
    return graph.invoke(new_state(dataset_id))


# --------------------------------------------------------------------------
# Chat graph
# --------------------------------------------------------------------------

def build_chat_graph(llm: Any, tools: list[Any], execute_tool: ToolExecutor):
    """Compile the chat graph: chat_agent <-> run_tools until the model answers."""

    def chat_agent_step(state: AgentState):
        return chat_agent_node(state, llm, tools)

    def run_tools_step(state: AgentState):
        return run_tools_node(state, execute_tool)

    graph = StateGraph(AgentState)
    graph.add_node("chat_agent", chat_agent_step)
    graph.add_node("run_tools", run_tools_step)
    graph.add_edge(START, "chat_agent")
    graph.add_conditional_edges("chat_agent", route_after_chat_agent, {"run_tools": "run_tools", "end": END})
    graph.add_edge("run_tools", "chat_agent")
    return graph.compile()


def compute_profile(store: DatasetStore, dataset_id: str) -> dict[str, Any]:
    """Profile and validation for a dataset, without an LLM. Compute once per dataset."""
    return profile_node(new_state(dataset_id), store)["profile"]


@dataclass
class ChatTurn:
    """The result of one question."""
    answer: str
    history: list[Any]                       # pass back in for the next question
    trace: list[dict[str, Any]] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)


def run_chat_turn(graph, dataset_id: str, profile: dict[str, Any], history: list[Any], question: str) -> ChatTurn:
    """Answer one question.

    history is the earlier questions and answers (plain text only). Tool calls and
    results are used within a turn but not carried to the next one, which keeps the
    prompt small. Trace and charts are for this question only.
    """
    state = new_state(dataset_id)
    state["profile"] = profile
    state["messages"] = list(history) + [HumanMessage(content=question)]
    final = graph.invoke(state)

    last = next((m for m in reversed(final["messages"]) if isinstance(m, AIMessage)), None)
    answer = message_text(last).strip() if last is not None else ""
    answer = answer or EMPTY_ANSWER

    new_history = list(history) + [HumanMessage(content=question), AIMessage(content=answer)]
    return ChatTurn(answer=answer, history=new_history, trace=final["trace"], charts=final["charts"])