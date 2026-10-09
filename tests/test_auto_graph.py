"""Tests for the auto-analysis graph. A scripted fake LLM and fake tools stand in for
Gemini and the analysis tools, so no API calls are made."""

import itertools

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.graph import build_auto_graph, new_state, run_auto_analysis
from src.agent.nodes import (
    CHART_PLACEHOLDER,
    MAX_CALLS_PER_STEP,
    RETRY_PROMPT,
    run_tools_node,
)
from src.loader import DatasetStore
from src.schemas import error_result, ok_result

CHART = '{"data":[1,2,3]}'
_ids = itertools.count(1)


# ------------------------------- test doubles -------------------------------

class ScriptedLLM:
    """Returns the scripted replies in order and records every call it receives."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def tool_calls(*specs):
    """An AI message asking for tools. specs are (name, args) pairs."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args,
                "id": f"call_{next(_ids)}", "type": "tool_call"}
            for name, args in specs
        ],
    )


def describe(args):
    if "bogus" in args.get("columns", []):
        return error_result("Unknown column(s): bogus.", hint="Valid columns: region, revenue")
    if "explode" in args.get("columns", []):
        raise RuntimeError("boom")
    return ok_result("Revenue mean is 100.", mean=100)


def chart(args):
    return ok_result("Bar chart of revenue by region.", chart_json=CHART)


class FakeTools:
    """Executor with the same shape as the registry's: (name, args, dataset_id) -> ToolResult."""

    def __init__(self):
        self.handlers = {"describe_stats": describe, "create_chart": chart}
        self.calls = []

    def __call__(self, name, args, dataset_id):
        self.calls.append((name, args, dataset_id))
        if name not in self.handlers:
            return error_result(f"Unknown tool '{name}'.", hint="Valid tools: " + ", ".join(self.handlers))
        return self.handlers[name](args)


def make_store(rows=40):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "region": rng.choice(["North", "South"], rows),
            "revenue": rng.uniform(10, 200, rows).round(2),
        }
    )
    store = DatasetStore()
    return store, store.add(df, "sales")


def run(replies, rows=40):
    store, dataset_id = make_store(rows)
    llm = ScriptedLLM(replies)
    tools = FakeTools()
    graph = build_auto_graph(store, llm, [], tools)
    return run_auto_analysis(graph, dataset_id), llm, tools, dataset_id


def text_of(messages):
    return "\n".join(str(m.content) for m in messages)


# ---------------------------------- happy path ----------------------------------

def test_happy_path():
    final, llm, tools, dataset_id = run(
        [
            tool_calls(("describe_stats", {"columns": [
                       "revenue"]}), ("create_chart", {"x": "region"})),
            AIMessage(content="Revenue averages 100."),
        ]
    )
    assert final["findings"] == "Revenue averages 100."
    assert [t["tool"] for t in final["trace"]] == [
        "profile_dataset", "validate_data", "describe_stats", "create_chart",
    ]
    assert all(t["ok"] for t in final["trace"])
    assert final["charts"] == [CHART]
    assert final["rounds"] == 1
    assert len(llm.calls) == 2  # one plan call, one summary call


def test_dataset_id_comes_from_state_not_from_the_llm():
    final, llm, tools, dataset_id = run(
        [tool_calls(("describe_stats", {"columns": ["revenue"]})), AIMessage(
            content="ok")]
    )
    assert [c[2] for c in tools.calls] == [dataset_id]


def test_plan_prompt_has_system_prompt_and_profile():
    final, llm, tools, dataset_id = run(
        [tool_calls(("describe_stats", {"columns": ["revenue"]})), AIMessage(
            content="ok")]
    )
    first_call = llm.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert isinstance(first_call[1], HumanMessage)
    assert '"rows":40' in first_call[1].content  # the real profile_dataset ran


def test_chart_json_is_kept_out_of_what_the_llm_sees():
    final, llm, tools, dataset_id = run(
        [tool_calls(("create_chart", {"x": "region"})),
         AIMessage(content="ok")]
    )
    seen_by_summarizer = text_of(llm.calls[1])
    assert CHART not in seen_by_summarizer
    assert CHART_PLACEHOLDER in seen_by_summarizer
    assert final["charts"] == [CHART]


def test_list_style_reply_becomes_plain_text():
    final, *_ = run(
        [
            tool_calls(("describe_stats", {"columns": ["revenue"]})),
            AIMessage(content=[{"type": "text", "text": "Findings here."}]),
        ]
    )
    assert final["findings"] == "Findings here."


# ------------------------------- retry behaviour -------------------------------

def test_failed_call_is_retried_once_and_the_model_sees_the_error():
    final, llm, tools, _ = run(
        [
            tool_calls(("describe_stats", {"columns": ["bogus"]})),
            tool_calls(("describe_stats", {"columns": ["revenue"]})),
            AIMessage(content="Done."),
        ]
    )
    assert len(llm.calls) == 3  # plan, plan again, summarize
    assert final["rounds"] == 2
    assert final["findings"] == "Done."
    assert [t["ok"] for t in final["trace"] if t["tool"]
            == "describe_stats"] == [False, True]

    second_plan = llm.calls[1]
    errors = [m for m in second_plan if isinstance(m, ToolMessage)]
    assert errors[0].status == "error" and "Valid columns" in errors[0].content
    assert any(isinstance(m, HumanMessage) and m.content ==
               RETRY_PROMPT for m in second_plan)

    summary_input = text_of(llm.calls[2])
    assert "bogus" not in summary_input  # the failure was fixed, so it is dropped
    assert "Revenue mean is 100" in summary_input


def test_failure_that_stays_unfixed_is_reported_to_the_summarizer():
    final, llm, tools, _ = run(
        [
            tool_calls(("describe_stats", {"columns": ["bogus"]})),
            tool_calls(
                ("describe_stats", {"columns": ["bogus"]}), ("create_chart", {"x": "region"})),
            AIMessage(content="Partial."),
        ]
    )
    assert len(llm.calls) == 3  # still stops after one retry
    assert final["rounds"] == 2
    summary_input = text_of(llm.calls[2])
    assert "could not be completed" in summary_input
    assert "Unknown column" in summary_input
    assert final["charts"] == [CHART]


def test_unknown_tool_name_is_an_error_the_model_can_fix():
    final, llm, tools, _ = run(
        [
            tool_calls(("make_coffee", {})),
            tool_calls(("describe_stats", {"columns": ["revenue"]})),
            AIMessage(content="ok"),
        ]
    )
    assert "Valid tools" in text_of(llm.calls[1])
    assert final["findings"] == "ok"


def test_a_tool_that_crashes_does_not_crash_the_graph():
    final, llm, tools, _ = run(
        [
            tool_calls(("describe_stats", {"columns": ["explode"]})),
            tool_calls(("describe_stats", {"columns": ["revenue"]})),
            AIMessage(content="ok"),
        ]
    )
    crashed = next(t for t in final["trace"] if t["tool"] == "describe_stats")
    assert crashed["ok"] is False and "boom" in crashed["summary"]
    assert final["findings"] == "ok"


# ---------------------------- the model gives no plan ----------------------------

def test_no_tool_calls_gives_a_fallback_without_inventing_findings():
    final, llm, tools, _ = run(
        [AIMessage(content="I cannot."), AIMessage(content="Still no.")], rows=40)
    assert len(llm.calls) == 2  # two plan attempts, and no summary call
    assert tools.calls == []
    assert "No analyses could be completed" in final["findings"]
    assert "40 rows x 2 columns" in final["findings"]
    assert final["trace"][-1]["summary"] == "The model made no tool calls."


def test_model_recovers_on_the_second_attempt_after_no_tool_calls():
    final, llm, tools, _ = run(
        [
            AIMessage(content="Let me think."),
            tool_calls(("describe_stats", {"columns": ["revenue"]})),
            AIMessage(content="Done."),
        ]
    )
    assert final["findings"] == "Done."
    assert len(llm.calls) == 3


# ------------------------------ run_tools_node alone ------------------------------

def test_extra_tool_calls_are_skipped_but_still_answered():
    store, dataset_id = make_store()
    ai = tool_calls(
        *[("describe_stats", {"columns": ["revenue"]})] * (MAX_CALLS_PER_STEP + 2))
    state = new_state(dataset_id)
    state["messages"] = [HumanMessage(content="go"), ai]
    tools = FakeTools()

    update = run_tools_node(state, tools)

    assert len(tools.calls) == MAX_CALLS_PER_STEP
    assert len(update["messages"]) == MAX_CALLS_PER_STEP + \
        2  # every call gets a reply
    assert [m.status for m in update["messages"]][-2:] == ["error", "error"]
    assert "too many" in update["messages"][-1].content


def test_every_tool_call_gets_a_matching_reply():
    store, dataset_id = make_store()
    ai = tool_calls(
        ("describe_stats", {"columns": ["revenue"]}), ("create_chart", {"x": "region"}))
    state = new_state(dataset_id)
    state["messages"] = [HumanMessage(content="go"), ai]
    update = run_tools_node(state, FakeTools())
    assert [m.tool_call_id for m in update["messages"]] == [c["id"]
                                                            for c in ai.tool_calls]
