"""Tests for the chat graph. A scripted fake LLM and fake tools stand in for Gemini and
the analysis tools, so no API calls are made."""

import itertools

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.graph import build_chat_graph, compute_profile, run_chat_turn
from src.agent.nodes import CHART_PLACEHOLDER, EMPTY_ANSWER, MAX_CHAT_ROUNDS
from src.loader import DatasetStore
from src.schemas import error_result, ok_result

CHART = '{"data":[1,2,3]}'
_ids = itertools.count(1)


class ScriptedLLM:
    """Replays scripted replies. Records each call and whether tools were bound to it."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []        # list of (messages, tools_bound)

    def bind_tools(self, tools):
        return _Bound(self)

    def invoke(self, messages, **kwargs):
        return self._reply(messages, bound=False)

    def _reply(self, messages, bound):
        self.calls.append((list(messages), bound))
        return self.replies.pop(0)


class _Bound:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, messages, **kwargs):
        return self.parent._reply(messages, bound=True)


def tool_calls(*specs):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": n, "args": a, "id": f"call_{next(_ids)}", "type": "tool_call"} for n, a in specs
        ],
    )


class FakeTools:
    def __init__(self):
        self.calls = []

    def __call__(self, name, args, dataset_id):
        self.calls.append((name, args, dataset_id))
        if args.get("group_by") == ["bogus"]:
            return error_result("Unknown column(s): bogus.", hint="Valid columns: region, revenue")
        if name == "create_chart":
            return ok_result("Bar chart made.", chart_json=CHART)
        return ok_result("North mean revenue is 123.", rows=[{"region": "North", "revenue_mean": 123}])


def make_store():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"region": rng.choice(["North", "South"], 40), "revenue": rng.uniform(10, 200, 40).round(2)})
    store = DatasetStore()
    return store, store.add(df, "sales")


def setup(replies):
    store, dataset_id = make_store()
    llm = ScriptedLLM(replies)
    tools = FakeTools()
    graph = build_chat_graph(llm, [], tools)
    profile = compute_profile(store, dataset_id)
    return graph, llm, tools, dataset_id, profile


def ask(setup_result, question, history=None):
    graph, llm, tools, dataset_id, profile = setup_result
    return run_chat_turn(graph, dataset_id, profile, history or [], question)


GA = ("group_aggregate", {"group_by": ["region"], "metrics": [{"column": "revenue"}]})


def test_compute_profile_has_profile_and_validation():
    *_, profile = setup([])
    assert profile["profile"]["rows"] == 40 and "validation" in profile


def test_question_answered_without_tools():
    s = setup([AIMessage(content="Hello! Ask me about revenue.")])
    turn = ask(s, "hi")
    assert turn.answer == "Hello! Ask me about revenue."
    assert s[2].calls == [] and turn.trace == [] and turn.charts == []


def test_tool_then_answer_and_dataset_id_comes_from_state():
    s = setup([tool_calls(GA), AIMessage(content="North averages 123.")])
    turn = ask(s, "average revenue by region?")
    assert turn.answer == "North averages 123."
    assert [c[2] for c in s[2].calls] == [s[3]]
    assert [t["tool"] for t in turn.trace] == ["group_aggregate"] and turn.trace[0]["ok"]
    assert len(s[1].calls) == 2


def test_system_prompt_has_rules_and_profile_and_the_question_comes_last():
    s = setup([AIMessage(content="ok")])
    ask(s, "what is in the data?")
    messages, bound = s[1].calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert "Every number" in messages[0].content and '"rows":40' in messages[0].content
    assert isinstance(messages[-1], HumanMessage) and messages[-1].content == "what is in the data?"
    assert bound is True


def test_chart_is_collected_and_kept_out_of_what_the_llm_sees():
    s = setup([tool_calls(("create_chart", {"chart_type": "bar", "x": "region"})), AIMessage(content="Here.")])
    turn = ask(s, "chart it")
    assert turn.charts == [CHART]
    second_call_text = "\n".join(str(m.content) for m in s[1].calls[1][0])
    assert CHART not in second_call_text and CHART_PLACEHOLDER in second_call_text


def test_failed_tool_call_is_seen_by_the_model_and_can_be_fixed():
    s = setup(
        [
            tool_calls(("group_aggregate", {"group_by": ["bogus"], "metrics": [{"column": "revenue"}]})),
            tool_calls(GA),
            AIMessage(content="Fixed."),
        ]
    )
    turn = ask(s, "q")
    assert turn.answer == "Fixed."
    assert [t["ok"] for t in turn.trace] == [False, True]
    errors = [m for m in s[1].calls[1][0] if isinstance(m, ToolMessage)]
    assert errors[0].status == "error" and "Valid columns" in errors[0].content


def test_tool_budget_forces_a_final_answer_without_tools():
    replies = [tool_calls(GA) for _ in range(MAX_CHAT_ROUNDS)] + [AIMessage(content="Best I can say.")]
    s = setup(replies)
    turn = ask(s, "q")
    assert turn.answer == "Best I can say."
    assert len(s[2].calls) == MAX_CHAT_ROUNDS
    assert [bound for _, bound in s[1].calls] == [True] * MAX_CHAT_ROUNDS + [False]
    assert "budget" in s[1].calls[-1][0][-1].content


def test_history_carries_plain_questions_and_answers_only():
    s = setup([tool_calls(GA), AIMessage(content="A1."), AIMessage(content="A2.")])
    first = ask(s, "Q1")
    assert [type(m) for m in first.history] == [HumanMessage, AIMessage]

    second = ask(s, "Q2", first.history)
    sent = s[1].calls[2][0]
    assert [m.content for m in sent if isinstance(m, (HumanMessage, AIMessage))] == ["Q1", "A1.", "Q2"]
    assert not any(isinstance(m, ToolMessage) for m in sent)
    assert second.trace == [] and second.charts == []  # per question, not accumulated
    assert [m.content for m in second.history] == ["Q1", "A1.", "Q2", "A2."]


def test_list_style_and_empty_replies():
    s = setup([AIMessage(content=[{"type": "text", "text": "Plain."}])])
    assert ask(s, "q").answer == "Plain."
    s = setup([AIMessage(content="")])
    assert ask(s, "q").answer == EMPTY_ANSWER


def test_chat_with_the_real_registry_end_to_end():
    from src.tools.registry import get_tools, make_executor

    store, dataset_id = make_store()
    llm = ScriptedLLM(
        [
            tool_calls(
                ("group_aggregate", {"group_by": ["region"], "metrics": [{"column": "revenue"}]}),
                ("create_chart", {"chart_type": "bar", "x": "region", "y": "revenue"}),
            ),
            AIMessage(content="Done."),
        ]
    )
    graph = build_chat_graph(llm, get_tools(), make_executor(store))
    turn = run_chat_turn(graph, dataset_id, compute_profile(store, dataset_id), [], "avg revenue by region, chart it")
    assert turn.answer == "Done." and len(turn.charts) == 1
    assert [t["ok"] for t in turn.trace] == [True, True]
