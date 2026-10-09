"""Tests for the tool registry and executor.

Sections 1 and 2 use a recording fake in place of the real tools, so they check only
the registry's own behaviour. Section 3 runs the REAL tools through the registry on a
small sales-like frame, so it also checks that the registry matches the tool functions.
"""

import dataclasses
import itertools
import json

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from src.agent.graph import build_auto_graph, run_auto_analysis
from src.loader import DatasetStore
from src.schemas import ok_result
from src.tools.registry import TOOLS, get_tools, make_executor

EXPECTED_TOOLS = {"describe_stats", "correlation_analysis", "detect_outliers", "create_chart"}
_ids = itertools.count(1)


def make_store():
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame(
        {
            "region": rng.choice(["North", "South", "East"], n),
            "units": rng.integers(1, 20, n),
            "revenue": rng.uniform(10, 500, n).round(2),
        }
    )
    store = DatasetStore()
    return store, store.add(df, "sales")


@pytest.fixture
def recorder(monkeypatch):
    """Replace every real tool with a fake that records how it was called."""
    calls = []

    def fake(store, dataset_id, **kwargs):
        calls.append({"store": store, "dataset_id": dataset_id, "kwargs": kwargs})
        return ok_result("fake")

    for name, spec in list(TOOLS.items()):
        monkeypatch.setitem(TOOLS, name, dataclasses.replace(spec, func=fake))
    return calls


# ============================ 1. the tool schemas ============================

def test_registry_has_the_expected_tools():
    tools = get_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(isinstance(t, StructuredTool) for t in tools)


def test_every_tool_has_a_useful_description():
    for tool in get_tools():
        assert len(tool.description) > 30, tool.name


def test_no_schema_exposes_dataset_id_or_store():
    for spec in TOOLS.values():
        properties = spec.args_model.model_json_schema()["properties"]
        assert "dataset_id" not in properties and "store" not in properties, spec.name


def test_required_arguments():
    assert TOOLS["detect_outliers"].args_model.model_json_schema()["required"] == ["column"]
    assert set(TOOLS["create_chart"].args_model.model_json_schema()["required"]) == {"chart_type", "x"}
    assert "required" not in TOOLS["describe_stats"].args_model.model_json_schema()


def test_tools_convert_to_gemini_declarations(monkeypatch):
    pytest.importorskip("langchain_google_genai")
    from langchain_google_genai import ChatGoogleGenerativeAI

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # no network call is made
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key="test-key")
    llm.bind_tools(get_tools())  # raises if a schema cannot be converted


# ====================== 2. the executor, with fake tools ======================

def test_executor_passes_store_and_dataset_id_from_the_graph(recorder):
    store, _ = make_store()
    result = make_executor(store)("describe_stats", {"columns": ["revenue"]}, "abc123")
    assert result.ok
    assert recorder[0]["dataset_id"] == "abc123"
    assert recorder[0]["store"] is store
    assert recorder[0]["kwargs"] == {"columns": ["revenue"]}


def test_arguments_the_model_left_out_are_not_passed(recorder):
    make_executor(DatasetStore())("describe_stats", {}, "abc")
    assert recorder[0]["kwargs"] == {}


def test_all_arguments_pass_through(recorder):
    args = {"column": "revenue", "method": "zscore", "threshold": 2.5}
    make_executor(DatasetStore())("detect_outliers", args, "abc")
    assert recorder[0]["kwargs"] == args


def test_a_dataset_id_sent_by_the_model_is_ignored(recorder):
    make_executor(DatasetStore())("describe_stats", {"dataset_id": "evil", "columns": ["a"]}, "real")
    assert recorder[0]["dataset_id"] == "real"
    assert recorder[0]["kwargs"] == {"columns": ["a"]}


def test_unknown_tool_lists_the_valid_ones(recorder):
    result = make_executor(DatasetStore())("make_coffee", {}, "abc")
    assert not result.ok and "make_coffee" in result.error
    assert "describe_stats" in result.hint
    assert recorder == []


def test_unknown_argument_is_rejected_with_the_valid_names(recorder):
    result = make_executor(DatasetStore())("detect_outliers", {"column": "a", "colour": "red"}, "abc")
    assert not result.ok and "colour" in result.error
    assert "threshold" in result.hint
    assert recorder == []


def test_bad_enum_value_is_rejected(recorder):
    result = make_executor(DatasetStore())("create_chart", {"chart_type": "pie", "x": "region"}, "abc")
    assert not result.ok and "chart_type" in result.error
    assert recorder == []


def test_missing_required_argument_is_rejected(recorder):
    result = make_executor(DatasetStore())("detect_outliers", {}, "abc")
    assert not result.ok and "column" in result.error
    assert recorder == []


def test_wrong_type_is_rejected(recorder):
    result = make_executor(DatasetStore())("detect_outliers", {"column": "a", "threshold": "lots"}, "abc")
    assert not result.ok and "threshold" in result.error
    assert recorder == []


def test_arguments_must_be_an_object(recorder):
    assert not make_executor(DatasetStore())("describe_stats", ["revenue"], "abc").ok
    assert recorder == []


# ==================== 3. the real tools, through the registry ====================

def test_real_describe_stats():
    store, dataset_id = make_store()
    result = make_executor(store)("describe_stats", {"columns": ["revenue"]}, dataset_id)
    assert result.ok, result.error
    assert result.data["columns"][0]["column"] == "revenue"


def test_real_correlation_analysis():
    store, dataset_id = make_store()
    assert make_executor(store)("correlation_analysis", {}, dataset_id).ok


def test_real_detect_outliers():
    store, dataset_id = make_store()
    result = make_executor(store)("detect_outliers", {"column": "revenue"}, dataset_id)
    assert result.ok, result.error


def test_real_create_chart_returns_figure_json_under_chart_json():
    store, dataset_id = make_store()
    args = {"chart_type": "bar", "x": "region", "y": "revenue", "aggregation": "mean"}
    result = make_executor(store)("create_chart", args, dataset_id)
    assert result.ok, result.error
    # The graph looks for the figure under data["chart_json"]. If this fails, the key differs.
    assert "chart_json" in result.data
    assert "data" in json.loads(result.data["chart_json"])


def test_real_tool_error_for_a_wrong_column_gives_a_hint_the_model_can_use():
    store, dataset_id = make_store()
    result = make_executor(store)("describe_stats", {"columns": ["nope"]}, dataset_id)
    assert not result.ok and result.hint


# ================= 4. whole graph, real registry, scripted LLM =================

class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        return self.replies.pop(0)


def tool_calls(*specs):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"c{next(_ids)}", "type": "tool_call"}
            for name, args in specs
        ],
    )


def test_graph_end_to_end_with_the_real_registry():
    store, dataset_id = make_store()
    llm = ScriptedLLM(
        [
            tool_calls(
                ("describe_stats", {"columns": ["revenue"]}),
                ("create_chart", {"chart_type": "bar", "x": "region", "y": "revenue", "aggregation": "mean"}),
            ),
            AIMessage(content="Summary."),
        ]
    )
    graph = build_auto_graph(store, llm, get_tools(), make_executor(store))
    final = run_auto_analysis(graph, dataset_id)

    assert final["findings"] == "Summary."
    assert len(final["charts"]) == 1
    assert [t["ok"] for t in final["trace"]] == [True, True, True, True]
