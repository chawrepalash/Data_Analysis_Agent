"""Tests for the Streamlit app, run headless with Streamlit's AppTest and a fake LLM."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.ui_helpers import cleaning_log_rows, column_rows, figures_from_json, fix_label, issue_rows, trace_rows  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
_ids = itertools.count(1)


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
        tool_calls=[{"name": n, "args": a,
                     "id": f"c{next(_ids)}", "type": "tool_call"} for n, a in specs],
    )


@pytest.fixture
def sample_dir(tmp_path, monkeypatch):
    """A working directory holding a small messy titanic-like sample at data/titanic.csv."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "Pclass": rng.integers(1, 4, n),
            "Sex": rng.choice(["male", "female"], n),
            "Age": np.where(rng.random(n) < 0.2, np.nan, rng.integers(1, 70, n)),
            "Fare": rng.uniform(5, 100, n).round(2),
            "Survived": rng.integers(0, 2, n),
        }
    )
    (tmp_path / "data").mkdir()
    df.to_csv(tmp_path / "data" / "titanic.csv", index=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def fake_llm(monkeypatch):
    """Make the app use a scripted model. Returns a function that sets the replies."""
    import src.agent.llm as llm_module

    holder = {}
    monkeypatch.setattr(llm_module, "get_llm",
                        lambda: holder["llm"], raising=False)

    def script(*replies):
        holder["llm"] = ScriptedLLM(replies)

    return script


def start():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def load_sample(at):
    next(b for b in at.button if "Titanic" in b.label).click().run()
    assert not at.exception, at.exception


# ------------------------------- pure helpers -------------------------------

def test_trace_rows():
    rows = trace_rows(
        [
            {"step": "profile", "tool": "profile_dataset", "args": {
                "dataset_id": "x"}, "ok": True, "summary": "s"},
            {"step": "run_tools", "tool": "describe_stats", "args": {
                "columns": ["a"]}, "ok": False, "summary": "bad"},
            {"step": "run_tools", "tool": None, "args": {},
                "ok": False, "summary": "none"},
        ]
    )
    assert rows[0]["arguments"] == "" and rows[0]["status"] == "ok"
    assert rows[1]["arguments"] == "columns=['a']" and rows[1]["status"] == "failed"
    assert rows[2]["tool"] == "(no tool called)"


def test_cleaning_log_rows_finds_the_log_under_any_key():
    rows = [{"fix_id": "a"}]
    assert cleaning_log_rows(
        {"change_log": rows, "shape_before": [1, 2]}) == rows
    assert cleaning_log_rows({"log": rows}) == rows
    assert cleaning_log_rows({"applied": rows, "n": 1}) == rows
    assert cleaning_log_rows({"before_shape": [1, 2]}) == []
    assert cleaning_log_rows({}) == []


def test_fix_label_tolerates_missing_fields():
    assert fix_label({"id": "f1", "action": "drop_duplicates"}
                     ) == "**drop duplicates**"
    assert "(7 values)" in fix_label(
        {"action": "fill_median", "column": "Age", "affected": 7})
    assert "(0 values)" in fix_label({"action": "x", "values_affected": 0})


def test_fix_label_and_profile_rows():
    label = fix_label({"action": "fill_median", "column": "Age",
                      "reason": "19.9% missing", "values_affected": 177})
    assert "fill median" in label and "`Age`" in label and "177" in label
    profile = {
        "profile": {"column_info": [{"name": "a", "kind": "numeric", "missing": 1, "missing_pct": 2.0, "unique": 3}]},
        "validation": {"issues": [{"severity": "high", "type": "t", "column": None, "detail": "d"}]},
    }
    assert column_rows(profile)[0]["column"] == "a"
    assert issue_rows(profile)[0]["column"] == ""


def test_figures_from_json_survives_a_bad_chart():
    good = '{"data":[{"type":"bar","x":[1],"y":[2]}],"layout":{}}'
    results = figures_from_json([good, "not json"])
    assert results[0][0] is not None and results[1][0] is None and results[1][1]


# --------------------------------- the app ---------------------------------

def test_empty_state_prompts_for_a_file(sample_dir):
    at = start()
    assert any("Upload a CSV" in i.value for i in at.info)
    assert len(at.tabs) == 0


def test_sample_loads_and_overview_shows_the_shape(sample_dir):
    at = start()
    load_sample(at)
    assert [t.label for t in at.tabs] == [
        "Overview", "Clean", "Auto-analysis", "Chat"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Rows"] == "80" and metrics["Columns"] == "5"


def test_cleaning_needs_approval_and_resets_analysis(sample_dir, fake_llm):
    at = start()
    load_sample(at)
    # Nothing is applied just by viewing the Clean tab.
    assert at.session_state["cleaned"] is False
    boxes = [c for c in at.checkbox if c.key and c.key.startswith("fix_")]
    assert boxes, "the sample has missing ages, so a fix should be proposed"
    assert all(c.value is False for c in boxes)

    # Apply is disabled until something is ticked.
    apply_button = next(b for b in at.button if b.label ==
                        "Apply selected fixes")
    assert apply_button.disabled

    boxes[0].check().run()
    next(b for b in at.button if b.label ==
         "Apply selected fixes").click().run()
    assert not at.exception, at.exception
    assert at.session_state["cleaned"] is True
    assert at.session_state["clean_log"].data["change_log"], "the change log should be kept for display"
    assert at.session_state["auto"] is None and at.session_state["chat_log"] == [
    ]
    assert any("Reset to the original" in b.label for b in at.button)

    next(b for b in at.button if "Reset to the original" in b.label).click().run()
    assert at.session_state["cleaned"] is False


def test_auto_analysis_shows_findings_charts_and_trace(sample_dir, fake_llm):
    fake_llm(
        tool_calls(
            ("describe_stats", {"columns": ["Fare"]}),
            ("group_aggregate", {"group_by": [
             "Sex"], "metrics": [{"column": "Survived"}]}),
            ("create_chart", {"chart_type": "bar",
             "x": "Sex", "y": "Survived"}),
        ),
        AIMessage(content="Women survived more often."),
    )
    at = start()
    load_sample(at)
    next(b for b in at.button if b.label == "Run auto-analysis").click().run()
    assert not at.exception, at.exception
    assert any("Women survived more often." in m.value for m in at.markdown)
    assert at.session_state["auto"]["charts"], "a chart should have been created"


def test_auto_analysis_without_an_api_key_shows_a_friendly_error(sample_dir, monkeypatch):
    import src.agent.llm as llm_module

    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(llm_module, "get_llm", boom, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = start()
    load_sample(at)
    next(b for b in at.button if b.label == "Run auto-analysis").click().run()
    assert not at.exception, at.exception
    assert any("GOOGLE_API_KEY" in e.value for e in at.error)


def test_busy_gemini_shows_a_warning_not_a_crash(sample_dir, monkeypatch):
    import src.agent.llm as llm_module

    class Busy:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, **kwargs):
            raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(llm_module, "get_llm", lambda: Busy(), raising=False)
    monkeypatch.setitem(
        llm_module.invoke_with_retry.__kwdefaults__, "sleep", lambda s: None)
    at = start()
    load_sample(at)
    next(b for b in at.button if b.label == "Run auto-analysis").click().run()
    assert not at.exception, at.exception
    assert any("busy" in w.value for w in at.warning)


def test_chat_answers_and_keeps_history(sample_dir, fake_llm):
    fake_llm(
        tool_calls(("group_aggregate", {"group_by": [
                   "Sex"], "metrics": [{"column": "Survived"}]})),
        AIMessage(content="Answer one."),
        AIMessage(content="Answer two."),
    )
    at = start()
    load_sample(at)
    at.chat_input[0].set_value("Survival by sex?").run()
    assert not at.exception, at.exception
    log = at.session_state["chat_log"]
    assert [e["role"] for e in log] == ["user", "assistant"]
    assert log[1]["content"] == "Answer one." and log[1]["trace"]

    at.chat_input[0].set_value("And again?").run()
    assert [e["content"] for e in at.session_state["chat_log"]] == [
        "Survival by sex?", "Answer one.", "And again?", "Answer two.",
    ]
    assert len(at.session_state["history"]) == 4
