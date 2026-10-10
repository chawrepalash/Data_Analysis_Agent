"""Tests for the eval harness and cases. No API calls."""

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from eval import run_eval
from eval.cases import CASES
from eval.harness import Case, Outcome, check_case, number_present, numbers_in, summarize


def titanic_like(n=400):
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "Survived": rng.integers(0, 2, n),
            "Pclass": rng.integers(1, 4, n),
            "Sex": rng.choice(["male", "female"], n),
            "Age": np.where(rng.random(n) < 0.2, np.nan, rng.integers(1, 70, n).astype(float)),
            "Fare": rng.exponential(30, n).round(2),
            "Embarked": rng.choice(["S", "C", "Q"], n),
        }
    )


# ------------------------------- number matching -------------------------------

def test_numbers_in_handles_commas_percent_and_negatives():
    assert numbers_in("1,234 people; 96.81% survived; r = -0.55.") == [1234.0, 96.81, -0.55]
    assert numbers_in("no digits here") == []


def test_rates_match_as_decimals_or_percentages():
    assert number_present(0.9681, [96.81])
    assert number_present(0.9681, [97])
    assert number_present(0.9681, [0.97])
    assert number_present(0.1889, [19])          # "19%" for 18.89%
    assert not number_present(0.1889, [25])


def test_counts_must_match_exactly():
    assert number_present(891, [891.0])
    assert not number_present(891, [894.0])


def test_negative_values_match_their_absolute_value():
    assert number_present(-0.5495, [-0.55])
    assert number_present(-0.5495, [0.55])
    assert not number_present(-0.5495, [0.9])


def test_other_numbers_allow_small_rounding():
    assert number_present(29.699, [29.7]) and number_present(29.699, [30])
    assert not number_present(29.699, [35])


# ------------------------------- scoring a case -------------------------------

def test_check_case_passes_and_fails_on_numbers():
    df = titanic_like()
    case = Case("c", "q", lambda d: [len(d)], any_tool=("describe_stats",))
    assert check_case(case, df, "There are 400 rows.", ["describe_stats"], 0) == []
    reasons = check_case(case, df, "There are 300 rows.", ["describe_stats"], 0)
    assert any("missing expected number 400" in r for r in reasons)


def test_check_case_requires_the_tool_and_the_chart():
    df = titanic_like()
    case = Case("c", "q", any_tool=("create_chart",), needs_chart=True)
    reasons = check_case(case, df, "Done.", [], 0)
    assert len(reasons) == 2
    assert check_case(case, df, "Done.", ["create_chart"], 1) == []


def test_refusal_cases_need_a_refusal_not_a_number():
    df = titanic_like()
    case = Case("r", "q", refusal=True)
    assert check_case(case, df, "The dataset does not contain ticket prices on Mars.", [], 0) == []
    assert check_case(case, df, "The average ticket price on Mars is 42.", [], 0)


def test_summarize_ignores_errors_in_accuracy():
    outcomes = [
        Outcome("a", True, tools=["x"], seconds=2),
        Outcome("b", False, ["wrong"], tools=["x", "y"], seconds=4),
        Outcome("c", False, error="503", seconds=1),
    ]
    summary = summarize(outcomes)
    assert summary["scored"] == 2 and summary["errors"] == 1 and summary["passed"] == 1
    assert summary["accuracy"] == 0.5 and summary["avg_seconds"] == 3.0 and summary["avg_tool_calls"] == 1.5


# ------------------------------ the case list itself ------------------------------

def test_case_list_is_well_formed():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)) and 15 <= len(ids) <= 25
    assert any(c.refusal for c in CASES) and any(c.needs_chart for c in CASES)


def test_every_truth_function_runs_and_gives_plain_numbers():
    df = titanic_like()
    for case in CASES:
        values = case.truth(df)
        assert all(isinstance(v, (int, float)) and not pd.isna(v) for v in values), case.id
        if not (case.refusal or case.needs_chart):
            assert values, f"{case.id} has nothing to check"


def test_cases_pass_when_the_answer_states_the_truth():
    df = titanic_like()
    for case in CASES:
        if case.refusal:
            continue
        answer = " and ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in case.truth(df))
        tools = list(case.any_tool)
        assert check_case(case, df, answer, tools, 1) == [], case.id


# ------------------------------ the runner ------------------------------

class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        return self.replies.pop(0)


def test_runner_scores_cases_and_writes_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_eval, "OUT_DIR", tmp_path / "eval")
    titanic_like().to_csv(tmp_path / "t.csv", index=False)
    llm = Scripted(
        [
            AIMessage(content="The dataset does not contain Mars prices."),    # refuse_mars: pass
            AIMessage(content="The average salary is 1000."),                  # refuse_salary: fail
        ]
    )
    sleeps = []
    summary = run_eval.main(str(tmp_path / "t.csv"), ["refuse_mars", "refuse_salary"], delay=0, llm=llm, sleep=sleeps.append)
    assert summary["passed"] == 1 and summary["scored"] == 2 and len(sleeps) == 1

    report = (tmp_path / "eval" / "RESULTS.md").read_text(encoding="utf-8")
    assert "1/2 correct (50%)" in report and "FAIL" in report and "pass" in report
    assert (tmp_path / "eval" / "results.json").exists()


def test_runner_counts_a_busy_api_as_an_error_not_a_failure(tmp_path, monkeypatch):
    import src.agent.llm as llm_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_eval, "OUT_DIR", tmp_path / "eval")
    monkeypatch.setitem(llm_module.invoke_with_retry.__kwdefaults__, "sleep", lambda s: None)
    titanic_like().to_csv(tmp_path / "t.csv", index=False)

    class Busy:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages, **kwargs):
            raise RuntimeError("503 UNAVAILABLE")

    summary = run_eval.main(str(tmp_path / "t.csv"), ["rows"], delay=0, llm=Busy(), sleep=lambda s: None)
    assert summary["errors"] == 1 and summary["scored"] == 0
