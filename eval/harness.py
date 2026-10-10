"""Scoring helpers for the eval. No LLM and no I/O here, so they are easy to test."""

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
REFUSAL_WORDS = (
    "not contain", "does not", "doesn't", "no column", "not available", "cannot", "can't",
    "unable", "not in the dataset", "no information", "isn't", "not present", "no such",
    "don't have", "do not have", "not found",
)


@dataclass
class Case:
    id: str
    question: str
    truth: Callable[[pd.DataFrame], list[float]] = lambda df: []   # numbers a right answer must contain
    any_tool: tuple[str, ...] = ()       # at least one of these tools must have run successfully
    needs_chart: bool = False
    refusal: bool = False                # the data cannot answer: the agent must say so
    note: str = ""


@dataclass
class Outcome:
    case_id: str
    passed: bool
    reasons: list[str] = field(default_factory=list)   # why it failed (empty if it passed)
    tools: list[str] = field(default_factory=list)
    seconds: float = 0.0
    answer: str = ""
    error: str | None = None


def numbers_in(text: str) -> list[float]:
    """Every number in a text, with thousands commas removed. '96.81%' gives 96.81."""
    out = []
    for raw in NUMBER.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def number_present(expected: float, found: list[float]) -> bool:
    """True if the expected value appears in the found numbers, allowing for rounding.

    - A whole-number count (an int) must match exactly.
    - A rate between -1 and 1 also matches as a percentage: 0.9681 matches 0.97, 96.81 or 97
      (percentages may be off by half a point, since models round).
    - A negative value also matches its absolute value ("a correlation of 0.55").
    - Any other number may differ by 1.5%.
    """
    if isinstance(expected, int) and not isinstance(expected, bool):
        return any(abs(x - expected) < 0.01 for x in found)

    candidates = [(expected, max(abs(expected) * 0.015, 0.006))]
    if -1.0 <= expected <= 1.0 and expected != 0:
        candidates = [(expected, max(abs(expected) * 0.01, 0.006)), (expected * 100, 0.5)]
    if expected < 0:
        candidates += [(-value, tol) for value, tol in candidates]
    return any(abs(x - value) <= tol for value, tol in candidates for x in found)


def check_case(case: Case, df: pd.DataFrame, answer: str, tools_ok: list[str], charts: int) -> list[str]:
    """Return the reasons the answer is wrong. An empty list means it passed."""
    reasons = []
    found = numbers_in(answer)

    if case.refusal:
        lowered = answer.lower()
        if not any(word in lowered for word in REFUSAL_WORDS):
            reasons.append("did not say the data cannot answer this")
        return reasons

    for expected in case.truth(df):
        if not number_present(expected, found):
            reasons.append(f"missing expected number {expected:.4g}")
    if case.any_tool and not any(tool in tools_ok for tool in case.any_tool):
        reasons.append(f"did not use any of: {', '.join(case.any_tool)}")
    if case.needs_chart and charts == 0:
        reasons.append("no chart was created")
    return reasons


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    scored = [o for o in outcomes if o.error is None]
    passed = sum(o.passed for o in scored)
    return {
        "total": len(outcomes),
        "scored": len(scored),
        "errors": len(outcomes) - len(scored),
        "passed": passed,
        "accuracy": round(passed / len(scored), 3) if scored else 0.0,
        "avg_seconds": round(sum(o.seconds for o in scored) / len(scored), 1) if scored else 0.0,
        "avg_tool_calls": round(sum(len(o.tools) for o in scored) / len(scored), 2) if scored else 0.0,
    }
