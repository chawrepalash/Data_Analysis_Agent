"""Run the eval: ask each question in eval/cases.py to the chat agent and score the answers.

    python -m eval.run_eval data/titanic.csv
    python -m eval.run_eval data/titanic.csv --only rows,rate_by_sex --delay 8

Each question starts a fresh conversation. Results go to eval/results.json and
eval/RESULTS.md. Gemini errors (busy, rate-limited) are counted as errors, not failures.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from eval.cases import CASES
from eval.harness import Outcome, check_case, summarize
from src.agent.graph import build_chat_graph, compute_profile, run_chat_turn
from src.loader import DatasetStore, load_file
from src.tools.registry import get_tools, make_executor

OUT_DIR = Path("eval")


def run_case(case, graph, df, dataset_id, profile) -> Outcome:
    started = time.time()
    try:
        turn = run_chat_turn(graph, dataset_id, profile, [], case.question)
    except Exception as exc:
        from src.agent.llm import is_transient

        if is_transient(exc):
            return Outcome(case.id, False, error=f"{type(exc).__name__}: {str(exc)[:120]}", seconds=time.time() - started)
        raise
    tools_ok = [s["tool"] for s in turn.trace if s["step"] == "run_tools" and s["ok"]]
    reasons = check_case(case, df, turn.answer, tools_ok, len(turn.charts))
    return Outcome(
        case.id, not reasons, reasons, [s["tool"] for s in turn.trace if s["step"] == "run_tools"],
        time.time() - started, turn.answer,
    )


def write_reports(outcomes, summary, model, data_name) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    questions = {c.id: c.question for c in CASES}
    (OUT_DIR / "results.json").write_text(
        json.dumps({"model": model, "data": data_name, "summary": summary,
                    "cases": [vars(o) | {"question": questions[o.case_id]} for o in outcomes]}, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# Eval results",
        "",
        f"Model: `{model}`  |  Data: `{data_name}`  |  "
        f"**{summary['passed']}/{summary['scored']} correct ({summary['accuracy']:.0%})**"
        + (f", {summary['errors']} errored (API busy)" if summary["errors"] else ""),
        f"Average {summary['avg_seconds']} s and {summary['avg_tool_calls']} tool calls per question.",
        "",
        "| # | Question | Result | Tools | Notes |",
        "|---|----------|--------|-------|-------|",
    ]
    for number, o in enumerate(outcomes, start=1):
        result = "error" if o.error else ("pass" if o.passed else "FAIL")
        notes = o.error or "; ".join(o.reasons)
        lines.append(f"| {number} | {questions[o.case_id]} | {result} | {', '.join(o.tools) or '-'} | {notes} |")
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(path: str, only: list[str] | None = None, delay: float = 6.0, llm=None, sleep=time.sleep) -> dict:
    if llm is None:
        from dotenv import load_dotenv

        load_dotenv()
        from src.agent.llm import get_llm

        llm = get_llm()

    store = DatasetStore()
    df = load_file(path)
    dataset_id = store.add(df, name=Path(path).name)
    profile = compute_profile(store, dataset_id)
    graph = build_chat_graph(llm, get_tools(), make_executor(store))

    cases = [c for c in CASES if not only or c.id in only]
    outcomes = []
    for number, case in enumerate(cases, start=1):
        outcome = run_case(case, graph, df, dataset_id, profile)
        outcomes.append(outcome)
        label = "ERROR" if outcome.error else ("pass " if outcome.passed else "FAIL ")
        print(f"[{number:>2}/{len(cases)}] {label} {case.id}  ({outcome.seconds:.0f}s)  "
              f"{outcome.error or '; '.join(outcome.reasons)}")
        if number < len(cases):
            sleep(delay)  # stay inside the free-tier rate limit

    summary = summarize(outcomes)
    try:
        from config import GEMINI_MODEL as model
    except Exception:
        model = "unknown"
    write_reports(outcomes, summary, model, Path(path).name)
    print(f"\n{summary['passed']}/{summary['scored']} correct ({summary['accuracy']:.0%}), "
          f"{summary['errors']} errored. Report: eval/RESULTS.md")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/titanic.csv")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--delay", type=float, default=6.0, help="seconds between questions")
    args = parser.parse_args()
    main(args.path, args.only.split(",") if args.only else None, args.delay)
