"""Run the eval: ask each question in eval/cases.py to the chat agent and score the answers.

    python -m eval.run_eval data/titanic.csv
    python -m eval.run_eval data/titanic.csv --rpm 12
    python -m eval.run_eval data/titanic.csv --failed

Each question starts a fresh conversation. Results go to eval/results.json and
eval/RESULTS.md. Model calls are paced to --rpm (default 12) instead of a fixed pause.
Re-run only what failed or errored with --failed; the new results are merged into the saved ones.
Gemini errors (busy, rate-limited) are counted as errors, not failures.
"""

import argparse
import json
import time
from pathlib import Path

from eval.cases import CASES
from eval.harness import Outcome, check_case, summarize
from eval.ratelimit import RateLimiter, RateLimitedLLM
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
    tools_ok = [s["tool"]
                for s in turn.trace if s["step"] == "run_tools" and s["ok"]]
    reasons = check_case(case, df, turn.answer, tools_ok, len(turn.charts))
    return Outcome(
        case.id, not reasons, reasons, [s["tool"]
                                        for s in turn.trace if s["step"] == "run_tools"],
        time.time() - started, turn.answer,
    )


def load_previous(directory: Path) -> dict[str, Outcome]:
    """Outcomes saved by the last run, keyed by case id. Empty if there are none."""
    try:
        saved = json.loads(
            (directory / "results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    keep = ("case_id", "passed", "reasons",
            "tools", "seconds", "answer", "error")
    return {c["case_id"]: Outcome(**{k: c[k] for k in keep if k in c}) for c in saved.get("cases", [])}


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
        lines.append(
            f"| {number} | {questions[o.case_id]} | {result} | {', '.join(o.tools) or '-'} | {notes} |")
    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(path: str, only: list[str] | None = None, delay: float = 0.0, llm=None, sleep=time.sleep,
         rpm: int = 12, failed: bool = False) -> dict:
    if llm is None:
        from dotenv import load_dotenv

        load_dotenv()
        from src.agent.llm import get_llm

        llm = get_llm()
    model_name = getattr(llm, "model", None) or getattr(
        llm, "model_name", None)  # what actually ran
    limiter = RateLimiter(rpm, sleep=sleep) if rpm else None
    if limiter:
        llm = RateLimitedLLM(llm, limiter)

    store = DatasetStore()
    df = load_file(path)
    dataset_id = store.add(df, name=Path(path).name)
    profile = compute_profile(store, dataset_id)
    graph = build_chat_graph(llm, get_tools(), make_executor(store))

    previous = load_previous(OUT_DIR) if (only or failed) else {}
    if failed:
        only = [
            c.id for c in CASES if c.id not in previous or previous[c.id].error or not previous[c.id].passed]
        if not only:
            print("Nothing to re-run: every case passed last time.")
            return summarize([previous[c.id] for c in CASES if c.id in previous])
    cases = [c for c in CASES if not only or c.id in only]

    results = dict(previous)
    for number, case in enumerate(cases, start=1):
        outcome = run_case(case, graph, df, dataset_id, profile)
        results[case.id] = outcome
        label = "ERROR" if outcome.error else (
            "pass " if outcome.passed else "FAIL ")
        print(f"[{number:>2}/{len(cases)}] {label} {case.id}  ({outcome.seconds:.0f}s)  "
              f"{outcome.error or '; '.join(outcome.reasons)}")
        if delay and number < len(cases):
            sleep(delay)

    outcomes = [results[c.id] for c in CASES if c.id in results]
    summary = summarize(outcomes)
    if not model_name:
        try:
            from config import GEMINI_MODEL as model_name
        except Exception:
            model_name = "unknown"
    write_reports(outcomes, summary, str(
        model_name).removeprefix("models/"), Path(path).name)
    waited = f" Waited {limiter.waited:.0f}s for the rate limit." if limiter and limiter.waited else ""
    print(f"\n{summary['passed']}/{summary['scored']} correct ({summary['accuracy']:.0%}), "
          f"{summary['errors']} errored.{waited} Report: eval/RESULTS.md")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/titanic.csv")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--failed", action="store_true",
                        help="re-run cases that failed, errored or never ran (also resumes an interrupted run)")
    parser.add_argument("--rpm", type=int, default=12,
                        help="max model calls per minute (0 = no limit)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="extra seconds between questions")
    args = parser.parse_args()
    main(args.path, args.only.split(",") if args.only else None,
         args.delay, rpm=args.rpm, failed=args.failed)
