"""Run the auto-analysis agent on a file and save the results.

    python run_auto.py data/titanic.csv

Prints what the agent did and what it found. Saves each chart as an HTML file in
out/ (open it in a browser) and the findings as out/findings.md.
"""

import sys
from pathlib import Path

import plotly.io as pio

from src.agent.graph import build_auto_graph, run_auto_analysis
from src.loader import DatasetStore, load_file
from src.tools.registry import get_tools, make_executor

OUT_DIR = Path("out")


def main(path: str, llm=None) -> dict:
    if llm is None:
        from dotenv import load_dotenv

        load_dotenv()
        from src.agent.llm import get_llm

        llm = get_llm()

    store = DatasetStore()
    dataset_id = store.add(load_file(path), name=Path(path).name)
    graph = build_auto_graph(store, llm, get_tools(), make_executor(store))

    try:
        final = run_auto_analysis(graph, dataset_id)
    except Exception as exc:
        from src.agent.llm import is_transient

        if is_transient(exc):
            print("\nGemini is busy or rate-limited right now. Wait a minute and run it again.")
            print(f"({type(exc).__name__}: {str(exc)[:160]})")
            sys.exit(1)
        raise

    print("\nWhat the agent did:")
    for step in final["trace"]:
        status = "ok  " if step["ok"] else "FAIL"
        args = step["args"] if step["step"] == "run_tools" else ""
        print(f"  [{status}] {step['tool']} {args}")
        print(f"         {str(step['summary'])[:110]}")

    print(f"\nLLM-planned rounds: {final['rounds']}   charts created: {len(final['charts'])}")

    OUT_DIR.mkdir(exist_ok=True)
    for number, chart_json in enumerate(final["charts"], start=1):
        target = OUT_DIR / f"chart_{number}.html"
        try:
            pio.from_json(chart_json).write_html(target, include_plotlyjs="cdn")
            print(f"  saved {target}")
        except Exception as exc:
            print(f"  chart {number} could not be saved: {exc}")

    (OUT_DIR / "findings.md").write_text(final["findings"], encoding="utf-8")
    print("\n" + "=" * 70 + "\nFINDINGS\n" + "=" * 70)
    print(final["findings"])
    return final


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/titanic.csv")
