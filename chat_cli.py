"""Chat with the agent about a file in the terminal.

    python chat_cli.py data/titanic.csv

Type a question and press Enter. Type quit to stop. Charts are saved to out/chat_N.html.
"""

import sys
from pathlib import Path

import plotly.io as pio

from src.agent.graph import build_chat_graph, compute_profile, run_chat_turn
from src.loader import DatasetStore, load_file
from src.tools.registry import get_tools, make_executor

OUT_DIR = Path("out")


def main(path: str, llm=None, read=input) -> None:
    if llm is None:
        from dotenv import load_dotenv

        load_dotenv()
        from src.agent.llm import get_llm

        llm = get_llm()

    store = DatasetStore()
    dataset_id = store.add(load_file(path), name=Path(path).name)
    profile = compute_profile(store, dataset_id)
    graph = build_chat_graph(llm, get_tools(), make_executor(store))

    print(f"Loaded {Path(path).name}. Ask a question, or type quit.")
    history, chart_count = [], 0
    while True:
        question = read("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        try:
            turn = run_chat_turn(graph, dataset_id, profile, history, question)
        except Exception as exc:
            from src.agent.llm import is_transient

            if is_transient(exc):
                print("Gemini is busy or rate-limited. Wait a minute and ask again.")
                continue
            raise
        history = turn.history

        for step in turn.trace:
            print(f"  [{'ok' if step['ok'] else 'FAIL'}] {step['tool']} {step['args']}")
        OUT_DIR.mkdir(exist_ok=True)
        for chart_json in turn.charts:
            chart_count += 1
            target = OUT_DIR / f"chat_{chart_count}.html"
            pio.from_json(chart_json).write_html(target, include_plotlyjs="cdn")
            print(f"  chart saved to {target}")
        print(f"\nAgent: {turn.answer}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/titanic.csv")
