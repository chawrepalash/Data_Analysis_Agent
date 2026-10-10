"""Small pure helpers for the Streamlit app, kept out of app.py so they can be tested."""

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


def trace_rows(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trace entries as flat rows for a table: status, tool, arguments, result."""
    rows = []
    for step in trace:
        args = step.get("args") or {}
        rows.append(
            {
                "status": "ok" if step.get("ok") else "failed",
                "tool": step.get("tool") or "(no tool called)",
                "arguments": ", ".join(f"{k}={v}" for k, v in args.items()) if step.get("step") == "run_tools" else "",
                "result": str(step.get("summary", ""))[:200],
            }
        )
    return rows


def column_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per column for the overview table."""
    rows = []
    for col in (profile.get("profile") or {}).get("column_info", []):
        rows.append(
            {
                "column": col["name"],
                "type": col["kind"],
                "missing": col["missing"],
                "missing %": col["missing_pct"],
                "unique": col["unique"],
                "min": col.get("min"),
                "max": col.get("max"),
            }
        )
    return rows


def issue_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    issues = (profile.get("validation") or {}).get("issues", [])
    return [
        {"severity": i["severity"], "type": i["type"],
            "column": i.get("column") or "", "detail": i["detail"]}
        for i in issues
    ]


AFFECTED_KEYS = ("values_affected", "rows_affected",
                 "affected", "rows_affected", "count", "n_affected")


def fix_label(fix: dict[str, Any]) -> str:
    """A readable checkbox label for a proposed cleaning fix.

    Reads every field with a fallback, so a fix without a reason or a count still
    gets a label instead of crashing the page.
    """
    where = f" `{fix['column']}`" if fix.get("column") else ""
    action = str(fix.get("action", "fix")).replace("_", " ")
    label = f"**{action}**{where}"
    if fix.get("reason"):
        label += f": {fix['reason']}"
    count = next((fix[k]
                 for k in AFFECTED_KEYS if fix.get(k) is not None), None)
    if count is not None:
        label += f" ({count} values)"
    return label


def cleaning_log_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-fix log from an apply_cleaning result, whatever key it is stored under.

    Looks for "change_log" and "log" first, then the first list of dicts in the result. Returns an
    empty list if there is none, so the page shows the summary line only.
    """
    candidates = [data.get("change_log"), data.get("log")
                  ] + list(data.values())
    for value in candidates:
        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
            return value
    return []


def figures_from_json(charts: list[str]) -> list[tuple[go.Figure | None, str | None]]:
    """Plotly figures from stored JSON. A chart that cannot be parsed gives (None, error)."""
    out = []
    for chart_json in charts:
        try:
            out.append((pio.from_json(chart_json), None))
        except Exception as exc:
            out.append((None, f"{type(exc).__name__}: {exc}"))
    return out
