"""AI Data Analysis Agent: Streamlit app.

    streamlit run app.py

Flow: load a file -> overview -> optional cleaning (you approve every fix) ->
auto-analysis -> chat. The LLM never sees the raw data, only a profile and tool results.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import MAX_FILE_MB, SUPPORTED_EXTENSIONS
from src.agent.graph import (
    build_auto_graph,
    build_chat_graph,
    compute_profile,
    run_auto_analysis,
    run_chat_turn,
)
from src.loader import DatasetStore, DataLoadError, load_bytes, load_file
from src.tools.cleaning import apply_cleaning, propose_cleaning
from src.tools.registry import get_tools, make_executor
from src.ui_helpers import cleaning_log_rows, column_rows, figures_from_json, fix_label, issue_rows, trace_rows

SAMPLE_PATH = "data/titanic.csv"

st.set_page_config(page_title="AI Data Analysis Agent", page_icon="📊", layout="wide")
load_dotenv()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

DEFAULTS = {
    "store": None,          # DatasetStore
    "dataset_id": None,
    "file_name": None,
    "profile": None,        # profile + validation, computed without the LLM
    "cleaned": False,
    "clean_log": None,      # result of the last apply_cleaning
    "auto": None,           # final state of the auto-analysis graph
    "history": [],          # plain Q&A messages sent back to the model
    "chat_log": [],         # what the chat tab shows: role, content, charts, trace
    "last_upload": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_analysis() -> None:
    """Forget results that were computed on data that has since changed."""
    st.session_state.auto = None
    st.session_state.history = []
    st.session_state.chat_log = []


def set_dataset(df, name: str) -> None:
    store = DatasetStore()
    dataset_id = store.add(df, name=name)
    st.session_state.update(
        store=store,
        dataset_id=dataset_id,
        file_name=name,
        profile=compute_profile(store, dataset_id),
        cleaned=False,
        clean_log=None,
    )
    reset_analysis()


def refresh_profile() -> None:
    state = st.session_state
    state.profile = compute_profile(state.store, state.dataset_id)
    reset_analysis()


def get_llm_or_none():
    """The Gemini client, created once. Shows a friendly error if it cannot be created."""
    if st.session_state.get("llm") is not None:
        return st.session_state.llm
    try:
        if not os.environ.get("GOOGLE_API_KEY"):
            try:
                os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
            except Exception:
                pass
        from src.agent.llm import get_llm

        st.session_state.llm = get_llm()
        return st.session_state.llm
    except Exception as exc:
        st.error(
            "Could not start the Gemini client. Put GOOGLE_API_KEY in your .env file "
            f"(or in Streamlit secrets).\n\n{type(exc).__name__}: {exc}"
        )
        return None


def run_guarded(action, *args):
    """Run an LLM action. A busy or rate-limited Gemini shows a warning instead of a crash."""
    from src.agent.llm import is_transient

    try:
        return action(*args)
    except Exception as exc:
        if is_transient(exc):
            st.warning("Gemini is busy or rate-limited right now. Wait a minute and try again.")
            return None
        raise


def show_charts(charts: list[str], key_prefix: str) -> None:
    for number, (figure, error) in enumerate(figures_from_json(charts), start=1):
        if figure is not None:
            st.plotly_chart(figure, width="stretch", key=f"{key_prefix}_{number}")
        else:
            st.warning(f"Chart {number} could not be shown ({error}).")


def show_trace(trace: list[dict], label: str = "What the agent did") -> None:
    with st.expander(label):
        st.dataframe(trace_rows(trace), hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Sidebar: load data
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Data Analysis Agent")
    st.caption("Upload a file, review the data, approve any cleaning, then analyze and ask questions.")

    upload = st.file_uploader(
        f"CSV or Excel, up to {MAX_FILE_MB} MB",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
    )
    if upload is not None:
        signature = (upload.name, upload.size)
        if st.session_state.last_upload != signature:
            try:
                set_dataset(load_bytes(upload.getvalue(), upload.name), upload.name)
                st.session_state.last_upload = signature
            except DataLoadError as exc:
                st.error(str(exc))

    if Path(SAMPLE_PATH).exists() and st.button("Try the Titanic sample"):
        set_dataset(load_file(SAMPLE_PATH), Path(SAMPLE_PATH).name)

    if st.session_state.dataset_id:
        st.divider()
        st.write(f"**Loaded:** {st.session_state.file_name}")
        st.write("**Version:** " + ("cleaned" if st.session_state.cleaned else "original"))
        if st.session_state.cleaned and st.button("Reset to the original upload"):
            st.session_state.store.reset(st.session_state.dataset_id)
            st.session_state.cleaned = False
            st.session_state.clean_log = None
            refresh_profile()
            st.rerun()


if not st.session_state.dataset_id:
    st.title("AI Data Analysis Agent")
    st.info("Upload a CSV or Excel file in the sidebar, or try the Titanic sample, to begin.")
    st.stop()

state = st.session_state
store, dataset_id, profile = state.store, state.dataset_id, state.profile
info = profile["profile"]

tab_overview, tab_clean, tab_auto, tab_chat = st.tabs(["Overview", "Clean", "Auto-analysis", "Chat"])


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{info['rows']:,}")
    c2.metric("Columns", info["columns"])
    c3.metric("Duplicate rows", info["duplicate_rows"])

    st.subheader("Columns")
    st.dataframe(column_rows(profile), hide_index=True, width="stretch")

    st.subheader("Data quality")
    issues = issue_rows(profile)
    if issues:
        st.dataframe(issues, hide_index=True, width="stretch")
    else:
        st.success("No data quality issues found.")

    st.subheader("First rows")
    st.dataframe(store.get(dataset_id).head(20), width="stretch")


# --------------------------------------------------------------------------
# Clean: proposals only, nothing changes until the user approves
# --------------------------------------------------------------------------

with tab_clean:
    st.write("The agent proposes fixes. Nothing changes until you tick the ones you want and apply them.")
    proposal = propose_cleaning(store, dataset_id)
    fixes = proposal.data.get("fixes", []) if proposal.ok else []

    if not proposal.ok:
        st.error(proposal.error)
    elif not fixes:
        st.success("No cleaning needed.")
    else:
        chosen = [
            fix["id"]
            for fix in fixes
            if st.checkbox(fix_label(fix), key=f"fix_{dataset_id}_{fix['id']}", value=False)
        ]
        if st.button("Apply selected fixes", disabled=not chosen):
            result = apply_cleaning(store, dataset_id, chosen)
            if result.ok:
                state.cleaned = True
                state.clean_log = result
                refresh_profile()
                st.rerun()
            else:
                st.error(f"{result.error} {result.hint or ''}")

    if state.clean_log is not None:
        st.success(state.clean_log.summary)
        log_rows = cleaning_log_rows(state.clean_log.data)
        if log_rows:
            st.dataframe(log_rows, hide_index=True, width="stretch")
        st.caption("Earlier analysis and chat were cleared because the data changed.")


# --------------------------------------------------------------------------
# Auto-analysis
# --------------------------------------------------------------------------

with tab_auto:
    if st.button("Run auto-analysis", type="primary"):
        llm = get_llm_or_none()
        if llm is not None:
            graph = build_auto_graph(store, llm, get_tools(), make_executor(store))
            with st.spinner("Analyzing... this takes about 10 to 30 seconds."):
                final = run_guarded(run_auto_analysis, graph, dataset_id)
            if final is not None:
                state.auto = final

    auto = state.auto
    if auto is None:
        st.caption("The agent picks analyses and charts, runs them with its tools, then writes the findings.")
    else:
        st.markdown(auto["findings"])
        show_charts(auto["charts"], "auto_chart")
        show_trace(auto["trace"])


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

with tab_chat:
    for turn_number, entry in enumerate(state.chat_log):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                show_charts(entry["charts"], f"chat_chart_{turn_number}")
                if entry["trace"]:
                    show_trace(entry["trace"], "Tools used")

    question = st.chat_input("Ask a question about your data")
    if question:
        state.chat_log.append({"role": "user", "content": question, "charts": [], "trace": []})
        llm = get_llm_or_none()
        if llm is not None:
            chat_graph = build_chat_graph(llm, get_tools(), make_executor(store))
            with st.spinner("Thinking..."):
                turn = run_guarded(run_chat_turn, chat_graph, dataset_id, profile, state.history, question)
            if turn is not None:
                state.history = turn.history
                state.chat_log.append(
                    {"role": "assistant", "content": turn.answer, "charts": turn.charts, "trace": turn.trace}
                )
                st.rerun()
        # Not answered (no client, or Gemini busy): drop the question so the user can ask again.
        # No rerun here, so the warning stays on screen.
        state.chat_log.pop()