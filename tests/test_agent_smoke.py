"""Smoke test for Gemini LLM and agent state initialization."""

from src.agent.prompts import PLANNER_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
from src.agent.state import AgentState
import os
import pytest
from dotenv import load_dotenv
from src.agent.llm import get_llm, message_text

load_dotenv()


def test_agent_state_keys():
    state: AgentState = {
        "dataset_id": "test_id",
        "profile": {"rows": 10, "columns": 2},
        "messages": [],
        "trace": [],
        "charts": [],
        "rounds": 0,
        "findings": "",
    }
    assert state["dataset_id"] == "test_id"
    assert state["rounds"] == 0


def test_prompts_configured():
    assert "Never compute numbers yourself" in PLANNER_SYSTEM_PROMPT
    assert "create_chart" in CHAT_SYSTEM_PROMPT


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set in environment")
def test_gemini_smoke_call():
    from src.agent.llm import get_llm, invoke_with_retry, is_transient, message_text

    llm = get_llm()
    try:
        res = invoke_with_retry(
            llm, "Reply with 'READY'", tries=3, base_delay=3)
    except Exception as exc:
        if is_transient(exc):
            pytest.skip(f"Gemini temporarily unavailable: {exc}")
        raise
    assert "READY" in message_text(res).upper()
