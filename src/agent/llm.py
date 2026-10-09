"""Gemini LLM initialization with exponential backoff retries."""

import os
import time
from typing import Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

import config

# .env GEMINI_MODEL overrides config.py (useful for testing different models)
MODEL_NAME = os.getenv("GEMINI_MODEL", config.GEMINI_MODEL)
TEMPERATURE = 0.0


def get_llm(model: str = MODEL_NAME, temperature: float = TEMPERATURE) -> ChatGoogleGenerativeAI:
    """Initialize the Google Gemini Chat model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. Add it to .env.")

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        max_retries=3,
    )


def message_text(message) -> str:
    """Plain text of an LLM reply, whether content is a string or a list of blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


TRANSIENT_CODES = {429, 500, 503, 504}
TRANSIENT_WORDS = ("503", "UNAVAILABLE", "429",
                   "RESOURCE_EXHAUSTED", "overloaded")


def is_transient(exc: Exception) -> bool:
    """True for errors that usually go away if you wait: overload and rate limits."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in TRANSIENT_CODES:
        return True
    return any(word in str(exc) for word in TRANSIENT_WORDS)


def invoke_with_retry(runnable, input_, *, tries=4, base_delay=2.0, sleep=time.sleep, **kwargs):
    """Call runnable.invoke, retrying transient errors with doubling waits (2s, 4s, 8s)."""
    for attempt in range(tries):
        try:
            return runnable.invoke(input_, **kwargs)
        except Exception as exc:
            if not is_transient(exc) or attempt == tries - 1:
                raise
            sleep(base_delay * 2**attempt)
