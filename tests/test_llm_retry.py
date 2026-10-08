import pytest

from src.agent.llm import invoke_with_retry, is_transient, message_text


class FakeError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class FlakyLLM:
    """Raises the given errors in order, then answers."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


class Msg:
    def __init__(self, content):
        self.content = content


# ---------- is_transient ----------

def test_transient_by_code():
    assert is_transient(FakeError("x", code=503))
    assert is_transient(FakeError("x", code=429))
    assert not is_transient(FakeError("x", code=400))


def test_transient_by_message():
    assert is_transient(Exception(
        "503 UNAVAILABLE. This model is currently experiencing high demand."))
    assert is_transient(Exception("429 RESOURCE_EXHAUSTED"))
    assert not is_transient(ValueError("unknown column"))


# ---------- invoke_with_retry ----------

def test_success_first_try_does_not_sleep():
    sleeps = []
    llm = FlakyLLM([])
    assert invoke_with_retry(llm, "hi", sleep=sleeps.append) == "ok"
    assert llm.calls == 1 and sleeps == []


def test_retries_then_succeeds_with_doubling_waits():
    sleeps = []
    llm = FlakyLLM([FakeError("busy", 503), FakeError("busy", 503)])
    assert invoke_with_retry(llm, "hi", base_delay=2,
                             sleep=sleeps.append) == "ok"
    assert llm.calls == 3 and sleeps == [2, 4]


def test_gives_up_after_max_tries_and_raises_the_last_error():
    sleeps = []
    llm = FlakyLLM([FakeError("busy", 503)] * 5)
    with pytest.raises(FakeError):
        invoke_with_retry(llm, "hi", tries=3, base_delay=1,
                          sleep=sleeps.append)
    assert llm.calls == 3 and sleeps == [1, 2]


def test_non_transient_error_is_not_retried():
    sleeps = []
    llm = FlakyLLM([ValueError("bad request")])
    with pytest.raises(ValueError):
        invoke_with_retry(llm, "hi", sleep=sleeps.append)
    assert llm.calls == 1 and sleeps == []


# ---------- message_text ----------

@pytest.mark.parametrize(
    "content, expected",
    [
        ("READY", "READY"),
        ([{"type": "text", "text": "READY"}], "READY"),
        (["RE", "ADY"], "READY"),
        ([{"type": "thinking", "thinking": "hmm"}, {
         "type": "text", "text": "READY"}], "READY"),
        ([], ""),
    ],
)
def test_message_text_handles_every_shape(content, expected):
    assert message_text(Msg(content)) == expected
