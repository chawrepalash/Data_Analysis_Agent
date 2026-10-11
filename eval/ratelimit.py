"""Keep LLM calls under a requests-per-minute limit.

Instead of sleeping a fixed time between questions (which wastes time on quick
questions and can still burst on slow ones), count the real model calls and wait
only when the next call would go over the limit.
"""

import time
from collections import deque
from typing import Any, Callable


class RateLimiter:
    """Allow at most max_calls in any rolling window (60 s by default)."""

    def __init__(self, max_calls: int, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls, self.window, self.clock, self.sleep = max_calls, window, clock, sleep
        self.stamps: deque[float] = deque()
        self.waited = 0.0

    def wait(self) -> None:
        while True:
            now = self.clock()
            while self.stamps and now - self.stamps[0] >= self.window:
                self.stamps.popleft()
            if len(self.stamps) < self.max_calls:
                self.stamps.append(now)
                return
            pause = self.window - (now - self.stamps[0]) + 0.1
            self.waited += pause
            self.sleep(pause)


class _Limited:
    def __init__(self, runnable: Any, limiter: RateLimiter):
        self._runnable, self._limiter = runnable, limiter

    def invoke(self, *args, **kwargs):
        self._limiter.wait()
        return self._runnable.invoke(*args, **kwargs)


class RateLimitedLLM(_Limited):
    """Wraps a chat model so every invoke, including on the result of bind_tools, is paced."""

    def bind_tools(self, tools):
        return _Limited(self._runnable.bind_tools(tools), self._limiter)
