"""Shared token-bucket rate limiter (one team-wide limiter at
120 RPM for Spark contributor, whose hard limit is 150 RPM per team, not per key)."""
from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, per_minute: float = 120.0, burst: int | None = None):
        self.rate = per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(1, int(per_minute // 10)))
        self.tokens = self.capacity
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one request token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)
