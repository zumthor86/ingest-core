"""In-process rate limiter — per application, no cross-process state (FR-006, D6).

One instance per process paces that process's calls; correctness of the shared
provider quota relies on the conductor's strict stage sequencing, not on any
shared token store.
"""
from __future__ import annotations

import asyncio
import threading
import time


class RateLimiter:
    """Minimum-interval pacer: at most ``max_calls`` acquires per ``per_seconds``.

    ``acquire()`` blocks the calling thread; ``aacquire()`` awaits instead so
    async callers (Hermes's ingest loop) never block the event loop. Slot
    reservation is thread-safe and monotonic-clock based.
    """

    def __init__(self, max_calls: int, per_seconds: float) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if per_seconds < 0:
            raise ValueError("per_seconds must be >= 0")
        self._interval = per_seconds / max_calls
        self._lock = threading.Lock()
        self._next_free = 0.0

    @property
    def interval(self) -> float:
        return self._interval

    def _reserve(self) -> float:
        """Reserve the next slot; return how long the caller must wait for it."""
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_free)
            self._next_free = start + self._interval
            return start - now

    def acquire(self) -> None:
        wait = self._reserve()
        if wait > 0:
            time.sleep(wait)

    async def aacquire(self) -> None:
        wait = self._reserve()
        if wait > 0:
            await asyncio.sleep(wait)
