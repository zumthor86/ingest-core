"""RateLimiter + with_retry behaviour (fast, deterministic)."""
from __future__ import annotations

import asyncio
import time

import pytest

from ingest_core.ratelimit import RateLimiter
from ingest_core.retry import with_retry
from ingest_core.types import EODHDLimitError


def test_ratelimiter_paces_successive_acquires():
    limiter = RateLimiter(max_calls=1, per_seconds=0.05)
    t0 = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.09  # two full intervals after the free first slot


def test_ratelimiter_async_acquire():
    limiter = RateLimiter(max_calls=1, per_seconds=0.03)

    async def burst():
        t0 = time.monotonic()
        await limiter.aacquire()
        await limiter.aacquire()
        return time.monotonic() - t0

    assert asyncio.run(burst()) >= 0.025


def test_ratelimiter_validates_args():
    with pytest.raises(ValueError):
        RateLimiter(0, 1.0)
    with pytest.raises(ValueError):
        RateLimiter(1, -1.0)


def test_with_retry_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise EODHDLimitError("throttled", retry_after=0.01)
        return "ok"

    assert with_retry(flaky, retries=3, backoff=0.01)() == "ok"
    assert attempts["n"] == 3


def test_with_retry_exhausts_and_reraises():
    def always_limited():
        raise EODHDLimitError("throttled", retry_after=0.01)

    with pytest.raises(EODHDLimitError):
        with_retry(always_limited, retries=2, backoff=0.01)()


def test_with_retry_does_not_catch_other_errors():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        with_retry(boom, retries=2, backoff=0.01)()
