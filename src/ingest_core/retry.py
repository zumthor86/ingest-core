"""Orchestration-agnostic retry/backoff wrapper (research D10).

For callers *outside* Prefect (scripts, ad-hoc refresh paths). Prefect flows
use their native ``retries=`` at the flow/task layer instead.
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar

from ingest_core.types import EODHDLimitError

T = TypeVar("T")


def with_retry(
    fn: Callable[..., T],
    *,
    retries: int = 3,
    backoff: float = 1.0,
    on: Tuple[Type[Exception], ...] = (EODHDLimitError,),
) -> Callable[..., T]:
    """Wrap ``fn`` with exponential backoff on the given exception types.

    Honours ``EODHDLimitError.retry_after`` when the provider supplies it;
    otherwise sleeps ``backoff * 2**attempt``. Re-raises after ``retries``
    exhausted.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        delay = backoff
        for attempt in range(retries + 1):
            try:
                return fn(*args, **kwargs)
            except on as exc:
                if attempt == retries:
                    raise
                wait = getattr(exc, "retry_after", None) or delay
                time.sleep(wait)
                delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    return wrapper
