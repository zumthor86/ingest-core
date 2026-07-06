"""Backfill plumbing: budget, windowed range fetch, pre-history walk, quota sizing.

Promoted from Hephaestus's ``scripts/range_backfill.py`` (feature 002) with
behaviour parity. Storage- and orchestration-agnostic (feature-003 boundary):
functions yield provider records or drive an app-supplied callback — no store
imports, no Prefect. The consuming app owns transform + merge-write.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterator, Optional

from ingest_core.clients.eodhd import build_options_eod_url, fetch_eodhd_page

__all__ = [
    "Budget",
    "PrehistoryResult",
    "iter_range_records",
    "run_prehistory_walk",
    "quota_remaining_requests",
    "probe_quota_reset",
]

_logger = logging.getLogger(__name__)

_PAGE_LIMIT = 1000
_OFFSET_CAP = 10_000


class Budget:
    """Request/row counters, thread-safe so concurrent fetch fan-outs share one budget.

    Semantics verbatim from ``scripts/range_backfill.py`` (parity anchor):
    ``exceeded()`` flips work to *deferred* — it never raises. ``max_requests=None``
    means unmetered (tests only; flows always size it).
    """

    def __init__(self, max_requests: Optional[int]) -> None:
        self.max = max_requests
        self.requests = 0
        self.rows = 0
        self._lock = threading.Lock()

    def exceeded(self) -> bool:
        return self.max is not None and self.requests >= self.max

    def add_request(self, n: int = 1) -> None:
        with self._lock:
            self.requests += n

    def add_rows(self, n: int) -> None:
        with self._lock:
            self.rows += n

    def remaining(self) -> Optional[int]:
        if self.max is None:
            return None
        return max(self.max - self.requests, 0)


def _paginate(
    symbol: str,
    start: date,
    end: date,
    option_type: Optional[str],
    api_token: Optional[str],
    budget: Budget,
    page_limit: int,
    offset_cap: int,
) -> tuple[list[dict], bool]:
    """Page one (symbol, range, type?) slice; returns (records, capped).

    Every page acquires one budget request. Mirrors the script: pagination runs a
    window to completion once started (budget overshoot is bounded by one window).
    """
    out: list[dict] = []
    off = 0
    while True:
        url = build_options_eod_url(
            symbol, start.isoformat(), end.isoformat(),
            page_offset=off, page_limit=page_limit, option_type=option_type,
        )
        recs = fetch_eodhd_page(url, api_token=api_token)["data"]
        budget.add_request()
        out.extend(recs)
        if len(recs) < page_limit:
            return out, False
        off += page_limit
        if off > offset_cap:
            return out, True


def iter_range_records(
    symbol: str,
    start: date,
    end: date,
    *,
    budget: Budget,
    api_token: Optional[str] = None,
    page_limit: int = _PAGE_LIMIT,
    offset_cap: int = _OFFSET_CAP,
) -> Iterator[tuple[date, date, list[dict]]]:
    """Yield ``(window_start, window_end, records)`` covering [start, end].

    Auto-halves any window whose pagination exceeds ``offset_cap`` (discarding the
    capped partial fetch, exactly like the script's ``backfill_range``); at a
    single-day window it splits by contract type instead. Yields per-window so
    callers can land records incrementally (bounded memory). Stops when the
    budget is exceeded — never raises for budget reasons.
    """
    if start > end or budget.exceeded():
        return
    records, capped = _paginate(symbol, start, end, None, api_token, budget, page_limit, offset_cap)
    if not capped:
        yield (start, end, records)
        return
    if (end - start).days <= 0:
        # Too dense for the offset cap even at one day: split call/put.
        out: list[dict] = []
        for option_type in ("call", "put"):
            recs, _capped = _paginate(symbol, start, end, option_type, api_token, budget, page_limit, offset_cap)
            out.extend(recs)
        yield (start, end, out)
        return
    # NOTE: deliberate fix vs the retired script, which used max(days // 2, 1):
    # for a capped 2-day window that made mid == end, recursing on the SAME
    # window until the budget died. days >= 1 here (single-day handled above),
    # so floor-half always yields a strictly smaller left window.
    mid = start + timedelta(days=(end - start).days // 2)
    yield from iter_range_records(
        symbol, start, mid,
        budget=budget, api_token=api_token, page_limit=page_limit, offset_cap=offset_cap,
    )
    yield from iter_range_records(
        symbol, mid + timedelta(days=1), end,
        budget=budget, api_token=api_token, page_limit=page_limit, offset_cap=offset_cap,
    )


@dataclass(frozen=True)
class PrehistoryResult:
    """Outcome of a pre-history walk.

    ``completed`` distinguishes a *finished* walk (listing boundary found, or the
    coverage floor reached) from a *budget-cut* one (resumes on a later run) —
    callers use it as the consolidation trigger (FR-016). ``boundary_window`` is
    the empty window that terminated a finished walk (``None`` when it finished
    by reaching the coverage floor, or was budget-cut); the caller MUST record
    per-day empty sentinels for a returned boundary window (FR-002) so the walk
    trigger self-extinguishes.
    """

    rows_landed: int
    boundary_window: Optional[tuple[date, date]]
    completed: bool


def run_prehistory_walk(
    land_window: Callable[[date, date], int],
    first: date,
    start: date,
    *,
    week_days: int = 7,
    budget: Budget,
) -> PrehistoryResult:
    """Walk backward from ``first − 1`` toward ``start`` in ``week_days`` windows.

    Stops at the first window for which ``land_window`` lands 0 rows (listing
    boundary), at the coverage floor, or when the budget is exceeded. All I/O
    goes through the callback — this function owns only the walk/stop semantics
    (script ``probe_prehistory``).
    """
    d = first - timedelta(days=1)
    total = 0
    while d >= start:
        if budget.exceeded():
            return PrehistoryResult(total, None, completed=False)
        ws = max(d - timedelta(days=week_days - 1), start)
        rows = land_window(ws, d)
        total += rows
        if rows == 0:
            _logger.info("prehistory stop at %s..%s (empty window -> assume not listed)", ws, d)
            return PrehistoryResult(total, (ws, d), completed=True)
        d = ws - timedelta(days=1)
    return PrehistoryResult(total, None, completed=True)  # coverage floor reached


def quota_remaining_requests(api_token: Optional[str] = None) -> Optional[int]:
    """Remaining Marketplace *requests* today (calls ÷ 10), or None if unavailable.

    Single home for the live-quota read (was duplicated between the range-backfill
    script and main_flow's preflight).
    """
    try:
        from ingest_core.clients.eodhd import get_eodhd_quota_status

        return int(get_eodhd_quota_status(api_token=api_token)["marketplace_requests_remaining"])
    except Exception:
        return None


def probe_quota_reset(api_token: Optional[str] = None) -> None:
    """Fire one tiny request to trigger EODHD's request-triggered daily quota reset.

    EODHD resets the daily Marketplace limit only when a request arrives >=24h
    after the window's first request — on a fresh quota day the usage endpoint can
    still *report* exhausted until something pokes the API. One minimal options
    request (PG, low volume, page_limit=1) flips it; if quota is genuinely gone,
    the probe just fails and callers proceed with the stale-low reading.
    """
    probe_date = (date.today() - timedelta(days=5)).isoformat()
    url = build_options_eod_url("PG", probe_date, probe_date, page_limit=1)
    try:
        fetch_eodhd_page(url, api_token=api_token)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("quota probe request failed (truly out of quota?): %s", exc)
