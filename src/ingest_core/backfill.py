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
    "pages_for_total",
    "run_prehistory_walk",
    "quota_remaining_requests",
    "probe_quota_reset",
]

_logger = logging.getLogger(__name__)

_PAGE_LIMIT = 1000
_OFFSET_CAP = 10_000


class Budget:
    """Request counters, thread-safe so concurrent fetch fan-outs share one budget.

    Semantics verbatim from ``scripts/range_backfill.py`` (parity anchor):
    ``exceeded()`` flips work to *deferred* — it never raises. ``max_requests=None``
    means unmetered (tests only; flows always size it).

    **Price the work before committing to it.** ``add_request()`` charges after a
    page is already fetched, which cannot prevent an overrun — it only records
    one. Every paging loop knows its true cost after page 1 (the vendor returns
    ``meta.total``), so the loop should ``try_reserve()`` the rest up front and
    page only if the claim succeeds. Reserving is atomic, so N concurrent workers
    cannot each independently decide the same last slots are free — the exact
    race that let a 1053-request budget spend 2089 on 2026-08-04, because
    ``exceeded()`` was only consulted between symbols while every in-flight
    symbol kept paging to completion regardless.

    **Known accounting gap (2026-08-05) — this counter UNDER-reports.** The
    reservation for a page whose fetch then raises is never released, which looks
    like a leak but is the safe direction, because the older claim below is wrong:

    Measured on 2026-08-05, a forward phase recorded 1,177 requests here while
    EODHD billed 3,790 — a 3.2x gap matching our retry volume (325 symbols failed,
    each after 6 retries). So 5xx responses appear to be **billed**, contradicting
    the vendor's own "no API calls charged" error text. Refunding on failure would
    therefore make this counter *less* accurate, not more: it would count only
    successes while the provider charges for every attempt, and a budget that
    under-counts is what lets a run overrun the real quota and hard-fail on 402.

    Correct accounting is per-HTTP-attempt, which means the retry loop in
    ``clients.eodhd`` must report attempts. Until that lands, this counter is a
    lower bound on what the provider charges — size budgets with headroom, and
    keep ``EODHD_5XX_MAX_RETRIES`` low so the gap stays bounded.
    """

    def __init__(self, max_requests: Optional[int]) -> None:
        self.max = max_requests
        self.requests = 0
        self.rows = 0
        self._lock = threading.Lock()

    def exceeded(self) -> bool:
        return self.max is not None and self.requests >= self.max

    def try_reserve(self, n: int = 1) -> bool:
        """Atomically claim *n* requests. False (claiming nothing) if they don't fit.

        Unmetered budgets always succeed. Callers that reserve an estimate must
        ``refund()`` whatever they don't spend.
        """
        if n <= 0:
            return True
        with self._lock:
            if self.max is not None and self.requests + n > self.max:
                return False
            self.requests += n
            return True

    def refund(self, n: int) -> None:
        """Return unspent reserved requests (e.g. a chain ended before its estimate)."""
        if n <= 0:
            return
        with self._lock:
            self.requests = max(self.requests - n, 0)

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


def pages_for_total(meta: dict, page_limit: int, offset_cap: int) -> Optional[int]:
    """Total pages a chain needs, from page 1's ``meta.total``; None if unknown.

    The single place that reads the vendor's page-count contract, so every
    ingest mode prices a symbol the same way instead of each re-deriving it.
    Capped at what ``page[offset]`` can actually reach — pages past the offset
    cap are unreachable and must not be budgeted for.
    """
    total = meta.get("total")
    if total is None:
        return None
    limit = int(meta.get("limit") or page_limit) or page_limit
    pages = max(-(-int(total) // limit), 1)  # ceil
    return min(pages, offset_cap // limit + 1)


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

    Prices the slice from page 1's ``meta.total`` and reserves the remaining
    pages before fetching them, so a window can no longer page to completion
    past an exhausted budget. A slice that will not fit stops after page 1 and
    reports ``capped`` — the caller already treats that as deferrable work.
    """
    out: list[dict] = []
    off = 0
    reserved_ahead = 0  # pages claimed up front but not yet fetched

    if not budget.try_reserve(1):
        return out, True  # no room even for page 1

    while True:
        url = build_options_eod_url(
            symbol, start.isoformat(), end.isoformat(),
            page_offset=off, page_limit=page_limit, option_type=option_type,
        )
        page = fetch_eodhd_page(url, api_token=api_token)
        recs = page["data"]
        out.extend(recs)

        if off == 0:
            # Price the rest of the slice once, from the vendor's own record
            # count, and claim it atomically — then page only what we hold.
            total_pages = pages_for_total(page.get("meta") or {}, page_limit, offset_cap)
            if total_pages is not None and total_pages > 1:
                if budget.try_reserve(total_pages - 1):
                    reserved_ahead = total_pages - 1
                else:
                    return out, True  # won't fit — defer the rest of this slice

        if len(recs) < page_limit:
            budget.refund(reserved_ahead)  # ended early; release the unspent claim
            return out, False
        off += page_limit
        if off > offset_cap:
            budget.refund(reserved_ahead)
            return out, True

        if reserved_ahead > 0:
            reserved_ahead -= 1  # spending a page we already hold
        elif not budget.try_reserve(1):
            # meta.total absent or understated — pay per page rather than overrun.
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
