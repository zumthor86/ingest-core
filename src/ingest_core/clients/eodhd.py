"""EODHD API client — the single shared implementation (FR-001/FR-002).

Two merged halves, promoted with behaviour parity (FR-005):

- **Options / Unicorn Bay** (from ``Hephaestus/processing/clients/eodhd.py``):
  URL builders, paginated page fetch with 429/5xx exponential backoff, record
  normalization, account quota status.
- **EOD / bulk / intraday** (from ``Hermes/io/ingest/eodhd/client.py``): plain
  price-history fetchers with limit-detection payload decoding.

No storage, no Polars/Parquet side effects beyond returning frames; callers own
persistence (FR-003/FR-004). The only intentional behaviour change from the
sources: the retries-exhausted throttle path raises :class:`EODHDLimitError`
(a ``RuntimeError`` subclass, so pre-extraction ``except RuntimeError`` call
sites behave identically).

Failures are typed by what the vendor actually said — :class:`EODHDThrottleError`
(429, minute window), :class:`EODHDQuotaExhaustedError` (402, daily budget) and
:class:`EODHDServerError` (5xx, vendor fault). All three subclass
:class:`EODHDLimitError`, so existing catch sites are unaffected, but callers
deciding whether work was *deferred* or *failed* must branch on the type — never
on the message text.
"""
import logging
import os
import threading
import time
import random
import re
import requests
import pandas as pd
import polars as pl
from datetime import date, datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode

from ingest_core.ratelimit import RateLimiter
from ingest_core.types import (
    EODHDLimitError,
    EODHDQuotaExhaustedError,
    EODHDServerError,
    EODHDThrottleError,
)

# Retry budget for HTTP 429 (the per-MINUTE request window). Minute-scale waits,
# few attempts: if we are pinned against the window the run's own pacing is wrong
# and the caller is better off deferring than sitting here.
_MAX_RETRIES = int(os.environ.get("EODHD_HTTP_MAX_RETRIES", "1"))
_BASE_DELAY = 60.0  # first retry waits ~60 s; doubles each attempt + jitter

# Retry budget for HTTP 5xx, which is a different animal entirely: EODHD documents
# it as "Server Error — retry after a short delay" and bills it 0 API calls, so
# retrying is both effective and free. Second-scale waits, more attempts.
# (Until 2026-07-27 5xx shared the 429 schedule above — one retry after a full
# 60 s — and was raised as a limit error, so transient vendor faults on the biggest
# option chains looked like quota exhaustion and were written off as deferred work.)
_SERVER_ERROR_MAX_RETRIES = int(os.environ.get("EODHD_5XX_MAX_RETRIES", "3"))
_SERVER_ERROR_BASE_DELAY = float(os.environ.get("EODHD_5XX_BASE_DELAY", "2.0"))
_PAGE_DELAY = 1.0   # seconds to sleep between paginated requests
# Ceiling on in-flight HTTP requests per process, across every caller in this module.
#
# Published limit: 1000 req/min. The old fixed value of 15 was justified as
# "Semaphore(15) + 1.0s per-page delay ≈ 900 req/min", but that arithmetic assumed the
# per-page sleep dominates. Two things make it wrong in practice: the /contracts path
# calls fetch_eodhd_page directly and never sleeps, and a real request costs seconds,
# not milliseconds. Measured on Hephaestus's 473-symbol forward pull (2026-08-01):
# 4006 requests / 2292 s at 15 concurrent = ~8.6 s per request-slot = 105 req/min —
# about 10% of the published limit, not 90%.
#
# EODHD_MAX_CONCURRENT_REQUESTS is a CEILING, not a fixed operating point: a burst of
# concurrent requests can trip the vendor's real (undocumented) burst-level enforcement
# even while the smoothed per-minute average has headroom (observed 2026-08-03 — a
# sustained 429 rate at 40 concurrent while X-RateLimit-Remaining stayed near-full).
# _AdaptiveConcurrencyGate below is an AIMD control loop over that ceiling: a congestion
# signal halves the live limit immediately (EODHD's own signal beats any number picked
# ahead of time), a run of clean responses grows it back by one at a time. Congestion
# means 429 *or* 5xx — on /options/contracts the vendor times a pushed request out into a
# 500 instead of shedding it with a 429 (2026-08-05), so listening only for 429 left the
# gate deaf to the wave that mattered. This is per-consumer, same as
# the static ceiling was: Hermes's equity ingest and Hephaestus's options ingest have very
# different request latencies, so each measures and sets its own ceiling via the environment.
# Resolved lazily on first request, NOT at import: the gate cannot be resized once built
# with a stale ceiling, and consumers routinely call load_dotenv() after importing this
# module (see Hephaestus's flows/main_flow.py). Reading at import time would silently pin
# the value to the pre-.env environment and there would be no error to notice — the run
# would just be slow. Every other knob here is re-read per call, so only this one needs
# the dance.
_DEFAULT_MAX_CONCURRENT_REQUESTS = 15
_DEFAULT_ADAPTIVE_MIN_CONCURRENT_REQUESTS = 2
_DEFAULT_ADAPTIVE_GROW_AFTER_CLEAN = 20


class _AdaptiveConcurrencyGate:
    """AIMD concurrency gate: halves the live limit on a congestion signal, grows it
    back by one after a streak of clean (HTTP 200) responses. Bounded to [floor, ceiling].

    **Both 429 and 5xx count as congestion.** 429 is the obvious one, but this vendor
    does not use it for the failure that actually matters: /options/contracts answers a
    large chain slowly (SPY 25-56s at rest vs GOOG ~2s) and, when pushed, times out
    server-side into a 500 rather than rejecting with a 429. Measured 2026-08-05 — at 40
    concurrent, 69% of 473 symbols failed with 500s while zero 429s were seen and
    X-RateLimit-Remaining stayed ~96% free; the same universe at concurrency 8 failed
    under 1%. Treating only 429 as congestion left the gate pinned at its ceiling for the
    entire outage, applying no backpressure on the one endpoint that needed it.

    threading.Semaphore can't be resized once built, so this tracks in-flight count
    and a mutable limit under a Condition instead of delegating to Semaphore.
    """

    def __init__(self, ceiling: int, floor: int) -> None:
        self._ceiling = ceiling
        self._floor = min(floor, ceiling)
        self._limit = ceiling
        self._in_flight = 0
        self._clean_streak = 0
        self._cond = threading.Condition()

    def __enter__(self) -> "_AdaptiveConcurrencyGate":
        with self._cond:
            while self._in_flight >= self._limit:
                self._cond.wait()
            self._in_flight += 1
        return self

    def __exit__(self, *exc_info: Any) -> None:
        with self._cond:
            self._in_flight -= 1
            self._cond.notify_all()

    def on_congestion(self, signal: str = "429") -> None:
        """Multiplicative decrease. ``signal`` labels the trigger for the log only."""
        with self._cond:
            new_limit = max(self._floor, self._limit // 2)
            if new_limit < self._limit:
                _logger.warning(
                    "[eodhd] adaptive concurrency: %s observed — shrinking %d -> %d",
                    signal, self._limit, new_limit,
                )
            self._limit = new_limit
            self._clean_streak = 0
            self._cond.notify_all()

    def on_clean_response(self) -> None:
        grow_after = int(os.environ.get(
            "EODHD_ADAPTIVE_GROW_AFTER_CLEAN", _DEFAULT_ADAPTIVE_GROW_AFTER_CLEAN,
        ))
        with self._cond:
            if self._limit >= self._ceiling:
                self._clean_streak = 0
                return
            self._clean_streak += 1
            if self._clean_streak >= grow_after:
                self._limit = min(self._ceiling, self._limit + 1)
                self._clean_streak = 0
                _logger.info(
                    "[eodhd] adaptive concurrency: clean streak — growing to %d", self._limit,
                )
                self._cond.notify_all()


_api_semaphore: Optional[_AdaptiveConcurrencyGate] = None
_api_semaphore_lock = threading.Lock()


def _get_api_semaphore() -> _AdaptiveConcurrencyGate:
    global _api_semaphore
    if _api_semaphore is None:
        with _api_semaphore_lock:
            if _api_semaphore is None:  # re-check: another thread may have won the race
                ceiling = int(os.environ.get(
                    "EODHD_MAX_CONCURRENT_REQUESTS", _DEFAULT_MAX_CONCURRENT_REQUESTS,
                ))
                if ceiling < 1:
                    raise ValueError("EODHD_MAX_CONCURRENT_REQUESTS must be >= 1")
                floor = int(os.environ.get(
                    "EODHD_ADAPTIVE_MIN_CONCURRENT_REQUESTS",
                    _DEFAULT_ADAPTIVE_MIN_CONCURRENT_REQUESTS,
                ))
                if floor < 1:
                    raise ValueError("EODHD_ADAPTIVE_MIN_CONCURRENT_REQUESTS must be >= 1")
                _logger.info(
                    "[eodhd] request concurrency ceiling: %d (adaptive floor %d)", ceiling, floor,
                )
                _api_semaphore = _AdaptiveConcurrencyGate(ceiling, floor)
    return _api_semaphore
_logger = logging.getLogger(__name__)
DEFAULT_OPTIONS_EOD_PAGE_LIMIT = 1000
MARKETPLACE_API_CALLS_PER_REQUEST = 10

_EODHD_BASE = "https://eodhd.com/api"

# Canonical EOD row shape shared by the fetchers and app-side stores.
EOD_COLS = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]
OHLCV_FLOAT_COLS = ["open", "high", "low", "close", "adjusted_close", "volume"]


def _resolve_api_token(api_token: Optional[str]) -> str:
    """Return the explicit or environment-provided EODHD API token.

    Raises:
        ValueError: When neither an argument nor EODHD_API_KEY is available.
    """
    resolved_token = api_token or os.getenv("EODHD_API_KEY")
    if not resolved_token:
        raise ValueError("EODHD API token not provided. Set EODHD_API_KEY or pass api_token.")
    return resolved_token


def _get_with_retry(url: str) -> requests.Response:
    """
    GET *url*, retrying on the conditions EODHD documents as retryable.

    Status handling follows the vendor's own error table (see the eodhd-api skill,
    ``references/general/rate-limits.md``) — these are three unrelated failures and
    are raised as three distinct exception types so callers can act on them:

    ==== ===================================== ==============================
    Code Meaning                               Handling
    ==== ===================================== ==============================
    429  per-MINUTE request window full        few minute-scale retries
    402  daily API *call* budget spent         no retry — defer the work
    5xx  vendor-side fault (0 API calls spent) several second-scale retries
    ==== ===================================== ==============================

    ``Retry-After`` is honoured whenever present. Anything else is a hard error.

    Note for budget accounting: 429s and 5xx are **billed 0 API calls**, so a
    caller metering the provider's daily quota must NOT charge for retried
    attempts — only a 200 actually costs quota. Cost control therefore belongs
    at the *decision* point (know a symbol's page count before committing to it,
    see ``Budget.try_reserve``), not in this retry loop.
    """
    throttle_delay = _BASE_DELAY
    server_delay = _SERVER_ERROR_BASE_DELAY
    throttle_attempts = 0
    server_attempts = 0

    while True:
        gate = _get_api_semaphore()
        with gate:
            resp = requests.get(url, timeout=(10, 60))
        if resp.status_code == 200:
            gate.on_clean_response()
            return resp

        retry_after_hdr = resp.headers.get("Retry-After")
        retry_after = float(retry_after_hdr) if retry_after_hdr else None

        if resp.status_code == 402:
            # Daily budget. Resets at midnight GMT (or the marketplace timeToReset),
            # never inside a run — so there is nothing to wait for here.
            raise EODHDQuotaExhaustedError(
                f"EODHD HTTP 402 — daily API call budget exhausted: {resp.text[:200]}",
                retry_after=retry_after,
            )

        if resp.status_code == 429:
            # X-RateLimit-* describe THIS MINUTE's request window, not the daily
            # call budget — label them as such so the message can't be misread.
            # Report the burst regardless of remaining retries: a 429 at 1199/1200
            # remaining is exactly the signal that our concurrency, not the minute
            # window, is what's actually full (2026-08-03).
            gate.on_congestion("429")
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            limit = resp.headers.get("X-RateLimit-Limit", "?")
            if throttle_attempts >= _MAX_RETRIES:
                raise EODHDThrottleError(
                    f"EODHD HTTP 429 after {throttle_attempts} retries — per-minute "
                    f"request window full (requests left this minute: "
                    f"{remaining}/{limit}): {resp.text[:200]}",
                    retry_after=retry_after,
                )
            wait = retry_after or throttle_delay + random.uniform(0, throttle_delay * 0.3)
            throttle_attempts += 1
            _logger.warning(
                "[eodhd] HTTP 429 minute-window full (%s/%s left) — retry %d/%d in %.0fs",
                remaining, limit, throttle_attempts, _MAX_RETRIES, wait,
            )
            time.sleep(wait)
            throttle_delay = min(throttle_delay * 2, 900)  # grow up to 15 min
            continue

        if resp.status_code >= 500:
            # Congestion signal, not just a fault. This vendor times a slow request out
            # into a 500 instead of shedding it with a 429, so on the endpoint most prone
            # to overload (/options/contracts) 5xx is the ONLY backpressure signal we get.
            # Fires per attempt, like the 429 path: a sustained wave should collapse the
            # limit toward the floor fast, and the clean-streak rule grows it back.
            gate.on_congestion(f"HTTP {resp.status_code}")
            if server_attempts >= _SERVER_ERROR_MAX_RETRIES:
                raise EODHDServerError(
                    f"EODHD HTTP {resp.status_code} after {server_attempts} retries "
                    f"— vendor-side fault, no API calls charged: {resp.text[:200]}",
                    retry_after=retry_after,
                )
            wait = retry_after or server_delay + random.uniform(0, server_delay * 0.3)
            server_attempts += 1
            _logger.warning(
                "[eodhd] HTTP %s (vendor-side) — retry %d/%d in %.1fs",
                resp.status_code, server_attempts, _SERVER_ERROR_MAX_RETRIES, wait,
            )
            time.sleep(wait)
            server_delay = min(server_delay * 2, 60.0)
            continue

        # Non-retryable error
        raise RuntimeError(f"EODHD HTTP {resp.status_code}: {resp.text[:200]}")


def fetch_eodhd_user_details(api_token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch account usage and subscription details from EODHD."""
    resolved_token = _resolve_api_token(api_token)
    response = _get_with_retry(
        f"https://eodhd.com/api/internal-user?api_token={resolved_token}"
    )
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected EODHD user-details payload shape")
    return payload


def get_eodhd_quota_status(api_token: Optional[str] = None) -> Dict[str, Any]:
    """Return normalized quota details for main and Marketplace EODHD usage.

    Informational only — NOT a cross-process gate (FR-006). The Marketplace
    fields are the ones relevant to the options pipeline because Unicorn Bay
    options endpoints consume Marketplace quota.
    """
    payload = fetch_eodhd_user_details(api_token=api_token)

    main_api_calls_used = int(payload.get("apiRequests") or 0)
    main_daily_limit = int(payload.get("dailyRateLimit") or 0)
    main_extra_limit = int(payload.get("extraLimit") or 0)
    main_api_calls_remaining = max(main_daily_limit - main_api_calls_used, 0)

    marketplace_payload = payload.get("availableMarketplaceDataFeeds")
    has_marketplace_access = isinstance(marketplace_payload, dict)
    if has_marketplace_access:
        marketplace_daily_limit = int(marketplace_payload.get("dailyRateLimit") or 0)
        marketplace_api_calls_used = int(marketplace_payload.get("requestsSpent") or 0)
        marketplace_api_calls_remaining = max(
            marketplace_daily_limit - marketplace_api_calls_used,
            0,
        )
        marketplace_time_to_reset = marketplace_payload.get("timeToReset")
        marketplace_subscriptions = list(marketplace_payload.get("subscriptions") or [])
    else:
        marketplace_daily_limit = 0
        marketplace_api_calls_used = 0
        marketplace_api_calls_remaining = 0
        marketplace_time_to_reset = None
        marketplace_subscriptions = []

    return {
        "main_api_calls_used": main_api_calls_used,
        "main_daily_limit": main_daily_limit,
        "main_api_calls_remaining": main_api_calls_remaining,
        "main_extra_limit": main_extra_limit,
        "main_usage_date": payload.get("apiRequestsDate"),
        "main_total_available_including_extra": main_api_calls_remaining + main_extra_limit,
        "marketplace_api_calls_per_request": MARKETPLACE_API_CALLS_PER_REQUEST,
        "has_marketplace_access": has_marketplace_access,
        "marketplace_daily_limit": marketplace_daily_limit,
        "marketplace_api_calls_used": marketplace_api_calls_used,
        "marketplace_api_calls_remaining": marketplace_api_calls_remaining,
        "marketplace_requests_remaining": (
            marketplace_api_calls_remaining // MARKETPLACE_API_CALLS_PER_REQUEST
        ),
        "marketplace_time_to_reset": marketplace_time_to_reset,
        "marketplace_subscriptions": marketplace_subscriptions,
        "raw": payload,
    }


def build_options_eod_url(
    underlying_symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page_limit: int = DEFAULT_OPTIONS_EOD_PAGE_LIMIT,
    page_offset: int = 0,
    compact: bool = False,
    fields: Optional[list[str]] = None,
    option_type: Optional[str] = None,
    sort: str = "-exp_date",
) -> str:
    """
    Build a Unicorn Bay options/eod URL using documented tradetime filters.

    When *start_date* and *end_date* are equal, the request uses
    ``filter[tradetime_eq]`` to target a single trading day. Otherwise it uses
    ``filter[tradetime_from]`` / ``filter[tradetime_to]``.

    *option_type* may be ``'call'`` or ``'put'`` to apply ``filter[type]``.

    *sort* defaults to ``-exp_date`` (longest expiry first), which the bulk ingest
    relies on. Pass ``exp_date`` when the caller needs the **near-dated** end of the
    chain instead: the response is capped at ``page[limit]`` (1000), so descending
    expiry can return a page consisting entirely of LEAPS and leave a caller that
    filters on a short DTE window with nothing at all — measured on SPY, where a
    descending page yields **zero** contracts inside a 5-200 DTE band on every date
    tested, while an ascending page yields 114-268.
    """
    base_symbol = underlying_symbol.replace(".US", "").upper()
    # EODHD expects BRK.B effectively as BRK-B (or just BRK-B) in some endpoints.
    # For the options/eod endpoint, if a dot is present, we likely need to swap it.
    base_symbol = base_symbol.replace(".", "-")

    params: list[tuple[str, str | int]] = [
        ("filter[underlying_symbol]", base_symbol),
        ("page[offset]", page_offset),
        ("page[limit]", min(page_limit, DEFAULT_OPTIONS_EOD_PAGE_LIMIT)),
        ("compact", 1 if compact else 0),
        ("sort", sort),
    ]

    if option_type:
        params.append(("filter[type]", option_type))

    if start_date and end_date and start_date == end_date:
        params.append(("filter[tradetime_eq]", start_date))
    else:
        if start_date:
            params.append(("filter[tradetime_from]", start_date))
        if end_date:
            params.append(("filter[tradetime_to]", end_date))

    if fields:
        params.append(("fields[options-eod]", ",".join(fields)))

    return f"https://eodhd.com/api/mp/unicornbay/options/eod?{urlencode(params)}"


def _normalize_underlying(underlying_symbol: str) -> str:
    """Vendor-normalized underlying (drop .US, upper, dot->dash for BRK.B etc.)."""
    return underlying_symbol.replace(".US", "").upper().replace(".", "-")


def build_options_contracts_url(
    underlying_symbol: str,
    page_offset: int = 0,
    page_limit: int = DEFAULT_OPTIONS_EOD_PAGE_LIMIT,
    fields: Optional[list[str]] = None,
    option_type: Optional[str] = None,
) -> str:
    """Build an options/contracts URL for the full *current* chain of an underlying.

    Returns one row per live contract (current EOD snapshot, incl. untraded) — used to
    capture today's chain and to enumerate the live expiry set. No tradetime filter.

    ``option_type`` ("call"/"put") halves the chain. Verified honoured 2026-08-05 (a
    filtered pull returned calls only), and it is the lever for the chains that sit on
    the vendor's ~56s server-side timeout: SPY answers in 25-56s and intermittently
    500s even at one request per 45 seconds, so fetching it as two smaller requests is
    what gets each under the limit. Note ``meta`` carries NO ``total`` on this endpoint,
    so a chain's size cannot be priced up front — narrowing the request is the only
    control available.
    """
    params: list[tuple[str, str | int]] = [
        ("filter[underlying_symbol]", _normalize_underlying(underlying_symbol)),
        ("page[offset]", page_offset),
        ("page[limit]", min(page_limit, DEFAULT_OPTIONS_EOD_PAGE_LIMIT)),
        ("compact", 0),
        ("sort", "-exp_date"),
    ]
    if option_type:
        params.append(("filter[type]", option_type))
    if fields:
        params.append(("fields[options-contracts]", ",".join(fields)))
    return f"https://eodhd.com/api/mp/unicornbay/options/contracts?{urlencode(params)}"


def build_options_eod_history_url(
    underlying_symbol: str,
    *,
    exp_date: Optional[str] = None,
    option_type: Optional[str] = None,
    strike_from: Optional[float] = None,
    strike_to: Optional[float] = None,
    page_offset: int = 0,
    page_limit: int = DEFAULT_OPTIONS_EOD_PAGE_LIMIT,
    fields: Optional[list[str]] = None,
) -> str:
    """Build an options/eod URL for the historical walk — scoped by expiry/type/strike, NOT
    tradetime. Default order is exp_date desc then snapshot desc, so paging an exp_date_eq
    slice yields newest snapshots first (walk-to-watermark). strike_from/to sub-chunk a slice
    that exceeds the 10k offset cap.
    """
    params: list[tuple[str, str | int | float]] = [
        ("filter[underlying_symbol]", _normalize_underlying(underlying_symbol)),
        ("page[offset]", page_offset),
        ("page[limit]", min(page_limit, DEFAULT_OPTIONS_EOD_PAGE_LIMIT)),
        ("compact", 0),
        ("sort", "-exp_date"),
    ]
    if exp_date:
        params.append(("filter[exp_date_eq]", exp_date))
    if option_type:
        params.append(("filter[type]", option_type))
    if strike_from is not None:
        params.append(("filter[strike_from]", strike_from))
    if strike_to is not None:
        params.append(("filter[strike_to]", strike_to))
    if fields:
        params.append(("fields[options-eod]", ",".join(fields)))
    return f"https://eodhd.com/api/mp/unicornbay/options/eod?{urlencode(params)}"


def build_options_contract_url(
    contract: str,
    fields: Optional[list[str]] = None,
) -> str:
    """Build a Unicorn Bay options/contracts URL for a single OCC contract."""
    params: list[tuple[str, str | int]] = [
        ("filter[contract]", contract.replace(" ", "")),
        ("page[offset]", 0),
        ("page[limit]", 1),
    ]

    if fields:
        params.append(("fields[options-contracts]", ",".join(fields)))

    return f"https://eodhd.com/api/mp/unicornbay/options/contracts?{urlencode(params)}"


def _normalize_eod_records(
    payload_data: list[Any],
    fields: Optional[list[str]],
) -> list[Dict[str, Any]]:
    """Normalize normal-mode and compact-mode options/eod rows to flat dicts."""
    records: list[Dict[str, Any]] = []
    for record in payload_data:
        if isinstance(record, dict):
            attributes = record.get("attributes")
            if isinstance(attributes, dict):
                flat = dict(attributes)
                # The options/eod id is "{contract}-{YYYY-MM-DD}"; its date suffix is the vendor's
                # canonical snapshot/session date — the authoritative source for trade_date
                # (tradetime is last-activity and stale; bid_date straddles midnight ET). The
                # options/contracts id has NO date suffix, so set session_date only when the id
                # genuinely ends in a date (else the transform falls back to ask_date ET).
                rid = record.get("id")
                if isinstance(rid, str) and len(rid) >= 11 and rid[-11] == "-":
                    try:
                        date.fromisoformat(rid[-10:])
                        flat["session_date"] = rid[-10:]
                    except ValueError:
                        pass
                records.append(flat)
            else:
                records.append(dict(record))
            continue
        if isinstance(record, list) and fields:
            records.append(dict(zip(fields, record)))
    return records


def fetch_eodhd_page(
    url: str,
    api_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch a single Unicorn Bay page and normalize its rows.

    The returned ``meta`` carries the vendor's ``total`` record count — that is
    what lets a caller price the rest of a chain from page 1 alone, instead of
    discovering the cost by paging blindly (see ``Budget.try_reserve``).
    """
    api_token = _resolve_api_token(api_token)

    separator = "&" if "?" in url else "?"
    url_with_token = f"{url}{separator}api_token={api_token}"
    response = _get_with_retry(url_with_token)
    payload = response.json()
    meta = payload.get("meta", {})
    fields = meta.get("fields")
    records = _normalize_eod_records(payload.get("data", []), fields)
    return {
        "meta": meta,
        "data": records,
        "next_url": payload.get("links", {}).get("next"),
    }


def fetch_eodhd_options(initial_url: str, api_token: Optional[str] = None) -> pl.DataFrame:
    """
    Fetch paginated EODHD options API data and return a Polars DataFrame.
    Fetches all pages until no more data is available.
    Returns raw API data with no transformations.

    Args:
        initial_url: EODHD options/eod endpoint URL built with build_options_eod_url(...)
        api_token: EODHD API authentication token (optional, defaults to env var EODHD_API_KEY)

    Returns:
        Polars DataFrame with raw EODHD options data (no transformations)

    Raises:
        ValueError: If API token not provided and EODHD_API_KEY not set
        RuntimeError: If API request fails
    """
    api_token = _resolve_api_token(api_token)

    next_url = initial_url
    records: List[Dict[str, Any]] = []

    while next_url:
        page = fetch_eodhd_page(next_url, api_token=api_token)
        data = page["data"]
        if not data:
            break
        records.extend(data)
        next_url = page["next_url"]
        if next_url:
            time.sleep(_PAGE_DELAY)  # avoid burst throttle between pages

    if not records:
        return pl.DataFrame([])

    # Accumulate per column while normalising datetime values — one pass, no
    # intermediate list of dicts.
    cols: dict[str, list] = {}
    for record in records:
        for key, value in record.items():
            if key not in cols:
                cols[key] = []
            if isinstance(value, datetime):
                cols[key].append(value.isoformat(sep=" "))
            elif isinstance(value, date):
                cols[key].append(value.isoformat())
            else:
                cols[key].append(value)

    # Build with type inference first (the vendor sometimes returns float values in
    # nominally-integer columns, e.g. open_interest=22.5, which a strict Int64
    # construction rejects). Then coerce the integer columns non-strictly — matching the
    # downstream transform_eod_data cast — so a stray float truncates instead of crashing
    # the whole fetch.
    # strict=False lets Polars pick a supertype when a column mixes int and float python
    # values (e.g. volume=[10, 7.0]); a strict build crashes on the first such column.
    _INT64_COLS = {"open_interest", "volume", "last_size", "bid_size", "ask_size"}
    df = pl.DataFrame(cols, strict=False)
    int_cols = [col for col in _INT64_COLS if col in df.columns]
    if int_cols:
        df = df.with_columns([pl.col(col).cast(pl.Int64, strict=False) for col in int_cols])
    return df


def fetch_eodhd_contract_snapshot(
    contract: str,
    api_token: Optional[str] = None,
    fields: Optional[list[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch the latest Unicorn Bay snapshot for a single OCC contract."""
    request_fields = fields or [
        "contract",
        "underlying_symbol",
        "bid_date",
        "ask_date",
        "tradetime",
        "delta",
        "gamma",
        "theta",
        "vega",
        "volatility",
        "midpoint",
    ]
    url = build_options_contract_url(contract, fields=request_fields)
    page = fetch_eodhd_page(url, api_token=api_token)
    data = page.get("data", [])
    return data[0] if data else None


# ---------------------------------------------------------------------------
# EOD / bulk / intraday price fetchers (promoted from Hermes io/ingest/eodhd/client.py)
# ---------------------------------------------------------------------------


def _payload_summary(payload: Any, max_length: int = 200) -> str:
    try:
        rendered = str(payload)
    except Exception:
        return "<unprintable payload>"
    rendered = re.sub(r"\s+", " ", rendered).strip()
    if len(rendered) <= max_length:
        return rendered
    return f"{rendered[:max_length - 3]}..."


def _is_limit_message(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("too many", "throttl", "rate limit", "quota", "api limit", "daily limit", "requests limit")
    )


def _decode_eodhd_payload(resp: requests.Response, request_label: str) -> Any:
    if resp.status_code == 429:
        raise EODHDThrottleError(f"EODHD throttled request for {request_label}: HTTP 429")
    response_text = (resp.text or "").strip()
    if resp.status_code in {402, 403} and _is_limit_message(response_text):
        # 402 is the documented daily-call-budget wall; 403 is an entitlement/auth
        # refusal that merely carries a limit-shaped message — not the same thing.
        err = EODHDQuotaExhaustedError if resp.status_code == 402 else EODHDLimitError
        raise err(
            f"EODHD rejected request for {request_label}: {_payload_summary(response_text)}"
        )
    try:
        payload = resp.json()
    except ValueError:
        if _is_limit_message(response_text):
            raise EODHDLimitError(
                f"EODHD rejected request for {request_label}: {_payload_summary(response_text)}"
            )
        raise
    if isinstance(payload, dict):
        summary = _payload_summary(payload)
        if _is_limit_message(summary):
            raise EODHDLimitError(f"EODHD rejected request for {request_label}: {summary}")
    return payload


def fetch_eod(
    ticker: str,
    exchange: str,
    history_from: str = "2000-01-01",
    history_to: Optional[str] = None,
    api_key: str = "",
) -> pd.DataFrame:
    key = api_key or os.environ.get("EODHD_API_KEY", "")
    if not key:
        _logger.error("EODHD_API_KEY is not set — cannot fetch %s.%s", ticker, exchange)
        return pd.DataFrame()

    symbol_full = f"{ticker}.{exchange}"
    date_to = history_to or date.today().isoformat()

    try:
        resp = requests.get(
            f"{_EODHD_BASE}/eod/{symbol_full}",
            params={"api_token": key, "fmt": "json", "from": history_from, "to": date_to},
            timeout=30,
        )
    except requests.RequestException as exc:
        _logger.error("Network error fetching %s: %s", symbol_full, exc)
        return pd.DataFrame()

    if resp.status_code == 404:
        _logger.warning("%s: 404 not found", symbol_full)
        return pd.DataFrame()
    if resp.status_code == 429:
        raise EODHDThrottleError(f"EODHD throttled request for {symbol_full}: HTTP 429")

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        _logger.error("HTTP error fetching %s: %s", symbol_full, exc)
        return pd.DataFrame()

    data = _decode_eodhd_payload(resp, symbol_full)
    if isinstance(data, dict):
        _logger.warning("%s: unexpected response payload: %s", symbol_full, _payload_summary(data))
        return pd.DataFrame()
    if not data:
        _logger.info("%s: no rows returned for requested range", symbol_full)
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = ticker.upper()
    df["exchange"] = exchange
    keep = EOD_COLS + ["symbol", "exchange"]
    df = df[[c for c in keep if c in df.columns]]
    for col in OHLCV_FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    return df


def fetch_bulk_eod_for_exchange(
    exchange_code: str,
    date_str: str,
    api_key: str = "",
) -> pd.DataFrame:
    key = api_key or os.environ.get("EODHD_API_KEY", "")
    if not key:
        return pd.DataFrame()
    try:
        resp = requests.get(
            f"{_EODHD_BASE}/eod-bulk-last-day/{exchange_code}",
            params={"api_token": key, "date": date_str, "fmt": "json"},
            timeout=60,
        )
        if resp.status_code == 404:
            _logger.debug("fetch_bulk_eod_for_exchange: 404 for %s on %s", exchange_code, date_str)
            return pd.DataFrame()
        if resp.status_code == 429:
            raise EODHDThrottleError(
                f"EODHD throttled bulk request for {exchange_code} on {date_str}: HTTP 429"
            )
        resp.raise_for_status()
        data = _decode_eodhd_payload(resp, f"{exchange_code} on {date_str}")
        if isinstance(data, dict):
            _logger.warning(
                "fetch_bulk_eod_for_exchange %s (%s): unexpected payload: %s",
                exchange_code, date_str, _payload_summary(data),
            )
            return pd.DataFrame()
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "code" in df.columns:
            df = df.rename(columns={"code": "symbol"})
        df["symbol"] = df["symbol"].str.upper()
        df["exchange"] = exchange_code
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        keep = EOD_COLS + ["symbol", "exchange"]
        df = df[[c for c in keep if c in df.columns]]
        for col in OHLCV_FLOAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype("float64")
        _logger.debug("fetch_bulk_eod_for_exchange: %s on %s → %d rows", exchange_code, date_str, len(df))
        return df
    except EODHDLimitError:
        raise
    except Exception as exc:
        _logger.warning("fetch_bulk_eod_for_exchange %s (%s): %s", exchange_code, date_str, exc)
        return pd.DataFrame()


def fetch_exchange_catalog(api_key: str = "") -> list[dict[str, Any]]:
    """Fetch the EODHD ``exchanges-list`` catalog (all exchanges + country/currency).

    Returns the raw list of exchange records, or an empty list on error / missing
    key. Raises :class:`EODHDLimitError` on throttling. Per-process caching and any
    domain shaping are the caller's responsibility.
    """
    key = api_key or os.environ.get("EODHD_API_KEY", "")
    if not key:
        _logger.error("EODHD_API_KEY is not set — cannot fetch exchange list")
        return []
    try:
        resp = requests.get(
            f"{_EODHD_BASE}/exchanges-list/",
            params={"api_token": key, "fmt": "json"},
            timeout=30,
        )
        if resp.status_code == 429:
            raise EODHDThrottleError("EODHD throttled exchanges-list request: HTTP 429")
        resp.raise_for_status()
        payload = _decode_eodhd_payload(resp, "exchanges-list")
        return payload if isinstance(payload, list) else []
    except EODHDLimitError:
        raise
    except Exception as exc:
        _logger.error("Failed to fetch EODHD exchange list: %s", exc)
        return []


def fetch_exchange_symbol_list(exchange_code: str, api_key: str = "") -> list[dict[str, Any]]:
    """Fetch one exchange's raw ``exchange-symbol-list`` records from EODHD.

    Returns the raw list of records (each carrying ``Code``/``Exchange``/``Type``/...),
    or an empty list on 404 / error / missing key. Raises :class:`EODHDLimitError`
    on throttling. Keying and record shaping are the caller's responsibility.
    """
    key = api_key or os.environ.get("EODHD_API_KEY", "")
    if not key:
        return []
    try:
        resp = requests.get(
            f"{_EODHD_BASE}/exchange-symbol-list/{exchange_code}",
            params={"api_token": key, "fmt": "json"},
            timeout=60,
        )
        if resp.status_code == 404:
            _logger.debug("exchange-symbol-list: 404 for exchange %s", exchange_code)
            return []
        if resp.status_code == 429:
            raise EODHDLimitError(
                f"EODHD throttled exchange-symbol-list request for {exchange_code}: HTTP 429"
            )
        resp.raise_for_status()
        payload = _decode_eodhd_payload(resp, f"exchange-symbol-list {exchange_code}")
        return payload if isinstance(payload, list) else []
    except EODHDLimitError:
        raise
    except Exception as exc:
        _logger.warning("Failed to fetch symbol list for exchange %s: %s", exchange_code, exc)
        return []


# Per-interval maximum window EODHD accepts in a single intraday request (days).
# Longer ranges are silently truncated by the API, so callers must chunk.
INTRADAY_MAX_WINDOW_DAYS: dict[str, int] = {"1m": 120, "5m": 600, "1h": 7200}

# Earliest date EODHD has intraday history for, per interval (US equities). 1-min
# goes back to 2004; 5-min and 1-hour only to October 2020. Requests before the
# floor return empty (still billed 5 calls), so backfills should clamp the start.
# None = no documented floor (1m is the deepest, though it varies by ticker).
INTRADAY_HISTORY_START: dict[str, "date | None"] = {
    "1m": None, "5m": date(2020, 10, 1), "1h": date(2020, 10, 1),
}

_INTRADAY_COLS = ["timestamp", "datetime", "open", "high", "low", "close", "volume"]
_INTRADAY_FLOAT_COLS = ["open", "high", "low", "close", "volume"]


def fetch_intraday(
    ticker: str,
    exchange: str,
    interval: str,
    from_ts: int,
    to_ts: int,
    api_key: str = "",
) -> pd.DataFrame:
    """Fetch intraday OHLCV bars for one symbol over a Unix-timestamp (UTC) window.

    ``interval`` is one of ``1m``/``5m``/``1h``. EODHD caps a single request at
    120/600/7200 days respectively (see ``INTRADAY_MAX_WINDOW_DAYS``); ranges longer
    than the cap are silently truncated by the API, so callers must chunk.

    Returns bars in **UTC** with columns
    ``[symbol, exchange, timestamp, datetime, open, high, low, close, volume]``
    (``timestamp`` is the candle's opening Unix time). Empty frame on no data / 404.
    Raises ``EODHDLimitError`` on throttling or quota exhaustion.

    Bars are **unadjusted** for splits/dividends — within-day log returns are unaffected
    (same-day constant scale cancels), but any cross-day return must be adjusted by the
    EOD ``adjusted_close/close`` factor. Each request consumes 5 API calls.
    """
    key = api_key or os.environ.get("EODHD_API_KEY", "")
    if not key:
        _logger.error("EODHD_API_KEY is not set — cannot fetch intraday %s.%s", ticker, exchange)
        return pd.DataFrame()

    if interval not in INTRADAY_MAX_WINDOW_DAYS:
        raise ValueError(f"fetch_intraday: unsupported interval {interval!r} (expected 1m/5m/1h)")

    symbol_full = f"{ticker}.{exchange}"
    span_days = (int(to_ts) - int(from_ts)) / 86400.0
    if span_days > INTRADAY_MAX_WINDOW_DAYS[interval]:
        _logger.warning(
            "fetch_intraday %s %s: requested %.0fd exceeds %dd cap — API will truncate; chunk the range",
            symbol_full, interval, span_days, INTRADAY_MAX_WINDOW_DAYS[interval],
        )

    try:
        resp = requests.get(
            f"{_EODHD_BASE}/intraday/{symbol_full}",
            params={
                "api_token": key,
                "interval": interval,
                "from": int(from_ts),
                "to": int(to_ts),
                "fmt": "json",
            },
            timeout=60,
        )
    except requests.RequestException as exc:
        _logger.error("Network error fetching intraday %s: %s", symbol_full, exc)
        return pd.DataFrame()

    if resp.status_code == 404:
        _logger.warning("intraday %s: 404 not found", symbol_full)
        return pd.DataFrame()
    if resp.status_code == 429:
        raise EODHDThrottleError(f"EODHD throttled intraday request for {symbol_full}: HTTP 429")

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        _logger.error("HTTP error fetching intraday %s: %s", symbol_full, exc)
        return pd.DataFrame()

    data = _decode_eodhd_payload(resp, f"intraday {symbol_full}")
    if isinstance(data, dict):
        _logger.warning("intraday %s: unexpected response payload: %s", symbol_full, _payload_summary(data))
        return pd.DataFrame()
    if not data:
        _logger.info("intraday %s: no bars returned for requested window", symbol_full)
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df[[c for c in _INTRADAY_COLS if c in df.columns]].copy()
    # EODHD 'datetime' is a UTC string; keep timestamp (int) as the canonical UTC clock.
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for col in _INTRADAY_FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    df["symbol"] = ticker.upper()
    df["exchange"] = exchange
    return df


# ---------------------------------------------------------------------------
# Contract facade (contracts/ingest-core-api.md)
# ---------------------------------------------------------------------------


class EODHDClient:
    """Thin object facade over the module functions, per the library contract.

    Holds the api key and an optional :class:`RateLimiter`; ``fetch_page``
    acquires the limiter before each HTTP request when one is supplied.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        rate_limiter: Optional[RateLimiter] = None,
        base_url: str = _EODHD_BASE,
    ) -> None:
        self.api_key = _resolve_api_token(api_key)
        self.rate_limiter = rate_limiter
        self.base_url = base_url

    def build_url(self, endpoint: str, **params: Any) -> str:
        if endpoint == "options-eod":
            return build_options_eod_url(**params)
        if endpoint == "options-contracts":
            return build_options_contracts_url(**params)
        if endpoint == "options-eod-history":
            return build_options_eod_history_url(**params)
        if endpoint == "options-contract":
            return build_options_contract_url(**params)
        raise ValueError(f"EODHDClient.build_url: unknown endpoint {endpoint!r}")

    def fetch_page(self, url: str) -> Dict[str, Any]:
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        return fetch_eodhd_page(url, api_token=self.api_key)

    def decode(self, page: Dict[str, Any]) -> list[Dict[str, Any]]:
        return list(page.get("data", []))

    def quota_status(self) -> Dict[str, Any]:
        return get_eodhd_quota_status(api_token=self.api_key)
