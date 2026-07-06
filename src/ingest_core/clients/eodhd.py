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
from ingest_core.types import EODHDLimitError

# Maximum retries and base delay (seconds) for 429 / 5xx responses.
# In practice a 429 here is the daily quota wall, not a transient service
# blip — retrying can't help, so default to ONE retry (was 4) and let the
# caller treat the failure as deferred work. Env-tunable for burst-throttle
# environments where waiting does help.
_MAX_RETRIES = int(os.environ.get("EODHD_HTTP_MAX_RETRIES", "1"))
_BASE_DELAY = 60.0  # first retry waits ~60 s; doubles each attempt + jitter
_PAGE_DELAY = 1.0   # seconds to sleep between paginated requests
# Published limit: 1000 req/min. Semaphore(15) + 1.0s per-page delay ≈ 900 req/min (10% headroom).
_MAX_CONCURRENT_REQUESTS = 15
_api_semaphore = threading.Semaphore(_MAX_CONCURRENT_REQUESTS)
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
    GET *url* with exponential back-off on HTTP 429 (and transient 5xx).

    Respects the ``Retry-After`` response header when present.  Raises
    ``EODHDLimitError`` after *_MAX_RETRIES* exhausted.
    """
    delay = _BASE_DELAY
    for attempt in range(_MAX_RETRIES + 1):
        with _api_semaphore:
            resp = requests.get(url, timeout=(10, 60))
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429 or resp.status_code >= 500:
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            limit = resp.headers.get("X-RateLimit-Limit", "?")
            retry_after = resp.headers.get("Retry-After")
            if attempt == _MAX_RETRIES:
                raise EODHDLimitError(
                    f"EODHD HTTP {resp.status_code} after {_MAX_RETRIES} retries "
                    f"(quota {remaining}/{limit}): {resp.text[:200]}",
                    retry_after=float(retry_after) if retry_after else None,
                )
            wait = float(retry_after) if retry_after else delay + random.uniform(0, delay * 0.3)
            _logger.warning(
                "[eodhd] HTTP %s (quota %s/%s) — retry %d/%d in %.0fs",
                resp.status_code, remaining, limit, attempt + 1, _MAX_RETRIES, wait,
            )
            time.sleep(wait)
            delay = min(delay * 2, 900)  # grow up to 15 min
            continue
        # Non-retryable error
        raise RuntimeError(f"EODHD HTTP {resp.status_code}: {resp.text[:200]}")
    # unreachable
    raise RuntimeError("Retry loop exhausted")


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
) -> str:
    """
    Build a Unicorn Bay options/eod URL using documented tradetime filters.

    When *start_date* and *end_date* are equal, the request uses
    ``filter[tradetime_eq]`` to target a single trading day. Otherwise it uses
    ``filter[tradetime_from]`` / ``filter[tradetime_to]``.

    *option_type* may be ``'call'`` or ``'put'`` to apply ``filter[type]``.
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
        ("sort", "-exp_date"),
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
) -> str:
    """Build an options/contracts URL for the full *current* chain of an underlying.

    Returns one row per live contract (current EOD snapshot, incl. untraded) — used to
    capture today's chain and to enumerate the live expiry set. No tradetime filter.
    """
    params: list[tuple[str, str | int]] = [
        ("filter[underlying_symbol]", _normalize_underlying(underlying_symbol)),
        ("page[offset]", page_offset),
        ("page[limit]", min(page_limit, DEFAULT_OPTIONS_EOD_PAGE_LIMIT)),
        ("compact", 0),
        ("sort", "-exp_date"),
    ]
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
    """Fetch a single Unicorn Bay page and normalize its rows."""
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
        raise EODHDLimitError(f"EODHD throttled request for {request_label}: HTTP 429")
    response_text = (resp.text or "").strip()
    if resp.status_code in {402, 403} and _is_limit_message(response_text):
        raise EODHDLimitError(
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
        raise EODHDLimitError(f"EODHD throttled request for {symbol_full}: HTTP 429")

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
            raise EODHDLimitError(
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


# Per-interval maximum window EODHD accepts in a single intraday request (days).
# Longer ranges are silently truncated by the API, so callers must chunk.
_INTRADAY_MAX_WINDOW_DAYS: dict[str, int] = {"1m": 120, "5m": 600, "1h": 7200}

# Earliest date EODHD has intraday history for, per interval (US equities). 1-min
# goes back to 2004; 5-min and 1-hour only to October 2020. Requests before the
# floor return empty (still billed 5 calls), so backfills should clamp the start.
# None = no documented floor (1m is the deepest, though it varies by ticker).
_INTRADAY_HISTORY_START: dict[str, "date | None"] = {
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
    120/600/7200 days respectively (see ``_INTRADAY_MAX_WINDOW_DAYS``); ranges longer
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

    if interval not in _INTRADAY_MAX_WINDOW_DAYS:
        raise ValueError(f"fetch_intraday: unsupported interval {interval!r} (expected 1m/5m/1h)")

    symbol_full = f"{ticker}.{exchange}"
    span_days = (int(to_ts) - int(from_ts)) / 86400.0
    if span_days > _INTRADAY_MAX_WINDOW_DAYS[interval]:
        _logger.warning(
            "fetch_intraday %s %s: requested %.0fd exceeds %dd cap — API will truncate; chunk the range",
            symbol_full, interval, span_days, _INTRADAY_MAX_WINDOW_DAYS[interval],
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
        raise EODHDLimitError(f"EODHD throttled intraday request for {symbol_full}: HTTP 429")

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
