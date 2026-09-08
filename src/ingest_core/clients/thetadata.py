"""ThetaData API client — the options-quote acquisition path (replaces EODHD options).

Acquisition only: returns Polars frames, owns no storage and no domain logic, same
boundary ``eodhd.py`` holds. Callers own persistence, schema mapping, and any
derivation (Hephaestus derives IV/greeks from these raw quotes — see
``processing/transforms/implied_vol.py``).

Why this module is so much smaller than ``eodhd.py``
---------------------------------------------------
EODHD's options ingest needed two endpoints (``/contracts`` for the live chain,
``/options/eod`` for a dated session), and almost all of that client's bulk exists
to reconcile them: session-id derivation because ``/contracts`` carries no date, an
AIMD concurrency gate because the vendor timed large chains out into 500s, a
per-request ``Budget`` because calls were metered at 10 API calls each, page
arithmetic against a 10,000-row offset cap, and call/put splitting to get under it.

``option_history_eod`` needs none of that. One call takes a symbol and a *date
range*, returns the full chain for every session in it — traded or not — with each
row stamped with the session it belongs to. Paid tiers are unmetered. So there is
no budget, no quota probe, no pagination, no session probe, and no adaptive gate:
just a concurrency ceiling and retries on transport faults.

Connection
----------
The ``thetadata`` package speaks gRPC to a **remote** host resolved during auth —
there is no local Theta Terminal daemon to run. Constructing the client performs a
network round-trip, so it is built lazily and cached per process.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from datetime import date
from typing import Any, Iterable, Optional, Sequence

import polars as pl

from ingest_core.types import (
    ThetaDataError,
    ThetaDataPermissionError,
    ThetaDataTransientError,
)

logger = logging.getLogger(__name__)

# Value tier documents 2 concurrent requests (Standard 4, Professional 8). This is a
# ceiling, not a target, and unlike EODHD's it is NOT adaptive: nothing observed in the
# vendor evaluation warranted a control loop. EODHD needed AIMD because /contracts fell
# over server-side under concurrency and shed load as 500s rather than 429s; ThetaData
# served a 14,100-row SPY chain in 1.86s and a 12-call full-history pull with zero
# rate-limit errors. Raise this only with a measured run behind it.
_DEFAULT_MAX_CONCURRENT = 2

# Transport faults only (see ThetaDataTransientError). Second-scale, because there is no
# metered budget being burned by a retry — only wall-clock time.
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 2.0

_client_lock = threading.Lock()
_client: Any = None
_gate: Optional[threading.Semaphore] = None
_gate_size: int = 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def _get_gate() -> threading.Semaphore:
    """Process-wide concurrency ceiling, resolved lazily on first use.

    Lazily and not at import for the same reason ``eodhd.py`` does it: a semaphore
    cannot be resized once built, and consumers routinely call ``load_dotenv()``
    *after* importing this module. Reading at import time would silently pin the
    value to the pre-.env environment, and there would be no error to notice — the
    run would just be slower or more aggressive than configured.
    """
    global _gate, _gate_size
    if _gate is None:
        with _client_lock:
            if _gate is None:
                _gate_size = max(1, _env_int("THETADATA_MAX_CONCURRENT", _DEFAULT_MAX_CONCURRENT))
                _gate = threading.Semaphore(_gate_size)
                logger.info("thetadata: concurrency ceiling %d", _gate_size)
    return _gate


def _import_thetadata() -> tuple[Any, Any]:
    """Import the optional vendor package, or explain how to install it.

    ``thetadata`` is an optional extra (see pyproject). A bare ModuleNotFoundError
    here reads as a broken install rather than a missing extra, which is a confusing
    place to land when the rest of ingest-core imported fine.
    """
    try:
        from thetadata import ThetaClient
        from thetadata.errors import NoDataFoundError
    except ImportError as exc:
        raise ThetaDataError(
            "the 'thetadata' package is not installed; ingest-core declares it as an "
            "optional extra: pip install -e '<path-to>/ingest-core[thetadata]'"
        ) from exc
    return ThetaClient, NoDataFoundError


def get_client() -> Any:
    """The shared :class:`thetadata.ThetaClient`, built on first use and cached.

    Constructing it authenticates over the network, so this is not free and must not
    be called per request.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                api_key = os.environ.get("THETADATA_API_KEY")
                if not api_key:
                    raise ThetaDataError(
                        "THETADATA_API_KEY is not set; cannot construct a ThetaData client"
                    )
                theta_client_cls, _ = _import_thetadata()
                _client = theta_client_cls(api_key=api_key, dataframe_type="polars")
                logger.info("thetadata: client authenticated")
    return _client


def reset_client() -> None:
    """Drop the cached client and concurrency gate. For tests and credential rotation."""
    global _client, _gate, _gate_size
    with _client_lock:
        _client = None
        _gate = None
        _gate_size = 0


def _classify(exc: BaseException) -> BaseException:
    """Map a vendor/transport exception onto this module's taxonomy.

    Branches on the gRPC status code, never on message text (constitution V). An
    unrecognised code becomes a plain :class:`ThetaDataError` rather than a transient
    one, so it fails closed: a caller retrying an unknown fault forever is worse than
    one surfacing it.
    """
    try:
        import grpc
    except ImportError:  # pragma: no cover - grpc ships with thetadata
        return ThetaDataError(str(exc))

    if not isinstance(exc, grpc.RpcError):
        return ThetaDataError(str(exc))

    code = exc.code() if hasattr(exc, "code") else None
    detail = exc.details() if hasattr(exc, "details") else str(exc)

    if code == grpc.StatusCode.PERMISSION_DENIED:
        return ThetaDataPermissionError(f"{code}: {detail}")
    if code in (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    ):
        return ThetaDataTransientError(f"{code}: {detail}")
    return ThetaDataError(f"{code}: {detail}")


def _call(method_name: str, **kwargs: Any) -> pl.DataFrame:
    """Invoke one client method under the concurrency gate, with transient retries.

    Returns an **empty frame** when the vendor reports no data. That is not an error
    condition here: an empty range is how a symbol's listing floor is discovered and
    how genuinely-empty sessions earn their store sentinel. Raising would turn a
    normal, informative outcome into a failure the caller has to unwind.
    """
    max_retries = _env_int("THETADATA_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
    base_delay = _DEFAULT_BASE_DELAY
    client = get_client()
    _, no_data_error = _import_thetadata()
    method = getattr(client, method_name)

    attempt = 0
    while True:
        try:
            with _get_gate():
                return method(**kwargs)
        except no_data_error:
            return pl.DataFrame()
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error below
            mapped = _classify(exc)
            if not isinstance(mapped, ThetaDataTransientError) or attempt >= max_retries:
                raise mapped from exc
            delay = base_delay * (2**attempt) + random.uniform(0, 1.0)
            logger.warning(
                "thetadata: %s transient failure (%s), retry %d/%d in %.1fs",
                method_name, mapped, attempt + 1, max_retries, delay,
            )
            time.sleep(delay)
            attempt += 1


def fetch_option_eod(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    right: str = "both",
) -> pl.DataFrame:
    """Full option chain EOD quotes for ``symbol`` over ``[start_date, end_date]``.

    One request covers the whole range and the whole chain. Returned columns:
    ``symbol, expiration, strike, right, created, last_trade, open, high, low, close,
    volume, count, bid_size, bid_exchange, bid, bid_condition, ask_size, ask_exchange,
    ask, ask_condition``.

    ``created`` is the datetime the vendor generated its national EOD report (17:15 ET
    each session) and is tz-aware ``America/New_York``. It is the authoritative session
    stamp — the field that removes every session-mislabel failure mode the EODHD path
    had to defend against. The quote is the last NBBO as of that instant.

    No open interest: that is a separate endpoint, deliberately not pulled (nothing
    downstream reads it).
    """
    return _call(
        "option_history_eod",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        expiration="*",
        right=right,
    )


def fetch_option_contracts(
    on_date: date,
    symbols: Optional[Sequence[str]] = None,
    *,
    request_type: str = "quote",
) -> pl.DataFrame:
    """Contracts traded or quoted on ``on_date`` — the dated chain listing (Value tier).

    Returns ``symbol, expiration, strike, right``. Accepts a symbol list, or ``None``
    for the entire market on that date. Unlike EODHD's ``/contracts`` this is genuinely
    point-in-time, which is what makes it usable for audit: "what existed that day" is
    answerable rather than inferred.
    """
    kwargs: dict[str, Any] = {"request_type": request_type, "date": on_date}
    if symbols is not None:
        kwargs["symbol"] = list(symbols)
    return _call("option_list_contracts", **kwargs)


def fetch_option_symbols() -> pl.DataFrame:
    """Every underlying symbol with listed options. Returns a single ``symbol`` column."""
    return _call("option_list_symbols")


def fetch_option_dates(
    symbol: str,
    expiration: date,
    *,
    request_type: str = "quote",
) -> pl.DataFrame:
    """Sessions with data for one ``(symbol, expiration)``. Returns a ``date`` column.

    Coverage auditing only — the ingest path never needs it, because a range pull
    reports its own coverage by what it returns.
    """
    return _call(
        "option_list_dates",
        request_type=request_type,
        symbol=symbol,
        expiration=expiration,
    )


def fetch_interest_rate_eod(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Risk-free rate history. ``symbol`` is e.g. ``SOFR``, ``TREASURY_M3``, ``TREASURY_Y1``.

    Free tier. This is the *fallback* rate source for IV derivation, not the primary
    one: the discount factor is normally implied per-expiry from put-call parity, which
    absorbs dividends and borrow cost as well. This endpoint only covers the case where
    a chain is too thin for that regression to be trustworthy.
    """
    return _call(
        "interest_rate_history_eod",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )


__all__ = [
    "fetch_interest_rate_eod",
    "fetch_option_contracts",
    "fetch_option_dates",
    "fetch_option_eod",
    "fetch_option_symbols",
    "get_client",
    "reset_client",
]
