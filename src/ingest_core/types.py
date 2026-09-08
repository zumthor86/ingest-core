"""Shared data-transfer types for the ingest plumbing (contracts/ingest-core-api.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional


class EODHDLimitError(RuntimeError):
    """Base: EODHD declined to serve the request.

    Subclasses RuntimeError so pre-extraction ``except RuntimeError`` call
    sites (Hephaestus pagination, Hermes backfill aborts) keep working
    unchanged — behaviour parity (FR-005).

    Prefer one of the subclasses below when raising. They exist because the three
    conditions have nothing in common operationally — one clears in under a minute,
    one clears at midnight, one is the vendor being broken — and callers were
    previously forced to guess between them by substring-matching the message.
    The base keeps its name only because Hermes's backfill and
    ``ingest_core.retry`` both catch it to mean "the provider said no".
    """

    def __init__(self, message: str = "", retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class EODHDThrottleError(EODHDLimitError):
    """HTTP 429 — the 1,000-requests-per-MINUTE window is full.

    Transient by construction: the window rolls every minute, so waiting works.
    This is *not* the daily call budget (see EODHDQuotaExhaustedError). The
    ``X-RateLimit-Remaining``/``-Limit`` headers describe this minute window only.
    """


class EODHDQuotaExhaustedError(EODHDLimitError):
    """HTTP 402 — the daily API *call* budget is spent.

    Resets at midnight GMT for subscription plans, or at the marketplace
    ``timeToReset``. Retrying within a run can never help, so the caller should
    defer the work rather than count it as a failed fetch.
    """


class EODHDServerError(EODHDLimitError):
    """HTTP 5xx — a vendor-side fault, and neither a limit nor a budget event.

    EODHD's own error table says "Server Error — retry after a short delay", and
    its consumption rules bill failed requests 0 API calls. Observed concentrated
    on the largest option chains (SPY/QQQ/NVDA/TSLA/...), where ``/contracts``
    appears to fall over server-side.

    Subclasses EODHDLimitError purely so existing ``except EODHDLimitError`` call
    sites keep their current behaviour; do not read the base name as a claim that
    a 5xx has anything to do with rate limiting.
    """


class ThetaDataError(RuntimeError):
    """Base: ThetaData declined to serve the request.

    Deliberately *not* a subclass of :class:`EODHDLimitError`. The two vendors
    fail in unrelated ways and a shared base would invite exactly the kind of
    catch-all handling that let EODHD's vendor 500s masquerade as quota
    exhaustion. Callers deciding whether work was deferred or failed branch on
    the subclasses below — never on message text.
    """


class ThetaDataPermissionError(ThetaDataError):
    """gRPC ``PERMISSION_DENIED`` — the subscription tier does not cover this request.

    Two distinct causes share this code and neither is retryable within a run:
    an endpoint above the current tier (greeks on Value), and a date older than
    the tier's history floor. Both are configuration facts, not transient
    conditions, so a caller must surface them rather than defer them — a run that
    quietly defers every request because the subscription lapsed would report a
    coverage shortfall with no indication of why.
    """


class ThetaDataTransientError(ThetaDataError):
    """A vendor-side or transport fault that retrying can plausibly fix.

    ``UNAVAILABLE``, ``DEADLINE_EXCEEDED``, ``INTERNAL``, ``RESOURCE_EXHAUSTED``.
    Unlike EODHD there is no metered call budget behind these, so a retry costs
    only wall-clock time.
    """


@dataclass(frozen=True)
class EODHDPage:
    """One decoded provider page + pagination cursor."""

    payload: Any
    offset: int = 0
    limit: int = 0
    has_more: bool = False


@dataclass(frozen=True)
class QuotaStatus:
    """Informational account quota — NOT a cross-process gate (FR-006)."""

    used: int
    limit: int
    remaining: int


@dataclass(frozen=True)
class BackfillRequest:
    symbol: str
    from_date: date
    to_date: date
    exchange: str = ""


@dataclass(frozen=True)
class BackfillResult:
    symbol: str
    rows: int
    missing_ranges: list[tuple[date, date]] = field(default_factory=list)
