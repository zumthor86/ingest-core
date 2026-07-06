"""Shared data-transfer types for the ingest plumbing (contracts/ingest-core-api.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional


class EODHDLimitError(RuntimeError):
    """Raised when EODHD indicates throttling or API quota exhaustion.

    Subclasses RuntimeError so pre-extraction ``except RuntimeError`` call
    sites (Hephaestus pagination, Hermes backfill aborts) keep working
    unchanged — behaviour parity (FR-005).
    """

    def __init__(self, message: str = "", retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


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
