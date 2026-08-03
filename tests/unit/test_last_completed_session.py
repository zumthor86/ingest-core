"""Hermetic tests for close-aware "latest trading day" resolution.

Anchored to NYSE's actual 2026-07-06 (Monday) session, whose schedule is
deterministic via pandas_market_calendars (no network). Regression coverage for
the bug found 2026-07-06: a mid-session intraday trigger must resolve to the
prior close, not the still-open "today".
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from ingest_core.calendar import last_completed_session


def test_mid_session_resolves_to_prior_close():
    # 2026-07-06 14:00 ET ~= 18:00 UTC, well inside the regular session (9:30-16:00 ET)
    mid_session = datetime(2026, 7, 6, 18, 0, tzinfo=timezone.utc)
    result = last_completed_session(as_of=mid_session, exchange="NYSE")
    assert result < date(2026, 7, 6)
    assert result == date(2026, 7, 2)  # prior NYSE session (Thursday; 7/3 was the observed holiday)


def test_before_open_resolves_to_prior_close():
    # 2026-07-06 08:00 UTC = 4:00 AM ET, before the session even opens
    pre_open = datetime(2026, 7, 6, 8, 0, tzinfo=timezone.utc)
    assert last_completed_session(as_of=pre_open, exchange="NYSE") == date(2026, 7, 2)


def test_after_close_plus_buffer_resolves_to_today():
    # NYSE closes 16:00 ET = 20:00 UTC; well after close + default buffer
    after_close = datetime(2026, 7, 6, 22, 0, tzinfo=timezone.utc)
    assert last_completed_session(
        as_of=after_close, exchange="NYSE", close_buffer_minutes=60
    ) == date(2026, 7, 6)


def test_close_buffer_delays_todays_availability():
    # Exactly at close (20:00 UTC): with a 60-min buffer, today isn't "complete" yet
    at_close = datetime(2026, 7, 6, 20, 0, tzinfo=timezone.utc)
    assert last_completed_session(
        as_of=at_close, exchange="NYSE", close_buffer_minutes=60
    ) == date(2026, 7, 2)
    # with a 0-min buffer, close-instant is already "complete"
    assert last_completed_session(
        as_of=at_close, exchange="NYSE", close_buffer_minutes=0
    ) == date(2026, 7, 6)


def test_weekend_returns_last_friday_close():
    saturday = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
    result = last_completed_session(as_of=saturday, exchange="NYSE")
    assert result.weekday() == 4  # Friday
    assert result < date(2026, 7, 11)


def test_naive_datetime_treated_as_utc():
    naive = datetime(2026, 7, 6, 18, 0)  # no tzinfo
    assert last_completed_session(as_of=naive, exchange="NYSE") == date(2026, 7, 2)


def test_default_as_of_uses_now():
    # Smoke test: no explicit as_of resolves without error and returns a real date
    result = last_completed_session(exchange="NYSE")
    assert isinstance(result, date)
    assert result <= date.today()
