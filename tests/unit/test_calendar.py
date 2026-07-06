"""Hermetic calendar tests. NYSE anchors are characterization values captured
from the pre-extraction implementations (Hermes calendar.py / Hephaestus
trading_calendar.py) on 2026-07-04 — parity per FR-005/SC-005."""
from __future__ import annotations

from datetime import date

from ingest_core.calendar import (
    collapse_to_ranges,
    get_calendar_day_offsets,
    get_early_close_sessions_set,
    get_trading_day_offsets,
    get_valid_sessions_set,
    missing_trading_days,
    parse_iso_date,
    trading_days_between,
)


def test_parse_iso_date():
    assert parse_iso_date(None) is None
    assert parse_iso_date("2026-03-13") == date(2026, 3, 13)
    assert parse_iso_date(date(2026, 3, 13)) == date(2026, 3, 13)


def test_trading_days_between_weekday_fallback():
    # No calendars supplied -> plain business days
    days = trading_days_between(date(2026, 6, 26), date(2026, 6, 30))
    assert days == [date(2026, 6, 26), date(2026, 6, 29), date(2026, 6, 30)]
    assert trading_days_between(date(2026, 6, 30), date(2026, 6, 26)) == []


def test_missing_trading_days_weekday_fallback():
    # Strictly between last_fetched and as_of (both exclusive)
    days = missing_trading_days("2026-06-25", "2026-06-30")
    assert days == [date(2026, 6, 26), date(2026, 6, 29)]
    assert missing_trading_days(None, "2026-06-30") == []


def test_collapse_to_ranges_weekday_fallback():
    ranges = collapse_to_ranges(
        [date(2026, 6, 26), date(2026, 6, 29), date(2026, 7, 1)]
    )
    # 6/26 -> 6/29 adjacent over the weekend; 7/1 split by missing 6/30
    assert ranges == [("2026-06-26", "2026-06-29"), ("2026-07-01", "2026-07-01")]


def test_nyse_trading_day_offsets_characterization():
    assert get_trading_day_offsets(date(2026, 3, 13)) == {
        21: date(2026, 4, 14),
        63: date(2026, 6, 12),
        126: date(2026, 9, 14),
        252: date(2027, 3, 16),
    }


def test_nyse_calendar_day_offsets_characterization():
    assert get_calendar_day_offsets(date(2026, 3, 13), intervals=[7, 30]) == {
        7: date(2026, 3, 20),
        30: date(2026, 4, 13),
    }


def test_nyse_sessions_characterization():
    # July-4th week 2026: 7/3 observed holiday (7/4 is a Saturday)
    assert sorted(get_valid_sessions_set(date(2026, 6, 26), date(2026, 7, 6))) == [
        date(2026, 6, 26), date(2026, 6, 29), date(2026, 6, 30),
        date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 6),
    ]


def test_nyse_early_close_characterization():
    # Day after Thanksgiving 2025 is a half session
    assert sorted(get_early_close_sessions_set(date(2025, 11, 24), date(2025, 11, 30))) == [
        date(2025, 11, 28)
    ]
