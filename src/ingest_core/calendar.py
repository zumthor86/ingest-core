"""Trading-calendar helpers: gap detection (from Hermes) + session/offset utilities (from Hephaestus).

Pure functions — no network, no storage. The gap-detection half is parameterized
by *resolved* calendar objects (a ``pandas_market_calendars`` calendar and/or a
``holidays`` country calendar) rather than an exchange code: exchange metadata
(the registry) stays app-owned in Hermes; this module owns only the date math.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
import pandas_market_calendars as mcal

# ---------------------------------------------------------------------------
# Gap detection (promoted from Hermes io/ingest/eodhd/calendar.py)
# ---------------------------------------------------------------------------


def parse_iso_date(value: str | date | None) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _weekday_trading_days(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        return []
    return [ts.date() for ts in pd.bdate_range(start=start_date, end=end_date)]


def _pmc_calendar_has_holidays_for_year(calendar, year: int) -> bool:
    try:
        holiday_dates = calendar.holidays().holidays
    except Exception:
        return True
    year_str = str(year)
    return any(str(d).startswith(year_str) for d in holiday_dates)


def trading_days_between(
    start_date: date,
    end_date: date,
    *,
    market_calendar=None,
    country_cal=None,
) -> list[date]:
    """Exchange sessions in the closed interval, best-effort by data source.

    Preference order: ``market_calendar`` schedule (with the stale-year holiday
    cross-check against ``country_cal``), then ``country_cal`` weekday/holiday
    filtering, then plain business days.
    """
    if start_date > end_date:
        return []

    if market_calendar is not None:
        try:
            schedule = market_calendar.schedule(start_date=start_date, end_date=end_date)
        except Exception:
            schedule = None
        if schedule is not None:
            if schedule.empty:
                return []
            days = [ts.date() for ts in schedule.index.normalize()]
            if country_cal is not None:
                request_years = {start_date.year, end_date.year}
                stale_years = {
                    y for y in request_years
                    if not _pmc_calendar_has_holidays_for_year(market_calendar, y)
                }
                if stale_years:
                    days = [
                        d for d in days
                        if d.year not in stale_years or d not in country_cal
                    ]
            return days

    if country_cal is not None:
        return [
            ts.date()
            for ts in pd.date_range(start=start_date, end=end_date)
            if ts.weekday() not in country_cal.weekend and ts.date() not in country_cal
        ]

    return _weekday_trading_days(start_date, end_date)


def missing_trading_days(
    last_fetched: str | date | None,
    as_of_date: str | date,
    *,
    market_calendar=None,
    country_cal=None,
) -> list[date]:
    """Sessions strictly between ``last_fetched`` and ``as_of_date`` (exclusive both ends)."""
    if last_fetched is None:
        return []
    last_fetched_date = parse_iso_date(last_fetched)
    run_date = parse_iso_date(as_of_date)
    if last_fetched_date is None or run_date is None:
        return []
    end_date = run_date - timedelta(days=1)
    start_date = last_fetched_date + timedelta(days=1)
    if start_date > end_date:
        return []
    return trading_days_between(
        start_date, end_date, market_calendar=market_calendar, country_cal=country_cal
    )


def collapse_to_ranges(
    missing_dates: list[date],
    *,
    market_calendar=None,
    country_cal=None,
) -> list[tuple[str, str]]:
    """Collapse dates into session-adjacent (from, to) ISO ranges."""
    if not missing_dates:
        return []
    sorted_dates = sorted(set(missing_dates))
    ranges: list[tuple[str, str]] = []
    range_start = sorted_dates[0]
    previous = sorted_dates[0]
    for current in sorted_dates[1:]:
        sessions_between = trading_days_between(
            previous + timedelta(days=1), current,
            market_calendar=market_calendar, country_cal=country_cal,
        )
        is_adjacent_session = len(sessions_between) == 1 and sessions_between[0] == current
        if not is_adjacent_session:
            ranges.append((range_start.isoformat(), previous.isoformat()))
            range_start = current
        previous = current
    ranges.append((range_start.isoformat(), previous.isoformat()))
    return ranges


# ---------------------------------------------------------------------------
# Session / offset utilities (promoted from Hephaestus processing/utils/trading_calendar.py)
# ---------------------------------------------------------------------------

# How far ahead to fetch the calendar schedule.
# 180 trading days ≈ 260 calendar days; 400 is a comfortable buffer.
_CALENDAR_LOOKAHEAD_DAYS = 400

CALENDAR_OFFSET_BASIS_CALENDAR_DAYS = "calendar_days"
CALENDAR_OFFSET_BASIS_TRADING_DAYS = "trading_days"

FUTURE_PRICE_INTERVALS = [7, 14, 21, 30, 45, 60, 90, 180]
IV_RANK_LOOKBACK_INTERVALS = [21, 63, 126, 252]
CALENDAR_OFFSET_INTERVALS_BY_BASIS = {
    CALENDAR_OFFSET_BASIS_CALENDAR_DAYS: FUTURE_PRICE_INTERVALS,
    CALENDAR_OFFSET_BASIS_TRADING_DAYS: IV_RANK_LOOKBACK_INTERVALS,
}


def _get_calendar(exchange: str):
    return mcal.get_calendar(exchange)


def _get_valid_session_dates(
    trade_date: date,
    exchange: str,
    lookahead_days: int,
) -> list[date]:
    """Return exchange-valid sessions strictly after the anchor trade date."""
    calendar = _get_calendar(exchange)
    end_date = trade_date + timedelta(days=lookahead_days)
    schedule = calendar.valid_days(
        start_date=trade_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    return [session.date() for session in schedule if session.date() > trade_date]


def get_trading_day_offsets(
    trade_date: date,
    intervals: Iterable[int] | None = None,
    exchange: str = "NYSE",
) -> dict[int, date]:
    """
    Return the exact trading date that is N trading days after trade_date.

    trade_date itself is NOT counted — the count starts from the next session.
    """
    resolved_intervals = sorted(set(intervals or IV_RANK_LOOKBACK_INTERVALS))
    if not resolved_intervals:
        return {}

    future_sessions = _get_valid_session_dates(trade_date, exchange, _CALENDAR_LOOKAHEAD_DAYS)

    max_needed = max(resolved_intervals)
    if len(future_sessions) < max_needed:
        raise ValueError(
            f"Only {len(future_sessions)} trading sessions found after {trade_date} "
            f"within {_CALENDAR_LOOKAHEAD_DAYS} calendar days; need at least {max_needed}. "
            "Increase _CALENDAR_LOOKAHEAD_DAYS."
        )

    # sessions are 0-indexed; N trading days ahead = index N-1
    return {n: future_sessions[n - 1] for n in resolved_intervals}


def get_calendar_day_offsets(
    trade_date: date,
    intervals: Iterable[int] | None = None,
    exchange: str = "NYSE",
) -> dict[int, date]:
    """Return the first trading date on or after trade_date + N calendar days."""
    resolved_intervals = sorted(set(intervals or FUTURE_PRICE_INTERVALS))
    if not resolved_intervals:
        return {}

    future_sessions = _get_valid_session_dates(trade_date, exchange, _CALENDAR_LOOKAHEAD_DAYS)
    if not future_sessions:
        raise ValueError(f"No trading sessions found after {trade_date} within lookahead window.")

    offsets: dict[int, date] = {}
    for interval in resolved_intervals:
        target_calendar_date = trade_date + timedelta(days=interval)
        target_trade_date = next(
            (session for session in future_sessions if session >= target_calendar_date),
            None,
        )
        if target_trade_date is None:
            raise ValueError(
                f"No trading session found on or after {target_calendar_date} "
                f"for anchor {trade_date} within {_CALENDAR_LOOKAHEAD_DAYS} days."
            )
        offsets[interval] = target_trade_date
    return offsets


def get_valid_sessions_set(
    start_date: date,
    end_date: date,
    exchange: str = "NYSE",
) -> set[date]:
    """Return the set of valid exchange session dates for the closed interval [start_date, end_date]."""
    calendar = _get_calendar(exchange)
    schedule = calendar.schedule(start_date=start_date.isoformat(), end_date=end_date.isoformat())
    return {ts.date() for ts in schedule.index}


def get_early_close_sessions_set(
    start_date: date,
    end_date: date,
    exchange: str = "NYSE",
) -> set[date]:
    """Return the set of early-close (half-day) sessions in [start_date, end_date].

    Early closes (e.g. the day after US Thanksgiving, Christmas/July-4th eves) list far
    fewer EOD option rows than a full session, so the thin-session detector must exclude
    them or they false-positive as half-ingested.
    """
    calendar = _get_calendar(exchange)
    schedule = calendar.schedule(start_date=start_date.isoformat(), end_date=end_date.isoformat())
    try:
        early = calendar.early_closes(schedule)
        return {ts.date() for ts in early.index}
    except Exception:
        # Fallback: a close before 16:00 exchange-local time is an early close.
        if "market_close" not in schedule.columns:
            return set()
        tz = getattr(calendar, "tz", "America/New_York")
        local_close = schedule["market_close"].dt.tz_convert(tz)
        return {ts.date() for ts, hour in zip(schedule.index, local_close.dt.hour) if hour < 16}


def last_completed_session(
    as_of: Optional[datetime] = None,
    exchange: str = "NYSE",
    close_buffer_minutes: int = 0,
) -> date:
    """Most recent exchange session that has actually **closed** as of ``as_of``.

    A plain "latest trading day" lookup (schedule ``on or before today``) returns
    *today* the instant today qualifies as a session — even mid-morning, before the
    exchange has opened, let alone closed. That silently breaks any same-day,
    intraday trigger of a pipeline stage expecting the just-completed session's
    EOD data: the provider cannot possibly have it yet, so a coverage/readiness
    check keyed on "today" reads 0% and fails every time it's tried, for as long
    as the market stays open (observed 2026-07-06: a manual ``hermes-ingest``
    trigger during market hours failed 3 retries in a row on
    ``eod_prices=0.0``, because the gate expected today's close as this ran).

    This resolves "latest trading day" as *last completed session* instead: if
    ``as_of`` falls before today's close (plus ``close_buffer_minutes`` — a small
    margin for the provider's own settle/posting lag), the **prior** session is
    returned; only once the current session has actually closed does "today"
    become the answer.

    ``close_buffer_minutes`` defaults to 0 (session-closed is sufficient); pass a
    larger value to also wait out a known vendor posting delay.
    """
    now = as_of if as_of is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    anchor_date = now.date()

    calendar = _get_calendar(exchange)
    schedule = calendar.schedule(
        start_date=(anchor_date - timedelta(days=10)).isoformat(),
        end_date=anchor_date.isoformat(),
    )
    if schedule.empty:
        raise RuntimeError(f"no {exchange} sessions found in the 10 days up to {anchor_date}")

    last_session_date = schedule.index[-1].date()
    if last_session_date == anchor_date:
        close_cutoff = schedule["market_close"].iloc[-1] + timedelta(minutes=close_buffer_minutes)
        if now < close_cutoff:
            if len(schedule) < 2:
                raise RuntimeError(
                    f"only one {exchange} session found in the 10 days up to {anchor_date}; "
                    "cannot fall back to a prior completed session"
                )
            return schedule.index[-2].date()  # today hasn't closed yet -> use the prior close
    return last_session_date


def get_calendar_offset_rows_by_basis(
    trade_date: date,
    exchange: str = "NYSE",
) -> dict[str, dict[int, date]]:
    """Return offset mappings for every supported basis used by the pipeline."""
    return {
        CALENDAR_OFFSET_BASIS_CALENDAR_DAYS: get_calendar_day_offsets(
            trade_date,
            intervals=FUTURE_PRICE_INTERVALS,
            exchange=exchange,
        ),
        CALENDAR_OFFSET_BASIS_TRADING_DAYS: get_trading_day_offsets(
            trade_date,
            intervals=IV_RANK_LOOKBACK_INTERVALS,
            exchange=exchange,
        ),
    }


def get_trading_date_offsets(
    trade_date: date,
    intervals: Iterable[int] | None = None,
    exchange: str = "NYSE",
) -> dict[int, date]:
    """Alias for trading-session offsets (kept: existing call sites in Hephaestus)."""
    return get_trading_day_offsets(trade_date, intervals=intervals, exchange=exchange)


def get_future_trading_dates(
    trade_date: date,
    exchange: str = "NYSE",
) -> dict[str, date]:
    """Wrapper for future-price callers using calendar-day horizons."""
    return {
        f"date_{offset_days}d": target_trade_date
        for offset_days, target_trade_date in get_calendar_day_offsets(
            trade_date,
            intervals=FUTURE_PRICE_INTERVALS,
            exchange=exchange,
        ).items()
    }
