"""Hermetic backfill-plumbing tests (feature 002).

Characterization anchors captured from ``Hephaestus/scripts/range_backfill.py``
before its retirement (SC-005): window auto-split decisions, walk stop rule,
and Budget accounting must match the script's behaviour exactly.
"""
from __future__ import annotations

import threading
from datetime import date

import pytest

from ingest_core.backfill import (
    Budget,
    PrehistoryResult,
    iter_range_records,
    probe_quota_reset,
    quota_remaining_requests,
    run_prehistory_walk,
)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_budget_counts_and_exceeds():
    b = Budget(3)
    assert not b.exceeded() and b.remaining() == 3
    b.add_request(2)
    b.add_rows(100)
    assert b.remaining() == 1 and not b.exceeded()
    b.add_request()
    assert b.exceeded() and b.remaining() == 0
    assert b.rows == 100


def test_budget_unmetered():
    b = Budget(None)
    b.add_request(10_000)
    assert not b.exceeded() and b.remaining() is None


def test_budget_thread_safe():
    b = Budget(None)

    def spin():
        for _ in range(1000):
            b.add_request()

    threads = [threading.Thread(target=spin) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert b.requests == 8000


# ---------------------------------------------------------------------------
# iter_range_records — fake provider
# ---------------------------------------------------------------------------


class FakeProvider:
    """Simulates options/eod pagination from the URL's window + offset.

    ``density[(start, end)]`` (or per-day fallback) controls total records for a
    window; pages come back in ``page_limit`` slices like the real endpoint.
    """

    def __init__(self, per_day: dict[date, int], page_limit: int = 5):
        self.per_day = per_day
        self.page_limit = page_limit
        self.calls: list[tuple[str, str, str, str | None, int]] = []

    def fetch(self, url: str, api_token=None):
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(url).query)
        sym = q["filter[underlying_symbol]"][0]
        if "filter[tradetime_eq]" in q:
            s = e = q["filter[tradetime_eq]"][0]
        else:
            s = q["filter[tradetime_from]"][0]
            e = q["filter[tradetime_to]"][0]
        otype = q.get("filter[type]", [None])[0]
        off = int(q["page[offset]"][0])
        self.calls.append((sym, s, e, otype, off))

        d0, d1 = date.fromisoformat(s), date.fromisoformat(e)
        total = 0
        d = d0
        while d <= d1:
            n = self.per_day.get(d, 0)
            total += n if otype is None else n // 2
            d = d.fromordinal(d.toordinal() + 1)
        window_records = [{"contract": f"{sym}-{i}"} for i in range(total)]
        page = window_records[off: off + self.page_limit]
        return {"meta": {}, "data": page, "next_url": None}


def _patch_provider(monkeypatch, provider, page_limit):
    monkeypatch.setattr("ingest_core.backfill.fetch_eodhd_page", provider.fetch)
    return provider


def test_range_single_window_no_split(monkeypatch):
    prov = _patch_provider(monkeypatch, FakeProvider({date(2026, 3, 2): 3, date(2026, 3, 3): 1}, page_limit=5), 5)
    bud = Budget(None)
    out = list(iter_range_records("ZM", date(2026, 3, 2), date(2026, 3, 3),
                                  budget=bud, page_limit=5, offset_cap=20))
    assert len(out) == 1
    ws, we, recs = out[0]
    assert (ws, we) == (date(2026, 3, 2), date(2026, 3, 3))
    assert len(recs) == 4
    assert bud.requests == 1  # one page


def test_range_auto_halves_on_offset_cap(monkeypatch):
    # 2-day window, 30 records/day, page_limit 5, cap 10 -> full window pages past
    # the cap (capped=True, partial discarded), then halves into two 1-day windows
    # which each page to completion (30 recs = 6 pages + terminating short page).
    per_day = {date(2026, 3, 2): 30, date(2026, 3, 3): 30}
    prov = _patch_provider(monkeypatch, FakeProvider(per_day, page_limit=5), 5)
    bud = Budget(None)
    out = list(iter_range_records("ZM", date(2026, 3, 2), date(2026, 3, 3),
                                  budget=bud, page_limit=5, offset_cap=10))
    assert [(ws, we, len(r)) for ws, we, r in out] == [
        (date(2026, 3, 2), date(2026, 3, 2), 30),
        (date(2026, 3, 3), date(2026, 3, 3), 30),
    ]
    # capped partial fetch was discarded and refetched — script-verbatim behaviour
    assert bud.requests > 12


def test_range_single_day_splits_by_type(monkeypatch):
    # One day too dense even alone: split into call/put slices (15 each).
    per_day = {date(2026, 3, 2): 30}
    prov = _patch_provider(monkeypatch, FakeProvider(per_day, page_limit=5), 5)
    out = list(iter_range_records("ZM", date(2026, 3, 2), date(2026, 3, 2),
                                  budget=Budget(None), page_limit=5, offset_cap=10))
    assert len(out) == 1
    ws, we, recs = out[0]
    assert (ws, we) == (date(2026, 3, 2), date(2026, 3, 2))
    assert len(recs) == 30  # 15 calls + 15 puts, no truncation
    assert any(c[3] == "call" for c in prov.calls) and any(c[3] == "put" for c in prov.calls)


def test_range_respects_budget(monkeypatch):
    per_day = {date(2026, 3, d): 3 for d in range(2, 9)}
    prov = _patch_provider(monkeypatch, FakeProvider(per_day, page_limit=5), 5)
    bud = Budget(0)  # already exceeded
    out = list(iter_range_records("ZM", date(2026, 3, 2), date(2026, 3, 8),
                                  budget=bud, page_limit=5, offset_cap=10))
    assert out == [] and prov.calls == []


# ---------------------------------------------------------------------------
# run_prehistory_walk — stop rule (script probe_prehistory parity)
# ---------------------------------------------------------------------------


def test_walk_stops_at_first_empty_window():
    landed: list[tuple[date, date]] = []

    def land(ws: date, we: date) -> int:
        landed.append((ws, we))
        return 10 if we >= date(2026, 2, 20) else 0  # listed ~mid-Feb

    res = run_prehistory_walk(land, date(2026, 3, 2), date(2026, 1, 1),
                              week_days=7, budget=Budget(None))
    assert isinstance(res, PrehistoryResult)
    assert res.rows_landed == 10 * (len(landed) - 1)
    assert res.boundary_window == landed[-1]  # the empty window terminated it
    assert res.completed is True  # boundary found -> walk finished (consolidation trigger)
    # windows walk strictly backward, adjacent, 7 days wide
    assert landed[0] == (date(2026, 2, 23), date(2026, 3, 1))
    assert landed[1][1] == landed[0][0] - timedelta_days(1)


def timedelta_days(n):
    from datetime import timedelta
    return timedelta(days=n)


def test_walk_stops_on_budget_without_boundary():
    bud = Budget(2)

    def land(ws, we):
        bud.add_request()
        return 5

    res = run_prehistory_walk(land, date(2026, 3, 2), date(2025, 1, 1), week_days=7, budget=bud)
    assert res.boundary_window is None  # budget stop -> resumable, no boundary claim
    assert res.completed is False  # budget-cut is NOT finished (FR-016 trigger must not fire)
    assert res.rows_landed == 10


def test_walk_respects_coverage_floor():
    calls = []

    def land(ws, we):
        calls.append((ws, we))
        return 1

    res = run_prehistory_walk(land, date(2026, 1, 10), date(2026, 1, 1), week_days=7, budget=Budget(None))
    assert res.boundary_window is None
    assert res.completed is True  # coverage floor reached -> finished, distinguishable from budget-cut
    assert calls[0][0] >= date(2026, 1, 1)
    assert all(ws >= date(2026, 1, 1) for ws, _ in calls)


# ---------------------------------------------------------------------------
# quota helpers
# ---------------------------------------------------------------------------


def test_quota_remaining_requests_parses(monkeypatch):
    monkeypatch.setattr(
        "ingest_core.clients.eodhd.get_eodhd_quota_status",
        lambda api_token=None: {"marketplace_requests_remaining": 42},
    )
    assert quota_remaining_requests() == 42


def test_quota_remaining_requests_none_on_error(monkeypatch):
    def boom(api_token=None):
        raise RuntimeError("nope")

    monkeypatch.setattr("ingest_core.clients.eodhd.get_eodhd_quota_status", boom)
    assert quota_remaining_requests() is None


def test_probe_quota_reset_swallows_errors(monkeypatch):
    def boom(url, api_token=None):
        raise RuntimeError("out of quota")

    monkeypatch.setattr("ingest_core.backfill.fetch_eodhd_page", boom)
    probe_quota_reset()  # must not raise
