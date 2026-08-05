"""Hermetic EODHD client tests — mocked HTTP, characterization-anchored URLs.

URL/normalization expected values were captured from the pre-extraction
Hephaestus implementation (parity, FR-005/SC-005).
"""
from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from ingest_core.clients import eodhd
from ingest_core.clients.eodhd import (
    EODHDClient,
    _decode_eodhd_payload,
    _normalize_eod_records,
    build_options_contract_url,
    build_options_contracts_url,
    build_options_eod_url,
    fetch_bulk_eod_for_exchange,
    fetch_eod,
)
from ingest_core.ratelimit import RateLimiter
from ingest_core.types import (
    EODHDLimitError,
    EODHDQuotaExhaustedError,
    EODHDServerError,
    EODHDThrottleError,
)


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def test_build_options_eod_url_single_day_characterization():
    assert build_options_eod_url("BRK.B", "2026-03-13", "2026-03-13") == (
        "https://eodhd.com/api/mp/unicornbay/options/eod?filter%5Bunderlying_symbol%5D=BRK-B"
        "&page%5Boffset%5D=0&page%5Blimit%5D=1000&compact=0&sort=-exp_date"
        "&filter%5Btradetime_eq%5D=2026-03-13"
    )


def test_build_options_eod_url_range_characterization():
    assert build_options_eod_url(
        "aapl.US", "2026-03-01", "2026-03-13", option_type="call", fields=["contract", "delta"]
    ) == (
        "https://eodhd.com/api/mp/unicornbay/options/eod?filter%5Bunderlying_symbol%5D=AAPL"
        "&page%5Boffset%5D=0&page%5Blimit%5D=1000&compact=0&sort=-exp_date"
        "&filter%5Btype%5D=call&filter%5Btradetime_from%5D=2026-03-01"
        "&filter%5Btradetime_to%5D=2026-03-13&fields%5Boptions-eod%5D=contract%2Cdelta"
    )


def test_build_options_contracts_urls_characterization():
    assert build_options_contracts_url("SPY", page_limit=500) == (
        "https://eodhd.com/api/mp/unicornbay/options/contracts?filter%5Bunderlying_symbol%5D=SPY"
        "&page%5Boffset%5D=0&page%5Blimit%5D=500&compact=0&sort=-exp_date"
    )
    assert build_options_contract_url("AAPL260320C00200000", fields=["contract", "delta"]) == (
        "https://eodhd.com/api/mp/unicornbay/options/contracts?filter%5Bcontract%5D=AAPL260320C00200000"
        "&page%5Boffset%5D=0&page%5Blimit%5D=1&fields%5Boptions-contracts%5D=contract%2Cdelta"
    )


def test_normalize_eod_records_characterization():
    records = _normalize_eod_records(
        [
            {"id": "AAPL260320C00200000-2026-03-13", "attributes": {"delta": 0.5}},
            {"id": "NODATE", "attributes": {"gamma": 0.1}},
            [0.5, 10],
            {"plain": 1},
        ],
        ["delta", "volume"],
    )
    assert records == [
        {"delta": 0.5, "session_date": "2026-03-13"},
        {"gamma": 0.1},
        {"delta": 0.5, "volume": 10},
        {"plain": 1},
    ]


def test_decode_payload_raises_limit_on_429():
    with pytest.raises(EODHDLimitError):
        _decode_eodhd_payload(_FakeResponse(status_code=429), "X")


def test_decode_payload_raises_limit_on_quota_message():
    resp = _FakeResponse(payload={"error": "API requests daily limit reached. Please upgrade."})
    with pytest.raises(EODHDLimitError):
        _decode_eodhd_payload(resp, "X")


def test_fetch_eod_shapes_frame(monkeypatch):
    rows = [
        {"date": "2026-03-12", "open": 1, "high": 2, "low": 0.5, "close": 1.5,
         "adjusted_close": 1.5, "volume": 100, "extra": "drop-me"},
        {"date": "2026-03-13", "open": 2, "high": 3, "low": 1.5, "close": 2.5,
         "adjusted_close": 2.5, "volume": 200, "extra": "drop-me"},
    ]
    monkeypatch.setattr(
        "ingest_core.clients.eodhd.requests.get",
        lambda *a, **k: _FakeResponse(payload=rows),
    )
    df = fetch_eod("aapl", "US", api_key="tok")
    assert list(df.columns) == ["date", "open", "high", "low", "close", "adjusted_close", "volume", "symbol", "exchange"]
    assert df["symbol"].tolist() == ["AAPL", "AAPL"]
    assert df["open"].dtype == "float64"
    assert len(df) == 2


def test_fetch_eod_raises_on_limit_payload(monkeypatch):
    monkeypatch.setattr(
        "ingest_core.clients.eodhd.requests.get",
        lambda *a, **k: _FakeResponse(
            payload={"error": "API requests daily limit reached. Please upgrade your plan."}
        ),
    )
    with pytest.raises(EODHDLimitError):
        fetch_eod("AAPL", "US", api_key="tok")


def test_fetch_bulk_raises_on_429(monkeypatch):
    monkeypatch.setattr(
        "ingest_core.clients.eodhd.requests.get",
        lambda *a, **k: _FakeResponse(status_code=429, payload={"status": "error"}),
    )
    with pytest.raises(EODHDLimitError):
        fetch_bulk_eod_for_exchange("US", "2026-05-01", api_key="tok")


def test_get_with_retry_raises_limit_error_subclassing_runtimeerror(monkeypatch):
    monkeypatch.setattr(eodhd, "_MAX_RETRIES", 0)
    monkeypatch.setattr(
        "ingest_core.clients.eodhd.requests.get",
        lambda *a, **k: _FakeResponse(status_code=429, text="slow down", headers={"Retry-After": "12"}),
    )
    with pytest.raises(EODHDLimitError) as exc_info:
        eodhd._get_with_retry("https://x")
    assert isinstance(exc_info.value, RuntimeError)  # parity with pre-extraction catches
    assert exc_info.value.retry_after == 12.0


def test_client_facade_paces_fetch_page(monkeypatch):
    calls: list[str] = []

    class _CountingLimiter(RateLimiter):
        def acquire(self):
            calls.append("acquire")

    monkeypatch.setattr(
        eodhd, "_get_with_retry",
        lambda url: _FakeResponse(payload={"meta": {}, "data": [], "links": {}}),
    )
    client = EODHDClient("tok", rate_limiter=_CountingLimiter(1, 0.0))
    page = client.fetch_page("https://eodhd.com/api/x")
    assert calls == ["acquire"]
    assert client.decode(page) == []
    with pytest.raises(ValueError):
        client.build_url("nope")


# ---------------------------------------------------------------------------
# _get_with_retry status classification (2026-07-27)
#
# 429 / 402 / 5xx are three unrelated failures. They previously shared one
# branch and one exception type, with the rate-limit headers stamped into every
# message — so a vendor 5xx was indistinguishable from a quota wall downstream.
# ---------------------------------------------------------------------------


def _patch_get(monkeypatch, responses):
    """Serve *responses* in order; record how many GETs were made."""
    calls = {"n": 0}

    def _fake_get(url, timeout=None):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[i]

    monkeypatch.setattr(eodhd.requests, "get", _fake_get)
    monkeypatch.setattr(eodhd.time, "sleep", lambda _s: None)
    return calls


def test_500_raises_server_error_not_a_limit_error(monkeypatch):
    """EODHD documents 5xx as 'retry after a short delay', billed 0 API calls —
    it is not a limit condition and must not claim to be one."""
    resp = _FakeResponse(
        status_code=500,
        text="Error occurred. Please contact support@eodhistoricaldata.com",
        headers={"X-RateLimit-Remaining": "1197", "X-RateLimit-Limit": "1200"},
    )
    calls = _patch_get(monkeypatch, [resp])

    with pytest.raises(EODHDServerError) as exc_info:
        eodhd._get_with_retry("https://eodhd.com/api/whatever")

    # Retried several times on a short schedule, not once after a minute.
    assert calls["n"] == eodhd._SERVER_ERROR_MAX_RETRIES + 1
    # The old message interpolated "quota {remaining}/{limit}" here, which is the
    # per-minute window and had nothing to do with the failure.
    assert "quota" not in str(exc_info.value).lower()


def test_429_raises_throttle_error_and_labels_the_minute_window(monkeypatch):
    resp = _FakeResponse(
        status_code=429,
        text="Too many requests",
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Limit": "1000"},
    )
    _patch_get(monkeypatch, [resp])

    with pytest.raises(EODHDThrottleError) as exc_info:
        eodhd._get_with_retry("https://eodhd.com/api/whatever")

    assert "per-minute" in str(exc_info.value)


def test_402_raises_quota_exhausted_and_does_not_retry(monkeypatch):
    """402 is the daily call budget; it resets at midnight GMT, never mid-run,
    so retrying is pure waste."""
    resp = _FakeResponse(status_code=402, text="Payment Required")
    calls = _patch_get(monkeypatch, [resp])

    with pytest.raises(EODHDQuotaExhaustedError):
        eodhd._get_with_retry("https://eodhd.com/api/whatever")

    assert calls["n"] == 1


def test_transient_500_then_success_returns_the_payload(monkeypatch):
    ok = _FakeResponse(status_code=200, payload={"ok": True})
    bad = _FakeResponse(status_code=503, text="upstream unavailable")
    _patch_get(monkeypatch, [bad, ok])

    assert eodhd._get_with_retry("https://eodhd.com/api/whatever") is ok


def test_all_typed_errors_remain_catchable_as_eodhd_limit_error():
    """Hermes's backfill and ingest_core.retry both catch EODHDLimitError to mean
    'the provider said no' — the new subclasses must not slip past them."""
    for cls in (EODHDThrottleError, EODHDQuotaExhaustedError, EODHDServerError):
        assert issubclass(cls, EODHDLimitError)
        assert issubclass(cls, RuntimeError)


# ---------------------------------------------------------------------------
# Adaptive concurrency gate (2026-08-03)
#
# A sustained 429 rate was observed at 40 concurrent requests while
# X-RateLimit-Remaining stayed near-full — the vendor's real burst-level
# enforcement, not the documented per-minute average, was the actual wall.
# EODHD_MAX_CONCURRENT_REQUESTS is now a ceiling the gate operates under, not
# a fixed point: it halves on a 429, grows back by one after a clean streak.
# ---------------------------------------------------------------------------


def test_adaptive_gate_halves_on_congestion_and_holds_the_floor():
    gate = eodhd._AdaptiveConcurrencyGate(ceiling=8, floor=2)

    gate.on_congestion()
    assert gate._limit == 4
    gate.on_congestion()
    assert gate._limit == 2  # floor
    gate.on_congestion()
    assert gate._limit == 2  # does not go below the floor


def test_adaptive_gate_grows_by_one_after_a_clean_streak(monkeypatch):
    monkeypatch.setenv("EODHD_ADAPTIVE_GROW_AFTER_CLEAN", "2")
    gate = eodhd._AdaptiveConcurrencyGate(ceiling=8, floor=2)
    gate._limit = 2

    gate.on_clean_response()
    assert gate._limit == 2  # streak 1 of 2 — not yet
    gate.on_clean_response()
    assert gate._limit == 3  # streak hit 2 — grows by exactly one

    gate._limit = 8
    gate.on_clean_response()
    assert gate._limit == 8  # never exceeds the ceiling


def test_adaptive_gate_blocks_admission_above_the_live_limit():
    gate = eodhd._AdaptiveConcurrencyGate(ceiling=8, floor=1)
    gate._limit = 1

    release_first = threading.Event()
    entered_second = threading.Event()

    def hold_first():
        with gate:
            release_first.wait(timeout=2)

    def try_second():
        with gate:
            entered_second.set()

    t1 = threading.Thread(target=hold_first)
    t1.start()
    time.sleep(0.05)  # let t1 acquire before t2 tries
    t2 = threading.Thread(target=try_second)
    t2.start()
    time.sleep(0.1)
    assert not entered_second.is_set(), "second entrant admitted past a limit of 1"

    release_first.set()
    t1.join(timeout=2)
    t2.join(timeout=2)
    assert entered_second.is_set(), "second entrant never admitted after the first released"


class _SpyGate:
    def __init__(self) -> None:
        self.throttled = 0
        self.clean = 0
        self.signals: list[str] = []

    def __enter__(self) -> "_SpyGate":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def on_congestion(self, signal: str = "429") -> None:
        self.throttled += 1
        self.signals.append(signal)

    def on_clean_response(self) -> None:
        self.clean += 1


def test_get_with_retry_reports_throttle_then_clean_response_to_the_gate(monkeypatch):
    spy = _SpyGate()
    monkeypatch.setattr(eodhd, "_get_api_semaphore", lambda: spy)
    resp_429 = _FakeResponse(
        status_code=429, text="slow down",
        headers={"X-RateLimit-Remaining": "1199", "X-RateLimit-Limit": "1200"},
    )
    resp_200 = _FakeResponse(status_code=200, payload={"ok": True})
    _patch_get(monkeypatch, [resp_429, resp_200])

    result = eodhd._get_with_retry("https://eodhd.com/api/whatever")

    assert result is resp_200
    assert spy.throttled == 1
    assert spy.clean == 1
    assert spy.signals == ["429"]


def test_get_with_retry_reports_5xx_to_the_gate_as_congestion(monkeypatch):
    """A 500 must apply backpressure, not just retry.

    Regression for 2026-08-05: /options/contracts times a large chain out into a
    500 rather than shedding it with a 429, so 5xx is the only congestion signal
    available on that endpoint. Because the gate only listened for 429, it stayed
    pinned at its ceiling of 40 through a 69% failure rate and never backed off.
    """
    monkeypatch.setenv("EODHD_5XX_BASE_DELAY", "0")
    spy = _SpyGate()
    monkeypatch.setattr(eodhd, "_get_api_semaphore", lambda: spy)
    resp_500 = _FakeResponse(status_code=500, text="Error occurred.")
    resp_200 = _FakeResponse(status_code=200, payload={"ok": True})
    _patch_get(monkeypatch, [resp_500, resp_500, resp_200])

    result = eodhd._get_with_retry("https://eodhd.com/api/whatever")

    assert result is resp_200
    assert spy.throttled == 2, "each 5xx attempt should shrink the gate"
    assert spy.signals == ["HTTP 500", "HTTP 500"]
    assert spy.clean == 1


def test_sustained_5xx_collapses_the_live_limit_toward_the_floor():
    """The behaviour that would have contained the 2026-08-05 incident."""
    gate = eodhd._AdaptiveConcurrencyGate(ceiling=40, floor=2)
    assert gate._limit == 40

    for _ in range(5):
        gate.on_congestion("HTTP 500")

    assert gate._limit == 2, "a sustained 500 wave must drive concurrency to the floor"
