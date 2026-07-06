"""Hermetic EODHD client tests — mocked HTTP, characterization-anchored URLs.

URL/normalization expected values were captured from the pre-extraction
Hephaestus implementation (parity, FR-005/SC-005).
"""
from __future__ import annotations

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
from ingest_core.types import EODHDLimitError


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
