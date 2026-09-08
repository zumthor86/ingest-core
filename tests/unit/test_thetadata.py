"""Hermetic ThetaData client tests — the vendor client is stubbed, no network.

What is worth testing here is the *classification* boundary, not the pass-through
call shapes: which vendor failures become which of our exception types, and — the
one with real downstream consequences — that "no data" is an empty frame rather than
an error. An empty range is how a symbol's listing floor is discovered and how a
genuinely-empty session earns its store sentinel; if that ever starts raising, the
ingest loop turns a normal outcome into a failure it has to unwind.
"""
from __future__ import annotations

import sys
import types
from datetime import date

import polars as pl
import pytest

from ingest_core.clients import thetadata as td
from ingest_core.types import (
    ThetaDataError,
    ThetaDataPermissionError,
    ThetaDataTransientError,
)


class _FakeStatusCode:
    """Stand-in for grpc.StatusCode members — identity is all _classify uses."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"StatusCode.{self.name}"


class _FakeRpcError(Exception):
    def __init__(self, code, details="boom"):
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _FakeNoDataFoundError(Exception):
    pass


@pytest.fixture
def fake_grpc(monkeypatch):
    """Install a fake ``grpc`` module whose StatusCode members _classify can match."""
    codes = types.SimpleNamespace(
        PERMISSION_DENIED=_FakeStatusCode("PERMISSION_DENIED"),
        UNAVAILABLE=_FakeStatusCode("UNAVAILABLE"),
        DEADLINE_EXCEEDED=_FakeStatusCode("DEADLINE_EXCEEDED"),
        INTERNAL=_FakeStatusCode("INTERNAL"),
        RESOURCE_EXHAUSTED=_FakeStatusCode("RESOURCE_EXHAUSTED"),
        NOT_FOUND=_FakeStatusCode("NOT_FOUND"),
        INVALID_ARGUMENT=_FakeStatusCode("INVALID_ARGUMENT"),
    )
    module = types.ModuleType("grpc")
    module.StatusCode = codes
    module.RpcError = _FakeRpcError
    monkeypatch.setitem(sys.modules, "grpc", module)
    return codes


@pytest.fixture
def stub_client(monkeypatch):
    """Replace the cached vendor client and the optional-import hook.

    Returns a recorder the test drives: set ``.result`` to a frame or ``.raises`` to
    an exception, then read ``.calls``.
    """

    class _Recorder:
        def __init__(self):
            self.calls = []
            self.result = pl.DataFrame({"symbol": ["AA"]})
            self.raises = None
            self.raise_times = 0

        def __getattr__(self, name):
            def _method(**kwargs):
                self.calls.append((name, kwargs))
                if self.raises is not None and len(self.calls) <= self.raise_times:
                    raise self.raises
                return self.result

            return _method

    recorder = _Recorder()
    monkeypatch.setattr(td, "_client", recorder)
    monkeypatch.setattr(td, "_import_thetadata", lambda: (object, _FakeNoDataFoundError))
    monkeypatch.setattr(td, "_gate", None)
    monkeypatch.setattr(td, "_gate_size", 0)
    monkeypatch.setenv("THETADATA_API_KEY", "test-key")
    yield recorder
    td.reset_client()


def test_no_data_returns_empty_frame_not_an_error(stub_client):
    """The load-bearing case: an empty range is information, not a failure."""
    stub_client.raises = _FakeNoDataFoundError("nothing here")
    stub_client.raise_times = 1

    out = td.fetch_option_eod("AA", date(2019, 1, 2), date(2019, 1, 3))

    assert isinstance(out, pl.DataFrame)
    assert out.is_empty()


def test_permission_denied_is_not_retried(stub_client, fake_grpc, monkeypatch):
    """A tier/history denial is a configuration fact — retrying can never fix it, and
    deferring it would report a coverage shortfall with no indication of the cause."""
    slept = []
    monkeypatch.setattr(td.time, "sleep", slept.append)
    stub_client.raises = _FakeRpcError(fake_grpc.PERMISSION_DENIED, "tier too low")
    stub_client.raise_times = 99

    with pytest.raises(ThetaDataPermissionError):
        td.fetch_option_eod("AA", date(2019, 1, 2), date(2019, 1, 3))

    assert slept == []
    assert len(stub_client.calls) == 1


def test_transient_error_retries_then_succeeds(stub_client, fake_grpc, monkeypatch):
    monkeypatch.setattr(td.time, "sleep", lambda _: None)
    stub_client.raises = _FakeRpcError(fake_grpc.UNAVAILABLE, "try later")
    stub_client.raise_times = 2

    out = td.fetch_option_eod("AA", date(2026, 8, 20), date(2026, 8, 20))

    assert not out.is_empty()
    assert len(stub_client.calls) == 3


def test_transient_error_gives_up_after_max_retries(stub_client, fake_grpc, monkeypatch):
    monkeypatch.setattr(td.time, "sleep", lambda _: None)
    monkeypatch.setenv("THETADATA_MAX_RETRIES", "2")
    stub_client.raises = _FakeRpcError(fake_grpc.UNAVAILABLE, "still down")
    stub_client.raise_times = 99

    with pytest.raises(ThetaDataTransientError):
        td.fetch_option_eod("AA", date(2026, 8, 20), date(2026, 8, 20))

    assert len(stub_client.calls) == 3  # initial + 2 retries


def test_unrecognised_status_fails_closed(stub_client, fake_grpc, monkeypatch):
    """An unknown code must NOT be treated as transient — retrying an unknown fault
    forever is worse than surfacing it (constitution V, fail closed)."""
    slept = []
    monkeypatch.setattr(td.time, "sleep", slept.append)
    stub_client.raises = _FakeRpcError(fake_grpc.INVALID_ARGUMENT, "bad request")
    stub_client.raise_times = 99

    with pytest.raises(ThetaDataError) as excinfo:
        td.fetch_option_eod("AA", date(2026, 8, 20), date(2026, 8, 20))

    assert not isinstance(excinfo.value, ThetaDataTransientError)
    assert slept == []


def test_classification_branches_on_code_never_message(stub_client, fake_grpc, monkeypatch):
    """A transient-sounding message under a denial code must still be a denial."""
    monkeypatch.setattr(td.time, "sleep", lambda _: None)
    stub_client.raises = _FakeRpcError(
        fake_grpc.PERMISSION_DENIED, "temporarily unavailable, please retry"
    )
    stub_client.raise_times = 99

    with pytest.raises(ThetaDataPermissionError):
        td.fetch_option_eod("AA", date(2026, 8, 20), date(2026, 8, 20))


def test_fetch_option_eod_requests_the_whole_chain(stub_client):
    td.fetch_option_eod("AA", date(2026, 8, 1), date(2026, 8, 20))

    name, kwargs = stub_client.calls[0]
    assert name == "option_history_eod"
    assert kwargs["expiration"] == "*"  # whole chain in one call — the entire design
    assert kwargs["symbol"] == "AA"
    assert kwargs["start_date"] == date(2026, 8, 1)
    assert kwargs["end_date"] == date(2026, 8, 20)


def test_fetch_option_contracts_omits_symbol_for_whole_market(stub_client):
    td.fetch_option_contracts(date(2026, 8, 20))
    _, kwargs = stub_client.calls[0]
    assert "symbol" not in kwargs

    td.fetch_option_contracts(date(2026, 8, 20), ["AA", "SPY"])
    _, kwargs = stub_client.calls[1]
    assert kwargs["symbol"] == ["AA", "SPY"]


def test_missing_api_key_raises_typed_error(monkeypatch):
    monkeypatch.delenv("THETADATA_API_KEY", raising=False)
    td.reset_client()
    with pytest.raises(ThetaDataError, match="THETADATA_API_KEY"):
        td.get_client()


def test_concurrency_ceiling_reads_env_lazily(stub_client, monkeypatch):
    """Resolved on first use, not at import — consumers call load_dotenv() after
    importing this module, and a semaphore cannot be resized once built."""
    monkeypatch.setenv("THETADATA_MAX_CONCURRENT", "5")
    td.fetch_option_symbols()
    assert td._gate_size == 5
