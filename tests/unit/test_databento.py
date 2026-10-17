"""Unit tests for clients/databento.py — hermetic, no network, no databento install needed."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ingest_core.clients import databento as d


class _ServerError(Exception):
    pass


@pytest.fixture()
def fake_db():
    mod = SimpleNamespace(BentoServerError=_ServerError)
    with patch.object(d, "_db", return_value=mod), patch.object(d.time, "sleep"):
        yield mod


def test_request_is_one_symbol_with_exclusive_end():
    assert d.request("GLBX.MDP3", "ohlcv-1m", "ES.v.0", "2026-09-19", "2026-09-26") == dict(
        dataset="GLBX.MDP3", schema="ohlcv-1m", stype_in="continuous", symbols=["ES.v.0"],
        start="2026-09-19", end="2026-09-26")


def test_server_errors_are_retried_then_raised(fake_db):
    hist = MagicMock()
    hist.metadata.get_cost.side_effect = [_ServerError("500"), 1.25]
    assert d.cost(hist, {"x": 1}) == 1.25
    hist.metadata.get_cost.side_effect = _ServerError("500")
    with pytest.raises(_ServerError):
        d.cost(hist, {"x": 1})


def test_download_lands_only_when_complete(fake_db, tmp_path):
    target = tmp_path / "ES" / "chunk.dbn.zst"

    def get_range(path, **_):
        path.write_bytes(b"dbn")

    hist = MagicMock()
    hist.timeseries.get_range.side_effect = get_range
    assert d.download(hist, {"symbols": ["ES.v.0"]}, target) == 3
    assert target.read_bytes() == b"dbn" and not list(tmp_path.rglob("*.part"))

    hist.timeseries.get_range.side_effect = OSError("connection reset")
    other = tmp_path / "ES" / "next.dbn.zst"
    with pytest.raises(OSError):
        d.download(hist, {"symbols": ["ES.v.0"]}, other)
    assert not other.exists()


def test_read_definitions_keeps_outright_futures_once(tmp_path):
    import pandas as pd

    frame = pd.DataFrame({
        "instrument_id": [11, 12, 11, 99],
        "raw_symbol": ["MCLX6", "MCLZ6", "MCLX6", "MCLX6-MCLZ6"],
        "asset": ["MCL", "MCL", "MCL", "MCL"],
        "instrument_class": ["F", "F", "F", "S"],
        "expiration": pd.to_datetime(["2026-10-19 18:30", "2026-11-17 19:30", "2026-10-19 18:30", "2026-10-19 18:30"],
                                     utc=True),
        "maturity_year": [2026, 2026, 2026, 2026],
        "maturity_month": [11, 12, 11, 11],
    })
    store = MagicMock()
    store.to_df.return_value = frame
    mod = SimpleNamespace(DBNStore=SimpleNamespace(from_file=lambda p: store))
    with patch.object(d, "_db", return_value=mod):
        out = d.read_definitions(tmp_path / "x.dbn.zst")
    assert out["raw_symbol"].to_list() == ["MCLX6", "MCLZ6"]
    assert str(out["expiration"][0]) == "2026-10-19"
