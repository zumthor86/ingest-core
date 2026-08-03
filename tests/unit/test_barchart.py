"""Hermetic Barchart client tests — CSV fixtures, no network."""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ingest_core.clients.barchart import (
    _apply_symbol_mappings,
    _file_date,
    get_top_symbols,
    parse_barchart_date,
    parse_barchart_value,
)


CSV_HEADER = "symbol,symbolName,lastPrice,priceChange,percentChange,optionsTotalVolume,scraped_at\n"


def _write_csv(folder, name, rows):
    path = folder / name
    path.write_text(CSV_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_get_top_symbols_orders_by_volume(tmp_path):
    _write_csv(tmp_path, "most_active_options_2026-03-13.csv", [
        'AAA,Alpha,"1,007.52",1,1%,"1,000",2026-03-13',
        'BBB,Beta,10,1,1%,"3,063,353",2026-03-13',
        'CCC,Gamma,10,1,1%,"2,000",2026-03-13',
    ])
    assert get_top_symbols(str(tmp_path), top_n=2) == ["BBB", "CCC"]


def test_get_top_symbols_as_of_picks_latest_eligible(tmp_path):
    _write_csv(tmp_path, "most_active_options_2026-03-12.csv", ['OLD,Old,1,1,1%,"5",x'])
    _write_csv(tmp_path, "most_active_options_2026-03-14.csv", ['NEW,New,1,1,1%,"5",x'])
    assert get_top_symbols(str(tmp_path), as_of=date(2026, 3, 13)) == ["OLD"]
    assert get_top_symbols(str(tmp_path)) == ["NEW"]


def test_get_top_symbols_raises_on_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_top_symbols(str(tmp_path / "nope"))


def test_file_date():
    assert _file_date("/x/most_active_options_2026-03-13.csv") == "2026-03-13"


def test_parse_barchart_value():
    assert parse_barchart_value("12.83%") == pytest.approx(0.1283)
    assert parse_barchart_value("2,123,529") == 2123529.0
    assert parse_barchart_value("-") is None
    assert parse_barchart_value(None) is None
    assert parse_barchart_value("unch") is None


def test_parse_barchart_date():
    assert parse_barchart_date("2026-03-13") == date(2026, 3, 13)
    assert parse_barchart_date("3/13/2026") == date(2026, 3, 13)
    assert parse_barchart_date("N/A") is None


def test_apply_symbol_mappings_injected_dict():
    df = pl.DataFrame({"symbol": ["BRK.B", "AAPL"]})
    out = _apply_symbol_mappings(df, {"BRK.B": "BRK-B"})
    assert out["symbol"].to_list() == ["BRK-B", "AAPL"]
    # no-op without mappings
    assert _apply_symbol_mappings(df, None)["symbol"].to_list() == ["BRK.B", "AAPL"]
