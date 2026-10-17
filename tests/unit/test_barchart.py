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


# ---- the Russell 1000 core-api pager


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Session:
    """Answers each GET with the next response, and remembers the page numbers asked for."""

    def __init__(self, responses):
        self.responses, self.pages, self.cookies = list(responses), [], []

    def get(self, url, headers=None, timeout=None):
        self.pages.append(int(url.rsplit("page=", 1)[1]))
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _rows(*symbols):
    return [{"symbol": s, "symbolName": f"{s} Inc", "marketCap": "1,000"} for s in symbols]


@pytest.fixture
def small_pages(monkeypatch):
    from ingest_core.clients import barchart
    monkeypatch.setattr(barchart, "_R1000_CORE_API_LIMIT", 2)
    return barchart


def test_r1000_pages_until_a_short_page(small_pages):
    s = _Session([_Resp(body={"data": _rows("AAA", "BBB")}), _Resp(body={"data": _rows("CCC")})])
    df, rows = small_pages._try_r1000_core_api(s, "https://ref")
    assert s.pages == [1, 2] and len(rows) == 3
    assert sorted(df["symbol"].to_list()) == ["AAA", "BBB", "CCC"] and "Name" in df.columns


def test_r1000_stops_at_the_stated_total(small_pages):
    s = _Session([_Resp(body={"data": _rows("AAA", "BBB"), "total": "2"})])
    _, rows = small_pages._try_r1000_core_api(s, "https://ref")
    assert s.pages == [1] and len(rows) == 2


def test_r1000_stops_at_an_empty_page_and_ignores_a_bad_total(small_pages):
    s = _Session([_Resp(body={"data": _rows("AAA", "BBB"), "total": "many"}), _Resp(body={"data": []})])
    _, rows = small_pages._try_r1000_core_api(s, "https://ref")
    assert s.pages == [1, 2] and len(rows) == 2


@pytest.mark.parametrize("answer", [
    RuntimeError("refused"), _Resp(status=403, text="blocked"), _Resp(body=ValueError("not json")),
    _Resp(body={"nodata": 1}), _Resp(body=["not", "a", "dict"]), _Resp(body={"data": []}),
])
def test_r1000_any_failure_or_no_rows_is_none(small_pages, answer):
    assert small_pages._try_r1000_core_api(_Session([answer]), "https://ref") is None


def test_r1000_a_failure_on_a_later_page_drops_the_earlier_pages(small_pages):
    s = _Session([_Resp(body={"data": _rows("AAA", "BBB")}), _Resp(status=500)])
    assert small_pages._try_r1000_core_api(s, "https://ref") is None
