"""Unit tests for clients/cftc.py: hermetic, no network."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import requests

from ingest_core.clients import cftc
from ingest_core.clients.cftc import CftcError, fetch_cot

LEGACY_HEADER = ("report_date_as_yyyy_mm_dd,cftc_contract_market_code,market_and_exchange_names,open_interest_all,"
                 "comm_positions_long_all,comm_positions_short_all,traders_comm_long_all,traders_comm_short_all,"
                 "noncomm_positions_long_all,noncomm_positions_short_all,noncomm_postions_spread_all,"
                 "traders_noncomm_long_all,traders_noncomm_short_all,traders_noncomm_spread_all,"
                 "nonrept_positions_long_all,nonrept_positions_short_all")
# WTI as of 2026-10-06, as the site served it on 2026-10-15.
WTI = ("2026-10-06T00:00:00.000,067651,WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE,1889422,"
       "894330,1020788,95,93,336706,237646,581197,123,62,120,77189,49791")


def _response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock()
    return response


def _legacy(*rows: str) -> MagicMock:
    return _response("\n".join([LEGACY_HEADER, *rows]) + "\n")


def test_each_report_row_becomes_one_row_per_trader_group():
    with patch("ingest_core.clients.cftc.requests.get", return_value=_legacy(WTI)):
        out = fetch_cot("legacy", since=date(2026, 10, 1))

    assert sorted(out["trader_group"]) == ["commercial", "noncommercial", "nonreportable"]
    assert set(out["as_of_date"]) == {date(2026, 10, 6)}
    assert set(out["open_interest"]) == {1889422}
    noncomm = out.filter(trader_group="noncommercial").row(0, named=True)
    assert (noncomm["positions_long"], noncomm["positions_short"], noncomm["positions_spread"]) == (336706, 237646, 581197)


def test_the_parts_add_up_to_open_interest():
    """Every long position belongs to exactly one group, so a wrong field map shows here."""
    with patch("ingest_core.clients.cftc.requests.get", return_value=_legacy(WTI)):
        out = fetch_cot("legacy")

    longs = out["positions_long"].sum() + out["positions_spread"].fill_null(0).sum()
    assert longs == 1889422


def test_a_count_the_report_lacks_is_null_not_zero():
    with patch("ingest_core.clients.cftc.requests.get", return_value=_legacy(WTI)):
        out = fetch_cot("legacy")

    comm = out.filter(trader_group="commercial").row(0, named=True)
    nonrept = out.filter(trader_group="nonreportable").row(0, named=True)
    assert comm["positions_spread"] is None
    assert nonrept["traders_long"] is None


def test_the_whole_market_spread_is_asked_for_not_the_old_crop_one():
    """Legacy has noncomm_positions_spread (old crop only) and the misspelt whole-market field."""
    with patch("ingest_core.clients.cftc.requests.get", return_value=_legacy(WTI)) as get:
        fetch_cot("legacy")

    select = get.call_args.kwargs["params"]["$select"].split(", ")
    assert "noncomm_postions_spread_all" in select
    assert "noncomm_positions_spread" not in select


def test_the_window_and_codes_are_sent_as_a_where_clause():
    with patch("ingest_core.clients.cftc.requests.get", return_value=_legacy(WTI)) as get:
        fetch_cot("legacy", since=date(2026, 9, 1), until=date(2026, 10, 6), codes=["067651", "088691"])

    where = get.call_args.kwargs["params"]["$where"]
    assert "report_date_as_yyyy_mm_dd >= '2026-09-01T00:00:00'" in where
    assert "report_date_as_yyyy_mm_dd <= '2026-10-06T00:00:00'" in where
    assert "cftc_contract_market_code in ('067651', '088691')" in where


def test_pages_are_read_until_a_short_page():
    pages = [_legacy(WTI, WTI.replace("2026-10-06", "2026-09-29")), _legacy(WTI.replace("2026-10-06", "2026-09-22"))]
    with patch.object(cftc, "PAGE_ROWS", 2), \
            patch("ingest_core.clients.cftc.requests.get", side_effect=pages) as get:
        out = fetch_cot("legacy")

    assert [c.kwargs["params"]["$offset"] for c in get.call_args_list] == [0, 2]
    assert out["as_of_date"].n_unique() == 3


def test_no_report_in_the_window_is_an_empty_typed_frame():
    with patch("ingest_core.clients.cftc.requests.get", return_value=_legacy()):
        out = fetch_cot("legacy", since=date(2030, 1, 1))

    assert out.is_empty()
    assert "trader_group" in out.columns and "as_of_date" in out.columns


def test_a_transport_failure_raises_rather_than_returning_empty():
    with patch("ingest_core.clients.cftc.requests.get", side_effect=requests.RequestException("reset")):
        with pytest.raises(CftcError, match="fetch failed"):
            fetch_cot("legacy")


def test_a_missing_field_raises():
    """A renamed vendor field must stop the load, not read as a column of nulls."""
    header = LEGACY_HEADER.replace("comm_positions_long_all,", "comm_long,", 1)
    with patch("ingest_core.clients.cftc.requests.get", return_value=_response(header + "\n" + WTI + "\n")):
        with pytest.raises(CftcError, match="lack the fields"):
            fetch_cot("legacy")


def test_every_report_maps_each_value_slot():
    for report, groups in cftc.GROUP_FIELDS.items():
        for group, spec in groups.items():
            assert len(spec) == 6, (report, group)
            assert spec[0] and spec[1], (report, group)


def test_an_unknown_report_is_rejected_before_the_request():
    with patch("ingest_core.clients.cftc.requests.get") as get:
        with pytest.raises(ValueError, match="unknown report"):
            fetch_cot("supplemental")
    get.assert_not_called()
