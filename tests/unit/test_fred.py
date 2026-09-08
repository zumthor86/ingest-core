"""Unit tests for clients/fred.py — hermetic, no network.

The client exists because ThetaData's interest-rate endpoint is free-tier and refuses any
date before 2024-01-01 while the options history it serves reaches back to 2016.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import requests

from ingest_core.clients.fred import FredError, fetch_treasury_yields


def _response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.raise_for_status = MagicMock()
    return response


CSV = "observation_date,DGS3MO\n2024-03-01,5.25\n2024-03-02,.\n2024-03-04,5.26\n"


def test_percent_is_converted_to_a_decimal_fraction():
    """FRED publishes 5.25 to mean 5.25%.

    A consumer feeding that straight into exp(-rT) would discount at 525%, so getting
    this wrong is not a rounding error.
    """
    with patch("ingest_core.clients.fred.requests.get", return_value=_response(CSV)):
        out = fetch_treasury_yields("DGS3MO", date(2024, 3, 1), date(2024, 3, 4))

    assert out["rate"].to_list() == pytest.approx([0.0525, 0.0526])


def test_unobserved_days_are_dropped_not_returned_as_null():
    """FRED writes "." for a day it holds no observation.

    Those rows are absent from the result rather than null, because how to fill a gap
    depends on what the rate is for -- that is the caller's decision.
    """
    with patch("ingest_core.clients.fred.requests.get", return_value=_response(CSV)):
        out = fetch_treasury_yields("DGS3MO", date(2024, 3, 1), date(2024, 3, 4))

    assert out["date"].to_list() == [date(2024, 3, 1), date(2024, 3, 4)]
    assert out["rate"].null_count() == 0


def test_the_frame_is_sorted_by_date():
    scrambled = "observation_date,DGS3MO\n2024-03-04,5.26\n2024-03-01,5.25\n"
    with patch("ingest_core.clients.fred.requests.get", return_value=_response(scrambled)):
        out = fetch_treasury_yields("DGS3MO", date(2024, 3, 1), date(2024, 3, 4))

    assert out["date"].to_list() == [date(2024, 3, 1), date(2024, 3, 4)]


def test_a_transport_failure_raises_rather_than_returning_empty():
    """A broken fetch must not be mistakable for a genuine absence of data."""
    with patch("ingest_core.clients.fred.requests.get",
               side_effect=requests.RequestException("connection reset")):
        with pytest.raises(FredError, match="fetch failed"):
            fetch_treasury_yields("DGS3MO", date(2024, 3, 1), date(2024, 3, 4))


def test_an_all_missing_series_raises():
    empty = "observation_date,DGS3MO\n2024-03-01,.\n2024-03-02,.\n"
    with patch("ingest_core.clients.fred.requests.get", return_value=_response(empty)):
        with pytest.raises(FredError, match="no observations"):
            fetch_treasury_yields("DGS3MO", date(2024, 3, 1), date(2024, 3, 2))


def test_a_reversed_window_is_rejected_before_the_request():
    with patch("ingest_core.clients.fred.requests.get") as get:
        with pytest.raises(ValueError, match="precedes"):
            fetch_treasury_yields("DGS3MO", date(2024, 3, 4), date(2024, 3, 1))
    get.assert_not_called()


def test_the_requested_window_is_passed_through_to_fred():
    with patch("ingest_core.clients.fred.requests.get", return_value=_response(CSV)) as get:
        fetch_treasury_yields("DGS1MO", date(2016, 1, 4), date(2016, 1, 8))

    params = get.call_args.kwargs["params"]
    assert params == {"id": "DGS1MO", "cosd": "2016-01-04", "coed": "2016-01-08"}
