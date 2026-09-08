"""FRED acquisition — US Treasury constant-maturity yields.

Exists because **ThetaData's interest-rate endpoint is free-tier and refuses any date
before 2024-01-01** ("Requesting Interest Rate data before 2024-01-01 with FREE
subscription"), while the options history it serves reaches back to 2016. A consumer
deriving a discount factor over that history therefore cannot get its rate from the same
vendor as its quotes, whatever series it asks for — SOFR, TREASURY_M3 and TREASURY_Y1 all
refuse identically.

FRED's CSV download needs no API key and covers the whole range, daily.

Returns frames and owns no storage, no derivation and no domain logic: which tenor to use,
how to fill non-trading days, and what to do when a fetch fails are all caller decisions.
"""
from __future__ import annotations

import io
import logging
from datetime import date

import polars as pl
import requests

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: Constant-maturity Treasury series, shortest first. These are the discount-curve
#: tenors; a consumer picks the one nearest its typical horizon, or interpolates.
TREASURY_SERIES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
}

_DEFAULT_TIMEOUT = 60


class FredError(RuntimeError):
    """A FRED fetch failed or returned something unusable."""


def fetch_treasury_yields(
    series_id: str,
    start_date: date,
    end_date: date,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
) -> pl.DataFrame:
    """Daily yields for one FRED series over ``[start_date, end_date]``.

    ``series_id`` is a FRED id such as ``DGS3MO`` — see :data:`TREASURY_SERIES`.

    Returns ``date`` (``pl.Date``) and ``rate`` (``pl.Float64``), **as a decimal fraction,
    not a percentage**: FRED publishes 5.31 to mean 5.31%, and a consumer that fed that
    straight into ``exp(-rT)`` would discount at 531%.

    **Rows exist only for days FRED actually observed.** Weekends, federal holidays and
    the occasional single-day gap are absent rather than null — FRED writes ``.`` for
    those and they are dropped here. Filling them is the caller's decision, because the
    right fill depends on what the rate is for; do not assume this frame is dense.

    Raises :class:`FredError` rather than returning an empty frame on a transport failure
    or an unparseable response, so a caller cannot mistake a broken fetch for a genuine
    absence of data.
    """
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} precedes start_date {start_date}")

    params = {
        "id": series_id,
        "cosd": start_date.isoformat(),
        "coed": end_date.isoformat(),
    }
    try:
        response = requests.get(FRED_CSV_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise FredError(f"FRED fetch failed for {series_id}: {exc}") from exc

    try:
        # FRED writes "." for a day it holds no observation; null_values turns those into
        # nulls so they can be dropped rather than parsed as a string and poisoning dtype
        # inference for the whole column.
        raw = pl.read_csv(io.StringIO(response.text), null_values=["."])
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same failure here
        raise FredError(f"FRED returned an unparseable CSV for {series_id}: {exc}") from exc

    if raw.width < 2:
        raise FredError(f"FRED returned no value column for {series_id}: {raw.columns}")

    date_col, value_col = raw.columns[0], raw.columns[1]
    frame = (
        raw.select([
            pl.col(date_col).cast(pl.Date).alias("date"),
            # Percent to decimal. FRED publishes 5.31 for 5.31%.
            (pl.col(value_col).cast(pl.Float64) / 100.0).alias("rate"),
        ])
        .filter(pl.col("rate").is_not_null())
        .sort("date")
    )

    if frame.is_empty():
        raise FredError(
            f"FRED returned no observations for {series_id} over "
            f"{start_date}..{end_date}"
        )

    logger.debug(
        "FRED %s: %d observations %s..%s",
        series_id, frame.height, frame["date"][0], frame["date"][-1],
    )
    return frame


__all__ = ["FRED_CSV_URL", "TREASURY_SERIES", "FredError", "fetch_treasury_yields"]
