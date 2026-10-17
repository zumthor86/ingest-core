"""CFTC Commitments of Traders: weekly futures positions by trader group.

The CFTC publishes every report free, with no key, on its public reporting site (a Socrata
store). Three reports, each for futures only or for futures and options combined:

* ``legacy`` (1986 on): commercial, noncommercial, nonreportable.
* ``disaggregated`` (2006-06 on, physical commodities): producer/merchant, swap dealer,
  managed money, other reportable, nonreportable.
* ``tff`` (2006-06 on, financial futures): dealer, asset manager, leveraged funds, other
  reportable, nonreportable.

**A report is as of a Tuesday and is released on a later day** (normally the Friday, 15:30
New York; later in a holiday week, and weeks later during a government shutdown). The
vendor gives only the Tuesday. The release date is the caller's to work out and store: a
consumer that uses Tuesday's positions on Tuesday uses data nobody had yet.

Returns frames and owns no storage. The vendor's field names are not regular (some end in
``_all``, some do not, and two are misspelt), so :data:`GROUP_FIELDS` maps each trader group
to its fields once, here.
"""
from __future__ import annotations

import io
import logging
from datetime import date
from typing import Optional

import polars as pl
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://publicreporting.cftc.gov/resource/{dataset}.csv"

#: (report, scope) -> Socrata dataset id. Checked against the site's metadata 2026-10-15.
DATASETS: dict[tuple[str, str], str] = {
    ("legacy", "futures"): "6dca-aqww",
    ("legacy", "combined"): "jun7-fc8e",
    ("disaggregated", "futures"): "72hh-3qpy",
    ("disaggregated", "combined"): "kh3c-gbw2",
    ("tff", "futures"): "gpe5-46if",
    ("tff", "combined"): "yw9f-hn96",
}

_NONREPT = ("nonrept_positions_long_all", "nonrept_positions_short_all", None, None, None, None)

#: report -> trader group -> (long, short, spread, traders long, traders short, traders spread).
#: None where the report has no such field. In the Legacy report ``noncomm_positions_spread``
#: (no "_all") is the old-crop figure; the whole-market one is the misspelt ``noncomm_postions_spread_all``.
GROUP_FIELDS: dict[str, dict[str, tuple[Optional[str], ...]]] = {
    "legacy": {
        "commercial": ("comm_positions_long_all", "comm_positions_short_all", None,
                       "traders_comm_long_all", "traders_comm_short_all", None),
        "noncommercial": ("noncomm_positions_long_all", "noncomm_positions_short_all", "noncomm_postions_spread_all",
                          "traders_noncomm_long_all", "traders_noncomm_short_all", "traders_noncomm_spread_all"),
        "nonreportable": _NONREPT,
    },
    "disaggregated": {
        "producer_merchant": ("prod_merc_positions_long", "prod_merc_positions_short", None,
                              "traders_prod_merc_long_all", "traders_prod_merc_short_all", None),
        "swap_dealer": ("swap_positions_long_all", "swap__positions_short_all", "swap__positions_spread_all",
                        "traders_swap_long_all", "traders_swap_short_all", "traders_swap_spread_all"),
        "managed_money": ("m_money_positions_long_all", "m_money_positions_short_all", "m_money_positions_spread",
                          "traders_m_money_long_all", "traders_m_money_short_all", "traders_m_money_spread_all"),
        "other_reportable": ("other_rept_positions_long", "other_rept_positions_short", "other_rept_positions_spread",
                             "traders_other_rept_long_all", "traders_other_rept_short", "traders_other_rept_spread"),
        "nonreportable": _NONREPT,
    },
    "tff": {
        "dealer": ("dealer_positions_long_all", "dealer_positions_short_all", "dealer_positions_spread_all",
                   "traders_dealer_long_all", "traders_dealer_short_all", "traders_dealer_spread_all"),
        "asset_manager": ("asset_mgr_positions_long", "asset_mgr_positions_short", "asset_mgr_positions_spread",
                          "traders_asset_mgr_long_all", "traders_asset_mgr_short_all", "traders_asset_mgr_spread"),
        "leveraged_funds": ("lev_money_positions_long", "lev_money_positions_short", "lev_money_positions_spread",
                            "traders_lev_money_long_all", "traders_lev_money_short_all", "traders_lev_money_spread"),
        "other_reportable": ("other_rept_positions_long", "other_rept_positions_short", "other_rept_positions_spread",
                             "traders_other_rept_long_all", "traders_other_rept_short", "traders_other_rept_spread"),
        "nonreportable": _NONREPT,
    },
}

_VALUES = ("positions_long", "positions_short", "positions_spread", "traders_long", "traders_short", "traders_spread")
_KEYS = {
    "report_date_as_yyyy_mm_dd": "as_of_date",
    "cftc_contract_market_code": "cftc_code",
    "market_and_exchange_names": "market_name",
    "open_interest_all": "open_interest",
}
PAGE_ROWS = 50_000
_DEFAULT_TIMEOUT = 120


class CftcError(RuntimeError):
    """A CFTC fetch failed or returned something unusable."""


def _fields(report: str) -> list[str]:
    names = list(_KEYS)
    for spec in GROUP_FIELDS[report].values():
        names += [f for f in spec if f is not None and f not in names]
    return names


def _where(since: Optional[date], until: Optional[date], codes: Optional[list[str]]) -> Optional[str]:
    parts = []
    if since is not None:
        parts.append(f"report_date_as_yyyy_mm_dd >= '{since.isoformat()}T00:00:00'")
    if until is not None:
        parts.append(f"report_date_as_yyyy_mm_dd <= '{until.isoformat()}T00:00:00'")
    if codes:
        quoted = ", ".join("'" + c.replace("'", "") + "'" for c in codes)
        parts.append(f"cftc_contract_market_code in ({quoted})")
    return " AND ".join(parts) or None


def _get_page(url: str, params: dict, timeout: int) -> pl.DataFrame:
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CftcError(f"CFTC fetch failed ({url}, offset {params.get('$offset')}): {exc}") from exc
    try:
        return pl.read_csv(io.StringIO(response.text), infer_schema_length=0)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same failure here
        raise CftcError(f"CFTC returned an unparseable CSV ({url}): {exc}") from exc


def _to_long(raw: pl.DataFrame, report: str) -> pl.DataFrame:
    """One row per (report row, trader group), values as Int64."""
    missing = [f for f in _fields(report) if f not in raw.columns]
    if missing:
        raise CftcError(f"CFTC {report} rows lack the fields {missing}")
    keys = [
        pl.col("report_date_as_yyyy_mm_dd").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("as_of_date"),
        pl.col("cftc_contract_market_code").str.strip_chars().alias("cftc_code"),
        pl.col("market_and_exchange_names").str.strip_chars().alias("market_name"),
        pl.col("open_interest_all").cast(pl.Int64).alias("open_interest"),
    ]
    frames = []
    for group, spec in GROUP_FIELDS[report].items():
        values = [(pl.col(f).cast(pl.Int64) if f else pl.lit(None, pl.Int64)).alias(v)
                  for f, v in zip(spec, _VALUES, strict=True)]
        frames.append(raw.select(*keys, pl.lit(group).alias("trader_group"), *values))
    return pl.concat(frames)


def fetch_cot(
    report: str,
    scope: str = "futures",
    *,
    since: Optional[date] = None,
    until: Optional[date] = None,
    codes: Optional[list[str]] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> pl.DataFrame:
    """Positions by trader group for the as-of dates in ``[since, until]`` (either end open).

    ``report`` is ``legacy``, ``disaggregated`` or ``tff``; ``scope`` is ``futures`` or
    ``combined``. ``codes`` limits it to these CFTC contract market codes (e.g. ``067651``, WTI).

    Returns one row per report row and trader group: ``as_of_date`` (the Tuesday, ``pl.Date``),
    ``cftc_code``, ``market_name``, ``open_interest``, ``trader_group``, ``positions_long``, ``positions_short``,
    ``positions_spread``, ``traders_long``, ``traders_short``, ``traders_spread``. A count the report
    does not have (a commercial spread, a nonreportable trader count) is null.

    An empty result is a real answer (no report in the window). A transport failure, an
    unparseable page or a missing field raises :class:`CftcError`.
    """
    if (report, scope) not in DATASETS:
        raise ValueError(f"unknown report {report!r} / scope {scope!r}; known: {sorted(DATASETS)}")
    if since and until and until < since:
        raise ValueError(f"until {until} precedes since {since}")

    url = BASE_URL.format(dataset=DATASETS[(report, scope)])
    params = {"$select": ", ".join(_fields(report)), "$order": ":id", "$limit": PAGE_ROWS}
    where = _where(since, until, codes)
    if where:
        params["$where"] = where

    pages, offset = [], 0
    while True:
        page = _get_page(url, {**params, "$offset": offset}, timeout)
        if page.height:
            pages.append(page)
        if page.height < PAGE_ROWS:
            break
        offset += PAGE_ROWS
    if not pages:
        return pl.DataFrame(schema={"as_of_date": pl.Date, "cftc_code": pl.Utf8, "market_name": pl.Utf8,
                                    "open_interest": pl.Int64, "trader_group": pl.Utf8,
                                    **dict.fromkeys(_VALUES, pl.Int64)})
    out = _to_long(pl.concat(pages), report)
    logger.debug("CFTC %s/%s: %d report rows, %s..%s", report, scope, out.height // len(GROUP_FIELDS[report]),
                 out["as_of_date"].min(), out["as_of_date"].max())
    return out


__all__ = ["BASE_URL", "DATASETS", "GROUP_FIELDS", "CftcError", "fetch_cot"]
