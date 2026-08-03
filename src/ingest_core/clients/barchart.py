"""Barchart acquisition — the single shared implementation (FR-001/FR-002).

Merged surface (deduplicated union of the two prior clients, D4):

- ``get_top_symbols`` — most-active CSV *reader* (from Hephaestus): ranks the
  top-N option symbols from the CSVs the fetcher below produces.
- ``fetch_most_active_options`` / ``fetch_russell1000_constituents`` — HTTP
  *fetchers* via the Barchart core-api JSON proxy (from Hermes).

Project-specific symbol mapping (Barchart→Polygon) is **injected** as a plain
``symbol_mappings`` dict by the caller — the mapping config file stays
app-owned in Hermes (FR-003: no project domain logic here). Storage (CSV
writes, ticker upserts, metrics stores) stays app-side (FR-004).
"""
from __future__ import annotations

import glob
import logging
import os
from datetime import date, datetime, UTC
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse, parse_qs

import polars as pl
import requests

logger = logging.getLogger(__name__)

BARCHART_MOST_ACTIVE_URL = (
    "https://www.barchart.com/options/most-active/stocks?viewName=main&orderBy=optionsTotalVolume&orderDir=desc"
)

# Russell 1000 constituents page
BARCHART_R1000_URL = "https://www.barchart.com/stocks/indices/russell/russell1000?viewName=174264"
_BARCHART_CORE_API_URL = "https://www.barchart.com/proxies/core-api/v1/quotes/get"
_R1000_CORE_API_LIMIT = 1000
_R1000_CORE_API_FIELDS = (
    "symbol,symbolName,marketCap,sicDescription,industryGroup,industry,earningsGrowth5y,annualNetIncome,peRatioForward,profitMargin,"
    "priceCashFlow,beta,historicVolatility30d,tradeTime,"
    "optionsWeightedImpliedVolatility,optionsImpliedVolatilityPercentile3m,"
    "optionsIntradayVs30dHistoricIV,optionsTotalVolume3m,optionsWeightedImpliedVolatility1m,"
    "optionsWeightedImpliedVolatility6m,nextEarningsDate,"
    # Extended metric fields persisted only to the long-format metrics store.
    "sharesOutstanding,dividendRate,dividendYield,"
    "optionsImpliedVolatilityRank1y,optionsPutCallOpenInterestRatio,optionsPutCallVolumeRatio,"
    "returnOnEquity,returnOnAssets,shortFloat,analystRating,"
    "eps,peRatio,pegRatio,priceBook,priceSales"
)

_R1000_EXPORT_COLUMN_MAP = {
    "symbol": "symbol",
    "symbolName": "Name",
    "marketCap": "Market Cap",
    "sicDescription": "SIC Description",
    "industryGroup": "Sector",
    "industry": "Industry",
    "earningsGrowth5y": "5Y Earn%",
    "annualNetIncome": "Net Income(a)",
    "peRatioForward": "P/E fwd",
    "profitMargin": "Profit%",
    "priceCashFlow": "Price/Cash Flow",
    "beta": "Beta",
    "historicVolatility30d": "30D His Vol",
    "tradeTime": "Time",
    "optionsWeightedImpliedVolatility": "Imp Vol",
    "optionsImpliedVolatilityPercentile3m": "3M IV Pctl",
    "optionsIntradayVs30dHistoricIV": "IV/HV",
    "optionsTotalVolume3m": "3M Total Vol",
    "optionsWeightedImpliedVolatility1m": "1M IV",
    "optionsWeightedImpliedVolatility6m": "6M IV",
    "nextEarningsDate": "Earnings Date",
}
_R1000_EXPORT_COLUMN_ORDER = [
    "symbol",
    "Name",
    "Sector",
    "Industry",
    "Market Cap",
    "SIC Description",
    "5Y Earn%",
    "Net Income(a)",
    "P/E fwd",
    "Profit%",
    "Price/Cash Flow",
    "Beta",
    "30D His Vol",
    "Time",
    "Imp Vol",
    "3M IV Pctl",
    "IV/HV",
    "3M Total Vol",
    "1M IV",
    "6M IV",
    "Earnings Date",
]


# ---------------------------------------------------------------------------
# Most-active CSV reader (promoted from Hephaestus processing/clients/barchart.py)
# ---------------------------------------------------------------------------


def get_top_symbols(
    folder: str,
    top_n: int = 300,
    as_of: Optional[date] = None,
) -> list[str]:
    """
    Return the top-N option symbols by total volume from the most-recent
    Barchart most-active CSV file in *folder*.

    Args:
        folder:  Directory containing ``most_active_options_YYYY-MM-DD.csv`` files.
        top_n:   Maximum number of symbols to return (default 300).
        as_of:   If supplied, use the latest file on or before this date.
                 Defaults to the most-recently dated file in the folder.

    Returns:
        List of base ticker symbols (e.g. ["NVDA", "TSLA", ...]) ordered by
        descending options volume, capped at *top_n*.

    Raises:
        FileNotFoundError: If no matching CSV files exist in *folder*.
        ValueError:        If the chosen file is empty or missing required columns.
    """
    pattern = os.path.join(folder, "most_active_options_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No most_active_options_*.csv files found in: {folder}"
        )

    if as_of is not None:
        cutoff = as_of.strftime("%Y-%m-%d")
        eligible = [f for f in files if _file_date(f) <= cutoff]
        if not eligible:
            raise FileNotFoundError(
                f"No most_active_options file on or before {cutoff} in: {folder}"
            )
        chosen = eligible[-1]  # latest eligible
    else:
        chosen = files[-1]  # most recent overall

    # Read numeric-looking columns as strings to handle comma-formatted values
    # like "1,007.52" (lastPrice) and "3,063,353" (optionsTotalVolume).
    df = pl.read_csv(
        chosen,
        schema_overrides={
            "lastPrice": pl.Utf8,
            "priceChange": pl.Utf8,
            "percentChange": pl.Utf8,
            "optionsTotalVolume": pl.Utf8,
        },
    )

    required = {"symbol", "optionsTotalVolume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV {chosen!r} is missing required columns: {missing}"
        )

    # Parse volume: stored as "3,063,353" (comma-formatted string)
    df = df.with_columns(
        pl.col("optionsTotalVolume")
        .cast(pl.Utf8)
        .str.replace_all(",", "")
        .cast(pl.Int64)
        .alias("volume_int")
    )

    top = (
        df.sort("volume_int", descending=True)
        .head(top_n)
        .select("symbol")
        .to_series()
        .to_list()
    )

    return top


def _file_date(path: str) -> str:
    """Extract the YYYY-MM-DD date string from a filename."""
    basename = os.path.basename(path)
    # most_active_options_2026-03-13.csv  ->  2026-03-13
    name_no_ext = os.path.splitext(basename)[0]
    return name_no_ext.split("_")[-1]


# ---------------------------------------------------------------------------
# HTTP fetchers (promoted from Hermes io/ingest/barchart.py; symbol mapping injected)
# ---------------------------------------------------------------------------


def _std_headers() -> dict:
    # Lightly spoof headers to avoid trivial bot blocks.
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.barchart.com/",
        "Connection": "keep-alive",
    }


def _get_cookie_value(session: requests.Session, name: str) -> Optional[str]:
    try:
        for c in session.cookies:
            if c.name == name and c.value:
                return c.value
    except Exception:
        return None
    return None


def _xsrf_header(session: requests.Session) -> Optional[dict]:
    token = _get_cookie_value(session, "XSRF-TOKEN") or _get_cookie_value(session, "xsrf-token")
    if not token:
        return None
    # Some frameworks require unquoted token value
    try:
        decoded = unquote(token)
    except Exception:
        decoded = token
    return {"X-XSRF-TOKEN": decoded}


def _apply_symbol_mappings(
    df: pl.DataFrame, mappings: Optional[Dict[str, str]] = None
) -> pl.DataFrame:
    """Apply caller-supplied symbol mappings (e.g. Barchart->Polygon) to a frame.

    Logs each symbol that gets mapped. No-op when ``mappings`` is falsy.
    """
    if df is None or df.is_empty() or "symbol" not in df.columns:
        return df
    if not mappings:
        return df

    # Track which symbols we actually map
    mapped_symbols = []

    # Build mapping expression for Polars
    # Use when-then-otherwise chain for each mapping
    expr = pl.col("symbol")
    for source_symbol, target_symbol in mappings.items():
        expr = pl.when(pl.col("symbol") == source_symbol).then(pl.lit(target_symbol)).otherwise(expr)
        # Check if this symbol exists in the dataframe
        if source_symbol in df.get_column("symbol").to_list():
            mapped_symbols.append(f"{source_symbol}->{target_symbol}")

    if mapped_symbols:
        logger.info("Mapped %d symbols: %s", len(mapped_symbols), ", ".join(mapped_symbols))

    return df.with_columns(expr.alias("symbol"))


def _try_core_api(session: requests.Session, timeout: int = 25) -> Optional[pl.DataFrame]:
    """Fetch Most Active via the primary core-api list used in practice.

    The endpoint silently returns an empty list unless a filter is supplied;
    `between(lastPrice,1,)=` keeps the call to ~$1+ stocks and unblocks results.
    """
    url = (
        "https://www.barchart.com/proxies/core-api/v1/quotes/get"
        "?list=options.mostActive.us"
        "&between(lastPrice,1,)="
        "&fields=symbol,symbolName,lastPrice,priceChange,percentChange,optionsTotalVolume"
        "&orderBy=optionsTotalVolume&orderDir=desc&limit=500&page=1"
    )
    headers = _std_headers() | {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }
    xsrf = _xsrf_header(session)
    if xsrf:
        headers.update(xsrf)
    try:
        r = session.get(url, headers=headers, timeout=timeout)
    except Exception as e:
        logger.warning("barchart.fetch: core_api_json request error: %s", e)
        return None
    if r.status_code != 200:
        logger.warning(
            "barchart.fetch: core_api_json status=%s body=%r",
            r.status_code, r.text[:200],
        )
        return None
    try:
        js = r.json()
    except Exception as e:
        logger.warning("barchart.fetch: core_api_json non-JSON body=%r err=%s", r.text[:200], e)
        return None
    data = js.get("data") if isinstance(js, dict) else None
    if not isinstance(data, list) or not data:
        logger.warning(
            "barchart.fetch: core_api_json empty data count=%s total=%s",
            js.get("count") if isinstance(js, dict) else None,
            js.get("total") if isinstance(js, dict) else None,
        )
        return None
    rows: List[Dict[str, Any]] = [item for item in data if isinstance(item, dict)]
    if not rows:
        return None
    df_pl = pl.DataFrame(rows)
    logger.debug("barchart.fetch: core_api_json rows=%d cols=%d", df_pl.height, len(df_pl.columns))
    return df_pl


def fetch_most_active_options(
    url: str = BARCHART_MOST_ACTIVE_URL,
    timeout: int = 25,
    symbol_mappings: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """Fetch Most Active Options table from Barchart and return as Polars DataFrame.

    The core-api requires the XSRF-TOKEN cookie set by any page request; a logged-in
    session is not required here, so we just GET the page to seed cookies.
    """
    sess = requests.Session()
    try:
        sess.get(url, headers=_std_headers(), timeout=timeout)
    except Exception as e:
        logger.warning("barchart.fetch: page GET to seed cookies failed: %s", e)
        return pl.DataFrame()
    df_pl = _try_core_api(sess, timeout=timeout)
    if df_pl is None or df_pl.is_empty():
        logger.warning("barchart.fetch: core_api_json returned no data")
        return pl.DataFrame()
    logger.info("barchart.fetch: approach_used=core_api_json")

    # Heuristic: ensure we have a 'symbol' column; try to detect common symbol headings
    if "symbol" not in df_pl.columns:
        for candidate in ["ticker", "root", "underlying", "contract"]:
            if candidate in df_pl.columns:
                df_pl = df_pl.rename({candidate: "symbol"})
                break

    # Apply caller-supplied symbol mappings
    df_pl = _apply_symbol_mappings(df_pl, symbol_mappings)

    df_pl = df_pl.with_columns(
        pl.lit(datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")).alias("scraped_at")
    )
    return df_pl


def _rename_r1000_columns(df: pl.DataFrame) -> pl.DataFrame:
    if df is None or df.is_empty():
        return df
    rename_map = {source: target for source, target in _R1000_EXPORT_COLUMN_MAP.items() if source in df.columns}
    out = df.rename(rename_map) if rename_map else df
    ordered_columns = [column for column in _R1000_EXPORT_COLUMN_ORDER if column in out.columns]
    remaining_columns = [column for column in out.columns if column not in ordered_columns]
    return out.select(ordered_columns + remaining_columns)


def parse_barchart_value(raw: Any) -> Optional[float]:
    """Parse a Barchart display value to float.

    Handles ``"12.83%"`` → ``0.1283``, ``"2,123,529"`` → ``2123529.0``,
    ``"-"``/``""``/``None`` → ``None``.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "N/A", "NA", "unch"):
        return None
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1]
    s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100.0 if is_pct else v


def parse_barchart_date(raw: Any) -> Optional[date]:
    """Parse Barchart date strings (typically ``YYYY-MM-DD`` or ``M/D/YYYY``)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "N/A"):
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _clean_r1000_df(
    df: pl.DataFrame, symbol_mappings: Optional[Dict[str, str]] = None
) -> pl.DataFrame:
    """Normalize and filter Russell 1000 symbols.

    - Uppercase symbols, strip leading hyphens and whitespace
    - Drop obvious non-stock entries (description contains ETF/ETN/Fund)
    - Drop duplicates and nulls
    - Apply caller-supplied symbol mappings
    """
    try:
        if "symbol" not in df.columns:
            return df
        out = df.with_columns([
            pl.col("symbol").cast(pl.Utf8).str.strip_chars().str.replace_all(r"^[-\s]+", "").str.to_uppercase(),
        ])
        # Heuristic filters
        desc_col = None
        for cand in ("description", "name", "company", "SymbolName", "symbolName"):
            if cand in out.columns:
                desc_col = cand
                break
        if desc_col:
            out = out.filter(~pl.col(desc_col).cast(pl.Utf8).str.contains(r"\b(ETF|ETFs|ETN|Fund)\b", literal=False, case_insensitive=True).fill_null(False))
        out = out.filter(pl.col("symbol").str.len_chars() > 0)
        out = out.unique(subset=["symbol"])  # drop dups

        # Apply caller-supplied symbol mappings
        out = _apply_symbol_mappings(out, symbol_mappings)

        return out
    except Exception as e:
        logger.error("Error in _clean_r1000_df: %s - %s", type(e).__name__, str(e), exc_info=True)
        return df


def _try_r1000_core_api(
    session: requests.Session,
    referer_url: str,
    timeout: int = 30,
    symbol_mappings: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[pl.DataFrame, List[Dict[str, Any]]]]:
    headers = _std_headers() | {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": referer_url,
        "X-Requested-With": "XMLHttpRequest",
    }
    xsrf = _xsrf_header(session)
    if xsrf:
        headers.update(xsrf)

    rows: List[Dict[str, Any]] = []
    page = 1
    total: Optional[int] = None

    while True:
        url = (
            f"{_BARCHART_CORE_API_URL}?list=stocks.markets.russell1000"
            f"&fields={_R1000_CORE_API_FIELDS}"
            f"&orderBy=symbol&orderDir=asc&limit={_R1000_CORE_API_LIMIT}&page={page}"
        )
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
        except Exception as e:
            logger.warning("barchart.r1000: core_api page=%d request error: %s", page, e)
            return None
        if resp.status_code != 200:
            logger.warning(
                "barchart.r1000: core_api page=%d status=%s body=%r",
                page,
                resp.status_code,
                resp.text[:200],
            )
            return None
        try:
            js = resp.json()
        except Exception as e:
            logger.warning("barchart.r1000: core_api page=%d non-JSON body=%r err=%s", page, resp.text[:200], e)
            return None

        data = js.get("data") if isinstance(js, dict) else None
        if not isinstance(data, list):
            logger.warning("barchart.r1000: core_api page=%d missing data payload", page)
            return None
        if not data:
            break

        rows.extend(item for item in data if isinstance(item, dict))
        logger.debug(
            "barchart.r1000: core_api page=%d fetched=%d count=%s total=%s",
            page,
            len(data),
            js.get("count") if isinstance(js, dict) else None,
            js.get("total") if isinstance(js, dict) else None,
        )

        if total is None and isinstance(js, dict):
            try:
                total = int(js.get("total")) if js.get("total") is not None else None
            except (TypeError, ValueError):
                total = None

        if total is not None and len(rows) >= total:
            break
        if len(data) < _R1000_CORE_API_LIMIT:
            break
        page += 1

    if not rows:
        return None

    df = pl.DataFrame(rows)
    df = _rename_r1000_columns(df)
    cleaned = _clean_r1000_df(df, symbol_mappings)
    logger.info("barchart.r1000: core_api cleaned rows=%d cols=%d", cleaned.height, len(cleaned.columns))
    return cleaned, rows


def fetch_russell1000_constituents(
    url: str = BARCHART_R1000_URL,
    timeout: int = 25,
    symbol_mappings: Optional[Dict[str, str]] = None,
) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
    """Fetch Russell 1000 constituents via anonymous paginated core-api JSON.

    Seeds session cookies from the public page, then pages through the core API.
    Returns ``(cleaned_df, raw_rows)``:

    - ``cleaned_df`` is normalized to the legacy Barchart CSV column names so
      existing CSV consumers and the ticker-table upsert continue to work
      unchanged.
    - ``raw_rows`` is the underlying camelCase response list, for callers that
      build derived frames (e.g. long-format metrics) without re-fetching.

    Both are empty/empty-list on failure.
    """
    sess = requests.Session()

    try:
        r = sess.get(url, headers=_std_headers(), timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"barchart.r1000: failed to fetch page: {e}")
        return pl.DataFrame(), []

    result = _try_r1000_core_api(
        sess, referer_url=url, timeout=max(timeout, 30), symbol_mappings=symbol_mappings
    )
    if result is None:
        logger.warning("barchart.r1000: core_api returned no data")
        return pl.DataFrame(), []
    df, rows = result
    if df.is_empty():
        logger.warning("barchart.r1000: core_api returned empty cleaned frame")
        return df, rows
    logger.info("barchart.r1000: approach_used=core_api rows=%d", df.height)
    return df, rows


def _extract_view_name(u: str) -> Optional[str]:
    try:
        q = parse_qs(urlparse(u).query)
        v = q.get("viewName") or q.get("view") or q.get("viewId")
        if v and isinstance(v, list) and v[0]:
            return v[0]
    except Exception:
        return None
    return None
