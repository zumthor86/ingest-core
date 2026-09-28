"""Databento Historical client: quote, download and decode OHLCV bars. Acquisition only.

The caller owns where downloaded files live and what is built from them; this module fetches bytes and
turns a DBN file into a Polars frame. Databento bills per byte served, so every request can be quoted
first (:func:`cost`, free) and a download writes to ``<path>.part`` and renames only when complete, so an
interrupted download never leaves a file that looks finished.

Needs the optional ``databento`` extra and ``DATABENTO_API_KEY`` in the environment.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import polars as pl

_logger = logging.getLogger(__name__)

BAR_COLUMNS = ["ts_event", "instrument_id", "open", "high", "low", "close", "volume"]


def _db():
    import databento  # optional extra: only the bar producer needs it

    return databento


def client():
    """A ``databento.Historical`` client keyed from ``DATABENTO_API_KEY``."""
    return _db().Historical()


def _retry(fn: Callable[..., Any], *args: Any, attempts: int = 6, **kwargs: Any) -> Any:
    """Databento answers the odd request with an empty 5xx; those are retried with a growing wait."""
    server_error = _db().BentoServerError
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except server_error as exc:
            if attempt == attempts - 1:
                raise
            wait = 15 * (attempt + 1)
            _logger.warning("databento server error (%s); retrying in %ss", exc, wait)
            time.sleep(wait)


def request(dataset: str, schema: str, symbol: str, start: str, end: str, stype_in: str = "continuous") -> dict:
    """Keyword arguments for one get_range / get_cost call. ``end`` is exclusive."""
    return dict(dataset=dataset, schema=schema, stype_in=stype_in, symbols=[symbol], start=start, end=end)


def cost(hist, req: dict) -> float:
    """US$ the request would bill. Free."""
    return float(_retry(hist.metadata.get_cost, **req))


def available_end(hist, dataset: str) -> str:
    """The exclusive end of what the dataset can serve now (ISO timestamp, UTC)."""
    return str(_retry(hist.metadata.get_dataset_range, dataset=dataset)["end"])


def download(hist, req: dict, path: Path) -> int:
    """Fetch ``req`` into ``path`` (DBN, zstd). Returns bytes written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    _retry(hist.timeseries.get_range, **req, path=tmp)
    tmp.replace(path)
    return path.stat().st_size


def read_bars(path: Path) -> pl.DataFrame:
    """A DBN OHLCV file as ``[ts_event (UTC), instrument_id, open, high, low, close, volume]``."""
    df = _db().DBNStore.from_file(Path(path)).to_df()
    if df.empty:
        return pl.DataFrame(schema={"ts_event": pl.Datetime("ns", "UTC"), "instrument_id": pl.UInt32, "open": pl.Float64,
                                    "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.UInt64})
    return pl.from_pandas(df.reset_index()[BAR_COLUMNS])
