"""The data-source seam.

Every spot feed satisfies :class:`SpotSource`. Swapping CSV for parquet, a
database or a broker API means adding one file under ``plugins/`` -- nothing
downstream of :func:`normalize` can tell the difference.

Modelled on ``screener.options.provider.OptionsProvider``, including the
try-in-order :class:`FallbackSpotSource`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from obt.session import IST, OHLC, stamps

LOG = logging.getLogger(__name__)


@runtime_checkable
class SpotSource(Protocol):
    """Load raw 1-minute spot bars for ``symbol``.

    Implementations return a frame satisfying the :func:`normalize` contract:
    a tz-aware ``Asia/Kolkata`` DatetimeIndex, sorted and unique, with float64
    ``open/high/low/close``. Session shaping is *not* the source's job --
    :mod:`obt.session` handles that uniformly.
    """

    def load(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame: ...


def normalize(frame: pd.DataFrame, *, index_col: str = "datetime") -> pd.DataFrame:
    """Coerce a freshly-read frame into the :class:`SpotSource` contract.

    Any column outside OHLC is dropped -- notably ``volume``, which is
    identically zero for an index and would otherwise invite a volume filter
    that silently matches nothing.
    """
    out = frame.copy()
    if index_col in out.columns:
        stamps = pd.to_datetime(out[index_col], utc=True, format="mixed")
        out = out.drop(columns=[index_col]).set_index(stamps.dt.tz_convert(IST))
    else:
        idx = pd.to_datetime(out.index, utc=True)
        out = out.set_index(idx.tz_convert(IST))

    missing = [column for column in OHLC if column not in out.columns]
    if missing:
        raise ValueError(f"spot frame is missing required columns: {missing}")

    out = out[list(OHLC)].astype("float64")
    out.index.name = "datetime"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def slice_dates(
    bars: pd.DataFrame, start: date | None, end: date | None
) -> pd.DataFrame:
    """Inclusive date-bounded slice, applied after :func:`normalize`."""
    if start is not None:
        bars = bars.loc[stamps(bars).date >= start]
    if end is not None:
        bars = bars.loc[stamps(bars).date <= end]
    return bars


class FallbackSpotSource:
    """Try sources in order; return the first that yields bars.

    A source that raises is logged and skipped rather than killing the run --
    the same degradation policy as ``FallbackOptionsProvider``.
    """

    def __init__(self, *sources: SpotSource) -> None:
        self.sources = tuple(sources)

    def load(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        for source in self.sources:
            try:
                bars = source.load(symbol, start, end)
            except Exception as exc:  # noqa: BLE001 - source boundary
                LOG.warning(
                    "spot source %s failed for %s: %s",
                    type(source).__name__,
                    symbol,
                    exc,
                )
                continue
            if not bars.empty:
                return bars
        raise RuntimeError(f"no spot source produced bars for {symbol!r}")
