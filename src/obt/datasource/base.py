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


#: Columns every option-chain feed must expose after :func:`normalize_option_chain`.
OPTION_CHAIN_COLUMNS: tuple[str, ...] = ("right", "strike", "expiry", "premium")


@runtime_checkable
class OptionChainSource(Protocol):
    """Load observed option quotes for ``symbol``.

    Returns a frame satisfying :func:`normalize_option_chain`: tz-aware IST
    DatetimeIndex, columns ``right`` (``call``/``put``), ``strike``, ``expiry``
    (python ``date``), and ``premium`` (float64 close). Multiple rows may share
    a timestamp (one per right, or a fuller chain later).

    This is the path that replaces Black-76 model prices in the engine when
    real quotes are available -- see :func:`obt.chain.pinned_leg_from_observed_chain`.
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


def normalize_option_chain(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a vendor option frame into the :class:`OptionChainSource` contract.

    Accepts either a pre-shaped frame (``right/strike/expiry/premium`` on a
    DatetimeIndex) or a raw CE/PE export with ``option_type`` and OHLC columns.
    Timestamps must already be tz-aware IST -- localize upstream (vendor
    files omit the offset; see ``nifty_index_csv.read_naive_ist_csv``).
    """
    out = frame.copy()
    if "datetime" in out.columns:
        idx = pd.to_datetime(out["datetime"])
        out = out.drop(columns=["datetime"])
        out.index = idx
    if out.index.tz is None:
        raise ValueError(
            "option chain timestamps must be tz-aware Asia/Kolkata; "
            "localize with read_naive_ist_csv (or equivalent) before normalize"
        )
    out.index = out.index.tz_convert(IST)
    out.index.name = "datetime"

    if "right" not in out.columns and "option_type" in out.columns:
        mapping = {"CE": "call", "PE": "put", "call": "call", "put": "put"}
        out["right"] = out["option_type"].map(mapping)
        bad = out["right"].isna()
        if bad.any():
            unknown = sorted({*out.loc[bad, "option_type"].astype(str)})
            raise ValueError(f"unknown option_type values: {unknown}")

    if "premium" not in out.columns:
        if "close" not in out.columns:
            raise ValueError(
                "option chain needs a 'premium' column or a 'close' quote to map"
            )
        out["premium"] = out["close"]

    if "expiry" not in out.columns:
        raise ValueError("option chain is missing required column: expiry")
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.date

    missing = [c for c in OPTION_CHAIN_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"option chain is missing required columns: {missing}")

    out = out[list(OPTION_CHAIN_COLUMNS)].copy()
    out["right"] = out["right"].astype(str)
    if not set(out["right"]).issubset({"call", "put"}):
        raise ValueError(
            f"option chain 'right' must be call/put; got {sorted(set(out['right']))}"
        )
    out["strike"] = out["strike"].astype("float64")
    out["premium"] = out["premium"].astype("float64")
    out = out.sort_index()
    return out


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
