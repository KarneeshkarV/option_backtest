"""NSE session shaping, shared by every data source.

Cleaning lives here rather than inside a source plugin so that a CSV feed, a
parquet dump and a broker API all produce an identically shaped frame. A source
plugin's only job is to *read* bytes; :func:`clean` decides what counts as a
tradeable bar.

Two rules here carry real weight:

- **Partial days are dropped.** A day with 125 bars instead of 375 would
  silently corrupt anything session-relative (opening ranges, end-of-day
  exits), producing plausible-looking but wrong results.
- **``bar_of_day`` is the anchor for gap safety.** The engine forces an exit on
  the last bar of every day, so a position can never span the multi-month holes
  in the data. That guarantee is structural, not per-strategy.
"""

from __future__ import annotations

from datetime import date, time
from typing import TypedDict

import pandas as pd

IST = "Asia/Kolkata"

#: NSE equity-derivatives regular session.
SESSION_START = time(9, 15)
SESSION_END = time(15, 29)

#: Bars in a complete 1-minute session (09:15..15:29 inclusive).
BARS_PER_SESSION = 375

#: Trading days per year, for annualizing volatility.
TRADING_DAYS_PER_YEAR = 252

OHLC: tuple[str, ...] = ("open", "high", "low", "close")


def stamps(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """``bars.index`` narrowed to a :class:`~pandas.DatetimeIndex`.

    pandas-stubs types ``DataFrame.index`` as ``Index[Any]``, which has no
    ``.date`` or ``.time``. Every frame in this package satisfies the
    :func:`obt.datasource.base.normalize` contract, so the narrowing is a fact
    -- but it is checked rather than asserted, because the one way it could be
    false is a caller skipping ``normalize`` entirely, and that deserves a clear
    error rather than an ``AttributeError`` three frames deeper.
    """
    index = bars.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"expected a DatetimeIndex, got {type(index).__name__}. "
            "Pass the frame through obt.datasource.base.normalize first."
        )
    return index


def filter_regular_session(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars inside 09:15-15:29.

    Drops Muhurat (Diwali) evening sessions, which are real trades but sit
    outside the session every strategy here assumes.
    """
    times = stamps(bars).time
    keep = (times >= SESSION_START) & (times <= SESSION_END)
    return bars.loc[keep]


def complete_days(
    bars: pd.DataFrame, *, min_bars: int = BARS_PER_SESSION
) -> pd.DataFrame:
    """Keep only days holding a full session's worth of bars."""
    days = pd.Series(stamps(bars).date, index=bars.index)
    return bars.loc[days.map(days.value_counts()) >= min_bars]


def add_session_cols(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach ``date`` and ``bar_of_day`` (0-based within each session)."""
    out = bars.copy()
    dates = pd.Index(stamps(out).date, name="date")
    out["date"] = dates
    out["bar_of_day"] = out.groupby(dates).cumcount().to_numpy()
    return out


def clean(bars: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: regular session -> complete days -> session columns."""
    return add_session_cols(complete_days(filter_regular_session(bars)))


def covered_periods(bars: pd.DataFrame) -> list[tuple[date, date]]:
    """Contiguous blocks of trading days present in ``bars``.

    Reported instead of a bare min/max so a sample with a nine-month hole in it
    can never be presented as continuous history.
    """
    days = sorted({*pd.Index(stamps(bars).date)})
    if not days:
        return []
    blocks: list[tuple[date, date]] = []
    block_start = previous = days[0]
    for day in days[1:]:
        # >7 calendar days apart is a real gap, not a weekend or a holiday run.
        if (day - previous).days > 7:
            blocks.append((block_start, previous))
            block_start = day
        previous = day
    blocks.append((block_start, previous))
    return blocks


class SessionSummary(TypedDict):
    """What cleaning removed. A TypedDict rather than ``dict[str, object]`` so
    the report can index it without every field collapsing to ``object``."""

    raw_bars: int
    kept_bars: int
    raw_days: int
    kept_days: int
    dropped_days: list[date]
    periods: list[tuple[date, date]]


def session_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> SessionSummary:
    """Counts describing what cleaning removed, for the report banner."""
    raw_days = {*pd.Index(stamps(raw).date)}
    kept_days = {*pd.Index(stamps(cleaned).date)}
    return {
        "raw_bars": int(len(raw)),
        "kept_bars": int(len(cleaned)),
        "raw_days": len(raw_days),
        "kept_days": len(kept_days),
        "dropped_days": sorted(raw_days - kept_days),
        "periods": covered_periods(cleaned),
    }
