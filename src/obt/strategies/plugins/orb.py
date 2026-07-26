"""Opening Range Breakout, expressed as long options.

The first ``range_bars`` minutes of the session define a high/low band. The
first close outside it takes a position in the direction of the break -- a long
call above, a long put below -- with a stop at the band's opposite edge. The
engine squares off at the close regardless.

Chosen as the reference strategy because it is completely unambiguous (no
fitted parameters beyond the range length), strictly intraday, and it exercises
every seam: it needs session structure, it picks between calls and puts per
day, and it produces enough trades to make the statistics mean something.

Whether it is *profitable* is not the point and should not be assumed -- ORB on
NIFTY after realistic option slippage is a hard test to pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from obt.chain import LegSpec
from obt.strategies.base import Signals
from obt.strategies.spec import strategy


@strategy(
    "orb",
    description="Opening range breakout, long ATM call/put in the break direction",
    defaults={"range_bars": 15, "strike_rule": "atm"},
)
def opening_range_breakout(
    bars: pd.DataFrame,
    *,
    range_bars: int = 15,
    strike_rule: str = "atm",
    lots: int = 1,
    lot_size: int | None = None,
    use_stop: bool = True,
) -> Signals:
    bar_of_day = bars["bar_of_day"].to_numpy()
    close = bars["close"].to_numpy()

    # The opening range: high/low over the first `range_bars` bars of each day,
    # broadcast back to every bar of that day.
    in_range = bars["bar_of_day"] < range_bars
    range_high = (
        bars["high"].where(in_range).groupby(bars["date"]).transform("max").to_numpy()
    )
    range_low = (
        bars["low"].where(in_range).groupby(bars["date"]).transform("min").to_numpy()
    )

    tradeable = bar_of_day >= range_bars
    broke_up = tradeable & (close > range_high)
    broke_down = tradeable & (close < range_low)

    # Only the first break of each day counts; later ones are the same move.
    first_up = _first_true_per_day(broke_up, bars["date"])
    first_down = _first_true_per_day(broke_down, bars["date"])

    # If both fire on the same day, the earlier one wins.
    up_idx = _first_index_per_day(first_up, bars["date"])
    down_idx = _first_index_per_day(first_down, bars["date"])
    up_wins = up_idx <= down_idx
    entries_up = first_up & up_wins
    entries_down = first_down & ~up_wins

    entries = entries_up | entries_down

    # Stop: price back through the far side of the opening range.
    if use_stop:
        direction = _direction_per_day(entries_up, entries_down, bars["date"])
        stopped_long = (direction == 1) & (close < range_low)
        stopped_short = (direction == -1) & (close > range_high)
        exits = stopped_long | stopped_short
    else:
        exits = np.zeros(len(bars), dtype=bool)

    legs = _legs_for(entries_up, entries_down, bars, strike_rule, lots, lot_size)
    return Signals(entries=entries, exits=exits, legs=legs)


def _first_true_per_day(mask: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Keep only the first True in each day."""
    series = pd.Series(mask, index=dates.index)
    cumulative = series.groupby(dates).cumsum()
    return (series & (cumulative == 1)).to_numpy()


def _first_index_per_day(mask: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Position of each day's first True, broadcast to that whole day.

    Days with no True get a large sentinel so comparisons treat them as
    "never happened" rather than "happened at bar 0".
    """
    positions = np.arange(len(mask))
    marked = pd.Series(np.where(mask, positions, np.iinfo(np.int64).max))
    return marked.groupby(dates.to_numpy()).transform("min").to_numpy()


def _direction_per_day(
    up: np.ndarray, down: np.ndarray, dates: pd.Series
) -> np.ndarray:
    """+1 on days that broke up, -1 on days that broke down, 0 otherwise."""
    signed = pd.Series(np.where(up, 1, np.where(down, -1, 0)))
    # Forward-fill the day's direction from its entry bar onward.
    return signed.groupby(dates.to_numpy()).transform("cumsum").to_numpy()


def _legs_for(
    up: np.ndarray,
    down: np.ndarray,
    bars: pd.DataFrame,
    strike_rule: str,
    lots: int,
    lot_size: int | None,
) -> pd.Series:
    """Per-bar leg choice: calls on upside breaks, puts on downside."""
    kwargs: dict[str, object] = {"strike_rule": strike_rule, "lots": lots}
    if lot_size is not None:
        kwargs["lot_size"] = lot_size
    call_leg = LegSpec(right="call", direction="long", **kwargs)  # type: ignore[arg-type]
    put_leg = LegSpec(right="put", direction="long", **kwargs)  # type: ignore[arg-type]

    # Built as a numpy object array: boolean-mask assignment on a Series with a
    # DatetimeIndex is interpreted as label indexing, not positional.
    values = np.empty(len(bars), dtype=object)
    values[:] = None
    for mask, leg in ((up, call_leg), (down, put_leg)):
        for position in np.flatnonzero(mask):
            values[position] = leg
    return pd.Series(values, index=bars.index, dtype="object")
