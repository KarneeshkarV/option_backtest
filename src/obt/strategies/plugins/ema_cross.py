"""EMA crossover: long ATM call when the fast EMA crosses above the slow.

Default periods are 5 and 20 (on the bar timeframe of the feed — 1-minute
here). Entry fires on a *bullish* cross (fast moves from at/below slow to
above). Exit fires on the reverse *bearish* cross. The engine still
force-closes every position at the session's last bar, so no trade spans a
data gap even if the death cross has not happened yet.

This is a pure long-premium signal. It is not expected to be profitable after
option costs and theta; it exists so the strategy seam can be exercised with a
classic indicator rule that is easy to verify by hand.
"""

from __future__ import annotations

import pandas as pd

from obt.chain import LegSpec
from obt.strategies.base import Signals
from obt.strategies.spec import strategy


@strategy(
    "ema_cross",
    description="Long ATM call on EMA(fast) cross above EMA(slow); exit on reverse cross",
    defaults={"fast": 5, "slow": 20, "strike_rule": "atm"},
)
def ema_crossover(
    bars: pd.DataFrame,
    *,
    fast: int = 5,
    slow: int = 20,
    strike_rule: str = "atm",
    lots: int = 1,
    lot_size: int | None = None,
) -> Signals:
    if fast < 1 or slow < 1:
        raise ValueError(f"EMA periods must be >= 1; got fast={fast}, slow={slow}")
    if fast >= slow:
        raise ValueError(
            f"fast EMA period must be shorter than slow; got fast={fast}, slow={slow}"
        )

    close = bars["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    # Cross events use the prior bar's relationship so a single bar that
    # jumps through the slow EMA still counts as one discrete cross.
    prev_fast = ema_fast.shift(1)
    prev_slow = ema_slow.shift(1)
    cross_up = (prev_fast <= prev_slow) & (ema_fast > ema_slow)
    cross_down = (prev_fast >= prev_slow) & (ema_fast < ema_slow)

    entries = cross_up.fillna(False).to_numpy(dtype=bool)
    exits = cross_down.fillna(False).to_numpy(dtype=bool)

    kwargs: dict[str, object] = {"strike_rule": strike_rule, "lots": lots}
    if lot_size is not None:
        kwargs["lot_size"] = lot_size
    leg = LegSpec(right="call", direction="long", **kwargs)  # type: ignore[arg-type]
    return Signals(entries=entries, exits=exits, leg=leg)
