"""Buy an ATM call at the open every day, hold to the close.

Deliberately mindless. It exists to prove the strategy seam works -- adding it
required one file and one import line, with no edits anywhere else -- and to
act as a control: it is a pure long-premium, long-theta-decay position, so it
should reliably *lose* money to decay and slippage. A backtester that shows
this making money has a bug, which makes it a useful canary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from obt.chain import LegSpec
from obt.strategies.base import Signals
from obt.strategies.spec import strategy


@strategy(
    "buy_open",
    description="Long ATM call from the opening bar to the close (decay control)",
    defaults={"entry_bar": 5},
)
def buy_at_open(
    bars: pd.DataFrame,
    *,
    entry_bar: int = 5,
    strike_rule: str = "atm",
    lots: int = 1,
    lot_size: int | None = None,
) -> Signals:
    entries = (bars["bar_of_day"] == entry_bar).to_numpy()
    # No discretionary exit: the engine's end-of-day square-off closes it.
    exits = np.zeros(len(bars), dtype=bool)

    kwargs: dict[str, object] = {"strike_rule": strike_rule, "lots": lots}
    if lot_size is not None:
        kwargs["lot_size"] = lot_size
    leg = LegSpec(right="call", direction="long", **kwargs)  # type: ignore[arg-type]
    return Signals(entries=entries, exits=exits, leg=leg)
