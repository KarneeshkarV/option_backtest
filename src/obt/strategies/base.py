"""Strategy callable types.

A strategy's whole job is to turn spot bars into entry/exit intent plus a
choice of option leg. It does **not** price anything, size anything, or handle
end-of-day squaring off -- the engine owns those, so that every strategy gets
the same guarantees and none can forget them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd

from obt.chain import LegSpec


class Signals(NamedTuple):
    """What a strategy emits for one run.

    ``entries`` and ``exits`` are per-bar boolean masks aligned to the input
    frame. They may be noisy -- repeated entries, exits while flat -- because
    :func:`obt.signals.resolve_trades` cleans them up.

    ``leg`` may vary per bar (``legs``) for strategies that pick calls or puts
    depending on direction; when it is constant, use ``leg``.
    """

    entries: np.ndarray
    exits: np.ndarray
    leg: LegSpec | None = None
    legs: pd.Series | None = None


StrategyFn = Callable[..., Signals]
