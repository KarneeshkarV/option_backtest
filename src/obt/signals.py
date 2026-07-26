"""Resolve raw entry/exit masks into actual positions.

This runs *before* pricing, and that ordering is the point. A strategy emits
noisy signals -- repeated entries while already long, exits with nothing open.
Pricing needs to know which bars genuinely *open* a position, because that is
where the option's strike gets pinned. Pinning on every raw entry signal would
roll the strike underneath a live position and silently fabricate P&L.

The end-of-day force-exit lives here rather than in each strategy, so no
strategy can forget it, and no position can span the multi-month gaps in the
data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def last_bar_of_day(bars: pd.DataFrame) -> np.ndarray:
    """Boolean mask marking each session's final bar."""
    dates = bars["date"].to_numpy()
    is_last = np.zeros(len(bars), dtype=bool)
    if len(bars):
        is_last[-1] = True
        is_last[:-1] = dates[:-1] != dates[1:]
    return is_last


def shift_signals(mask: pd.Series | np.ndarray) -> np.ndarray:
    """Delay a signal by one bar.

    A signal computed from bar ``t``'s close cannot be acted on until bar
    ``t+1``. screener's vbt sweep applies the same shift; without it every
    backtest here would quietly trade on information it did not have.
    """
    values = np.asarray(mask, dtype=bool)
    out = np.zeros_like(values)
    out[1:] = values[:-1]
    return out


def resolve_trades(
    entries: np.ndarray,
    exits: np.ndarray,
    is_last_bar: np.ndarray,
) -> dict[str, np.ndarray]:
    """Walk signals into non-overlapping, same-day trades.

    Returns ``open_mask``, ``close_mask`` and ``trade_id`` (-1 while flat).
    A position is never carried past its session's last bar.
    """
    entries = np.asarray(entries, dtype=bool)
    exits = np.asarray(exits, dtype=bool)
    is_last_bar = np.asarray(is_last_bar, dtype=bool)
    n = len(entries)

    open_mask = np.zeros(n, dtype=bool)
    close_mask = np.zeros(n, dtype=bool)
    trade_id = np.full(n, -1, dtype=np.int64)

    in_position = False
    current = -1
    for i in range(n):
        if in_position:
            trade_id[i] = current
            if exits[i] or is_last_bar[i]:
                close_mask[i] = True
                in_position = False
            continue
        # Opening on the final bar would buy something with no bars left to
        # hold it, so skip it rather than book an instant round-trip of costs.
        if entries[i] and not is_last_bar[i]:
            in_position = True
            current += 1
            open_mask[i] = True
            trade_id[i] = current

    return {
        "open_mask": open_mask,
        "close_mask": close_mask,
        "trade_id": trade_id,
    }
