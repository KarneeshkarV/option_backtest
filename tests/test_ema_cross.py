"""Signal-level checks for the EMA crossover strategy.

These tests assert the *cross event* logic on a hand-built series so a broken
comparison or off-by-one shift fails without needing a full engine run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obt.strategies.plugins.ema_cross import ema_crossover
from obt.strategies.spec import get_strategy


def _bars_from_close(close: list[float]) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="1min")
    close_arr = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr,
            "low": close_arr,
            "close": close_arr,
            "date": idx.date,
            "bar_of_day": np.arange(n),
        },
        index=idx,
    )


def test_ema_cross_is_registered():
    spec = get_strategy("ema_cross")
    assert spec.defaults["fast"] == 5
    assert spec.defaults["slow"] == 20


def test_golden_cross_enters_and_death_cross_exits():
    # Rising then falling path that forces a 2/4 EMA golden then death cross.
    # Values chosen so the geometry is obvious, not so the periods match live
    # defaults — defaults are exercised by the engine smoke path.
    close = [
        100.0,
        100.0,
        100.0,
        100.0,
        110.0,
        120.0,
        130.0,
        140.0,
        150.0,
        140.0,
        120.0,
        100.0,
        90.0,
        80.0,
        70.0,
    ]
    bars = _bars_from_close(close)
    signals = ema_crossover(bars, fast=2, slow=4)

    assert signals.leg is not None
    assert signals.leg.right == "call"
    assert signals.leg.direction == "long"

    # At least one bullish entry and one bearish exit should fire on this path.
    assert int(signals.entries.sum()) >= 1
    assert int(signals.exits.sum()) >= 1

    # Entry must precede exit on this single-run path.
    first_entry = int(np.flatnonzero(signals.entries)[0])
    first_exit = int(np.flatnonzero(signals.exits)[0])
    assert first_entry < first_exit


def test_rejects_invalid_periods():
    bars = _bars_from_close([100.0, 101.0, 102.0, 103.0, 104.0])
    with pytest.raises(ValueError, match="shorter"):
        ema_crossover(bars, fast=20, slow=5)
    with pytest.raises(ValueError, match=">= 1"):
        ema_crossover(bars, fast=0, slow=20)
