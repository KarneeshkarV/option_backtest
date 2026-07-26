"""Shared fixtures.

Most tests run on a small synthetic frame rather than the 20 MB CSV: they are
testing logic, not data, and a 3-day frame makes failures readable. The tests
that genuinely need the real file are marked ``realdata`` and skip when it is
absent, so the suite passes on a machine without it.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from obt.session import BARS_PER_SESSION, IST


def make_session(day: date, *, start_price: float = 20_000.0, seed: int = 0):
    """One complete 375-bar session of plausible 1-minute bars."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 6.0, BARS_PER_SESSION).cumsum()
    close = start_price + steps
    open_ = np.concatenate(([start_price], close[:-1]))
    spread = np.abs(rng.normal(0.0, 4.0, BARS_PER_SESSION)) + 1.0
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    first = datetime(day.year, day.month, day.day, 9, 15)
    index = pd.date_range(first, periods=BARS_PER_SESSION, freq="1min", tz=IST)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=index
    )


@pytest.fixture
def raw_bars() -> pd.DataFrame:
    """Three complete sessions plus one deliberately incomplete one."""
    frames = []
    day = date(2024, 1, 1)
    for i in range(3):
        frames.append(make_session(day + timedelta(days=i), seed=i))
    # A short session that `complete_days` must drop.
    partial = make_session(day + timedelta(days=3), seed=9).iloc[:120]
    frames.append(partial)
    out = pd.concat(frames)
    out.index.name = "datetime"
    return out


@pytest.fixture
def two_week_bars() -> pd.DataFrame:
    """Ten consecutive complete weekday sessions, 2024-01-01 to 2024-01-12.

    Long enough to contain two real Thursday expiries, so tests can tell an
    expiry that genuinely fell inside the data from one pinned to its edge.
    """
    from obt import session

    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(12)]
    frames = [
        make_session(day, seed=i) for i, day in enumerate(days) if day.weekday() < 5
    ]
    out = pd.concat(frames)
    out.index.name = "datetime"
    return session.clean(out)


@pytest.fixture
def bars(raw_bars: pd.DataFrame) -> pd.DataFrame:
    from obt import session

    return session.clean(raw_bars)
