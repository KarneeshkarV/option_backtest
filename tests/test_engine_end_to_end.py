"""End-to-end ``engine.run`` coverage: the guarantees ``engine.py``'s own
module docstring lists, checked all the way through to
``portfolio.trades.records_readable`` rather than at the unit level.

``tests/test_engine_legs.py`` (Agent C) already runs ``engine.run`` end to
end for per-leg direction, size, disambiguated columns, and a frozen strike
within one fill pair -- read that file first. This file complements it with
the guarantees that file does not assert:

- resolved entry/exit **timestamps** match the shifted signal exactly
  (guarantee 1), not just entry/exit *dates*
- same-day closure holds across an entire multi-trade run, not just one
  example trade
- a position never spans a data gap, proven through the full engine (spot
  bars -> calendar -> strike selection -> pinned premiums -> vectorbt),
  where ``tests/test_session_and_gaps.py`` (Agent B) already proves it at
  the ``resolve_trades``/``shift_signals`` unit level
- the strike/expiry freeze holds across *every* trade in a run, and
  genuinely re-pins (differs) between separate trades rather than
  coincidentally staying constant
- the EOD force-exit (guarantee 2) actually fires when a strategy supplies
  no exit signal at all
- unwarmed (NaN-IV) sessions produce zero trades and are named in
  ``result.warnings`` -- finding 4's engine-level behaviour, currently
  untested anywhere

A trivial, test-only ``VolModel`` is used throughout (a flat IV, with an
explicit per-day NaN override for the warm-up test) rather than the real
``gk_vrp`` model, so these tests pin down ``obt.engine``'s own contract and
do not depend on ``gk_vrp``'s warm-up semantics, which is under independent,
concurrent development elsewhere in this tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from obt import engine
from obt.chain import LegSpec
from obt.session import BARS_PER_SESSION, IST, clean, covered_periods
from obt.strategies.base import Signals
from obt.strategies.spec import strategy


@dataclass(frozen=True)
class _FlatVol:
    """Flat ATM IV everywhere, with an optional set of sessions forced to a
    NaN IV -- the shape ``engine.run`` must treat as "not tradeable yet"
    (finding 4), independent of *why* a real vol model might emit NaN.
    """

    level: float = 0.20
    cold_days: frozenset = field(default_factory=frozenset)

    @property
    def label(self) -> str:
        return f"flat({self.level})"

    def atm_iv(self, bars: pd.DataFrame) -> pd.Series:
        cold = bars["date"].isin(self.cold_days).to_numpy()
        return pd.Series(np.where(cold, np.nan, self.level), index=bars.index)

    def iv(
        self,
        atm: np.ndarray,
        spot: np.ndarray,
        strike: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        atm = np.asarray(atm, dtype="float64")
        return np.where(np.isnan(atm), np.nan, self.level).astype("float64")


def _make_session(
    day: date, *, start_price: float = 20_000.0, seed: int = 0
) -> pd.DataFrame:
    """One complete 375-bar session, independent of conftest's own fixture
    builder so this file stays self-contained."""
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


@pytest.fixture(scope="module")
def e2e_bars() -> pd.DataFrame:
    """Three consecutive complete sessions, no gap. Each day starts at a
    different price level (not just a different random seed) so the ATM
    strike is guaranteed to differ across sessions -- needed to prove
    genuine per-trade re-pinning rather than relying on a random walk
    happening to cross a 50-point boundary by the entry bar.
    """
    day0 = date(2024, 1, 1)
    frames = [
        _make_session(
            day0 + timedelta(days=i), start_price=20_000.0 + i * 500.0, seed=i
        )
        for i in range(3)
    ]
    out = pd.concat(frames)
    out.index.name = "datetime"
    return clean(out)


@pytest.fixture(scope="module")
def gap_bars() -> pd.DataFrame:
    """Two sessions, a ~5-month hole, then two more sessions."""
    early = [date(2024, 1, 1), date(2024, 1, 2)]
    late = [date(2024, 6, 3), date(2024, 6, 4)]
    frames = [_make_session(day, seed=i) for i, day in enumerate(early + late)]
    out = pd.concat(frames)
    out.index.name = "datetime"
    cleaned = clean(out)
    # Confirm the fixture actually contains the gap it claims to.
    assert len(covered_periods(cleaned)) == 2
    return cleaned


_LEG = LegSpec(right="call", direction="long", strike_rule="atm", lots=1)


@strategy(
    "engine_e2e_test_strategy",
    description="Test-only: entry at a fixed bar-of-day, every session (or a "
    "caller-restricted subset of dates), one fixed LegSpec. No exit signal "
    "is ever emitted -- every trade's close is produced solely by the "
    "engine's own end-of-day force-exit (guarantee 2).",
    defaults={"entry_bar": 10, "on_dates": None},
)
def _e2e_strategy(
    bars: pd.DataFrame,
    *,
    entry_bar: int = 10,
    on_dates: tuple[date, ...] | None = None,
) -> Signals:
    entries = bars["bar_of_day"].to_numpy() == entry_bar
    if on_dates is not None:
        entries = entries & bars["date"].isin(on_dates).to_numpy()
    exits = np.zeros(len(bars), dtype=bool)
    return Signals(entries=entries, exits=exits, leg=_LEG)


def test_entry_and_exit_timestamps_match_the_shifted_signal(e2e_bars):
    """Guarantee 1, checked at the level of exact TIMESTAMPS, not just
    dates: a raw signal at bar-of-day ``signal_bar`` on a single session
    must resolve to a fill at bar-of-day ``signal_bar + 1`` -- the exact
    next bar's timestamp, taken straight off ``bars.index`` -- and the exit
    (with no exit signal supplied) must land on that session's last bar.

    Fails if the one-bar shift in ``obt.signals.shift_signals`` is ever
    applied twice, dropped, or off by a bar as it flows through
    ``engine.run``.
    """
    signal_bar = 5
    first_day = sorted({*e2e_bars["date"]})[0]

    result = engine.run(
        e2e_bars,
        "engine_e2e_test_strategy",
        vol=_FlatVol(),
        params={"entry_bar": signal_bar, "on_dates": (first_day,)},
    )

    records = result.portfolio.trades.records_readable
    assert len(records) == 1

    day_bars = e2e_bars.loc[e2e_bars["date"] == first_day]
    expected_entry_ts = day_bars.index[signal_bar + 1]
    expected_exit_ts = day_bars.index[-1]

    assert pd.Timestamp(records["Entry Timestamp"].iloc[0]) == expected_entry_ts
    assert pd.Timestamp(records["Exit Timestamp"].iloc[0]) == expected_exit_ts


def test_same_day_closure_holds_across_every_trade_in_the_run(e2e_bars):
    """Same-day closure asserted as a blanket property over an entire
    multi-session run (one trade per day here), not just spot-checked on a
    single example trade.
    """
    result = engine.run(
        e2e_bars, "engine_e2e_test_strategy", vol=_FlatVol(), params={"entry_bar": 10}
    )
    records = result.portfolio.trades.records_readable
    assert len(records) == e2e_bars["date"].nunique()

    for _, row in records.iterrows():
        entry_date = pd.Timestamp(row["Entry Timestamp"]).date()
        exit_date = pd.Timestamp(row["Exit Timestamp"]).date()
        assert entry_date == exit_date


def test_position_never_spans_a_data_gap(gap_bars):
    """The engine's guarantee 2, proven through the FULL pipeline -- spot
    bars, calendar, strike selection, pinned premiums, vectorbt -- across a
    real multi-month hole, complementing the unit-level proof in
    ``tests/test_session_and_gaps.py`` (which exercises ``resolve_trades``
    and ``shift_signals`` directly, not ``engine.run``).

    Every trade must close the same day it opened, and every trade's dates
    must sit inside a single contiguous covered block -- never straddling
    the hole.
    """
    result = engine.run(
        gap_bars, "engine_e2e_test_strategy", vol=_FlatVol(), params={"entry_bar": 10}
    )
    records = result.portfolio.trades.records_readable
    assert len(records) == gap_bars["date"].nunique() == 4

    blocks = covered_periods(gap_bars)
    assert len(blocks) == 2

    for _, row in records.iterrows():
        entry_date = pd.Timestamp(row["Entry Timestamp"]).date()
        exit_date = pd.Timestamp(row["Exit Timestamp"]).date()
        assert entry_date == exit_date  # never spans anything, gap included

        containing = [
            (start, end) for start, end in blocks if start <= entry_date <= end
        ]
        assert len(containing) == 1, (
            f"trade on {entry_date} does not sit inside exactly one covered "
            f"block {blocks}"
        )


def test_force_exit_fires_on_the_sessions_final_bar(e2e_bars):
    """Guarantee 2 directly: with no exit signal ever supplied, every
    trade's exit timestamp must be that session's actual final bar.
    """
    result = engine.run(
        e2e_bars, "engine_e2e_test_strategy", vol=_FlatVol(), params={"entry_bar": 10}
    )
    records = result.portfolio.trades.records_readable
    assert len(records) > 0

    for _, row in records.iterrows():
        entry_date = pd.Timestamp(row["Entry Timestamp"]).date()
        expected_last_bar = e2e_bars.loc[e2e_bars["date"] == entry_date].index[-1]
        assert pd.Timestamp(row["Exit Timestamp"]) == expected_last_bar


def test_strike_and_expiry_frozen_within_every_trade_and_re_pinned_across_them(
    e2e_bars,
):
    """Generalizes Agent C's single-trade frozen-strike check
    (``test_strike_and_expiry_frozen_between_a_fill_pair`` in
    ``test_engine_legs.py``) to EVERY trade in a multi-trade run, and adds
    the complementary half that test does not cover: that the strike
    genuinely differs across separate trades when spot has moved, proving
    real per-trade re-pinning rather than a single frozen constant that
    happens to look pinned within one example.
    """
    result = engine.run(
        e2e_bars, "engine_e2e_test_strategy", vol=_FlatVol(), params={"entry_bar": 10}
    )
    records = result.portfolio.trades.records_readable
    assert len(records) >= 2

    frame = result.leg_frame[_LEG.label]
    strikes_per_trade = []
    for _, row in records.iterrows():
        entry_ts = pd.Timestamp(row["Entry Timestamp"])
        exit_ts = pd.Timestamp(row["Exit Timestamp"])
        window = frame.loc[entry_ts:exit_ts]
        assert len(window) > 1
        assert window["strike"].nunique() == 1
        assert window["expiry"].nunique() == 1
        strikes_per_trade.append(window["strike"].iloc[0])

    # Different sessions' random-walked spot paths must not all land on the
    # same ATM strike -- otherwise "frozen within a trade" could trivially
    # be satisfied by one strike frozen for the whole run.
    assert len(set(strikes_per_trade)) > 1


def test_unwarmed_sessions_produce_no_trades_and_are_reported(e2e_bars):
    """Finding 4's engine-level behaviour: a session with no vol-model IV
    yet must contribute zero trades, and ``result.warnings`` must say so --
    currently untested anywhere in the suite.
    """
    cold_day = sorted({*e2e_bars["date"]})[0]
    vol = _FlatVol(cold_days=frozenset({cold_day}))

    result = engine.run(
        e2e_bars, "engine_e2e_test_strategy", vol=vol, params={"entry_bar": 10}
    )
    records = result.portfolio.trades.records_readable

    entry_dates = {pd.Timestamp(ts).date() for ts in records["Entry Timestamp"]}
    assert cold_day not in entry_dates
    assert len(records) == e2e_bars["date"].nunique() - 1

    total_sessions = e2e_bars["date"].nunique()
    expected = f"1 of {total_sessions} sessions carry no vol-model IV yet"
    assert any(expected in w for w in result.warnings), result.warnings
