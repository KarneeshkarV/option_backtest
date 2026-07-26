"""Session cleaning and the gap guarantee.

The dataset has a nine-month hole in 2022 and several shorter ones. The whole
design rests on one claim: no position can span a gap. These tests check the
mechanism that makes it true, rather than trusting that strategies remember.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from obt import session
from obt.session import BARS_PER_SESSION, IST
from obt.signals import last_bar_of_day, resolve_trades, shift_signals


def test_complete_days_drops_partial_sessions(raw_bars):
    cleaned = session.complete_days(session.filter_regular_session(raw_bars))
    counts = cleaned.groupby(cleaned.index.date).size()
    assert (counts == BARS_PER_SESSION).all()
    assert len({*cleaned.index.date}) == 3  # the 120-bar day is gone


def test_filter_regular_session_drops_after_hours(raw_bars):
    """Muhurat-style evening bars sit outside the assumed session."""
    evening = raw_bars.iloc[:5].copy()
    evening.index = pd.date_range(
        datetime(2024, 1, 1, 18, 10), periods=5, freq="1min", tz=IST
    )
    with_evening = pd.concat([raw_bars, evening]).sort_index()

    kept = session.filter_regular_session(with_evening)
    assert len(kept) == len(raw_bars)
    assert kept.index.time.max() <= session.SESSION_END


def test_bar_of_day_is_zero_based_and_complete(bars):
    for _, day in bars.groupby("date"):
        assert day["bar_of_day"].iloc[0] == 0
        assert day["bar_of_day"].iloc[-1] == BARS_PER_SESSION - 1


def test_covered_periods_splits_on_a_real_gap():
    """A multi-month hole must surface as separate blocks, not one range."""
    frames = []
    for day in (date(2024, 1, 1), date(2024, 1, 2), date(2024, 6, 3)):
        index = pd.date_range(
            datetime(day.year, day.month, day.day, 9, 15),
            periods=BARS_PER_SESSION,
            freq="1min",
            tz=IST,
        )
        frames.append(pd.DataFrame({"close": 1.0}, index=index))
    periods = session.covered_periods(pd.concat(frames))

    assert len(periods) == 2
    assert periods[0] == (date(2024, 1, 1), date(2024, 1, 2))
    assert periods[1] == (date(2024, 6, 3), date(2024, 6, 3))


def test_shift_signals_delays_by_one_bar():
    mask = np.array([False, True, False, False])
    np.testing.assert_array_equal(
        shift_signals(mask), np.array([False, False, True, False])
    )


def test_last_bar_of_day_marks_each_session_end(bars):
    is_last = last_bar_of_day(bars)
    assert is_last.sum() == len({*bars["date"]})
    assert bars.loc[is_last, "bar_of_day"].eq(BARS_PER_SESSION - 1).all()


def test_positions_never_span_a_day(bars):
    """The core gap guarantee, stated directly."""
    entries = (bars["bar_of_day"] == 100).to_numpy()
    exits = np.zeros(len(bars), dtype=bool)  # strategy never exits
    resolved = resolve_trades(entries, exits, last_bar_of_day(bars))

    dates = bars["date"].to_numpy()
    opens = np.flatnonzero(resolved["open_mask"])
    closes = np.flatnonzero(resolved["close_mask"])
    assert len(opens) == len(closes) > 0
    for open_i, close_i in zip(opens, closes, strict=True):
        assert dates[open_i] == dates[close_i]


def test_forced_exit_happens_even_without_an_exit_signal(bars):
    entries = (bars["bar_of_day"] == 10).to_numpy()
    resolved = resolve_trades(
        entries, np.zeros(len(bars), dtype=bool), last_bar_of_day(bars)
    )
    assert resolved["close_mask"].sum() == resolved["open_mask"].sum()


def test_repeated_entries_do_not_reopen_a_live_position(bars):
    """Every bar signals entry; only one position may exist per day."""
    entries = np.ones(len(bars), dtype=bool)
    resolved = resolve_trades(
        entries, np.zeros(len(bars), dtype=bool), last_bar_of_day(bars)
    )
    assert resolved["open_mask"].sum() == len({*bars["date"]})


def test_entry_on_final_bar_is_skipped(bars):
    """Opening with no bars left to hold would just book a round trip of costs."""
    entries = last_bar_of_day(bars)
    resolved = resolve_trades(
        entries, np.zeros(len(bars), dtype=bool), last_bar_of_day(bars)
    )
    assert resolved["open_mask"].sum() == 0


def test_trade_ids_are_contiguous_and_flat_is_negative(bars):
    entries = (bars["bar_of_day"] == 50).to_numpy()
    resolved = resolve_trades(
        entries, np.zeros(len(bars), dtype=bool), last_bar_of_day(bars)
    )
    ids = resolved["trade_id"]
    active = ids[ids >= 0]
    assert active.min() == 0
    assert np.array_equal(np.unique(active), np.arange(len({*bars["date"]})))


@pytest.mark.parametrize("gap_days", [1, 270])
def test_no_trade_bridges_a_data_gap(gap_days):
    """Explicitly the 2022-shaped failure: a hole in the middle of the sample."""
    frames = []
    for offset in (0, gap_days):
        day = date(2024, 1, 1) + timedelta(days=offset)
        index = pd.date_range(
            datetime(day.year, day.month, day.day, 9, 15),
            periods=BARS_PER_SESSION,
            freq="1min",
            tz=IST,
        )
        frames.append(
            pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                },
                index=index,
            )
        )
    bars = session.add_session_cols(pd.concat(frames))

    entries = (bars["bar_of_day"] == 5).to_numpy()
    resolved = resolve_trades(
        entries, np.zeros(len(bars), dtype=bool), last_bar_of_day(bars)
    )
    dates = bars["date"].to_numpy()
    opens = np.flatnonzero(resolved["open_mask"])
    closes = np.flatnonzero(resolved["close_mask"])
    for open_i, close_i in zip(opens, closes, strict=True):
        assert dates[open_i] == dates[close_i]
