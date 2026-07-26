"""Expiry calendar, including the weekday switch and holiday rollback.

A wrong expiry silently misprices every option -- nothing raises, the numbers
just quietly become fiction. These tests are the only thing standing between
that and a plausible-looking equity curve.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from obt.calendar import ExpiryCalendar, expiry_weekday, tau_years
from obt.session import BARS_PER_SESSION


def weekdays_between(start: date, end: date) -> list[date]:
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


@pytest.fixture
def calendar() -> ExpiryCalendar:
    return ExpiryCalendar(weekdays_between(date(2024, 1, 1), date(2026, 6, 30)))


def test_expiry_weekday_switches_to_tuesday():
    assert expiry_weekday(date(2024, 6, 1)) == 3  # Thursday
    assert expiry_weekday(date(2025, 8, 31)) == 3
    assert expiry_weekday(date(2025, 9, 1)) == 1  # Tuesday
    assert expiry_weekday(date(2026, 1, 5)) == 1


def test_weekly_expiry_before_switch_is_thursday(calendar):
    expiry = calendar.weekly_expiry_for(date(2024, 1, 2))
    assert expiry == date(2024, 1, 4)
    assert expiry.weekday() == 3


def test_weekly_expiry_after_switch_is_tuesday(calendar):
    expiry = calendar.weekly_expiry_for(date(2025, 9, 3))
    assert expiry == date(2025, 9, 9)
    assert expiry.weekday() == 1


def test_expiry_day_resolves_to_itself(calendar):
    """A trade opened on expiry day expires that day, not next week."""
    assert calendar.weekly_expiry_for(date(2024, 1, 4)) == date(2024, 1, 4)


def test_day_after_expiry_rolls_to_next_week(calendar):
    assert calendar.weekly_expiry_for(date(2024, 1, 5)) == date(2024, 1, 11)


def test_holiday_expiry_rolls_back_to_previous_session():
    """When the nominal expiry is a holiday, expiry moves earlier, not later."""
    days = [d for d in weekdays_between(date(2024, 1, 1), date(2024, 3, 31))]
    holiday = date(2024, 1, 25)  # a Thursday
    days.remove(holiday)
    calendar = ExpiryCalendar(days)

    expiry = calendar.weekly_expiry_for(date(2024, 1, 22))
    assert expiry == date(2024, 1, 24)
    assert expiry < holiday


def test_sessions_between_counts_inclusively(calendar):
    assert calendar.sessions_between(date(2024, 1, 1), date(2024, 1, 1)) == 1
    assert calendar.sessions_between(date(2024, 1, 1), date(2024, 1, 5)) == 5


def test_tau_shrinks_within_the_day(bars):
    calendar = ExpiryCalendar.from_bars(bars)
    from obt.calendar import expiry_series

    expiry = expiry_series(bars, calendar)
    tau = tau_years(bars, expiry, calendar)

    first_day = bars["date"] == bars["date"].iloc[0]
    day_tau = tau[first_day.to_numpy()]
    assert (np.diff(day_tau) < 0).all(), "time to expiry must decrease each bar"
    assert (day_tau > 0).all()


def test_tau_uses_trading_time_not_calendar_time(bars):
    """One session of decay is 1/252 of a year, whatever the wall clock says."""
    calendar = ExpiryCalendar.from_bars(bars)
    from obt.calendar import expiry_series

    expiry = expiry_series(bars, calendar)
    tau = tau_years(bars, expiry, calendar)

    opens = bars["bar_of_day"] == 0
    tau_at_opens = tau[opens.to_numpy()]
    # Consecutive session opens differ by exactly one trading day of tau,
    # except where the expiry contract rolls.
    steps = np.diff(tau_at_opens)
    one_session = 1.0 / 252.0
    assert np.any(np.isclose(np.abs(steps), one_session, atol=1e-9))


def test_tau_at_final_bar_is_one_bar_not_zero(bars):
    calendar = ExpiryCalendar.from_bars(bars)
    from obt.calendar import expiry_series

    expiry = expiry_series(bars, calendar)
    tau = tau_years(bars, expiry, calendar)
    # Last bar of an expiry day: one bar of trading time remains.
    smallest = tau.min()
    assert 0 < smallest <= (1.0 / BARS_PER_SESSION) / 252.0 + 1e-12


def test_block_edge_sessions_flags_the_data_boundary(bars):
    """The calendar cannot tell "exchange shut" from "our file ends here".

    The three-session fixture stops mid-week, so its final week's expiry rolls
    back onto the last bar of data. That must be reported, not silently priced.
    """
    from obt.calendar import block_edge_sessions

    calendar = ExpiryCalendar.from_bars(bars)
    flagged = block_edge_sessions(bars, calendar)
    last_day = max({*bars["date"]})
    assert flagged, "a truncated sample must flag its trailing edge"
    assert all(calendar.weekly_expiry_for(day) == last_day for day in flagged)


def test_block_edge_flag_spares_expiries_that_fall_inside_the_data(two_week_bars):
    """Only the trailing edge is an artifact; real expiries must not be flagged.

    2024-01-04 and 2024-01-11 are genuine Thursday expiries sitting inside the
    sample, so every session up to the 11th resolves cleanly. Only 2024-01-12,
    which looks forward past the end of the file, is pinned to the boundary.
    """
    from obt.calendar import block_edge_sessions

    calendar = ExpiryCalendar.from_bars(two_week_bars)
    assert block_edge_sessions(two_week_bars, calendar) == [date(2024, 1, 12)]


def test_empty_calendar_rejected():
    with pytest.raises(ValueError, match="at least one trading day"):
        ExpiryCalendar([])
