"""NIFTY weekly expiry calendar and trading-time to expiry.

Two things here quietly corrupt everything downstream if they are wrong, so
both are explicit and both are tested:

**The expiry weekday changed.** NIFTY weekly options expired on Thursday for
most of this sample and moved to Tuesday. It is encoded as a dated rule table
rather than a hardcoded weekday, so the switch is one line to correct and
visible in review.

The transition was announced in NSE/FAOP/68685 (2025-06-23), but that circular
was superseded two days later by NSE/FAOP/68747 (2025-06-25) -- a "partial
modification" using a different mechanism. 68685 alone would have this switch
elongate the already-listed Thursday contracts forward to the following week's
Tuesday; 68747 instead left existing Thursday contracts (through 2025-08-28)
unchanged and had already pre-listed the new Tuesday contracts (02-Sep,
09-Sep, ...) on the ordinary rolling schedule, so the real transition is a
clean cutover, not an elongation. The effective boundary is 2025-08-29 (the
first trading day the new Tuesday contracts govern), not 2025-09-01 -- verify
*both* circular numbers, not just the first, before trusting results that
straddle this date; a wrong switch date silently misprices options rather
than raising anything.

**Holidays come from the data, not a hardcoded list.** The set of trading days
is whatever the loaded bars contain. If a nominal expiry has no session, expiry
rolls back to the previous trading day, which is the NSE rule. This also means
the calendar is automatically correct for the multi-month holes in this
dataset instead of inventing expiries inside them.

Time to expiry is measured in **trading time** (bars remaining / 375 / 252),
not calendar time. Calendar time badly overstates overnight theta for weekly
options -- a Friday-to-Monday hold is one trading day of decay, not three.

**Pricing and fills happen at each bar's close** (see ``chain.pinned_leg`` and
the premium frame in ``engine.py``), so bar ``i``'s minute has already elapsed
by the time it is priced. Time remaining today counts only bars strictly
after the current one, which makes tau exactly zero on the expiry session's
final bar rather than one bar short of it.
"""

from __future__ import annotations

import bisect
from datetime import date, timedelta

import numpy as np
import pandas as pd

from obt.session import BARS_PER_SESSION, TRADING_DAYS_PER_YEAR, stamps

#: ``(effective_from, weekday)`` with Monday=0. Most recent applicable wins.
#: Thursday=3 historically; Tuesday=1 from 2025-08-29 -- the first trade date
#: governed by the new-regime contracts per NSE/FAOP/68747 (2025-06-25),
#: which superseded NSE/FAOP/68685 (2025-06-23). NOT 2025-09-01: that date is
#: three calendar days (one trading day) too late, and NOT 2025-08-28, which
#: would wrongly flip 2025-08-28's own resolution (it is the last date the
#: old Thursday regime governs, per the circular).
WEEKLY_EXPIRY_RULES: tuple[tuple[date, int], ...] = (
    (date(1900, 1, 1), 3),
    (date(2025, 8, 29), 1),
)


def expiry_weekday(as_of: date) -> int:
    """Nominal weekly-expiry weekday in force on ``as_of`` (Monday=0)."""
    weekday = WEEKLY_EXPIRY_RULES[0][1]
    for effective_from, day in WEEKLY_EXPIRY_RULES:
        if as_of >= effective_from:
            weekday = day
    return weekday


class ExpiryCalendar:
    """Weekly expiries derived from the trading days actually present."""

    def __init__(self, trading_days: list[date]) -> None:
        if not trading_days:
            raise ValueError("expiry calendar needs at least one trading day")
        self._days = sorted(set(trading_days))
        self._day_set = set(self._days)
        self._index = {day: i for i, day in enumerate(self._days)}

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> ExpiryCalendar:
        return cls(sorted({*pd.Index(stamps(bars).date)}))

    @property
    def trading_days(self) -> list[date]:
        return list(self._days)

    def previous_trading_day(self, day: date) -> date | None:
        """The latest trading day on or before ``day``."""
        position = bisect.bisect_right(self._days, day) - 1
        return self._days[position] if position >= 0 else None

    def weekly_expiry_for(self, as_of: date) -> date | None:
        """Expiry of the front weekly contract for a trade opened on ``as_of``.

        Walks forward to the nominal expiry weekday, then rolls **back** to the
        previous trading day when that date is a holiday. Returns ``None`` when
        the resolved expiry would fall outside the data -- callers must treat
        that as "not tradeable", never as zero time to expiry.
        """
        weekday = expiry_weekday(as_of)
        ahead = (weekday - as_of.weekday()) % 7
        nominal = as_of + timedelta(days=ahead)

        resolved = nominal if nominal in self._day_set else None
        if resolved is None:
            candidate = self.previous_trading_day(nominal)
            # Only accept the rollback if it is still on or after `as_of`;
            # otherwise this week's contract already expired -- go to next week.
            resolved = (
                candidate if candidate is not None and candidate >= as_of else None
            )
        if resolved is None:
            return self._next_week_expiry(as_of, nominal)
        return resolved

    def _next_week_expiry(self, as_of: date, nominal: date) -> date | None:
        nominal += timedelta(days=7)
        if nominal in self._day_set:
            return nominal
        candidate = self.previous_trading_day(nominal)
        if candidate is not None and candidate > as_of:
            return candidate
        return None

    def sessions_between(self, start: date, end: date) -> int | None:
        """Count of trading sessions from ``start`` to ``end`` inclusive."""
        if start not in self._index or end not in self._index:
            return None
        return self._index[end] - self._index[start] + 1


def tau_years(
    bars: pd.DataFrame,
    expiry: pd.Series,
    calendar: ExpiryCalendar,
    *,
    calendar_time: bool = False,
) -> np.ndarray:
    """Trading-time years to expiry for every bar.

    Pricing and fills use each bar's CLOSE (``chain.pinned_leg``, the premium
    frame vectorbt trades in ``engine.py``), so by the time bar ``i`` is priced
    that bar's minute is already gone. Time remaining today therefore counts
    only bars strictly after the current one -- on the expiry session's final
    bar there are none left, so tau is exactly zero and Black-76 (see
    ``pricing.black76.price``, which returns intrinsic value for
    ``tau <= _MIN_TAU``) prices it as pure intrinsic rather than decaying into
    it a bar late.

    Set ``calendar_time=True`` to use wall-clock days/365 instead -- available
    for comparison, but it overstates decay for anything held overnight.
    """
    dates = bars["date"].to_numpy()
    bar_of_day = bars["bar_of_day"].to_numpy()
    expiry_dates = expiry.to_numpy()

    if calendar_time:
        days = np.array(
            [
                (e - d).days if e is not None else np.nan
                for d, e in zip(dates, expiry_dates, strict=True)
            ],
            dtype="float64",
        )
        remaining_today = (BARS_PER_SESSION - 1 - bar_of_day) / BARS_PER_SESSION
        return np.maximum((days + remaining_today) / 365.0, 0.0)

    # Trading time: whole sessions strictly after today, plus today's remainder.
    sessions = np.array(
        [
            calendar.sessions_between(d, e) if e is not None else np.nan
            for d, e in zip(dates, expiry_dates, strict=True)
        ],
        dtype="float64",
    )
    full_sessions_ahead = sessions - 1.0
    remaining_today = (BARS_PER_SESSION - 1 - bar_of_day) / BARS_PER_SESSION
    total_sessions = full_sessions_ahead + remaining_today
    return np.maximum(total_sessions / TRADING_DAYS_PER_YEAR, 0.0)


def expiry_series(bars: pd.DataFrame, calendar: ExpiryCalendar) -> pd.Series:
    """Front weekly expiry for each bar, resolved once per date."""
    unique_dates = sorted({*bars["date"]})
    mapping = {day: calendar.weekly_expiry_for(day) for day in unique_dates}
    return bars["date"].map(mapping)


def block_edge_sessions(bars: pd.DataFrame, calendar: ExpiryCalendar) -> list[date]:
    """Sessions whose resolved expiry is an artifact of where the data stops.

    Deriving holidays from data presence is right for real holidays and wrong at
    the edge of a covered block: the calendar cannot distinguish "the exchange
    was shut" from "our file ends here", so it rolls the expiry back onto the
    block's last session. Those bars get a shorter time to expiry than the real
    contract had, and therefore cheaper options -- silently, with nothing
    raising.

    Small in this dataset (17 of 877 sessions) but exactly the kind of quiet
    corruption the calendar exists to prevent, so the engine surfaces it rather
    than leaving it to be found later.
    """
    from obt.session import covered_periods

    block_ends = {end for _, end in covered_periods(bars)}
    return [
        day
        for day in sorted({*bars["date"]})
        if (resolved := calendar.weekly_expiry_for(day)) in block_ends
        and resolved.weekday() != expiry_weekday(day)
    ]
