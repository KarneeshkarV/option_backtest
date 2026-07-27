"""``gk_vrp``: EWMA-in-variance-space, warmup, and clamp validation.

Three defects, three groups of tests:

- The EWMA must smooth *variance*, then take one sqrt at the end -- never
  smooth an already-annualized vol series (sqrt is concave, so that
  understates the level and damps regime changes).
- Warmup must actually honour ``seed_days``, must never let a session's IV
  depend on that same session's own high/low/close, and must reset at every
  gap reported by :func:`obt.session.covered_periods` rather than smuggling a
  pre-gap level into the first post-gap bar.
- ``iv_floor`` must not be silently accepted above ``iv_cap``.

The GK formula and the within-day aggregation are not under test here --
they're covered ground; see the module docstring in
``obt.vol.plugins.gk_vrp``.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from conftest import make_session
from pydantic import ValidationError

from obt import session
from obt.session import IST, TRADING_DAYS_PER_YEAR
from obt.vol.plugins.gk_vrp import _GK_C, GkVrpVolModel

# Wide enough that the iv_floor/iv_cap clamp never binds on the small
# synthetic values these tests construct -- the clamp itself is tested
# separately, below.
_NO_CLAMP = {"iv_floor": 1e-9, "iv_cap": 10.0}


def _one_bar_frame(rows: list[tuple[date, float, float, float, float]]) -> pd.DataFrame:
    """One session per row, reduced to a single 09:15 bar carrying its O/H/L/C.

    `garman_klass_daily` only ever looks at each session's first open, max
    high, min low and last close, so a single bar per day is a faithful (and
    hand-calculable) stand-in for a full 375-bar session.
    """
    index = [
        pd.Timestamp(day.year, day.month, day.day, 9, 15, tz=IST) for day, *_ in rows
    ]
    frame = pd.DataFrame(
        [{"open": o, "high": h, "low": low, "close": c} for _, o, h, low, c in rows],
        index=pd.DatetimeIndex(index),
    )
    return session.add_session_cols(frame)


def _gk_variance(open_: float, high: float, low: float, close: float) -> float:
    """Independent hand-calculation of one day's (non-annualized) GK variance."""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    return max(0.0, 0.5 * log_hl**2 - _GK_C * log_co**2)


def test_ewma_smooths_variance_not_vol():
    """Core Finding-1 regression: sqrt(EWMA(variance)), not EWMA(sqrt(variance)).

    Fails against the pre-fix code, which computed
    ``garman_klass_daily(...).ewm(...).mean()`` -- i.e. averaged the
    already-annualized vol series.
    """
    rows = [
        (date(2024, 1, 1), 100.0, 108.0, 96.0, 101.0),
        (date(2024, 1, 2), 100.0, 103.0, 99.0, 100.5),
        (date(2024, 1, 3), 100.0, 115.0, 90.0, 105.0),
        (date(2024, 1, 4), 100.0, 101.0, 99.5, 100.2),
        (date(2024, 1, 5), 100.0, 110.0, 95.0, 103.0),
    ]
    bars = _one_bar_frame(rows)
    halflife = 2.5

    model = GkVrpVolModel(
        halflife_days=halflife, seed_days=1, vrp_mult=1.0, **_NO_CLAMP
    )
    atm = model.atm_iv(bars)

    variance = pd.Series(
        [_gk_variance(o, h, low, c) for _, o, h, low, c in rows],
        index=[r[0] for r in rows],
    )

    # Correct: smooth in variance space, shift one day, sqrt once at the end.
    expected_variance = variance.ewm(halflife=halflife, min_periods=1).mean().shift(1)
    expected_atm = np.sqrt(expected_variance * TRADING_DAYS_PER_YEAR)

    daily_atm = atm.groupby(bars["date"]).first()
    # Day one has no D-1 inside the series -- unwarmed, must be NaN.
    assert pd.isna(daily_atm.iloc[0])
    np.testing.assert_allclose(
        daily_atm.iloc[1:].to_numpy(), expected_atm.iloc[1:].to_numpy(), rtol=1e-10
    )

    # Wrong (the old bug): smooth already-annualized vol, not variance.
    vol = np.sqrt(variance * TRADING_DAYS_PER_YEAR)
    old_wrong_atm = vol.ewm(halflife=halflife, min_periods=1).mean().shift(1).bfill()
    assert not np.allclose(
        daily_atm.iloc[1:].to_numpy(), old_wrong_atm.iloc[1:].to_numpy()
    )


def test_seed_days_is_honoured():
    """seed_days=1 vs seed_days=10 must produce different output.

    Fails against the pre-fix code: `seed_days` was declared on `VolParams`
    but never read by `atm_iv`, so any two values produced byte-identical
    series.
    """
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(15)]
    frames = [make_session(day, seed=i) for i, day in enumerate(days)]
    bars = session.add_session_cols(pd.concat(frames))

    warm_fast = GkVrpVolModel(seed_days=1, **_NO_CLAMP).atm_iv(bars)
    warm_slow = GkVrpVolModel(seed_days=10, **_NO_CLAMP).atm_iv(bars)

    assert not warm_fast.equals(warm_slow)

    per_day_fast = warm_fast.groupby(bars["date"]).first()
    per_day_slow = warm_slow.groupby(bars["date"]).first()

    # seed_days=1: only the very first session (no D-1 at all) is unwarmed.
    assert per_day_fast.isna().sum() == 1
    # seed_days=10: the first 10 sessions of the block are unwarmed.
    assert per_day_slow.isna().sum() == 10
    assert per_day_slow.iloc[10:].notna().all()


def test_first_session_iv_does_not_depend_on_its_own_range():
    """No lookahead: day one's own high/low/close must not leak into day one's IV.

    Fails against the pre-fix code, where `smoothed.shift(1).bfill()` seeded
    the first day's IV from that same day's own full-session GK estimate --
    two frames differing only in day one's range produced different day-one
    IVs there.
    """
    tame_day1 = (date(2024, 1, 1), 100.0, 101.0, 99.0, 100.2)
    wild_day1 = (date(2024, 1, 1), 100.0, 150.0, 60.0, 120.0)
    common_tail = [
        (date(2024, 1, 2), 100.0, 103.0, 99.0, 101.0),
        (date(2024, 1, 3), 100.0, 104.0, 97.0, 102.0),
    ]

    # Sanity: the two day-one candles really do imply very different GK vol,
    # so if it leaked through, the two variants would visibly disagree.
    tame_var = _gk_variance(*tame_day1[1:])
    wild_var = _gk_variance(*wild_day1[1:])
    assert not np.isclose(tame_var, wild_var)

    model = GkVrpVolModel(seed_days=1, halflife_days=5.0, vrp_mult=1.0, **_NO_CLAMP)
    frame_tame = _one_bar_frame([tame_day1, *common_tail])
    frame_wild = _one_bar_frame([wild_day1, *common_tail])

    atm_tame = model.atm_iv(frame_tame).groupby(frame_tame["date"]).first()
    atm_wild = model.atm_iv(frame_wild).groupby(frame_wild["date"]).first()

    # Both unwarmed on day one -- identical (NaN) regardless of the range
    # that day actually had.
    assert pd.isna(atm_tame.iloc[0])
    assert pd.isna(atm_wild.iloc[0])


def test_ewma_resets_across_a_covered_period_gap():
    """A >7-day hole must not hand its pre-gap smoothed level to the first
    post-gap bar.

    Fails against the pre-fix code, which ran one continuous EWMA over the
    whole series: the first bar after a gap inherited whatever the EWMA had
    smoothed to just before the hole, however hot or cold that was.
    """
    rows = [
        (date(2024, 1, 1), 100.0, 101.0, 99.0, 100.3),
        (date(2024, 1, 2), 100.0, 101.5, 98.7, 100.5),
        # A deliberately hot day right before the gap.
        (date(2024, 1, 3), 100.0, 160.0, 50.0, 140.0),
        # >7 calendar days later -- a new covered block.
        (date(2024, 3, 1), 100.0, 101.0, 99.0, 100.4),
        (date(2024, 3, 4), 100.0, 102.0, 98.0, 101.0),
    ]
    frame = _one_bar_frame(rows)
    assert len(session.covered_periods(frame)) == 2  # sanity: the gap registers

    model = GkVrpVolModel(seed_days=1, halflife_days=5.0, vrp_mult=1.0, **_NO_CLAMP)
    atm = model.atm_iv(frame).groupby(frame["date"]).first()

    # Block two's first session has no D-1 *within its own block* -- unwarmed,
    # not a carry-over from the hot day before the gap.
    assert pd.isna(atm.iloc[3])

    # Block two's second session must reflect only block two's first day --
    # never the hot day3 that preceded the gap.
    block2_day1_variance = _gk_variance(*rows[3][1:])
    expected = np.sqrt(block2_day1_variance * TRADING_DAYS_PER_YEAR)
    assert atm.iloc[4] == pytest.approx(expected, rel=1e-10)


def test_iv_clamp_equal_bounds_accepted():
    model = GkVrpVolModel(iv_floor=0.10, iv_cap=0.10)
    assert model.params.iv_floor == model.params.iv_cap == 0.10


def test_iv_clamp_reversed_bounds_rejected():
    with pytest.raises(ValidationError):
        GkVrpVolModel(iv_floor=0.80, iv_cap=0.06)
