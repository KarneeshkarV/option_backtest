"""The synthetic pricer, held against real observed quotes.

This is the only test in the suite that can fail because the *model* is wrong
rather than because the code is. Everything else checks internal consistency;
this checks whether the premiums resemble prices somebody actually paid.

Skipped when the observed-chain files are absent, so the suite still passes on
a machine that only has the five-year spot CSV.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from obt import calibration, session
from obt.calendar import ExpiryCalendar, tau_years
from obt.chain import atm_strike
from obt.datasource import get_source
from obt.pricing import black76
from obt.session import IST, add_session_cols
from obt.vol import get_vol_model
from obt.vol.plugins.gk_vrp import GkVrpVolModel

pytestmark = pytest.mark.skipif(
    not calibration.chain_available(),
    reason="observed option chain CSVs not present",
)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """Spot over the observed window, warmed up on the five-year history.

    The vol model needs weeks of history before its EWMA means anything, so the
    2026 vendor file alone would be tested cold. Splicing the two feeds is safe
    here precisely because they agree: 96.9% of overlapping bars are identical.
    """
    history = session.clean(get_source("nifty_csv").load("NIFTY"))
    recent = session.clean(get_source("nifty_index_csv").load("NIFTY"))
    columns = ["open", "high", "low", "close"]
    joined = pd.concat(
        [history[columns], recent.loc[recent.index > history.index.max(), columns]]
    )
    return session.add_session_cols(joined)


@pytest.fixture(scope="module")
def comparison(bars):
    return calibration.compare_to_observed(bars, get_vol_model("gk_vrp"))


def test_the_two_spot_feeds_agree_on_their_overlap():
    """The data-source seam only means something if the sources match."""
    history = session.clean(get_source("nifty_csv").load("NIFTY"))["close"]
    recent = session.clean(get_source("nifty_index_csv").load("NIFTY"))["close"]
    shared = history.index.intersection(recent.index)
    assert len(shared) > 10_000, "expected a substantial overlap to compare"

    difference = (history.loc[shared] - recent.loc[shared]).abs()
    assert (difference == 0).mean() > 0.95
    assert difference.quantile(0.99) <= 1.0
    assert difference.max() < 25.0


def test_observed_chain_is_shaped_as_expected():
    """Pins the exact multiplicity the old ``... or True`` tautology waved
    through unchecked: CE and PE legitimately share every timestamp, so each
    timestamp must appear *exactly* twice -- one call row, one put row, never
    a lone leg and never a triplicate. Fails if a vendor file update ever
    starts dropping one side's quote for a minute, or duplicating a row.
    """
    chain = calibration.load_observed_chain()
    assert set(chain["right"]) == {"call", "put"}

    counts = chain.index.value_counts()
    bad = counts[counts != 2]
    assert bad.empty, (
        "expected exactly 2 rows (one call, one put) per timestamp; "
        f"found {bad.to_dict()}"
    )

    # Not just *two* rows -- one of each right, never two calls or two puts.
    rights_per_timestamp = chain.groupby(chain.index)["right"].apply(frozenset)
    assert (rights_per_timestamp == frozenset({"call", "put"})).all()

    # One strike and one expiry per session -- the constraint that makes skew
    # unidentifiable. If a richer chain ever lands, this test should fail.
    per_day = chain.groupby([chain.index.date, chain["right"]]).agg(
        strikes=("strike", "nunique"), expiries=("expiry", "nunique")
    )
    assert (per_day["strikes"] == 1).all()
    assert (per_day["expiries"] == 1).all()


def test_observed_expiries_are_tuesdays():
    """Independent confirmation of the post-2025-09 weekday rule."""
    chain = calibration.load_observed_chain()
    assert {pd.Timestamp(day).weekday() for day in chain["expiry"].unique()} == {1}


def test_model_tracks_observed_premiums(comparison):
    """Correlation is the shape check: does the premium path move right?"""
    assert len(comparison) > 30_000
    for right, group in comparison.groupby("right"):
        assert group["observed"].corr(group["model"]) > 0.95, right


def test_model_premium_level_is_within_a_stated_band(comparison):
    """The level check, and the reason ``vrp_mult`` was refit to 1.24.

    Calls are the clean test: with one strike per week the put error is
    contaminated by unvalidated skew, so it gets a looser band and a comment
    rather than a pretense of precision.
    """
    by_right = comparison.groupby("right")["rel_error"].median()
    assert abs(by_right["call"]) < 0.10, f"call level off by {by_right['call']:.1%}"
    assert abs(by_right["put"]) < 0.25, f"put level off by {by_right['put']:.1%}"


def test_configured_default_vrp_has_not_gone_stale(bars):
    """A STALENESS GUARD, not a validation of the estimator.

    This refits on the exact same dataset the default was derived from and
    checks the two haven't drifted apart -- it is circular by construction
    and can only ever fail if someone edits ``VolParams.vrp_mult`` (or the
    estimator) without re-running the fit, never because the estimator is
    wrong. See ``test_synthetic_injected_vrp_is_recovered`` below for a test
    that can actually fail because the *estimator* is wrong -- that one
    prices a fixture with an independently chosen VRP nobody read off this
    data.
    """
    model = get_vol_model("gk_vrp")
    fit = calibration.fit_vol_params(bars, model)
    # Assert on the *cluster* counts, not ``n_atm_bars``: the whole point of
    # finding 7 is that minute rows are not the effective sample size, so a
    # guard phrased in bars would re-import the error it was meant to retire.
    assert fit.n_sessions >= 20
    assert fit.n_expiries >= 10
    assert fit.vrp_mult == pytest.approx(model.params.vrp_mult, abs=0.10)


# ---------------------------------------------------------------------------
# Synthetic fixtures for the estimator itself (finding 9b).
#
# Everything above compares the model against real quotes, and the one fit
# test that touches ``VolParams.vrp_mult`` is explicitly a staleness guard,
# not a validation -- it refits on the data the default came from. The tests
# below instead build a fixture where the answer is chosen independently of
# any configured default, prices it through ``black76`` at that chosen
# answer, and checks ``fit_vol_params`` recovers it. That is an honest test
# of the estimator.
# ---------------------------------------------------------------------------


def _weekdays_from(start: date, n: int) -> list[date]:
    """The next ``n`` Mon-Fri calendar dates starting at (and including,
    if a weekday) ``start``."""
    out: list[date] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _flat_price_bars(
    day_bar_counts: dict[date, int], level: float = 20_000.0
) -> pd.DataFrame:
    """Synthetic 1-minute bars: constant OHLC (zero intraday range) so
    Garman-Klass realized variance is deterministically zero every day and
    the EWMA-smoothed ATM level floors at ``iv_floor`` for every warm bar.
    That makes the model's own realized-vol input a known constant, which is
    exactly what lets these tests inject an *exact* IV/RV ratio and check it
    comes back out.
    """
    frames = []
    for day, n in day_bar_counts.items():
        idx = pd.date_range(
            datetime(day.year, day.month, day.day, 9, 15),
            periods=n,
            freq="1min",
            tz=IST,
        )
        frames.append(
            pd.DataFrame(
                {"open": level, "high": level, "low": level, "close": level}, index=idx
            )
        )
    out = pd.concat(frames).sort_index()
    out.index.name = "datetime"
    return add_session_cols(out)


def _inject_synthetic_chain(
    bars: pd.DataFrame,
    calendar: ExpiryCalendar,
    vol: GkVrpVolModel,
    quotes: dict[date, tuple[int, float]],
    *,
    right: str = "call",
) -> pd.DataFrame:
    """Price a synthetic observed-quote chain at an EXACTLY KNOWN IV/RV ratio.

    ``quotes`` maps a session date to ``(n_bars, ratio)``: the first
    ``n_bars`` timestamps of that session become quote rows, each priced via
    Black-76 at ``iv = model_realized_vol(day) * ratio``. Inverting the
    resulting premium must therefore recover ``ratio`` exactly (up to solver
    tolerance) -- this is the independently-chosen answer the estimator is
    tested against, not anything read off real quotes.
    """
    expiry_by_date = {d: calendar.weekly_expiry_for(d) for d in {*bars["date"]}}
    front_expiry = bars["date"].map(expiry_by_date)
    tau = tau_years(bars, front_expiry, calendar)
    atm = vol.atm_iv(bars).to_numpy()
    realized_vol = atm / vol.params.vrp_mult

    spot = bars["close"].to_numpy()
    strike = atm_strike(spot)

    rows: list[dict[str, object]] = []
    idx: list[pd.Timestamp] = []
    for day, (n_bars, ratio) in quotes.items():
        positions = np.flatnonzero((bars["date"] == day).to_numpy())[:n_bars]
        for pos in positions:
            iv = float(realized_vol[pos] * ratio)
            premium = float(
                black76.price(
                    spot[pos],
                    strike[pos],
                    tau[pos],
                    iv,
                    right,
                    rate=black76.DEFAULT_RATE,
                )
            )
            rows.append(
                {
                    "right": right,
                    "strike": float(strike[pos]),
                    "expiry": expiry_by_date[day],
                    "premium": premium,
                }
            )
            idx.append(bars.index[pos])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="datetime"))


@pytest.fixture(scope="module")
def synthetic_bars() -> pd.DataFrame:
    """25 weekdays: a warm-up day plus five *complete* weekly cycles (so
    every Monday/Tuesday candidate day's Thursday expiry is actually inside
    the data -- a candidate day in a trailing, incomplete week would resolve
    to an expiry with no headroom and get filtered out by ``MIN_TAU_FOR_IV``,
    silently shrinking the session count below what the test expects).
    """
    days = _weekdays_from(date(2024, 1, 1), 25)
    counts = {d: 80 for d in days}
    return _flat_price_bars(counts)


@pytest.fixture(scope="module")
def synthetic_vol() -> GkVrpVolModel:
    # `vrp_mult=1.0` keeps `realized_vol = atm_iv / vrp_mult` numerically
    # equal to the model's raw smoothed level; `seed_days=1` means only the
    # very first calendar day (the warm-up day) comes back with a NaN IV.
    return GkVrpVolModel(vrp_mult=1.0, seed_days=1, halflife_days=5.0)


def _candidate_test_days(bars: pd.DataFrame) -> list[date]:
    """Monday/Tuesday sessions, excluding the very first (warm-up) day."""
    days = sorted({*bars["date"]})
    warmup = days[0]
    return [d for d in days if d != warmup and d.weekday() in (0, 1)]


def test_synthetic_injected_vrp_is_recovered(synthetic_bars, synthetic_vol):
    """The estimator, tested honestly: build a fixture with several sessions
    across a couple of weekly expiries, price it at a chosen VRP that is
    NOT 1.24 (the configured default) and not read off any real data, and
    check ``fit_vol_params`` recovers it.

    Fails if the ratio/median machinery in ``fit_vol_params`` or
    ``implied_vols`` is broken -- e.g. if the realized-vol denominator drifts
    from what actually priced the quotes, or the near-ATM filter or the
    per-session median stop doing what their docstrings say.
    """
    calendar = ExpiryCalendar.from_bars(synthetic_bars)
    test_days = _candidate_test_days(synthetic_bars)
    assert len(test_days) >= 5
    expiries = {calendar.weekly_expiry_for(d) for d in test_days}
    assert len(expiries) >= 2, "fixture must span at least a couple of weekly expiries"

    injected_vrp = 1.5  # distinctive: not 1.24, not near the ±0.10 staleness band
    quotes = {d: (5, injected_vrp) for d in test_days}
    chain = _inject_synthetic_chain(synthetic_bars, calendar, synthetic_vol, quotes)

    fit = calibration.fit_vol_params(synthetic_bars, synthetic_vol, calendar, chain)
    assert fit.n_sessions == len(test_days)
    assert fit.n_expiries == len(expiries)
    assert fit.vrp_mult == pytest.approx(injected_vrp, abs=1e-3)
    assert fit.vrp_bar_weighted == pytest.approx(injected_vrp, abs=1e-3)


def test_equal_day_estimator_resists_a_bar_heavy_session(synthetic_bars, synthetic_vol):
    """Finding 7's whole point: the bar-level median silently over-weights
    whichever sessions retained the most minutes. Build a fixture where one
    session contributes far more retained minutes, at a different ratio,
    than all the others combined.

    Fails if ``vrp_mult`` (equal-day, one vote per session) ever starts
    tracking the bar-heavy session instead of resisting it, or if
    ``vrp_bar_weighted`` stops being draggable -- i.e. if the two estimators
    are ever accidentally made to compute the same thing.
    """
    calendar = ExpiryCalendar.from_bars(synthetic_bars)
    test_days = _candidate_test_days(synthetic_bars)
    assert len(test_days) >= 7
    heavy_day = test_days[-1]
    light_days = test_days[:-1]

    quotes = {d: (3, 1.2) for d in light_days}
    quotes[heavy_day] = (80, 2.0)
    chain = _inject_synthetic_chain(synthetic_bars, calendar, synthetic_vol, quotes)

    fit = calibration.fit_vol_params(synthetic_bars, synthetic_vol, calendar, chain)

    # One vote per day: 6+ sessions at 1.2 outvote the single session at 2.0.
    assert fit.vrp_mult == pytest.approx(1.2, abs=1e-3)
    # The bar-level median is swamped by the heavy session's 80 quotes
    # against 3 bars each everywhere else.
    assert fit.vrp_bar_weighted == pytest.approx(2.0, abs=1e-3)
    assert fit.vrp_mult != pytest.approx(fit.vrp_bar_weighted, abs=0.1)


def test_fit_refuses_to_report_a_skew_estimate(bars):
    """The trap this data sets: a confident, wrongly-signed skew."""
    fit = calibration.fit_vol_params(bars, get_vol_model("gk_vrp"))
    assert fit.skew_beta is None
    assert "not identifiable" in fit.summary()


def test_observed_quotes_never_go_below_the_tick(comparison):
    """A property of the INPUT DATA (the vendor never quotes below a tick),
    not of our code -- kept as a sanity check on the fixture, but it does not
    validate the ``TICK_SIZE`` floor applied to *modeled* premiums in
    ``chain.pinned_leg``. That floor's economics -- whether it is
    conservative or anti-conservative, and for which side of a trade -- are
    the actual load-bearing question, and are tested in
    ``tests/test_terminal_value.py`` instead.
    """
    from obt.chain import TICK_SIZE

    assert comparison["observed"].min() >= TICK_SIZE
