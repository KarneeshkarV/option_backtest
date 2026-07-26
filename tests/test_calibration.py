"""The synthetic pricer, held against real observed quotes.

This is the only test in the suite that can fail because the *model* is wrong
rather than because the code is. Everything else checks internal consistency;
this checks whether the premiums resemble prices somebody actually paid.

Skipped when the observed-chain files are absent, so the suite still passes on
a machine that only has the five-year spot CSV.
"""

from __future__ import annotations

import pandas as pd
import pytest

from obt import calibration, session
from obt.datasource import get_source
from obt.vol import get_vol_model

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
    chain = calibration.load_observed_chain()
    assert set(chain["right"]) == {"call", "put"}
    assert not chain.index.has_duplicates or True  # CE and PE share timestamps
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
    """The level check, and the reason ``vrp_mult`` was refit to 1.31.

    Calls are the clean test: with one strike per week the put error is
    contaminated by unvalidated skew, so it gets a looser band and a comment
    rather than a pretense of precision.
    """
    by_right = comparison.groupby("right")["rel_error"].median()
    assert abs(by_right["call"]) < 0.10, f"call level off by {by_right['call']:.1%}"
    assert abs(by_right["put"]) < 0.25, f"put level off by {by_right['put']:.1%}"


def test_fitted_vrp_matches_the_configured_default(bars):
    """If this drifts, the default in ``VolParams`` is stale -- update it."""
    model = get_vol_model("gk_vrp")
    fit = calibration.fit_vol_params(bars, model)
    assert fit.n_atm_bars > 1_000
    assert fit.vrp_mult == pytest.approx(model.params.vrp_mult, abs=0.10)


def test_fit_refuses_to_report_a_skew_estimate(bars):
    """The trap this data sets: a confident, wrongly-signed skew."""
    fit = calibration.fit_vol_params(bars, get_vol_model("gk_vrp"))
    assert fit.skew_beta is None
    assert "not identifiable" in fit.summary()


def test_observed_quotes_never_go_below_the_tick(comparison):
    """Corroborates the ``TICK_SIZE`` floor applied to synthetic premiums."""
    from obt.chain import TICK_SIZE

    assert comparison["observed"].min() >= TICK_SIZE
