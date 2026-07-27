"""Realized-vol-driven IV: Garman-Klass, EWMA-smoothed, scaled by a risk premium.

The chain of reasoning, since every link is an assumption worth arguing with:

- **Garman-Klass** estimates each day's variance from that day's O/H/L/C. It is
  far more efficient than close-to-close (it uses the range), and we have the
  full OHLC. It is computed strictly within a day, so the multi-month holes in
  the data never enter an estimate.
- **EWMA across days** smooths the noisy daily estimate into a vol level.
- **Variance risk premium.** Options systematically trade above realized vol;
  ``vrp_mult`` is that wedge. It is the single most consequential number in
  this package -- it alone decides whether premium-selling looks profitable.
- **Skew.** NIFTY puts trade above calls. ``skew_beta`` tilts IV by
  log-moneyness.

The ATM series is **shifted one day forward**: the IV applied to bars on day D
comes from data through day D-1. Without that shift the model would price
today's options using today's high and low, which is a lookahead leak that
flatters every result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from obt.session import TRADING_DAYS_PER_YEAR, covered_periods
from obt.vol.spec import vol_model

#: Garman-Klass constant on the close-to-open term.
_GK_C = 2.0 * np.log(2.0) - 1.0


class VolParams(BaseModel):
    """Every knob the vol model exposes. Frozen so a run cannot mutate it."""

    vrp_mult: float = Field(default=1.24, gt=0)
    """IV = realized vol x this. The dominant assumption; sweep it anyway.

    **Measured, not guessed -- but far less precisely than it looks.**
    :func:`obt.calibration.fit_vol_params` inverts observed NIFTY ATM quotes to
    implied vol and divides by this model's own realized-vol input. The
    estimator is a median of *per-session* medians -- one vote per trading day
    -- because the retained rows are one-minute quotes that share a single
    daily realized-vol denominator and autocorrelate heavily within a session.
    They are not independent draws, and the older bar-level median presented
    7,108 of them as if they were.

    The honest sample is **23 sessions across 13 weekly expiries**, giving
    1.244 with a session-bootstrap 95% band of roughly 1.20-1.26. Reweightings
    disagree by more than that band is wide: bar-weighted 1.226, per-expiry
    1.257, calls 1.18 vs puts 1.25, and 1.33 at two sessions to expiry against
    1.20 at four. Held-out later cycles give 1.199 against 1.254 on the earlier
    ones, so the level drifts with regime. **Read this as "about 1.2, one
    regime, three months", not as three significant figures.**

    Previously 1.31, fitted before :func:`_garman_klass_daily_variance` existed:
    the EWMA then smoothed vol instead of variance and understated realized vol
    by about 7%, so the multiplier had silently absorbed that bias. Correcting
    the estimator raised realized vol and the multiplier had to come down with
    it, or the same correction would have been counted twice.
    """

    skew_beta: float = Field(default=-1.2)
    """iv(K) = atm * (1 + beta * log(K/S)). Negative lifts downside puts.

    Still a guess. The observed-quote files carry one strike per weekly cycle,
    which cannot identify a smile at all -- see :mod:`obt.calibration`. Real
    puts price roughly 20% above this model at 2+ sessions to expiry, so the
    true tilt is probably steeper, but nothing here measures it.
    """

    term_slope: float = Field(default=0.0, ge=0)
    """Optional IV bump as expiry approaches. 0 disables it."""

    halflife_days: float = Field(default=10.0, gt=0)
    """EWMA halflife over daily realized-variance estimates."""

    iv_floor: float = Field(default=0.06, gt=0)
    iv_cap: float = Field(default=0.80, gt=0)
    """Clamps, so a quiet stretch cannot produce near-zero premiums."""

    seed_days: int = Field(default=10, ge=1)
    """Sessions of prior history required before a bar is considered warm.

    A bar in a session short of ``seed_days`` predecessors (counting from the
    start of its covered data block -- see :func:`obt.session.covered_periods`)
    gets a NaN ATM IV rather than a level derived from too little history.
    """

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _check_clamp_bounds(self) -> VolParams:
        if self.iv_cap < self.iv_floor:
            raise ValueError(
                f"iv_cap ({self.iv_cap}) must be >= iv_floor ({self.iv_floor}); "
                "a reversed clamp silently flattens the whole IV surface to "
                "iv_cap."
            )
        return self


def _garman_klass_daily_variance(bars: pd.DataFrame) -> pd.Series:
    """Per-day (non-annualized) realized variance from intraday OHLC.

    Split out from :func:`garman_klass_daily` so smoothing can happen in
    variance units. An EWMA average belongs over the variance, not over its
    square root: sqrt is concave, so averaging already-annualized vols (as the
    original code did) computes ``E[sqrt(V)]``, which sits systematically
    below the correct ``sqrt(E[V])`` (Jensen's inequality) and damps regime
    changes.
    """
    grouped = bars.groupby(bars["date"])
    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
        }
    )
    log_hl = np.log(daily["high"] / daily["low"])
    log_co = np.log(daily["close"] / daily["open"])
    variance = 0.5 * log_hl**2 - _GK_C * log_co**2
    # GK can go slightly negative on a doji-ish day; clip rather than NaN out.
    return variance.clip(lower=0.0)


def _annualize_vol(variance: pd.Series) -> pd.Series:
    """sqrt(variance * trading days), factored out so the type stays ``Series``.

    ``np.sqrt`` on a bare expression loses the static ``Series`` type to mypy
    (it resolves to an ndarray overload), which then breaks the pandas-style
    ``.clip(lower=..., upper=...)`` call downstream. Routing through a
    function with an explicit ``-> pd.Series`` return annotation is the same
    trick :func:`garman_klass_daily` already relied on.
    """
    return np.sqrt(variance * TRADING_DAYS_PER_YEAR)


def garman_klass_daily(bars: pd.DataFrame) -> pd.Series:
    """Per-day annualized realized vol from intraday OHLC.

    Each day is reduced to its session open, high, low and close, then fed
    through the Garman-Klass estimator. Aggregating within the day first is
    what keeps overnight gaps -- and the multi-month data holes -- out of the
    estimate entirely.

    Public contract preserved: this still returns annualized VOL (sqrt of
    variance), same as before. :meth:`GkVrpVolModel.atm_iv` no longer calls
    this directly -- it smooths :func:`_garman_klass_daily_variance` instead,
    in variance units, and only takes the square root once at the end.
    """
    variance = _garman_klass_daily_variance(bars)
    return _annualize_vol(variance)


class GkVrpVolModel:
    """Garman-Klass realized vol, EWMA-smoothed, scaled to an IV."""

    def __init__(self, **kwargs: float) -> None:
        self.params = VolParams(**kwargs)  # type: ignore[arg-type]

    @property
    def label(self) -> str:
        p = self.params
        return (
            f"gk_vrp(vrp={p.vrp_mult:g}, skew={p.skew_beta:g}, hl={p.halflife_days:g}d)"
        )

    def atm_iv(self, bars: pd.DataFrame) -> pd.Series:
        p = self.params
        daily_variance = _garman_klass_daily_variance(bars)

        # EWMA per covered block, not across the whole series: a 300-day hole
        # is not "yesterday" and must not hand its pre-gap smoothed level to
        # the first session after the hole. `covered_periods` gives the block
        # boundaries; each block's EWMA starts cold.
        blocks: list[pd.Series] = []
        for start, end in covered_periods(bars):
            block = daily_variance.loc[
                (daily_variance.index >= start) & (daily_variance.index <= end)
            ]
            # min_periods=seed_days actually honours seed_days: a bar needs
            # that many prior sessions of history before it is "warm".
            smoothed = block.ewm(
                halflife=p.halflife_days, min_periods=p.seed_days
            ).mean()
            # Shift one day: bars on day D are priced with variance smoothed
            # through D-1 only. No bfill -- a leading NaN (the block's first
            # `seed_days` sessions have no D-1 within the block) stays NaN
            # rather than being seeded from that session's own full-day GK
            # estimate, which would be a lookahead leak on day one.
            blocks.append(smoothed.shift(1))
        smoothed_variance = (
            pd.concat(blocks) if blocks else daily_variance.iloc[0:0].astype("float64")
        )

        smoothed_vol = _annualize_vol(smoothed_variance)
        atm = (smoothed_vol * p.vrp_mult).clip(lower=p.iv_floor, upper=p.iv_cap)
        return bars["date"].map(atm).astype("float64")

    def iv(
        self,
        atm: np.ndarray,
        spot: np.ndarray,
        strike: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        p = self.params
        with np.errstate(divide="ignore", invalid="ignore"):
            log_moneyness = np.log(np.where(spot > 0, strike / spot, 1.0))
        log_moneyness = np.nan_to_num(log_moneyness, nan=0.0, posinf=0.0, neginf=0.0)
        surface = atm * (1.0 + p.skew_beta * log_moneyness)
        if p.term_slope:
            # Shorter expiry -> richer vol; bounded so tau->0 stays finite.
            surface = surface * (1.0 + p.term_slope * np.exp(-tau * 52.0))
        return np.clip(surface, p.iv_floor, p.iv_cap)


@vol_model(
    "gk_vrp",
    description="Garman-Klass realized vol x variance risk premium, with skew",
)
def _build(**kwargs: float) -> GkVrpVolModel:
    return GkVrpVolModel(**kwargs)
