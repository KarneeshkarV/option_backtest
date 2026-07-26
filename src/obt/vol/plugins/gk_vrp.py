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
from pydantic import BaseModel, ConfigDict, Field

from obt.session import TRADING_DAYS_PER_YEAR
from obt.vol.spec import vol_model

#: Garman-Klass constant on the close-to-open term.
_GK_C = 2.0 * np.log(2.0) - 1.0


class VolParams(BaseModel):
    """Every knob the vol model exposes. Frozen so a run cannot mutate it."""

    vrp_mult: float = Field(default=1.31, gt=0)
    """IV = realized vol x this. The dominant assumption; sweep it anyway.

    **Measured, not guessed.** :func:`obt.calibration.fit_vol_params` inverts
    60 sessions of observed NIFTY ATM quotes to implied vol and divides by this
    model's own realized-vol input; the median over 7,108 near-ATM bars is 1.31.
    The original 1.15 was a textbook figure and it under-priced real calls by
    about 7%. Three months is a thin sample and one volatility regime, so treat
    1.31 as a better-anchored estimate rather than a settled constant.
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
    """Days used to seed the EWMA before the series is considered warm."""

    model_config = ConfigDict(frozen=True)


def garman_klass_daily(bars: pd.DataFrame) -> pd.Series:
    """Per-day annualized realized vol from intraday OHLC.

    Each day is reduced to its session open, high, low and close, then fed
    through the Garman-Klass estimator. Aggregating within the day first is
    what keeps overnight gaps -- and the multi-month data holes -- out of the
    estimate entirely.
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
    variance = variance.clip(lower=0.0)
    return np.sqrt(variance * TRADING_DAYS_PER_YEAR)


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
        daily_rv = garman_klass_daily(bars)
        smoothed = daily_rv.ewm(halflife=p.halflife_days, min_periods=1).mean()
        # Shift one day: bars on day D are priced with information through D-1.
        smoothed = smoothed.shift(1)
        # Seed the first day (and any leading NaN) with the first real estimate.
        smoothed = smoothed.bfill()
        atm = (smoothed * p.vrp_mult).clip(lower=p.iv_floor, upper=p.iv_cap)
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
