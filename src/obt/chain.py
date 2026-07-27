"""Option series: strike selection, expiry pinning, premium paths.

This module is where spot becomes an option, and it is the single most
dangerous file in the package. The danger has one shape: **a premium series
whose strike moves while a position is open is meaningless**, and it looks
completely plausible in a chart. Two synthetic helpers and one observed path
keep that distinction impossible to blur by accident:

- :func:`rolling_atm` re-picks the strike every bar. Use it to *study* how
  premiums behave. It is not tradeable and must never reach a portfolio.
- :func:`pinned_leg` freezes strike and expiry at the open and prices with
  Black-76. **Valid for P&L only as a model**, not as market truth.
- :func:`pinned_leg_from_observed_chain` freezes strike and expiry the same
  way, but reads **observed** premiums from an :class:`OptionChainSource`.
  This is the path that matches ``screener``'s rule: P&L uses traded quotes.

Everything downstream consumes the same output shape
(``premium/strike/iv/tau/expiry``), so the engine does not care which path
produced the frame.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from obt.calendar import ExpiryCalendar, expiry_series, tau_years
from obt.pricing import black76
from obt.vol.base import VolModel

Right = Literal["call", "put"]
Direction = Literal["long", "short"]

#: NIFTY strikes are listed on a 50-point grid.
STRIKE_STEP = 50.0

#: NSE quotes option premiums on a 5-paise tick and nothing trades below it.
#: Black-76 has no such floor: a far-OTM call with one bar of life left prices
#: to exactly 0.0, which is both untradeable in reality and rejected outright by
#: vectorbt ("order.price must be finite and greater than 0"). Floor the
#: tradeable series at one tick.
#:
#: **The direction of that error is not symmetric, and it is not conservative
#: for a long.** A long closes by *selling*: floored, it receives Rs 0.05 for
#: something truly worth 0, so its P&L is overstated by one tick times
#: quantity. A short closes by *buying back*, so the same floor costs it the
#: same amount and understates its P&L. Both shipped strategies are long, so
#: the bias currently flatters the headline numbers: at a 75-unit lot it is
#: Rs 3.75 per trade that expires at the floor, which is 156 of the 792 `orb`
#: trades -- about Rs 585, or 0.03% of starting capital. Small, but it points
#: the optimistic way for longs, not the conservative way an earlier version
#: of this comment claimed.
TICK_SIZE = 0.05

#: Fallback lot size when no point-in-time lot history is available.
#: screener refuses to fabricate that history, and so do we -- this is a
#: stated assumption, surfaced in the report banner, not a lookup.
DEFAULT_LOT_SIZE = 75


class LegSpec(BaseModel):
    """One option leg: what to buy or sell, and which strike.

    ``strike_rule`` is one of:

    - ``"atm"`` -- nearest listed strike to spot
    - ``"otm:N"`` / ``"itm:N"`` -- N strike steps out of / into the money,
      interpreted relative to ``right`` so the same spec means the same thing
      for calls and puts
    - ``"delta:X"`` -- the listed strike whose |delta| is closest to X
    """

    model_config = ConfigDict(frozen=True)

    right: Right
    direction: Direction = "long"
    strike_rule: str = "atm"
    lots: int = Field(default=1, ge=1)
    lot_size: int = Field(default=DEFAULT_LOT_SIZE, ge=1)
    strike_step: float = Field(default=STRIKE_STEP, gt=0)

    @property
    def label(self) -> str:
        return f"{self.direction} {self.strike_rule} {self.right}"

    @property
    def signed_quantity(self) -> int:
        qty = self.lots * self.lot_size
        return qty if self.direction == "long" else -qty


def lot_size_for(as_of: date, *, fallback: int = DEFAULT_LOT_SIZE) -> tuple[int, bool]:
    """NIFTY lot size on ``as_of``. Returns ``(size, from_history)``.

    Reads screener's user-maintained point-in-time file when present. That file
    usually is not present, and screener is explicit about why: historical F&O
    lot sizes are not reliably downloadable, so it never fabricates them.
    Neither do we -- the caller gets ``from_history=False`` and the report says
    a flat assumed lot size was used.
    """
    from screener.options.lot_history import historical_lot_sizes

    sizes = historical_lot_sizes(as_of)
    size = sizes.get("NIFTY")
    if size:
        return int(size), True
    return fallback, False


def atm_strike(spot: np.ndarray, step: float = STRIKE_STEP) -> np.ndarray:
    """Nearest listed strike."""
    return np.round(np.asarray(spot, dtype="float64") / step) * step


def _offset_strikes(
    spot: np.ndarray, right: Right, steps: int, step: float, *, into: bool
) -> np.ndarray:
    """Move ``steps`` strikes in/out of the money for the given right."""
    base = atm_strike(spot, step)
    # Calls go OTM upward, puts go OTM downward.
    outward = 1.0 if right == "call" else -1.0
    direction = -outward if into else outward
    return base + direction * steps * step


def _delta_targeted_strikes(
    spot: np.ndarray,
    tau: np.ndarray,
    atm_iv: np.ndarray,
    right: Right,
    target: float,
    vol: VolModel,
    step: float,
    *,
    search_steps: int = 40,
) -> np.ndarray:
    """Listed strike whose |delta| sits closest to ``target``.

    Evaluated over a grid of candidate strikes around ATM rather than solved
    analytically, because strikes are discrete anyway -- the grid *is* the
    answer space.
    """
    base = atm_strike(spot, step)
    offsets = np.arange(-search_steps, search_steps + 1, dtype="float64") * step
    candidates = base[:, None] + offsets[None, :]

    spot_grid = np.repeat(spot[:, None], candidates.shape[1], axis=1)
    tau_grid = np.repeat(tau[:, None], candidates.shape[1], axis=1)
    atm_grid = np.repeat(atm_iv[:, None], candidates.shape[1], axis=1)
    iv_grid = vol.iv(atm_grid, spot_grid, candidates, tau_grid)

    deltas = np.abs(black76.delta(spot_grid, candidates, tau_grid, iv_grid, right))
    best = np.argmin(np.abs(deltas - abs(target)), axis=1)
    return candidates[np.arange(len(spot)), best]


def select_strikes(
    spot: np.ndarray,
    tau: np.ndarray,
    atm_iv: np.ndarray,
    leg: LegSpec,
    vol: VolModel,
) -> np.ndarray:
    """Resolve ``leg.strike_rule`` into concrete strikes."""
    rule = leg.strike_rule.strip().lower()
    if rule == "atm":
        return atm_strike(spot, leg.strike_step)
    for prefix, into in (("otm:", False), ("itm:", True)):
        if rule.startswith(prefix):
            steps = int(rule[len(prefix) :])
            return _offset_strikes(spot, leg.right, steps, leg.strike_step, into=into)
    if rule.startswith("delta:"):
        target = float(rule[len("delta:") :])
        return _delta_targeted_strikes(
            spot, tau, atm_iv, leg.right, target, vol, leg.strike_step
        )
    raise ValueError(
        f"unknown strike rule {leg.strike_rule!r}; "
        "expected 'atm', 'otm:N', 'itm:N' or 'delta:X'"
    )


def rolling_atm(
    bars: pd.DataFrame,
    vol: VolModel,
    calendar: ExpiryCalendar,
    right: Right = "call",
    *,
    rate: float = black76.DEFAULT_RATE,
) -> pd.DataFrame:
    """Continuous ATM premium series, strike re-picked every bar.

    **Not tradeable.** The strike silently rolls from bar to bar, so any P&L
    computed from this series is fiction -- it measures a contract nobody could
    hold. Use it for studying premium behaviour, calibrating the vol model, or
    plotting. For anything that touches a portfolio, use :func:`pinned_leg`.
    """
    spot = bars["close"].to_numpy()
    expiry = expiry_series(bars, calendar)
    tau = tau_years(bars, expiry, calendar)
    atm_iv = vol.atm_iv(bars).to_numpy()
    strike = atm_strike(spot)
    iv = vol.iv(atm_iv, spot, strike, tau)
    premium = black76.price(spot, strike, tau, iv, right, rate=rate)
    return pd.DataFrame(
        {
            "premium": premium,
            "strike": strike,
            "iv": iv,
            "tau": tau,
            "expiry": expiry.to_numpy(),
        },
        index=bars.index,
    )


def pinned_leg(
    bars: pd.DataFrame,
    open_mask: np.ndarray,
    leg: LegSpec,
    vol: VolModel,
    calendar: ExpiryCalendar,
    *,
    rate: float = black76.DEFAULT_RATE,
    calendar_time: bool = False,
) -> pd.DataFrame:
    """Premium path with strike and expiry frozen at each position open.

    ``open_mask`` must come from :func:`obt.signals.resolve_trades` -- it marks
    bars that genuinely open a position, not raw entry signals. At each such
    bar the strike and expiry are chosen and then carried forward until the
    next open.

    Between trades the series still carries the previous trade's strike, so it
    can jump at an open. That is harmless *only* because the engine guarantees
    the portfolio is flat at those bars; it is the reason the end-of-day
    force-exit is not optional.
    """
    spot = bars["close"].to_numpy()
    open_mask = np.asarray(open_mask, dtype=bool)

    front_expiry = expiry_series(bars, calendar)
    front_tau = tau_years(bars, front_expiry, calendar, calendar_time=calendar_time)
    atm_iv = vol.atm_iv(bars).to_numpy()

    chosen = select_strikes(spot, front_tau, atm_iv, leg, vol)

    # Pin at opens, then carry forward. `ffill` on a float Series is the
    # cheapest correct way to express "hold until the next open".
    strike_at_open = pd.Series(np.where(open_mask, chosen, np.nan), index=bars.index)
    strike = strike_at_open.ffill().bfill().to_numpy()

    expiry_at_open = front_expiry.where(pd.Series(open_mask, index=bars.index))
    expiry = expiry_at_open.ffill().bfill()

    tau = tau_years(bars, expiry, calendar, calendar_time=calendar_time)
    iv = vol.iv(atm_iv, spot, strike, tau)
    premium = np.maximum(
        black76.price(spot, strike, tau, iv, leg.right, rate=rate), TICK_SIZE
    )

    return pd.DataFrame(
        {
            "premium": premium,
            "strike": strike,
            "iv": iv,
            "tau": tau,
            "expiry": expiry.to_numpy(),
        },
        index=bars.index,
    )


def pinned_leg_from_observed_chain(
    bars: pd.DataFrame,
    open_mask: np.ndarray,
    leg: LegSpec,
    chain: pd.DataFrame,
    calendar: ExpiryCalendar,
    *,
    calendar_time: bool = False,
) -> pd.DataFrame:
    """Premium path from **observed** quotes, strike/expiry frozen at each open.

    Same pinning contract as :func:`pinned_leg`, but premiums come from
    ``chain`` (columns ``right, strike, expiry, premium`` on a tz-aware IST
    index -- see :func:`obt.datasource.normalize_option_chain`) rather than
    Black-76. IV is left as NaN: the quote is the price; we do not invert it
    on the hot path.

    The shipped ATM CSVs carry one strike per weekly cycle. Non-``atm``
    ``strike_rule`` values are therefore rejected here: there is no OTM/ITM
    ladder in the file to look up. When a fuller chain arrives, extend the
    lookup; do not silently fall back to the model.
    """
    if leg.strike_rule.strip().lower() != "atm":
        raise ValueError(
            f"observed option chain only supports strike_rule='atm' "
            f"(got {leg.strike_rule!r}); the CE/PE files carry one ATM strike "
            "per weekly cycle, not an OTM/ITM ladder"
        )

    open_mask = np.asarray(open_mask, dtype=bool)
    right_chain = chain.loc[chain["right"] == leg.right]
    if right_chain.empty:
        raise ValueError(f"observed option chain has no rows for right={leg.right!r}")

    # One quote per timestamp for this right. Duplicate stamps would mean the
    # feed grew a multi-strike chain without this function learning to pick.
    if right_chain.index.has_duplicates:
        counts = right_chain.index.value_counts()
        bad = counts[counts > 1]
        raise ValueError(
            f"observed chain has multiple {leg.right} quotes at the same "
            f"timestamp (e.g. {bad.index[0]}); pinned_leg_from_observed_chain "
            "expects one ATM row per right per bar"
        )

    # Align quotes onto the spot index. Some sessions have fewer option bars
    # than spot bars; fill gaps only within the same session so a missing
    # minute does not borrow yesterday's premium across the overnight gap.
    by_day = bars["date"]
    strike_series = (
        right_chain["strike"]
        .reindex(bars.index)
        .groupby(by_day)
        .ffill()
        .groupby(by_day)
        .bfill()
    )
    expiry_series_ = (
        right_chain["expiry"]
        .reindex(bars.index)
        .groupby(by_day)
        .ffill()
        .groupby(by_day)
        .bfill()
    )
    premium_series = (
        right_chain["premium"]
        .reindex(bars.index)
        .groupby(by_day)
        .ffill()
        .groupby(by_day)
        .bfill()
    )

    # Pin at resolved opens, then carry forward -- same ffill pattern as the
    # synthetic path. Premium follows the *pinned* contract: with one ATM row
    # per right the reindexed series already is that contract within a cycle,
    # and EOD force-exit keeps every trade inside a single session.
    strike_at_open = strike_series.where(open_mask)
    strike = strike_at_open.ffill().bfill().to_numpy(dtype="float64")

    expiry_at_open = expiry_series_.where(open_mask)
    expiry = expiry_at_open.ffill().bfill()

    premium = premium_series.to_numpy(dtype="float64")
    # Same tick floor as the synthetic path: vectorbt rejects a non-positive
    # price. Real quotes already sit on the 5-paise grid; this only clamps
    # the rare exact-zero print.
    premium = np.where(np.isfinite(premium), np.maximum(premium, TICK_SIZE), premium)

    tau = tau_years(bars, expiry, calendar, calendar_time=calendar_time)
    iv = np.full(len(bars), np.nan, dtype="float64")

    return pd.DataFrame(
        {
            "premium": premium,
            "strike": strike,
            "iv": iv,
            "tau": tau,
            "expiry": expiry.to_numpy(),
        },
        index=bars.index,
    )
