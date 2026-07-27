"""Measure the synthetic vol model against real observed option quotes.

Everything else in this package prices options from a model. This module is the
only place that reads prices somebody actually paid, and its whole job is to
answer one question: *how wrong are we?*

The reference data is 60 sessions of 1-minute NIFTY ATM CE/PE quotes
(2026-04-22 to 2026-07-21). It is small, but it is real, and it converts the two
load-bearing guesses in :mod:`obt.vol.plugins.gk_vrp` into measurements:

- ``vrp_mult`` **is identified** by this data, but only to about one decimal
  place. Inverting the near-ATM bars and dividing by the model's realized-vol
  input gives 1.244. The bars look like a large sample and are not: they are
  minutes drawn from **23 sessions across 13 weekly expiries**, sharing one
  realized-vol denominator per day. The fit therefore clusters -- one vote per
  session -- and reports a session-bootstrap band alongside the point estimate.
- ``skew_beta`` **is not identified** by this data, and pretending otherwise is
  the trap here. The files carry exactly one strike per weekly cycle, so the
  moneyness spread comes from spot drifting away from a pinned strike over
  days, not from a smile across strikes at an instant. Regressing IV on
  log-moneyness therefore measures the leverage effect through *time* and
  returns a positive slope -- the wrong sign for an equity index. See
  :func:`fit_vol_params`, which refuses to report a skew estimate for that
  reason.

Two further limits on what this data can prove, both worth stating before
anyone reads a number off it:

- The strike is ATM at the *start* of each weekly cycle, not intraday ATM. By
  the end of a cycle spot has drifted a median 144 points from it (max 525).
- Three months and one strike per week is not a sample you can fit a surface to.
  It bounds the level. It does not validate the shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from obt.calendar import ExpiryCalendar, tau_years
from obt.datasource.plugins import nifty_atm_options_csv as _atm_options
from obt.pricing import black76
from obt.session import TRADING_DAYS_PER_YEAR
from obt.vol.base import VolModel

# Re-exports so callers keep using obt.calibration.ENV_VAR / CHAIN_FILES.
CHAIN_FILES = _atm_options.CHAIN_FILES
ENV_VAR = _atm_options.ENV_VAR
chain_files_available = _atm_options.chain_files_available
default_chain_dir = _atm_options.default_chain_dir

#: Bars nearer than this to expiry are excluded from IV inversion. Inside the
#: last two sessions a near-worthless option's price is dominated by the 5-paise
#: tick and its vega is nearly zero, so implied vol is numerically
#: unidentifiable -- inverting anyway produces confident nonsense.
MIN_TAU_FOR_IV = 2.0 / 252.0

#: Quotes at or below this are tick noise, not information.
MIN_PREMIUM_FOR_IV = 1.0

#: Only bars this close to the strike inform the ATM level.
ATM_LOG_MONEYNESS = 0.002


def chain_dir() -> Path:
    """Directory holding the observed CE/PE CSVs.

    Override with ``$OBT_OPTION_CHAIN_DIR``.
    """
    return default_chain_dir()


def chain_available() -> bool:
    """Whether the observed-quote files are present on this machine."""
    return chain_files_available(chain_dir())


def load_observed_chain(directory: Path | None = None) -> pd.DataFrame:
    """Observed CE and PE quotes, one row per bar per right.

    Thin wrapper over the ``nifty_atm_options_csv`` option-source plugin so
    calibration and the engine share one reader. Returns columns
    ``right, strike, expiry, premium`` on a tz-aware IST index, with ``right``
    values matching :mod:`obt.pricing.black76`.
    """
    from obt.datasource import get_option_source

    return get_option_source("nifty_atm_options_csv", path=directory).load("NIFTY")


def _align(
    bars: pd.DataFrame, chain: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to bars they share."""
    shared = chain.index.intersection(bars.index)
    if shared.empty:
        raise ValueError(
            "observed chain and spot bars do not overlap. The chain covers "
            f"{chain.index.min().date()}..{chain.index.max().date()} and the "
            f"bars cover {bars.index.min().date()}..{bars.index.max().date()}."
        )
    return bars.loc[shared[~shared.duplicated()]], chain.loc[shared]


def compare_to_observed(
    bars: pd.DataFrame,
    vol: VolModel,
    calendar: ExpiryCalendar | None = None,
    chain: pd.DataFrame | None = None,
    *,
    rate: float = black76.DEFAULT_RATE,
) -> pd.DataFrame:
    """Price each observed contract with ``vol`` and return both prices.

    The comparison is like-for-like by construction: strike, expiry and right
    all come from the observed row, so the only thing under test is the vol
    model and the pricer.
    """
    chain = load_observed_chain() if chain is None else chain
    calendar = calendar or ExpiryCalendar.from_bars(bars)
    spot_bars, chain = _align(bars, chain)

    atm_series = pd.Series(vol.atm_iv(bars).to_numpy(), index=bars.index)
    aligned_spot = spot_bars["close"].reindex(chain.index)
    tau = tau_years(
        bars.reindex(chain.index),
        pd.Series(chain["expiry"].to_numpy(), index=chain.index),
        calendar,
    )

    spot = aligned_spot.to_numpy()
    strike = chain["strike"].to_numpy()
    atm = atm_series.reindex(chain.index).to_numpy()

    model = np.empty(len(chain), dtype="float64")
    for right in ("call", "put"):
        mask = (chain["right"] == right).to_numpy()
        if not mask.any():
            continue
        iv = vol.iv(atm[mask], spot[mask], strike[mask], tau[mask])
        model[mask] = black76.price(
            spot[mask], strike[mask], tau[mask], iv, right, rate=rate
        )

    observed = chain["premium"].to_numpy()
    out = pd.DataFrame(
        {
            "right": chain["right"].to_numpy(),
            "strike": strike,
            # Carried through so the fit can cluster on the contract a quote
            # belongs to, rather than treating every minute as its own draw.
            "expiry": chain["expiry"].to_numpy(),
            "spot": spot,
            "tau": tau,
            "observed": observed,
            "model": model,
            "error": model - observed,
            # Relative error is floored at one rupee so a 5-paise quote cannot
            # manufacture a 2000% "error" and swamp every summary statistic.
            "rel_error": (model - observed) / np.maximum(observed, 1.0),
        },
        index=chain.index,
    )
    return out.dropna(subset=["observed", "model", "spot"])


def implied_vols(
    bars: pd.DataFrame,
    vol: VolModel,
    calendar: ExpiryCalendar | None = None,
    chain: pd.DataFrame | None = None,
    *,
    rate: float = black76.DEFAULT_RATE,
) -> pd.DataFrame:
    """Invert observed premiums to implied vol, beside the model's own inputs.

    Columns: ``right, implied_iv, realized_vol, log_moneyness, tau``.
    ``realized_vol`` is the model's pre-VRP input, so ``implied_iv /
    realized_vol`` is a direct estimate of the variance risk premium.
    """
    from screener.options.greeks import implied_volatility

    chain = load_observed_chain() if chain is None else chain
    calendar = calendar or ExpiryCalendar.from_bars(bars)
    priced = compare_to_observed(bars, vol, calendar, chain, rate=rate)

    usable = priced[
        (priced["tau"] > MIN_TAU_FOR_IV) & (priced["observed"] > MIN_PREMIUM_FOR_IV)
    ]
    if usable.empty:
        return pd.DataFrame(
            columns=["right", "implied_iv", "realized_vol", "log_moneyness", "tau"]
        )

    solved = [
        implied_volatility(row.observed, row.spot, row.strike, row.tau, rate, row.right)
        for row in usable.itertuples()
    ]

    # Undo the VRP multiplier to recover the realized-vol input the model
    # started from. `vrp_mult` is a gk_vrp concept; a model without one is
    # compared against its own ATM level instead.
    atm = pd.Series(vol.atm_iv(bars).to_numpy(), index=bars.index).reindex(usable.index)
    multiplier = float(getattr(getattr(vol, "params", None), "vrp_mult", 1.0))

    out = pd.DataFrame(
        {
            "right": usable["right"].to_numpy(),
            "implied_iv": np.array(
                [np.nan if value is None else value for value in solved]
            ),
            "realized_vol": atm.to_numpy() / multiplier,
            "log_moneyness": np.log(
                usable["strike"].to_numpy() / usable["spot"].to_numpy()
            ),
            "tau": usable["tau"].to_numpy(),
            # The two cluster keys. Minute bars within a session share one
            # realized-vol denominator all day, and every session in a weekly
            # cycle shares one contract, so neither is an independent draw.
            "session": pd.Index(usable.index).date,
            "expiry": usable["expiry"].to_numpy(),
        },
        index=usable.index,
    )
    return out.dropna()


#: Weekly expiry cycles held back from the fit for out-of-sample validation.
#: The fit is reported on the earlier cycles and checked against these, so a
#: number that only describes the window it was fitted on is visible as such.
OOS_EXPIRY_CYCLES = 5

#: Bootstrap resamples used for the cluster-level uncertainty band.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


def _equal_day_median(atm: pd.DataFrame) -> float:
    """Median of per-session medians: one vote per trading day.

    The bar-level median answers "what is the typical retained *minute*",
    which is not the quantity anyone wants -- retained minutes are unevenly
    spread across days and expiries, so it silently weights whichever sessions
    happened to survive the filters most often.
    """
    return float(atm.groupby("session")["ratio"].median().median())


@dataclass(frozen=True)
class VolFit:
    """What the observed chain does and does not tell us about the vol model."""

    vrp_mult: float
    """Median of per-session medians of ``implied_iv / realized_vol``.

    One vote per trading day, not per minute bar -- see :func:`_equal_day_median`
    and the ``n_sessions`` field, which is the honest sample size.
    """

    n_atm_bars: int
    n_bars: int
    n_sessions: int
    """Trading days contributing at least one near-ATM bar. **This, not
    ``n_atm_bars``, is the effective sample size.**"""

    n_expiries: int
    """Distinct weekly contracts represented. Sessions within one cycle share a
    contract, so this bounds the independent information even below
    ``n_sessions``."""

    median_implied_iv: float
    median_realized_vol: float
    vrp_bar_weighted: float
    """The old bar-level estimator, kept only for comparison."""

    vrp_by_expiry: float
    ci_low: float
    ci_high: float
    """95% band from a bootstrap resampling whole **sessions**, which respects
    the within-session autocorrelation instead of assuming minutes are draws."""

    by_right: dict[str, float]
    by_tau_bucket: dict[str, float]
    vrp_in_sample: float
    vrp_out_of_sample: float
    n_sessions_in: int
    n_sessions_out: int

    skew_beta: None = None
    """Always ``None``. One strike per week cannot identify a smile -- see the
    module docstring. Left in the dataclass so the absence is explicit rather
    than something a caller has to notice is missing."""

    def summary(self) -> str:
        rights = ", ".join(f"{k} {v:.3f}" for k, v in sorted(self.by_right.items()))
        buckets = ", ".join(
            f"{k} {v:.3f}" for k, v in sorted(self.by_tau_bucket.items())
        )
        return (
            f"vrp_mult = {self.vrp_mult:.3f}  "
            f"[95% CI {self.ci_low:.3f}-{self.ci_high:.3f}]\n"
            f"  estimator  : median of per-session medians (one vote per day)\n"
            f"  sample     : {self.n_sessions} sessions / {self.n_expiries} weekly "
            f"expiries -- NOT {self.n_atm_bars:,} independent bars. The "
            f"{self.n_atm_bars:,} near-ATM rows (of {self.n_bars:,} solvable) are "
            "minutes sharing one realized-vol denominator per day.\n"
            f"  levels     : median implied IV {self.median_implied_iv:.1%} / "
            f"realized vol {self.median_realized_vol:.1%}\n"
            f"  reweighted : bar-weighted {self.vrp_bar_weighted:.3f}, "
            f"per-expiry {self.vrp_by_expiry:.3f}\n"
            f"  by right   : {rights}\n"
            f"  by tau     : {buckets}\n"
            f"  in/out     : {self.vrp_in_sample:.3f} on the first "
            f"{self.n_sessions_in} sessions vs {self.vrp_out_of_sample:.3f} on the "
            f"{self.n_sessions_out} held-out later sessions\n"
            "skew_beta = not identifiable from this data (one strike per "
            "weekly cycle); the configured value stands unvalidated."
        )


def fit_vol_params(
    bars: pd.DataFrame,
    vol: VolModel,
    calendar: ExpiryCalendar | None = None,
    chain: pd.DataFrame | None = None,
) -> VolFit:
    """Estimate ``vrp_mult`` from observed quotes. Does not estimate skew.

    The estimator is a median of per-session medians rather than a median over
    minute bars. Both are robust, but they answer different questions and the
    bar-level one overstates its own precision by three orders of magnitude:
    the retained rows are one-minute quotes whose ratio autocorrelates within a
    session and which share a single daily realized-vol denominator, so they
    are nothing like independent observations.
    """
    table = implied_vols(bars, vol, calendar, chain)
    if table.empty:
        raise ValueError("no observed bars yielded a solvable implied vol")

    atm = table[table["log_moneyness"].abs() < ATM_LOG_MONEYNESS].copy()
    if atm.empty:
        raise ValueError(
            "no near-ATM bars in the overlap; cannot identify the ATM vol level"
        )
    atm["ratio"] = atm["implied_iv"] / atm["realized_vol"]

    daily = atm.groupby("session")["ratio"].median()
    cycles = np.sort(atm["expiry"].unique())
    # Hold back the latest cycles entirely, so the reported in/out gap is a
    # genuine forward check rather than a resampling of the same weeks.
    split = max(len(cycles) - OOS_EXPIRY_CYCLES, 1)
    in_sample = atm[atm["expiry"] <= cycles[split - 1]]
    out_sample = atm[atm["expiry"] > cycles[split - 1]]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sessions = daily.index.to_numpy()
    draws = [
        np.median(daily.loc[rng.choice(sessions, len(sessions), replace=True)])
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])

    trading_days = (atm["tau"] * TRADING_DAYS_PER_YEAR).round()
    tau_medians = atm.groupby(trading_days)["ratio"].median()
    by_tau = {
        f"{int(days)}d": float(value)
        for days, value in zip(
            tau_medians.index.astype("int64").tolist(),
            tau_medians.to_numpy(),
            strict=True,
        )
    }
    return VolFit(
        vrp_mult=_equal_day_median(atm),
        n_atm_bars=len(atm),
        n_bars=len(table),
        n_sessions=int(atm["session"].nunique()),
        n_expiries=int(atm["expiry"].nunique()),
        median_implied_iv=float(atm["implied_iv"].median()),
        median_realized_vol=float(atm["realized_vol"].median()),
        vrp_bar_weighted=float(atm["ratio"].median()),
        vrp_by_expiry=float(atm.groupby("expiry")["ratio"].median().median()),
        ci_low=float(low),
        ci_high=float(high),
        by_right={
            str(k): float(v) for k, v in atm.groupby("right")["ratio"].median().items()
        },
        by_tau_bucket=by_tau,
        vrp_in_sample=_equal_day_median(in_sample) if len(in_sample) else float("nan"),
        vrp_out_of_sample=(
            _equal_day_median(out_sample) if len(out_sample) else float("nan")
        ),
        n_sessions_in=int(in_sample["session"].nunique()),
        n_sessions_out=int(out_sample["session"].nunique()),
    )
