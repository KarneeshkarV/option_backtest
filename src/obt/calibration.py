"""Measure the synthetic vol model against real observed option quotes.

Everything else in this package prices options from a model. This module is the
only place that reads prices somebody actually paid, and its whole job is to
answer one question: *how wrong are we?*

The reference data is 60 sessions of 1-minute NIFTY ATM CE/PE quotes
(2026-04-22 to 2026-07-21). It is small, but it is real, and it converts the two
load-bearing guesses in :mod:`obt.vol.plugins.gk_vrp` into measurements:

- ``vrp_mult`` **is identified** by this data. Inverting 7,108 near-ATM bars to
  implied vol and dividing by the model's realized-vol input gives 1.31.
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

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from obt.calendar import ExpiryCalendar, tau_years
from obt.datasource.plugins.nifty_index_csv import read_naive_ist_csv
from obt.pricing import black76
from obt.vol.base import VolModel

ENV_VAR = "OBT_OPTION_CHAIN_DIR"

#: ``right -> filename``. The vendor also ships a combined
#: ``NIFTY_ATM_options_1min_*.csv``; it is byte-for-byte ``concat(CE, PE)``, so
#: reading it as well would double-count every bar.
CHAIN_FILES = {
    "call": "NIFTY_ATM_CE_1min_2026-04-22_2026-07-21.csv",
    "put": "NIFTY_ATM_PE_1min_2026-04-22_2026-07-21.csv",
}

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
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2]


def chain_available() -> bool:
    """Whether the observed-quote files are present on this machine."""
    directory = chain_dir()
    return all((directory / name).exists() for name in CHAIN_FILES.values())


def load_observed_chain(directory: Path | None = None) -> pd.DataFrame:
    """Observed CE and PE quotes, one row per bar per right.

    Returns columns ``right, strike, expiry, premium`` on a tz-aware IST index,
    with ``right`` values matching :mod:`obt.pricing.black76`.
    """
    directory = directory or chain_dir()
    frames = []
    for right, filename in CHAIN_FILES.items():
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"observed chain file not found at {path}. "
                f"Set ${ENV_VAR} to the directory holding them."
            )
        raw = read_naive_ist_csv(path)
        frames.append(
            pd.DataFrame(
                {
                    "right": right,
                    "strike": raw["strike"].astype("float64").to_numpy(),
                    "expiry": pd.to_datetime(raw["expiry"]).dt.date.to_numpy(),
                    "premium": raw["close"].astype("float64").to_numpy(),
                },
                index=pd.DatetimeIndex(raw["datetime"], name="datetime"),
            )
        )
    # `read_naive_ist_csv` has already localized the timestamps to IST, which
    # matters: these bars are matched against spot by exact timestamp, so a
    # missing offset would shift every one of them by 5h30m and silently align
    # each quote with the wrong bar.
    return pd.concat(frames).sort_index()


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
        },
        index=usable.index,
    )
    return out.dropna()


@dataclass(frozen=True)
class VolFit:
    """What the observed chain does and does not tell us about the vol model."""

    vrp_mult: float
    """Median ``implied_iv / realized_vol`` on near-ATM bars. Identified."""

    n_atm_bars: int
    n_bars: int
    median_implied_iv: float
    median_realized_vol: float

    skew_beta: None = None
    """Always ``None``. One strike per week cannot identify a smile -- see the
    module docstring. Left in the dataclass so the absence is explicit rather
    than something a caller has to notice is missing."""

    def summary(self) -> str:
        return (
            f"vrp_mult = {self.vrp_mult:.3f} "
            f"(median implied IV {self.median_implied_iv:.1%} / realized vol "
            f"{self.median_realized_vol:.1%}, {self.n_atm_bars:,} near-ATM bars "
            f"of {self.n_bars:,} solvable)\n"
            "skew_beta = not identifiable from this data (one strike per "
            "weekly cycle); the configured value stands unvalidated."
        )


def fit_vol_params(
    bars: pd.DataFrame,
    vol: VolModel,
    calendar: ExpiryCalendar | None = None,
    chain: pd.DataFrame | None = None,
) -> VolFit:
    """Estimate ``vrp_mult`` from observed quotes. Does not estimate skew."""
    table = implied_vols(bars, vol, calendar, chain)
    if table.empty:
        raise ValueError("no observed bars yielded a solvable implied vol")

    atm = table[table["log_moneyness"].abs() < ATM_LOG_MONEYNESS]
    if atm.empty:
        raise ValueError(
            "no near-ATM bars in the overlap; cannot identify the ATM vol level"
        )
    return VolFit(
        vrp_mult=float((atm["implied_iv"] / atm["realized_vol"]).median()),
        n_atm_bars=len(atm),
        n_bars=len(table),
        median_implied_iv=float(atm["implied_iv"].median()),
        median_realized_vol=float(atm["realized_vol"].median()),
    )
