"""Hold the synthetic vol model up against real observed option quotes.

    uv run python scripts/calibrate.py

Prints the fitted variance risk premium and the residual pricing error by right
and by time to expiry. This is the report that decides whether ``vrp_mult`` in
:class:`obt.vol.plugins.gk_vrp.VolParams` is still the right number.
"""

from __future__ import annotations

import pandas as pd

from obt import calibration, session
from obt.datasource import get_source
from obt.vol import get_vol_model

OHLC = ["open", "high", "low", "close"]


def spliced_spot() -> pd.DataFrame:
    """Five-year history extended with the 2026 vendor feed.

    The vol model needs weeks of history before its EWMA means anything, so the
    observed window cannot be evaluated on its own data alone. Splicing is
    defensible only because the feeds agree on their 37-session overlap --
    :func:`tests.test_calibration.test_the_two_spot_feeds_agree_on_their_overlap`
    is what keeps that true.
    """
    history = session.clean(get_source("nifty_csv").load("NIFTY"))
    recent = session.clean(get_source("nifty_index_csv").load("NIFTY"))
    tail = recent.loc[recent.index > history.index.max(), OHLC]
    return session.add_session_cols(pd.concat([history[OHLC], tail]))


def main() -> int:
    if not calibration.chain_available():
        print(
            "observed option chain CSVs not found. Set "
            f"${calibration.ENV_VAR} to the directory holding them."
        )
        return 2

    bars = spliced_spot()
    model = get_vol_model("gk_vrp")

    print("VOL MODEL CALIBRATION")
    print(f"  model    : {model.label}")
    print(f"  spot     : {bars.index.min().date()} -> {bars.index.max().date()}")
    print()

    fit = calibration.fit_vol_params(bars, model)
    print(fit.summary())
    print()

    comparison = calibration.compare_to_observed(bars, model)
    print(f"RESIDUAL PRICING ERROR ({len(comparison):,} observed bars)")
    for right, group in comparison.groupby("right"):
        print(
            f"  {right:<5}: median {group['rel_error'].median():+7.2%} "
            f"| MAE Rs{group['error'].abs().mean():6.1f} "
            f"| corr {group['observed'].corr(group['model']):.4f}"
        )
    print()

    print("MEDIAN RELATIVE ERROR BY TRADING DAYS TO EXPIRY")
    days = (comparison["tau"] * 252).round().clip(upper=6)
    table = comparison.groupby(["right", days])["rel_error"].median().unstack()
    print(table.to_string(float_format=lambda v: f"{v:+.1%}"))
    print()
    print(
        "Reminder: only the ATM *level* is identified here. One strike per\n"
        "weekly cycle cannot measure skew, so the put column carries an\n"
        "unvalidated `skew_beta` and should not be read as a model error."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
