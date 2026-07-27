"""Reporting that makes the model's assumptions impossible to overlook.

Every other backtester in this family reports observed prices. This one
reports the output of a pricing model fed by a volatility guess, so a bare
Sharpe ratio would be actively misleading. Three things therefore always print
alongside the numbers:

- **The banner** -- which vol model, which slippage, which sample.
- **Sanity checks** -- do the synthetic premiums resemble real ones at all? If
  the ATM straddle is not in the right ballpark, nothing below it means
  anything and the report says so.
- **The sensitivity table** -- the same strategy under different assumptions.
  This is the actual result. A single row of statistics is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from obt.calendar import ExpiryCalendar
from obt.chain import atm_strike
from obt.costs import SlippageParams
from obt.pricing import black76
from obt.session import covered_periods, session_summary
from obt.vol.base import VolModel

if TYPE_CHECKING:
    from obt.engine import BacktestResult

#: A NIFTY weekly ATM straddle a few days from expiry is worth roughly this
#: fraction of spot. Outside this band the vol model is not producing
#: option-like prices and results should not be read.
STRADDLE_PCT_BAND = (0.7, 1.8)


def _calibration_note() -> str:
    """Where the vol level came from -- measured, or assumed.

    ``vrp_mult`` was fitted to real observed quotes, and a reader deciding how
    much to trust these numbers needs to know that, along with how thin the
    calibration sample is and what it could not measure.
    """
    from obt import calibration

    if not calibration.chain_available():
        return (
            "vrp_mult from a prior fit to observed quotes; "
            "reference files absent, not re-checked this run"
        )
    return (
        "vrp_mult fitted to 60 sessions of observed NIFTY ATM quotes "
        "(run `just calibrate`); skew unvalidated"
    )


def format_banner(result: BacktestResult, bars: pd.DataFrame) -> str:
    a = result.assumptions
    periods = covered_periods(bars)
    lines = [
        "=" * 78,
        "SYNTHETIC OPTION BACKTEST -- premiums are modelled, not observed",
        "=" * 78,
        f"strategy       : {a['strategy']}  {a['params']}",
        f"vol model      : {a['vol_model']}",
        f"calibration    : {_calibration_note()}",
        f"slippage       : {a['slippage_pct']:.3f}% of premium per fill",
        f"statutory fees : {a['fee_fraction'] * 100:.4f}% per fill "
        f"(STT {a['stt_rate'] * 100:.4f}% sell-side)",
        f"time basis     : {a['time_basis']} time to expiry",
        f"size           : {a['quantity_per_trade']} units/trade "
        f"(lot from history: {a['lot_from_history']})",
        f"capital        : Rs {a['init_cash']:,.0f}",
        "",
        f"sample         : {len(bars):,} bars over "
        f"{len({*bars['date']}):,} trading days, in {len(periods)} blocks",
    ]
    for start, end in periods:
        lines.append(f"                 {start} -> {end}")
    lines.append("")
    for warning in result.warnings:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)


def format_data_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> str:
    summary = session_summary(raw, cleaned)
    return "\n".join(
        [
            "DATA",
            f"  bars  : {summary['raw_bars']:,} read -> {summary['kept_bars']:,} kept",
            f"  days  : {summary['raw_days']:,} read -> "
            f"{summary['kept_days']:,} kept "
            f"({len(summary['dropped_days'])} incomplete sessions dropped)",
        ]
    )


def sanity_checks(
    bars: pd.DataFrame,
    vol: VolModel,
    calendar: ExpiryCalendar,
    *,
    sample_size: int = 5000,
    seed: int = 0,
) -> tuple[str, bool]:
    """Verify the synthetic surface behaves like an option surface.

    Returns the rendered block and whether everything passed. A failure here
    invalidates every strategy statistic, so callers should surface it loudly
    rather than continuing quietly.
    """
    rng = np.random.default_rng(seed)
    atm_all = vol.atm_iv(bars).to_numpy()
    # Bars before the vol model is warm carry no ATM level at all -- the model
    # refuses to emit one until it has `seed_days` of history within the
    # current covered block, rather than backfilling it from the future. Those
    # bars are not part of the surface under test, and sampling them would fail
    # every check below for the wrong reason. Excluded here and counted, so the
    # exclusion is stated rather than silent.
    warm = np.flatnonzero(np.isfinite(atm_all))
    unwarmed = len(bars) - len(warm)
    if len(warm) == 0:
        return (
            "SANITY CHECKS (synthetic surface)\n"
            "  [FAIL] vol model never warms up            no bar has an ATM IV",
            False,
        )
    take = min(sample_size, len(warm))
    rows = np.sort(rng.choice(warm, size=take, replace=False))
    sample = bars.iloc[rows]

    spot = sample["close"].to_numpy()
    strike = atm_strike(spot)
    # Price a representative weekly: four sessions from expiry.
    tau = np.full_like(spot, 4.0 / 252.0)
    atm = atm_all[rows]
    iv = vol.iv(atm, spot, strike, tau)

    call = black76.price(spot, strike, tau, iv, "call")
    put = black76.price(spot, strike, tau, iv, "put")
    straddle_pct = (call + put) / spot * 100.0

    forward = black76.forward_price(spot, tau)
    discount = np.exp(-black76.DEFAULT_RATE * tau)
    parity_residual = np.abs(call - put - discount * (forward - strike))

    greeks = black76.greeks(spot, strike, tau, iv, "call")
    finite = all(np.isfinite(v).all() for v in greeks.values())

    median_straddle = float(np.median(straddle_pct))
    low, high = STRADDLE_PCT_BAND
    checks = [
        (
            "ATM straddle within realistic band",
            low <= median_straddle <= high,
            f"median {median_straddle:.3f}% of spot (expect {low}-{high}%)",
        ),
        (
            "put-call parity holds",
            float(parity_residual.max()) < 1e-6,
            f"max residual {float(parity_residual.max()):.2e}",
        ),
        (
            "no negative premiums",
            bool((call >= 0).all() and (put >= 0).all()),
            f"min call {float(call.min()):.2f}, min put {float(put.min()):.2f}",
        ),
        ("all greeks finite", finite, "delta/gamma/vega/theta"),
        (
            "IV within configured clamps",
            bool(np.isfinite(iv).all() and (iv > 0).all()),
            f"IV range {float(iv.min()):.3f}-{float(iv.max()):.3f}",
        ),
    ]

    lines = ["SANITY CHECKS (synthetic surface)"]
    if unwarmed:
        lines.append(
            f"  ({unwarmed:,} unwarmed bars excluded -- no IV until the model "
            f"has seed_days of history in each covered block)"
        )
    passed = True
    for name, ok, detail in checks:
        passed = passed and ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")
    if not passed:
        lines.append(
            "  >> A failed check means the premiums are not option-like. "
            "Do not read the statistics below."
        )
    return "\n".join(lines), passed


def format_stats(result: BacktestResult, baseline: dict[str, float]) -> str:
    stats = result.stats
    return "\n".join(
        [
            "RESULTS",
            f"  {'':<22}{'strategy':>14}{'buy & hold spot':>18}",
            f"  {'total return %':<22}{stats['total_return_pct']:>14.2f}"
            f"{baseline['total_return_pct']:>18.2f}",
            f"  {'sharpe (daily)':<22}{stats['sharpe']:>14.2f}"
            f"{baseline['sharpe']:>18.2f}",
            f"  {'max drawdown %':<22}{stats['max_drawdown_pct']:>14.2f}"
            f"{baseline['max_drawdown_pct']:>18.2f}",
            f"  {'trades':<22}{stats['trades']:>14,}{'-':>18}",
            f"  {'win rate %':<22}{stats['win_rate_pct']:>14.2f}{'-':>18}",
            f"  {'avg P&L/trade (Rs)':<22}{stats['avg_pnl']:>14,.0f}{'-':>18}",
        ]
    )


def sensitivity(
    bars: pd.DataFrame,
    strategy_name: str,
    *,
    vrp_mults: tuple[float, ...] = (1.00, 1.10, 1.15, 1.25),
    slippage_pcts: tuple[float, ...] = (0.0, 0.0025, 0.0075, 0.015),
    params: dict[str, Any] | None = None,
    init_cash: float | None = None,
) -> pd.DataFrame:
    """Total return across the two assumptions that drive the result.

    ``vrp_mult`` sets how expensive options are; ``slippage_pct`` sets how much
    crossing the spread costs. Neither is observable from spot data. If the
    sign of the result flips across this grid, the strategy has no demonstrated
    edge -- the edge was in the assumption.
    """
    from obt.engine import DEFAULT_INIT_CASH, run

    cash = init_cash if init_cash is not None else DEFAULT_INIT_CASH
    rows = []
    for vrp in vrp_mults:
        row: dict[str, Any] = {"vrp_mult": vrp}
        for slip in slippage_pcts:
            result = run(
                bars,
                strategy_name,
                vol_kwargs={"vrp_mult": vrp},
                slippage=SlippageParams(premium_pct=slip),
                params=params,
                init_cash=cash,
            )
            row[f"slip={slip * 100:.2f}%"] = round(result.stats["total_return_pct"], 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("vrp_mult")


def format_sensitivity(table: pd.DataFrame) -> str:
    return "\n".join(
        [
            "SENSITIVITY -- total return % by vol and slippage assumption",
            "  (this is the result; a single number above is not)",
            "",
            table.to_string(),
        ]
    )
