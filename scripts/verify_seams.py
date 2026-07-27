"""Prove the three seams actually swap, rather than merely looking like they do.

    uv run python scripts/verify_seams.py

A pluggable architecture is a claim, and claims about code should be executable.
Each check swaps exactly one component and states what must and must not change:

1. **Data source.** Dump the CSV to parquet, load it back through a different
   source class, run the same backtest. Every statistic must be *identical* --
   the source may change how bytes arrive, never what they mean.
2. **Strategy.** A second strategy must run with no edits outside
   ``strategies/plugins/``.
3. **Vol model.** Swapping ``gk_vrp`` for ``constant`` must change premiums (so
   the model is genuinely load-bearing) while leaving the trade *schedule*
   untouched -- entries come from spot, and no vol model may influence when a
   strategy trades.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from obt import engine, session
from obt.datasource import get_source
from obt.datasource.plugins.parquet_spot import write_parquet
from obt.strategies import strategy_names
from obt.vol import get_vol_model, vol_model_names

OHLC = ["open", "high", "low", "close"]
STATS = ("total_return_pct", "sharpe", "max_drawdown_pct", "trades", "win_rate_pct")


def report(name: str, passed: bool, detail: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def check_data_source_seam(bars) -> bool:
    print("1. DATA SOURCE -- csv vs parquet must be indistinguishable")
    with tempfile.TemporaryDirectory() as tmp:
        path = write_parquet(bars[OHLC], Path(tmp) / "spot.parquet")
        reloaded = session.clean(get_source("parquet_spot", path=path).load("NIFTY"))

    same_shape = reloaded[OHLC].equals(bars[OHLC])
    ok = report(
        "round-trip",
        same_shape,
        f"{len(reloaded):,} bars, frames equal={same_shape}",
    )

    csv_stats = engine.run(bars, "orb").stats
    parquet_stats = engine.run(reloaded, "orb").stats
    identical = all(
        csv_stats[key] == parquet_stats[key]
        for key in STATS
        if not (isinstance(csv_stats[key], float) and np.isnan(csv_stats[key]))
    )
    return (
        report(
            "backtest equality",
            identical,
            f"return {csv_stats['total_return_pct']:.4f}% via csv, "
            f"{parquet_stats['total_return_pct']:.4f}% via parquet",
        )
        and ok
    )


def check_strategy_seam(bars) -> bool:
    print("\n2. STRATEGY -- a second plugin runs with no engine changes")
    names = strategy_names()
    ok = report("registry", len(names) >= 2, f"registered: {names}")

    results = {name: engine.run(bars, name) for name in ("orb", "buy_open")}
    distinct = (
        results["orb"].stats["total_return_pct"]
        != results["buy_open"].stats["total_return_pct"]
    )
    for name, result in results.items():
        print(
            f"         {name:<9} {result.stats['trades']:>4} trades, "
            f"{result.stats['total_return_pct']:+8.2f}%"
        )
    return report("independent results", distinct, "strategies differ") and ok


def check_vol_seam(bars) -> bool:
    print("\n3. VOL MODEL -- swapping it changes premiums, never the timing")
    ok = report("registry", len(vol_model_names()) >= 2, f"{vol_model_names()}")

    gk = engine.run(bars, "orb", vol="gk_vrp")
    flat = engine.run(bars, "orb", vol="constant")

    # The signals come from spot, so a vol model can never move a trade to a
    # different bar. It can, legitimately, remove bars it refuses to price at
    # all: gk_vrp emits no IV until it is warm within a covered block, while
    # `constant` is warm everywhere. So the honest invariant is not "identical
    # schedules" but "gk_vrp's trades are a subset of `constant`'s, landing on
    # exactly the same timestamps wherever both can price".
    gk_schedule = gk.trades[["entry_time", "exit_time"]].reset_index(drop=True)
    flat_schedule = flat.trades[["entry_time", "exit_time"]]
    shared = flat_schedule[
        flat_schedule["entry_time"].isin(gk_schedule["entry_time"])
    ].reset_index(drop=True)
    same_schedule = gk_schedule.equals(shared)
    ok = (
        report(
            "trade timing unchanged where both models price",
            same_schedule,
            f"{len(gk.trades):,} of {len(flat.trades):,} trades survive "
            "gk_vrp's warmup, on identical entry/exit timestamps",
        )
        and ok
    )

    gk_premium = gk.leg_frame.xs("premium", axis=1, level=1).to_numpy()
    flat_premium = flat.leg_frame.xs("premium", axis=1, level=1).to_numpy()
    changed = not np.allclose(gk_premium, flat_premium)
    detail = (
        f"median premium Rs{np.nanmedian(gk_premium):.1f} "
        f"({get_vol_model('gk_vrp').label}) vs "
        f"Rs{np.nanmedian(flat_premium):.1f} ({get_vol_model('constant').label})"
    )
    return report("premiums changed", changed, detail) and ok


def main() -> int:
    bars = session.clean(get_source("nifty_csv").load("NIFTY"))
    print(
        f"SEAM VERIFICATION over {len(bars):,} bars, "
        f"{bars['date'].nunique():,} sessions\n"
    )
    checks = [
        check_data_source_seam(bars),
        check_strategy_seam(bars),
        check_vol_seam(bars),
    ]
    print()
    if all(checks):
        print("All seams verified.")
        return 0
    print("SEAM VERIFICATION FAILED -- the architecture is not doing what it claims.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
