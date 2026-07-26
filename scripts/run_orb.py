"""Run the reference ORB strategy end to end and print a full report.

    uv run python scripts/run_orb.py [strategy] [--quick] [--no-sensitivity]

``--quick`` restricts the sample to the final contiguous block, which makes
iteration fast; the sensitivity grid is 16 full backtests and takes a while
over the whole sample.
"""

from __future__ import annotations

import sys

from obt import engine, report, session
from obt.calendar import ExpiryCalendar
from obt.datasource import get_source
from obt.strategies import strategy_names
from obt.vol import get_vol_model


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    strategy = args[0] if args else "orb"

    if strategy not in strategy_names():
        print(f"unknown strategy {strategy!r}; known: {strategy_names()}")
        return 2

    raw = get_source("nifty_csv").load("NIFTY")
    bars = session.clean(raw)
    if "--quick" in flags:
        last_start, _ = session.covered_periods(bars)[-1]
        bars = bars.loc[bars["date"] >= last_start]

    result = engine.run(bars, strategy)
    vol = get_vol_model("gk_vrp")
    calendar = ExpiryCalendar.from_bars(bars)

    print(report.format_banner(result, bars))
    print()
    print(report.format_data_summary(raw, bars))
    print()
    checks, passed = report.sanity_checks(bars, vol, calendar)
    print(checks)
    print()
    print(report.format_stats(result, engine.buy_and_hold_spot(bars)))
    print()

    if "--no-sensitivity" not in flags:
        table = report.sensitivity(bars, strategy)
        print(report.format_sensitivity(table))
        print()

    if not passed:
        print("Sanity checks failed -- results above are not trustworthy.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
