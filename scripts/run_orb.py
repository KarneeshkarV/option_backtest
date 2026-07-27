"""Run the reference ORB strategy end to end and print a full report.

    uv run python scripts/run_orb.py [strategy] [flags]

Flags: ``--quick``, ``--no-sensitivity``, ``--observed``.

``--quick`` restricts the sample to the final contiguous block, which makes
iteration fast; the sensitivity grid is 16 full backtests and takes a while
over the whole sample.

``--observed`` prices fills from the local ATM CE/PE CSVs
(``option_source=nifty_atm_options_csv``) instead of Black-76. Spot comes from
the overlapping 2026 vendor index feed, since that is the window the option
files cover. Sensitivity is skipped (it only varies the synthetic vol model).
"""

from __future__ import annotations

import sys

from obt import engine, report, session
from obt.calendar import ExpiryCalendar
from obt.datasource import get_option_source, get_source, option_source_names
from obt.strategies import strategy_names
from obt.vol import get_vol_model


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    strategy = args[0] if args else "orb"

    if strategy not in strategy_names():
        print(f"unknown strategy {strategy!r}; known: {strategy_names()}")
        return 2

    observed = "--observed" in flags
    if observed:
        # Option quotes only exist on the 2026 vendor window; pair them with
        # that window's spot feed so timestamps line up exactly.
        raw = get_source("nifty_index_csv").load("NIFTY")
        bars = session.clean(raw)
        option_source = "nifty_atm_options_csv"
        if option_source not in option_source_names():
            print(f"option source {option_source!r} is not registered")
            return 2
        # Fail early with a clear message if the CSVs are missing.
        get_option_source(option_source).load("NIFTY")
        result = engine.run(bars, strategy, option_source=option_source)
    else:
        raw = get_source("nifty_csv").load("NIFTY")
        bars = session.clean(raw)
        if "--quick" in flags:
            last_start, _ = session.covered_periods(bars)[-1]
            bars = bars.loc[bars["date"] >= last_start]
        result = engine.run(bars, strategy)

    print(report.format_banner(result, bars))
    print()
    print(report.format_data_summary(raw, bars))
    print()

    passed = True
    if not observed:
        vol = get_vol_model("gk_vrp")
        calendar = ExpiryCalendar.from_bars(bars)
        checks, passed = report.sanity_checks(bars, vol, calendar)
        print(checks)
        print()
    else:
        print("SANITY CHECKS skipped -- observed premiums, not a synthetic surface")
        print()

    print(report.format_stats(result, engine.buy_and_hold_spot(bars)))
    print()

    if not observed and "--no-sensitivity" not in flags:
        table = report.sensitivity(bars, strategy)
        print(report.format_sensitivity(table))
        print()

    if not passed:
        print("Sanity checks failed -- results above are not trustworthy.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
