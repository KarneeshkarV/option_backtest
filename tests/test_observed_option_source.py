"""Observed option-chain data source and engine path (no Black-76).

Skipped when the ATM CE/PE CSVs are absent so machines with only the
five-year spot file still pass the suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obt import calibration, engine, session
from obt.calendar import ExpiryCalendar
from obt.chain import LegSpec, pinned_leg_from_observed_chain
from obt.datasource import (
    get_option_source,
    get_source,
    normalize_option_chain,
    option_source_names,
)
from obt.session import IST

pytestmark = pytest.mark.skipif(
    not calibration.chain_available(),
    reason="observed option chain CSVs not present",
)


@pytest.fixture(scope="module")
def index_bars() -> pd.DataFrame:
    return session.clean(get_source("nifty_index_csv").load("NIFTY"))


@pytest.fixture(scope="module")
def observed_chain():
    return get_option_source("nifty_atm_options_csv").load("NIFTY")


def test_nifty_atm_options_csv_is_registered():
    assert "nifty_atm_options_csv" in option_source_names()


def test_observed_chain_source_matches_calibration_loader(observed_chain):
    via_calibration = calibration.load_observed_chain()
    assert set(observed_chain.columns) == {"right", "strike", "expiry", "premium"}
    assert observed_chain.index.equals(via_calibration.index)
    pd.testing.assert_frame_equal(
        observed_chain.reset_index(drop=True),
        via_calibration.reset_index(drop=True),
    )


def test_normalize_option_chain_maps_ce_pe_and_close():
    raw = pd.DataFrame(
        {
            "datetime": pd.DatetimeIndex(
                ["2026-04-22 09:15:00", "2026-04-22 09:15:00"], tz=IST
            ),
            "option_type": ["CE", "PE"],
            "strike": [24450, 24450],
            "expiry": ["2026-04-28", "2026-04-28"],
            "close": [100.0, 90.0],
        }
    )
    out = normalize_option_chain(raw)
    assert list(out["right"]) == ["call", "put"]
    assert list(out["premium"]) == [100.0, 90.0]


def test_engine_run_with_observed_premiums_is_reproducible(index_bars):
    r1 = engine.run(index_bars, "orb", option_source="nifty_atm_options_csv")
    r2 = engine.run(index_bars, "orb", option_source="nifty_atm_options_csv")
    assert r1.stats == r2.stats
    assert len(r1.trades) > 0
    assert r1.assumptions["premiums"] == "observed"
    assert r1.assumptions["option_source"] == "nifty_atm_options_csv"
    assert r1.assumptions["vol_model"] == "n/a (observed premiums)"
    assert any("OBSERVED" in w for w in r1.warnings)


def test_observed_entry_premium_matches_csv_quote(index_bars, observed_chain):
    """Fills must use the traded close, not a model price."""
    result = engine.run(index_bars, "orb", option_source="nifty_atm_options_csv")
    assert len(result.trades) > 0

    # First trade: look up the premium vectorbt used at entry.
    row = result.trades.iloc[0]
    entry_ts = pd.Timestamp(row["entry_time"])
    entry_price = float(row["entry_price"])

    # Recover which right from the leg frame column that is open at entry.
    frame = result.leg_frame
    # MultiIndex columns: (leg_label, field)
    premium_cols = [c for c in frame.columns if c[1] == "premium"]
    matched = False
    for leg_label, _ in premium_cols:
        prem = float(frame[(leg_label, "premium")].loc[entry_ts])
        # Slippage is applied by vectorbt on top of the close series; the
        # series value itself must equal the CSV quote (before slippage).
        right = "call" if "call" in leg_label else "put"
        quotes = observed_chain.loc[
            (observed_chain.index == entry_ts) & (observed_chain["right"] == right),
            "premium",
        ]
        if quotes.empty:
            continue
        quote = float(quotes.iloc[0])
        assert prem == pytest.approx(max(quote, 0.05), rel=0, abs=1e-9)
        # Entry fill moves against the trader by slippage; series is pre-slip.
        slipped = max(quote, 0.05) * 1.0075
        near_quote = entry_price == pytest.approx(quote, rel=0.02)
        near_slipped = entry_price == pytest.approx(slipped, rel=1e-3)
        assert near_quote or near_slipped
        matched = True
        break
    assert matched, "could not match first trade entry to an observed quote"


def test_observed_path_rejects_non_atm_strike_rule(index_bars, observed_chain):
    calendar = ExpiryCalendar.from_bars(index_bars)
    open_mask = np.zeros(len(index_bars), dtype=bool)
    open_mask[20] = True
    leg = LegSpec(right="call", direction="long", strike_rule="otm:1")
    with pytest.raises(ValueError, match="only supports strike_rule='atm'"):
        pinned_leg_from_observed_chain(
            index_bars, open_mask, leg, observed_chain, calendar
        )


def test_observed_and_synthetic_paths_differ(index_bars):
    """Sanity: observed P&L is not accidentally still Black-76."""
    observed = engine.run(index_bars, "orb", option_source="nifty_atm_options_csv")
    synthetic = engine.run(index_bars, "orb")
    # Same strategy on the same spot window should not produce bit-identical
    # returns once the premium source changes (unless by extreme coincidence).
    assert observed.stats["total_return_pct"] != synthetic.stats["total_return_pct"]
    assert observed.assumptions["premiums"] == "observed"
    assert synthetic.assumptions["premiums"] == "black76"


def test_synthetic_path_unchanged_without_option_source(index_bars):
    """Default engine.run still uses Black-76 (no accidental wire-up)."""
    result = engine.run(index_bars, "orb")
    assert result.assumptions.get("premiums", "black76") == "black76"
    assert result.assumptions["option_source"] is None
    assert any("Black-76" in w for w in result.warnings)
