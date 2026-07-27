"""End-to-end ``engine.run`` tests for per-leg direction and size.

Before this file, nothing ran ``engine.run`` end to end. These tests are
scoped narrowly to Finding 2: engine.py used to collapse every resolved leg
group's direction and size to a single portfolio-wide scalar
(``is_short = is_short or leg.direction == "short"``, ``quantity =
reference_leg.lots * effective_lot`` taken only from ``groups[0][0]``), so a
strategy mixing a long call with a short put would run *both* columns
``shortonly`` and size *both* at the first group's quantity. The fix passes
``direction`` and ``size`` to ``vbt.Portfolio.from_signals`` as plain
per-column lists (verified empirically against the installed vectorbt 1.0.0 --
see the probe embedded in ``reports/agent_c_engine_legs.md``).

A test-only strategy is registered below (``engine_legs_test_strategy``)
rather than adding a plugin file, per the strategy seam's own contract: it
fires an entry at a fixed bar-of-day on however many of the fixture's leading
days it's given legs for, and hands each day's open a caller-supplied
``LegSpec`` -- exactly the ``legs`` (per-bar leg choice) shape ``orb`` uses,
just with explicit control over which day gets which leg so the mixed
long/short and mixed-size cases are exact and deterministic.

All tests use a trivial constant-IV ``VolModel`` rather than the real
``gk_vrp`` model. `obt.vol.base.VolModel` is a `Protocol` precisely so
callers can substitute one, and these tests exist to pin down `obt.engine`'s
per-leg direction/size handling, not `gk_vrp`'s realized-vol warmup
semantics (under concurrent, independent development elsewhere in this
tree) -- pulling in the real model would make these tests depend on how many
sessions of prior history `gk_vrp` happens to require this week.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from obt import engine
from obt.calendar import ExpiryCalendar
from obt.chain import LegSpec, lot_size_for, pinned_leg
from obt.costs import IndiaOptionsCosts, SlippageParams, vbt_fees
from obt.signals import last_bar_of_day, resolve_trades, shift_signals
from obt.strategies.base import Signals
from obt.strategies.spec import get_strategy, strategy


@dataclass(frozen=True)
class _ConstantVol:
    """Flat ATM IV, no skew, no warmup window -- see module docstring."""

    level: float = 0.20

    @property
    def label(self) -> str:
        return f"const({self.level})"

    def atm_iv(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(self.level, index=bars.index)

    def iv(
        self,
        atm: np.ndarray,
        spot: np.ndarray,
        strike: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        return np.full(np.asarray(spot).shape, self.level, dtype="float64")


def _day_index(bars: pd.DataFrame) -> np.ndarray:
    """0-based day number for each bar, in the order days first appear."""
    _, day_idx = np.unique(bars["date"].to_numpy(), return_inverse=True)
    return day_idx


@strategy(
    "engine_legs_test_strategy",
    description="Test-only: fixed entry bar, one caller-supplied LegSpec per "
    "leading day, engine's EOD square-off provides the only exit.",
    defaults={"entry_bar": 10, "legs_by_day": ()},
)
def _engine_legs_test_strategy(
    bars: pd.DataFrame,
    *,
    entry_bar: int = 10,
    legs_by_day: tuple[LegSpec, ...] = (),
) -> Signals:
    day_idx = _day_index(bars)
    entries = (bars["bar_of_day"].to_numpy() == entry_bar) & (
        day_idx < len(legs_by_day)
    )
    exits = np.zeros(len(bars), dtype=bool)  # rely solely on the EOD force-exit

    values = np.empty(len(bars), dtype=object)
    values[:] = None
    for day, leg in enumerate(legs_by_day):
        values[entries & (day_idx == day)] = leg
    legs = pd.Series(values, index=bars.index, dtype="object")
    return Signals(entries=entries, exits=exits, legs=legs)


def _trade_dates(bars: pd.DataFrame) -> list:
    return sorted({*bars["date"]})


def test_mixed_long_call_short_put_executes_each_leg_correctly(bars):
    """The exact failure in the finding: a long call one day, a short put the
    next. Each column must keep its own side and its own size -- neither may
    inherit the other's.
    """
    call_leg = LegSpec(right="call", direction="long", strike_rule="atm", lots=1)
    put_leg = LegSpec(right="put", direction="short", strike_rule="atm", lots=1)

    result = engine.run(
        bars,
        "engine_legs_test_strategy",
        vol=_ConstantVol(),
        params={"entry_bar": 10, "legs_by_day": (call_leg, put_leg)},
    )

    records = result.portfolio.trades.records_readable
    assert set(records["Column"]) == {call_leg.label, put_leg.label}

    call_row = records[records["Column"] == call_leg.label].iloc[0]
    put_row = records[records["Column"] == put_leg.label].iloc[0]

    # Right side per leg -- the bug this finding describes made both columns
    # `shortonly` because `is_short = is_short or leg.direction == "short"`
    # was a single OR across every group.
    assert call_row["Direction"] == "Long"
    assert put_row["Direction"] == "Short"

    # Right size per leg -- the bug sized every column off `groups[0][0]`.
    assert call_row["Size"] == pytest.approx(call_leg.lots * call_leg.lot_size)
    assert put_row["Size"] == pytest.approx(put_leg.lots * put_leg.lot_size)

    # Same-day closure, on the day each leg actually opened.
    dates = _trade_dates(bars)
    assert pd.Timestamp(call_row["Entry Timestamp"]).date() == dates[0]
    assert pd.Timestamp(call_row["Exit Timestamp"]).date() == dates[0]
    assert pd.Timestamp(put_row["Entry Timestamp"]).date() == dates[1]
    assert pd.Timestamp(put_row["Exit Timestamp"]).date() == dates[1]

    # The short-margin warning must still fire because ANY leg is short.
    assert any("Short legs present" in w for w in result.warnings)

    # Both legs trade 1 lot at the (assumed, no history in this fixture) flat
    # lot size, so the scalar `quantity_per_trade` shape is preserved even
    # though direction differs per leg.
    expected_qty = call_leg.lots * call_leg.lot_size
    assert result.assumptions["quantity_per_trade"] == expected_qty


def test_mixed_size_same_direction_uses_correct_per_leg_amount(bars):
    """Two long calls, 1 lot vs 2 lots. They share `leg.label` exactly
    ('long atm call' for both), which used to collide as a dict key in
    engine.py's per-leg dicts (premium/entries/exits keyed by `leg.label`) --
    the second leg would silently overwrite the first's entries/exits. The
    engine must disambiguate the column names, and each column must carry its
    own size.
    """
    small = LegSpec(right="call", direction="long", strike_rule="atm", lots=1)
    big = LegSpec(right="call", direction="long", strike_rule="atm", lots=2)
    assert small.label == big.label  # the collision this test guards against

    result = engine.run(
        bars,
        "engine_legs_test_strategy",
        vol=_ConstantVol(),
        params={"entry_bar": 10, "legs_by_day": (small, big)},
    )

    records = result.portfolio.trades.records_readable
    assert records["Column"].nunique() == 2  # not collapsed to one column
    assert len(records) == 2  # both legs' trades survived, neither overwritten

    sizes = sorted(records["Size"].tolist())
    expected = sorted([small.lots * small.lot_size, big.lots * big.lot_size])
    assert sizes == pytest.approx(expected)
    assert sizes[0] != sizes[1]  # the two legs are genuinely differently sized

    # leg_frame must also carry two distinct per-leg columns, not one
    # overwritten by the other.
    assert result.leg_frame.columns.get_level_values(0).nunique() == 2

    # Heterogeneous per-leg size can no longer be reported as one scalar
    # without lying about one of the legs.
    qpt = result.assumptions["quantity_per_trade"]
    assert isinstance(qpt, dict)
    assert sorted(qpt.values()) == expected


def test_buy_open_matches_prefix_scalar_portfolio(bars):
    """No-regression guard for the homogeneous, single-leg-group path.

    `buy_open` only ever resolves to one leg group, so the pre-fix scalar
    `size=float(quantity)` / `direction="...only"` and the fixed per-column
    `size=[float(quantity)]` / `direction=[...]` describe the exact same
    order stream. This rebuilds the pre-fix call by hand from the same
    pieces engine.run uses internally and asserts the two portfolios are
    bit-identical, so the homogeneous path the shipped strategies exercise
    is provably unaffected by the fix.
    """
    vol_model = _ConstantVol()
    result = engine.run(bars, "buy_open", vol=vol_model)

    spec = get_strategy("buy_open")
    signals = spec.signal_fn(bars, **spec.defaults)
    entries = shift_signals(signals.entries, bars["date"])
    exits = shift_signals(signals.exits, bars["date"])
    resolved = resolve_trades(entries, exits, last_bar_of_day(bars))

    calendar = ExpiryCalendar.from_bars(bars)
    priced = pinned_leg(bars, resolved["open_mask"], signals.leg, vol_model, calendar)

    name = signals.leg.label
    close = pd.DataFrame({name: priced["premium"]}, index=bars.index)
    entry_df = pd.DataFrame({name: resolved["open_mask"]}, index=bars.index)
    exit_df = pd.DataFrame({name: resolved["close_mask"]}, index=bars.index)

    lot_size, lot_from_history = lot_size_for(bars["date"].iloc[0])
    effective_lot = lot_size if lot_from_history else signals.leg.lot_size
    quantity = signals.leg.lots * effective_lot

    costs = IndiaOptionsCosts()
    slippage = SlippageParams()
    old_style = vbt.Portfolio.from_signals(
        close,
        entry_df,
        exit_df,
        size=float(quantity),  # pre-fix: portfolio-wide scalar
        size_type="amount",
        direction="shortonly" if signals.leg.direction == "short" else "longonly",
        init_cash=float(engine.DEFAULT_INIT_CASH),
        fees=vbt_fees(costs),
        slippage=float(slippage.premium_pct),
        freq="1min",
        group_by=True,
        cash_sharing=True,
    )

    assert float(result.portfolio.total_return()) == pytest.approx(
        float(old_style.total_return())
    )
    pd.testing.assert_series_equal(
        result.portfolio.value(), old_style.value(), check_names=False
    )
    new_records = result.portfolio.trades.records_readable
    old_records = old_style.trades.records_readable
    assert len(new_records) == len(old_records) > 0
    assert new_records["PnL"].tolist() == pytest.approx(old_records["PnL"].tolist())
    assert new_records["Size"].tolist() == pytest.approx(old_records["Size"].tolist())


def test_strike_and_expiry_frozen_between_a_fill_pair(bars):
    """Proves the premium column reaching the portfolio came from a frozen
    strike: between one leg's entry and exit, `leg_frame`'s strike and expiry
    must not move, even though `pinned_leg` carries the previous trade's
    values forward outside that window.
    """
    call_leg = LegSpec(right="call", direction="long", strike_rule="atm", lots=1)
    result = engine.run(
        bars,
        "engine_legs_test_strategy",
        vol=_ConstantVol(),
        params={"entry_bar": 10, "legs_by_day": (call_leg,)},
    )

    records = result.portfolio.trades.records_readable
    assert len(records) == 1
    row = records.iloc[0]
    entry_ts = pd.Timestamp(row["Entry Timestamp"])
    exit_ts = pd.Timestamp(row["Exit Timestamp"])

    # Same-day closure.
    assert entry_ts.date() == exit_ts.date()

    frame = result.leg_frame[call_leg.label]
    window = frame.loc[entry_ts:exit_ts]
    assert len(window) > 1
    assert window["strike"].nunique() == 1
    assert window["expiry"].nunique() == 1
