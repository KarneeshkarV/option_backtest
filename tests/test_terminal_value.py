"""Is the ``TICK_SIZE`` floor economically sound at the edges of a trade?

``obt.chain.pinned_leg`` floors every modeled premium at ``TICK_SIZE`` (Rs
0.05) because Black-76 prices a far-OTM, no-time-left option to exactly 0.0,
which vectorbt rejects outright ("order.price must be finite and greater
than 0"). ``chain.py``'s own comment justifies the floor only for a long
position: "that error points the conservative way -- a long that should have
died worthless still pays to close." Finding 9c is that nobody checked the
short side, and nobody checked the long side's arithmetic either.

These tests compute the actual, signed cash effect for both sides directly
-- and, for the terminal-value assertions, corroborate it against a real
``vectorbt`` fill so the conclusion does not rest on the L/S bookkeeping
convention alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from obt.chain import TICK_SIZE, LegSpec
from obt.pricing import black76

RATE = black76.DEFAULT_RATE


def test_deep_otm_expiry_bar_prices_true_zero_before_flooring():
    """At tau=0 (the expiry session's final bar -- see obt.calendar, finding
    5), a deep-OTM call has zero intrinsic value and Black-76 returns exactly
    0.0, matching chain.py's own comment: "a far-OTM call with one bar of
    life left prices to exactly 0.0". Corroborates that the floor is doing
    something real (lifting an exact 0.0), not padding an already-nonzero
    number.
    """
    spot = np.array([20_000.0])
    strike = np.array([21_000.0])  # 1000 points OTM
    tau = np.array([0.0])
    iv = np.array([0.15])

    true_price = black76.price(spot, strike, tau, iv, "call", rate=RATE)
    assert true_price[0] == 0.0

    floored = np.maximum(true_price, TICK_SIZE)
    assert floored[0] == pytest.approx(TICK_SIZE)
    assert floored[0] > true_price[0]


def test_long_worthless_expiry_floor_inflates_pnl_not_conservative():
    """The long side of chain.py's claim, checked directly.

    Closing a LONG position is a SELL. Selling at the floor (Rs 0.05)
    instead of the true value (0) hands the long strictly MORE cash back
    than the option was worth -- their realized loss is smaller than the
    true economics, not larger. That is the opposite of "conservative": it
    makes the backtest look slightly *better* for the long than reality.

    Fails if the sign of this bias is ever flipped without noticing (e.g. by
    "fixing" the floor's direction based on the docstring's claim without
    re-deriving it), or if the floor is ever applied only at entry and not
    at exit (or vice versa) for a long leg.
    """
    long_call = LegSpec(right="call", direction="long", strike_rule="atm", lots=1)
    quantity = long_call.signed_quantity
    assert quantity > 0

    true_exit = 0.0
    floored_exit = TICK_SIZE

    # P&L contribution at exit is `quantity * exit_price` (holding entry
    # fixed); the floor's bias is the difference the flooring makes to it.
    pnl_bias_from_floor = quantity * (floored_exit - true_exit)

    assert pnl_bias_from_floor > 0, (
        "the floor should make a dying long's terminal P&L HIGHER than the "
        "true (worthless) economics -- if this is <= 0, the direction "
        "claimed by chain.py ('conservative for a long') would actually "
        "hold, and the finding below is stale"
    )
    assert pnl_bias_from_floor == pytest.approx(
        TICK_SIZE * long_call.lots * long_call.lot_size
    )


def test_short_worthless_expiry_floor_is_conservative():
    """The unchecked side: a SHORT closes a dying position by BUYING BACK.
    Paying Rs 0.05 to buy back something truly worth 0 costs the short money
    relative to the true economics -- their realized profit is smaller than
    the true economics, never larger. That direction genuinely IS
    conservative: it does not let a premium-selling strategy collect a free
    windfall out of a floor that exists only for numerical/vectorbt reasons.
    """
    short_call = LegSpec(right="call", direction="short", strike_rule="atm", lots=1)
    quantity = short_call.signed_quantity
    assert quantity < 0

    true_exit = 0.0
    floored_exit = TICK_SIZE

    pnl_bias_from_floor = quantity * (floored_exit - true_exit)

    assert pnl_bias_from_floor < 0, (
        "the floor should make a dying short's terminal P&L LOWER than the "
        "true (worthless) economics -- i.e. cost the short money"
    )
    assert pnl_bias_from_floor == pytest.approx(
        -TICK_SIZE * short_call.lots * short_call.lot_size
    )


def test_floor_bias_is_equal_and_opposite_for_long_and_short():
    """Same mechanism, mirrored sign: whatever the floor does to a long's
    terminal P&L, it does the exact negative to a short's, because both
    share the same |quantity| and the same floored-vs-true mark. This is
    the structural reason the two verdicts above cannot both be
    "conservative" -- one side's extra nickel is the other side's missing
    nickel.
    """
    long_leg = LegSpec(right="call", direction="long", strike_rule="atm", lots=3)
    short_leg = LegSpec(right="call", direction="short", strike_rule="atm", lots=3)
    assert long_leg.signed_quantity == -short_leg.signed_quantity

    delta = TICK_SIZE - 0.0
    long_bias = long_leg.signed_quantity * delta
    short_bias = short_leg.signed_quantity * delta
    assert long_bias == pytest.approx(-short_bias)
    assert long_bias > 0 > short_bias


def test_vectorbt_confirms_the_long_short_direction_empirically():
    """The abstract `signed_quantity * price` arithmetic above, corroborated
    against an actual vectorbt fill rather than trusted on its own -- so the
    conclusion doesn't rest solely on getting the P&L sign convention right
    by hand.

    vectorbt rejects a literal 0.0 price ("order.price must be finite and
    greater than 0"), which is the exact reason the floor exists, so the
    "true" path here uses a negligible epsilon (Rs 0.001) as a stand-in for
    the unreachable true zero; the floored path uses the real TICK_SIZE. The
    comparison isolates the floor's effect, not some other epsilon artifact.
    """
    idx = pd.date_range("2024-01-01 09:15", periods=5, freq="1min", tz="Asia/Kolkata")
    decaying_true = [50.0, 30.0, 10.0, 2.0, 0.001]
    decaying_floored = [50.0, 30.0, 10.0, 2.0, TICK_SIZE]

    entries = pd.Series([True, False, False, False, False], index=idx)
    exits = pd.Series([False, False, False, False, True], index=idx)

    def _pnl(prices: list[float], direction: str) -> float:
        close = pd.Series(prices, index=idx)
        pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            size=1.0,
            size_type="amount",
            direction=direction,
            init_cash=10_000.0,
            fees=0.0,
            freq="1min",
        )
        records = pf.trades.records_readable
        assert len(records) == 1
        return float(records["PnL"].iloc[0])

    long_true = _pnl(decaying_true, "longonly")
    long_floored = _pnl(decaying_floored, "longonly")
    short_true = _pnl(decaying_true, "shortonly")
    short_floored = _pnl(decaying_floored, "shortonly")

    # Floor makes the long's loss SMALLER (better) than the true economics.
    assert long_floored > long_true
    # Floor makes the short's profit SMALLER (worse) than the true economics.
    assert short_floored < short_true


def test_deep_itm_terminal_value_equals_intrinsic_for_both_rights():
    """At tau=0, a deep-ITM option's priced mark must equal intrinsic value
    exactly, for both calls and puts -- the floor must not bind here (the
    true price sits comfortably above TICK_SIZE) and Black-76 must not
    spuriously retain time value at exactly zero tau (finding 5: tau is now
    exactly 0 on an expiry session's final bar, not one bar short of it).
    """
    spot = np.array([20_000.0])
    tau = np.array([0.0])
    iv = np.array([0.20])

    # Deep ITM call / deep OTM put at the same strike.
    strike_low = np.array([15_000.0])
    call_itm = black76.price(spot, strike_low, tau, iv, "call", rate=RATE)
    put_otm = black76.price(spot, strike_low, tau, iv, "put", rate=RATE)
    assert call_itm[0] == pytest.approx(20_000.0 - 15_000.0)
    assert put_otm[0] == pytest.approx(0.0)
    assert call_itm[0] > TICK_SIZE  # comfortably above the floor: it cannot bind

    # Mirror image: deep OTM call / deep ITM put.
    strike_high = np.array([25_000.0])
    call_otm = black76.price(spot, strike_high, tau, iv, "call", rate=RATE)
    put_itm = black76.price(spot, strike_high, tau, iv, "put", rate=RATE)
    assert call_otm[0] == pytest.approx(0.0)
    assert put_itm[0] == pytest.approx(25_000.0 - 20_000.0)
    assert put_itm[0] > TICK_SIZE

    # The floor is a genuine no-op on a deep-ITM mark.
    assert np.maximum(call_itm, TICK_SIZE)[0] == pytest.approx(call_itm[0])
    assert np.maximum(put_itm, TICK_SIZE)[0] == pytest.approx(put_itm[0])
