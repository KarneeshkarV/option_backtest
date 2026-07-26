"""Pin the vectorized pricer against screener's scalar implementation.

``screener.options.greeks`` is the oracle: independently written, scalar, and
already trusted by another backtester. If this file passes, the array
vectorization introduced no error, which is the only thing that could plausibly
go wrong in translating it.
"""

from __future__ import annotations

import numpy as np
import pytest
from screener.options.greeks import black_scholes_greeks, black_scholes_price

from obt.pricing import black76

RATE = 0.065


@pytest.fixture
def grid():
    rng = np.random.default_rng(20260726)
    spot = rng.uniform(15_000, 25_000, 300)
    strike = np.round(spot * rng.uniform(0.88, 1.12, 300) / 50) * 50
    tau = rng.uniform(1e-4, 0.1, 300)
    vol = rng.uniform(0.06, 0.6, 300)
    return spot, strike, tau, vol


@pytest.mark.parametrize("right", ["call", "put"])
def test_price_matches_scalar_oracle(grid, right):
    spot, strike, tau, vol = grid
    mine = black76.price(spot, strike, tau, vol, right, rate=RATE)
    reference = np.array(
        [
            black_scholes_price(s, k, t, RATE, v, right)
            for s, k, t, v in zip(spot, strike, tau, vol, strict=True)
        ]
    )
    np.testing.assert_allclose(mine, reference, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("right", ["call", "put"])
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta"])
def test_greeks_match_scalar_oracle(grid, right, greek):
    spot, strike, tau, vol = grid
    mine = black76.greeks(spot, strike, tau, vol, right, rate=RATE)[greek]
    reference = np.array(
        [
            black_scholes_greeks(s, k, t, RATE, v, right)[greek]
            for s, k, t, v in zip(spot, strike, tau, vol, strict=True)
        ]
    )
    np.testing.assert_allclose(mine, reference, rtol=1e-7, atol=1e-8)


def test_put_call_parity(grid):
    spot, strike, tau, vol = grid
    call = black76.price(spot, strike, tau, vol, "call", rate=RATE)
    put = black76.price(spot, strike, tau, vol, "put", rate=RATE)
    forward = black76.forward_price(spot, tau, RATE)
    discount = np.exp(-RATE * tau)
    np.testing.assert_allclose(call - put, discount * (forward - strike), atol=1e-8)


@pytest.mark.parametrize(
    ("right", "spot", "strike", "expected"),
    [
        ("call", 20_000.0, 19_500.0, 500.0),
        ("call", 20_000.0, 20_500.0, 0.0),
        ("put", 20_000.0, 20_500.0, 500.0),
        ("put", 20_000.0, 19_500.0, 0.0),
    ],
)
def test_zero_tau_is_intrinsic(right, spot, strike, expected):
    """Expiry-day bars must degrade to intrinsic, not divide by zero."""
    value = black76.price(
        np.array([spot]), np.array([strike]), np.array([0.0]), np.array([0.2]), right
    )
    assert value[0] == pytest.approx(expected)


def test_premiums_never_negative():
    """Deep OTM with tiny tau is where rounding could push a price below zero."""
    spot = np.full(50, 20_000.0)
    strike = np.linspace(24_000, 30_000, 50)
    tau = np.full(50, 1e-6)
    vol = np.full(50, 0.1)
    assert (black76.price(spot, strike, tau, vol, "call") >= 0).all()


def test_greeks_finite_at_expiry():
    spot = np.array([20_000.0, 20_000.0])
    strike = np.array([20_000.0, 21_000.0])
    tau = np.array([0.0, 0.0])
    vol = np.array([0.15, 0.15])
    greeks = black76.greeks(spot, strike, tau, vol, "call")
    for values in greeks.values():
        assert np.isfinite(values).all()


def test_unknown_right_rejected():
    with pytest.raises(ValueError, match="call.*put"):
        black76.price(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.1]),
            np.array([0.2]),
            "straddle",  # type: ignore[arg-type]
        )
