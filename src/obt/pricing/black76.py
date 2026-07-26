"""Vectorized Black-76 option pricing and greeks.

``screener.options.greeks`` already implements this correctly, but scalar
per-call -- unusable across 330k bars x a parameter sweep. This module is the
array version, and ``tests/test_pricing.py`` pins it against those scalar
functions as an oracle. That test is the reason to trust the vectorization.

Two choices worth stating:

- **Priced off the forward.** ``F = S*exp((r-q)T)``, then discounted at ``r``.
  For ``q=0`` this is algebraically identical to Black-Scholes on spot, so the
  oracle test is exact rather than approximate; the forward formulation just
  makes the carry assumption explicit and easy to change when a real futures
  basis is available.
- **``tau -> 0`` returns intrinsic value.** Expiry-day bars are precisely where
  a naive implementation divides by zero, and they matter here because weekly
  options spend a fifth of their life on expiry day.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import ndtr

Right = Literal["call", "put"]

#: Indian risk-free proxy; a parameter everywhere it is used.
DEFAULT_RATE = 0.065
DEFAULT_DIVIDEND_YIELD = 0.0

#: Below this many years to expiry, treat the option as expiring now.
_MIN_TAU = 1e-9
_MIN_VOL = 1e-9


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _d1_d2(
    forward: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(d1, d2, vol*sqrt(tau))`` with degenerate cells made safe."""
    root_tau = np.sqrt(tau)
    vol_root = np.maximum(vol * root_tau, _MIN_VOL)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_fk = np.log(np.where(strike > 0, forward / strike, 1.0))
    d1 = (log_fk + 0.5 * vol_root * vol_root) / vol_root
    d2 = d1 - vol_root
    return d1, d2, vol_root


def forward_price(
    spot: np.ndarray,
    tau: np.ndarray,
    rate: float = DEFAULT_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
) -> np.ndarray:
    return spot * np.exp((rate - dividend_yield) * tau)


def price(
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    vol: np.ndarray,
    right: Right,
    *,
    rate: float = DEFAULT_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
) -> np.ndarray:
    """Black-76 premium. Arrays broadcast; ``tau`` is in years."""
    spot, strike, tau, vol = np.broadcast_arrays(
        *(np.asarray(a, dtype="float64") for a in (spot, strike, tau, vol))
    )
    tau_safe = np.maximum(tau, _MIN_TAU)
    forward = forward_price(spot, tau_safe, rate, dividend_yield)
    d1, d2, _ = _d1_d2(forward, strike, tau_safe, vol)
    discount = np.exp(-rate * tau_safe)

    if right == "call":
        value = discount * (forward * ndtr(d1) - strike * ndtr(d2))
        intrinsic = np.maximum(spot - strike, 0.0)
    elif right == "put":
        value = discount * (strike * ndtr(-d2) - forward * ndtr(-d1))
        intrinsic = np.maximum(strike - spot, 0.0)
    else:
        raise ValueError(f"right must be 'call' or 'put', got {right!r}")

    expired = (tau <= _MIN_TAU) | (vol <= _MIN_VOL)
    value = np.where(expired, intrinsic, value)
    # Never return a negative premium, whatever rounding did.
    return np.maximum(value, 0.0)


def greeks(
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    vol: np.ndarray,
    right: Right,
    *,
    rate: float = DEFAULT_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
) -> dict[str, np.ndarray]:
    """Delta, gamma, vega and theta.

    ``vega`` is per 1.00 of vol (not per vol point) and ``theta`` is per year,
    matching ``screener.options.greeks`` so the two can be compared directly.
    """
    spot, strike, tau, vol = np.broadcast_arrays(
        *(np.asarray(a, dtype="float64") for a in (spot, strike, tau, vol))
    )
    tau_safe = np.maximum(tau, _MIN_TAU)
    forward = forward_price(spot, tau_safe, rate, dividend_yield)
    d1, d2, vol_root = _d1_d2(forward, strike, tau_safe, vol)
    root_tau = np.sqrt(tau_safe)
    carry_discount = np.exp(-dividend_yield * tau_safe)
    discount = np.exp(-rate * tau_safe)
    pdf_d1 = _norm_pdf(d1)

    gamma = carry_discount * pdf_d1 / (spot * vol * root_tau)
    vega = spot * carry_discount * pdf_d1 * root_tau
    common_theta = -(spot * carry_discount * pdf_d1 * vol) / (2.0 * root_tau)

    if right == "call":
        delta = carry_discount * ndtr(d1)
        theta = (
            common_theta
            + dividend_yield * spot * carry_discount * ndtr(d1)
            - rate * strike * discount * ndtr(d2)
        )
    elif right == "put":
        delta = -carry_discount * ndtr(-d1)
        theta = (
            common_theta
            - dividend_yield * spot * carry_discount * ndtr(-d1)
            + rate * strike * discount * ndtr(-d2)
        )
    else:
        raise ValueError(f"right must be 'call' or 'put', got {right!r}")

    expired = (tau <= _MIN_TAU) | (vol <= _MIN_VOL)
    zero = np.zeros_like(spot)
    return {
        "delta": np.where(expired, zero, delta),
        "gamma": np.where(expired, zero, gamma),
        "vega": np.where(expired, zero, vega),
        "theta": np.where(expired, zero, theta),
    }


def delta(
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    vol: np.ndarray,
    right: Right,
    *,
    rate: float = DEFAULT_RATE,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
) -> np.ndarray:
    """Delta only -- the hot path for delta-targeted strike selection."""
    return greeks(
        spot, strike, tau, vol, right, rate=rate, dividend_yield=dividend_yield
    )["delta"]
