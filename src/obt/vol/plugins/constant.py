"""Flat IV across every strike, expiry and date.

Deliberately naive. It exists as the control: run a strategy under this and
under ``gk_vrp`` and the difference isolates exactly how much of the result the
volatility model is responsible for. It is blind to regime, so it will badly
misprice premiums around crashes and event days -- do not read its output as a
forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from obt.vol.spec import vol_model


class ConstantVolModel:
    """One IV number, everywhere."""

    def __init__(self, iv: float = 0.14) -> None:
        if iv <= 0:
            raise ValueError("constant iv must be positive")
        self.level = float(iv)

    @property
    def label(self) -> str:
        return f"constant(iv={self.level:g})"

    def atm_iv(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(self.level, index=bars.index, dtype="float64")

    def iv(
        self,
        atm: np.ndarray,
        spot: np.ndarray,
        strike: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        return np.full_like(np.asarray(strike, dtype="float64"), self.level)


@vol_model("constant", description="Flat IV baseline, for isolating vol-model effects")
def _build(iv: float = 0.14) -> ConstantVolModel:
    return ConstantVolModel(iv)
