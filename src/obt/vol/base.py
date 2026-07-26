"""The volatility seam.

There is no IV in the input data -- only spot -- so implied vol has to be
modelled. A :class:`VolModel` does that in two separable pieces:

1. :meth:`VolModel.atm_iv` -- an at-the-money level per bar, derived from the
   price history.
2. :meth:`VolModel.iv` -- how that level varies across strike and expiry
   (skew, term structure).

Splitting them keeps the expensive history-dependent part computed once per
run while the per-strike part stays a cheap array op.

**This is the largest source of model risk in the package.** Premiums, and
therefore every P&L number, are only as good as the assumption baked in here,
which is why :mod:`obt.report` sweeps the parameters rather than quoting a
single result.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class VolModel(Protocol):
    """Produce annualized implied volatilities from spot history."""

    #: Short label for the report banner, e.g. ``"gk_vrp(vrp=1.15)"``.
    @property
    def label(self) -> str: ...

    def atm_iv(self, bars: pd.DataFrame) -> pd.Series:
        """Annualized ATM IV per bar, indexed like ``bars``.

        Must not peek: the value at bar ``t`` may only use information
        available strictly before ``t``.
        """
        ...

    def iv(
        self,
        atm: np.ndarray,
        spot: np.ndarray,
        strike: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        """Spread the ATM level across strike/expiry. ``tau`` is in years."""
        ...
