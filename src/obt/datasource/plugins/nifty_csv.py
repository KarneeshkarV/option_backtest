"""NIFTY 1-minute spot bars from a local CSV.

Expects the header ``datetime,open,high,low,close,volume`` with ISO timestamps
carrying a ``+05:30`` offset. ``volume`` is read and discarded: the file
records it as ``0`` on every row, because a spot index has no traded volume.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from obt.datasource.base import normalize, slice_dates
from obt.datasource.spec import data_source

#: Overridable so tests and other machines need not share one layout.
ENV_VAR = "OBT_NIFTY_CSV"
DEFAULT_FILENAME = "NIFTY_1MIN_5YEAR (1).csv"


def default_csv_path() -> Path:
    """``$OBT_NIFTY_CSV`` if set, else the bundled file at the repo root."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    # src/obt/datasource/plugins/nifty_csv.py -> repo root is five levels up.
    return Path(__file__).resolve().parents[4] / DEFAULT_FILENAME


class NiftyCsvSource:
    """Read spot bars from a CSV on disk."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_csv_path()

    def load(
        self,
        symbol: str = "NIFTY",
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"NIFTY CSV not found at {self.path}. Set ${ENV_VAR} to point at it."
            )
        raw = pd.read_csv(self.path)
        return slice_dates(normalize(raw), start, end)


@data_source("nifty_csv", description="NIFTY 1-minute spot OHLC from a local CSV")
def _build(path: str | Path | None = None) -> NiftyCsvSource:
    return NiftyCsvSource(path)
