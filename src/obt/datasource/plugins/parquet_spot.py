"""Spot bars from a parquet file.

Exists mainly to keep the data-source seam honest: the verification step dumps
the CSV to parquet, loads it back through this source, and asserts the backtest
is byte-identical. If that ever fails, the seam has leaked.

It is also the faster path for repeated runs -- parquet loads the 330k-row
frame in a fraction of the CSV parse time.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from obt.datasource.base import normalize, slice_dates
from obt.datasource.spec import data_source


class ParquetSpotSource:
    """Read spot bars from a parquet file written by :func:`write_parquet`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(
        self,
        symbol: str = "NIFTY",
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"parquet spot file not found: {self.path}")
        raw = pd.read_parquet(self.path)
        return slice_dates(normalize(raw), start, end)


def write_parquet(bars: pd.DataFrame, path: str | Path) -> Path:
    """Persist normalized bars so :class:`ParquetSpotSource` can read them."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(target)
    return target


@data_source("parquet_spot", description="Spot OHLC from a local parquet file")
def _build(path: str | Path) -> ParquetSpotSource:
    return ParquetSpotSource(path)
