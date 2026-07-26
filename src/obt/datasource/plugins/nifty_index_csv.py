"""NIFTY50 spot bars from the 2026 vendor export.

A second, independently-sourced spot feed covering 2026-04-22 to 2026-07-21.
It exists for two reasons beyond its own 62 sessions:

- It **overlaps the five-year file by 37 sessions**, which turns the data-source
  seam into a real cross-check rather than a design claim. On the overlap the
  two feeds agree exactly on 96.9% of bars and within 0.5 points on 99.0% --
  close enough to treat as the same underlying, far enough apart to be worth
  knowing about.
- It carries the spot leg of the observed option chain in :mod:`obt.calibration`,
  so the calibration never has to mix vendors within a single comparison.

Format differs from the five-year file: separate ``date``/``time`` columns, a
combined ``datetime`` without a UTC offset (timestamps are IST wall clock), a
``symbol`` column, and real ``volume``. Bars run to 15:39 on some days;
:func:`obt.session.filter_regular_session` trims them.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from obt.datasource.base import normalize, slice_dates
from obt.datasource.spec import data_source
from obt.session import IST

ENV_VAR = "OBT_NIFTY_INDEX_CSV"
DEFAULT_FILENAME = "NIFTY_index_1min_2026-04-22_2026-07-21.csv"


def default_csv_path() -> Path:
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[4] / DEFAULT_FILENAME


def read_naive_ist_csv(path: Path) -> pd.DataFrame:
    """Read a vendor CSV whose ``datetime`` column has no offset.

    The timestamps are IST wall clock with the offset simply omitted. Localizing
    is therefore correct; letting :func:`normalize` parse them as UTC would
    silently shift every bar by 5h30m and quietly move trades across sessions.
    """
    raw = pd.read_csv(path)
    stamps = pd.to_datetime(raw["datetime"])
    if stamps.dt.tz is None:
        stamps = stamps.dt.tz_localize(IST)
    return raw.assign(datetime=stamps)


class NiftyIndexCsvSource:
    """Read 2026 spot bars from the vendor CSV."""

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
                f"NIFTY index CSV not found at {self.path}. "
                f"Set ${ENV_VAR} to point at it."
            )
        raw = read_naive_ist_csv(self.path)
        return slice_dates(normalize(raw), start, end)


@data_source(
    "nifty_index_csv",
    description="NIFTY50 1-minute spot OHLC, 2026 vendor export",
)
def _build(path: str | Path | None = None) -> NiftyIndexCsvSource:
    return NiftyIndexCsvSource(path)
