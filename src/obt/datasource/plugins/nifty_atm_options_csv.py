"""Observed NIFTY ATM option quotes from the local CE/PE CSV pair.

These are the only real option prices in the package. They cover
2026-04-22 → 2026-07-21 (about 60 sessions): one call and one put per bar,
strike pinned at the weekly cycle's opening ATM. They are what
:func:`obt.engine.run` uses when ``option_source="nifty_atm_options_csv"`` so
P&L is measured on traded premiums rather than Black-76 model output.

The vendor also ships a combined ``NIFTY_ATM_options_1min_*.csv`` that is
``concat(CE, PE)`` (same rows, often a different order). Reading it *in
addition* would double-count every bar, so this source only opens the two
per-right files.

Environment override: ``$OBT_OPTION_CHAIN_DIR`` points at the directory holding
the CE/PE files (shared with :mod:`obt.calibration`).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from obt.datasource.base import normalize_option_chain, slice_dates
from obt.datasource.plugins.nifty_index_csv import read_naive_ist_csv
from obt.datasource.spec import option_source

ENV_VAR = "OBT_OPTION_CHAIN_DIR"

#: ``right -> filename``. Combined options CSV is deliberately omitted.
CHAIN_FILES = {
    "call": "NIFTY_ATM_CE_1min_2026-04-22_2026-07-21.csv",
    "put": "NIFTY_ATM_PE_1min_2026-04-22_2026-07-21.csv",
}


def default_chain_dir() -> Path:
    """``$OBT_OPTION_CHAIN_DIR`` if set, else the repo root (five levels up)."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[4]


def chain_files_available(directory: Path | None = None) -> bool:
    """Whether both CE and PE quote files are present."""
    directory = directory or default_chain_dir()
    return all((directory / name).exists() for name in CHAIN_FILES.values())


class NiftyAtmOptionsCsvSource:
    """Read observed ATM CE/PE 1-minute quotes from a directory of CSVs."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_chain_dir()

    def load(
        self,
        symbol: str = "NIFTY",
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        del symbol  # single-underlying feed; kept for OptionChainSource parity
        frames: list[pd.DataFrame] = []
        for right, filename in CHAIN_FILES.items():
            file_path = self.path / filename
            if not file_path.exists():
                raise FileNotFoundError(
                    f"observed option chain file not found at {file_path}. "
                    f"Set ${ENV_VAR} to the directory holding the CE/PE CSVs."
                )
            raw = read_naive_ist_csv(file_path)
            raw = raw.assign(right=right)
            frames.append(raw)
        combined = pd.concat(frames, ignore_index=True)
        chain = normalize_option_chain(combined)
        return slice_dates(chain, start, end)


@option_source(
    "nifty_atm_options_csv",
    description="Observed NIFTY ATM CE/PE 1-minute quotes from local CSVs",
)
def _build(path: str | Path | None = None) -> NiftyAtmOptionsCsvSource:
    return NiftyAtmOptionsCsvSource(path)
