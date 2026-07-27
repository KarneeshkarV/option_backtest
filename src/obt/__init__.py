"""Modular intraday NIFTY option backtester.

Pluggable seams follow the registry/plugin pattern proven in ``screener``:
spot data sources, **observed option-chain sources**, strategies, and vol
models. Pricing models (:mod:`obt.pricing`) are swappable the same way.

**Default path: premiums are Black-76 model output** from spot + a vol model.
Pass ``option_source="nifty_atm_options_csv"`` to :func:`obt.engine.run` (or
``just run-orb-observed``) to price fills from the local ATM CE/PE CSVs
instead -- that is the path that matches ``screener``'s rule ("P&L always
uses observed chain prices, never model prices").
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
