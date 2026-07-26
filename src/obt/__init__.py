"""Modular intraday NIFTY option backtester.

Two seams are pluggable, both following the registry/plugin pattern proven in
``screener``: data sources (:mod:`obt.datasource`) and strategies
(:mod:`obt.strategies`). Vol models (:mod:`obt.vol`) and pricing models
(:mod:`obt.pricing`) are swappable the same way.

**Premiums here are model output, not observed quotes.** The only available
input is NIFTY index spot, so option prices are synthesized with Black-76 from
a realized-vol-derived IV. ``screener.options.structures`` states the opposite
rule for its own backtester ("P&L always uses observed chain prices, never
model prices") -- this package deliberately inverts it, which is why every
report carries an assumption banner and a parameter-sensitivity table.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
