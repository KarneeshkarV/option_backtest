"""Plugin directory for spot and option-chain data sources.

Each ``*.py`` file registers either a ``@data_source`` (spot OHLC) or an
``@option_source`` (observed option quotes). Add a plugin by dropping a
module here and importing it from ``obt.datasource.spec.discover_plugins``.
"""
