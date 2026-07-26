"""Pluggable spot data sources."""

from obt.datasource.base import FallbackSpotSource, SpotSource, normalize
from obt.datasource.spec import SOURCES, data_source, get_source, source_names

__all__ = [
    "SOURCES",
    "FallbackSpotSource",
    "SpotSource",
    "data_source",
    "get_source",
    "normalize",
    "source_names",
]
