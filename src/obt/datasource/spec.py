"""Data-source registry and the ``@data_source`` decorator.

Mirrors ``screener.strategies.spec``: a frozen pydantic spec, a decorator that
registers it, and an explicit-import :func:`discover_plugins` so decorators in
``plugins/`` fire on first import. A test asserts every module in that
directory is actually imported here -- the one failure mode of the
explicit-import approach is adding a plugin and forgetting the import line.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, SkipValidation, field_validator
from screener._registry import Registry

from obt._view import DerivedView
from obt.datasource.base import SpotSource

F = TypeVar("F", bound=Callable[..., SpotSource])


class SourceSpec(BaseModel):
    """A named, lazily-constructed spot data source."""

    name: str
    factory: SkipValidation[Callable[..., SpotSource]]
    description: str = ""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("data source name must not be empty")
        return normalized


registry: Registry[SourceSpec] = Registry("data source")

#: ``name -> factory`` live projection of :data:`registry`.
SOURCES: DerivedView[SourceSpec, Callable[..., SpotSource]] = DerivedView(
    registry, lambda spec: spec.factory
)


def data_source(name: str, *, description: str = "", **meta: Any) -> Callable[[F], F]:
    """Register a zero-or-more-arg factory returning a :class:`SpotSource`."""

    def _wrap(value: F) -> F:
        registry.add(
            name,
            SourceSpec(name=name, factory=value, description=description),
            **meta,
        )
        return value

    return _wrap


def discover_plugins() -> None:
    """Import every plugin module so its ``@data_source`` decorator fires."""
    from obt.datasource.plugins import (  # noqa: F401
        nifty_csv,
        nifty_index_csv,
        parquet_spot,
    )


def get_source(name: str, **kwargs: Any) -> SpotSource:
    """Build a registered source by name."""
    discover_plugins()
    return registry.get(name).factory(**kwargs)


def source_names() -> list[str]:
    discover_plugins()
    return sorted(registry.names())
