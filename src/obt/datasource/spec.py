"""Data-source registry and the ``@data_source`` / ``@option_source`` decorators.

Mirrors ``screener.strategies.spec``: a frozen pydantic spec, a decorator that
registers it, and an explicit-import :func:`discover_plugins` so decorators in
``plugins/`` fire on first import. A test asserts every module in that
directory is actually imported here -- the one failure mode of the
explicit-import approach is adding a plugin and forgetting the import line.

Two registries live here because the return shapes differ:

- :data:`registry` / ``@data_source`` -- spot OHLC (:class:`SpotSource`)
- :data:`option_registry` / ``@option_source`` -- observed option quotes
  (:class:`OptionChainSource`), used by the engine instead of Black-76 when
  real premiums are available
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, SkipValidation, field_validator
from screener._registry import Registry

from obt._view import DerivedView
from obt.datasource.base import OptionChainSource, SpotSource

F = TypeVar("F", bound=Callable[..., SpotSource])
G = TypeVar("G", bound=Callable[..., OptionChainSource])


def _normalize_registry_name(value: str, *, kind: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{kind} name must not be empty")
    return normalized


class SourceSpec(BaseModel):
    """A named, lazily-constructed spot data source."""

    name: str
    factory: SkipValidation[Callable[..., SpotSource]]
    description: str = ""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        return _normalize_registry_name(value, kind="data source")


class OptionSourceSpec(BaseModel):
    """A named, lazily-constructed observed option-chain source."""

    name: str
    factory: SkipValidation[Callable[..., OptionChainSource]]
    description: str = ""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        return _normalize_registry_name(value, kind="option source")


registry: Registry[SourceSpec] = Registry("data source")
option_registry: Registry[OptionSourceSpec] = Registry("option source")

#: ``name -> factory`` live projection of :data:`registry`.
SOURCES: DerivedView[SourceSpec, Callable[..., SpotSource]] = DerivedView(
    registry, lambda spec: spec.factory
)

#: ``name -> factory`` live projection of :data:`option_registry`.
OPTION_SOURCES: DerivedView[OptionSourceSpec, Callable[..., OptionChainSource]] = (
    DerivedView(option_registry, lambda spec: spec.factory)
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


def option_source(name: str, *, description: str = "", **meta: Any) -> Callable[[G], G]:
    """Register a factory returning an :class:`OptionChainSource`.

    Observed option premiums for engine P&L -- not spot OHLC. Pass the name to
    :func:`obt.engine.run` via ``option_source=...`` to skip Black-76 entirely.
    """

    def _wrap(value: G) -> G:
        option_registry.add(
            name,
            OptionSourceSpec(name=name, factory=value, description=description),
            **meta,
        )
        return value

    return _wrap


def discover_plugins() -> None:
    """Import every plugin module so its registry decorators fire."""
    from obt.datasource.plugins import (  # noqa: F401
        nifty_atm_options_csv,
        nifty_csv,
        nifty_index_csv,
        parquet_spot,
    )


def get_source(name: str, **kwargs: Any) -> SpotSource:
    """Build a registered spot source by name."""
    discover_plugins()
    return registry.get(name).factory(**kwargs)


def get_option_source(name: str, **kwargs: Any) -> OptionChainSource:
    """Build a registered option-chain source by name."""
    discover_plugins()
    return option_registry.get(name).factory(**kwargs)


def source_names() -> list[str]:
    discover_plugins()
    return sorted(registry.names())


def option_source_names() -> list[str]:
    discover_plugins()
    return sorted(option_registry.names())
