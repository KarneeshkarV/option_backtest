"""Strategy registry and the ``@strategy`` decorator.

Mirrors ``screener.strategies.spec``. Adding a strategy is one file in
``plugins/`` plus one import line in :func:`discover_plugins` -- and a test
asserts those two never drift apart, which is the only way this pattern
actually fails in practice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, SkipValidation, field_validator
from screener._registry import Registry

from obt._view import DerivedView
from obt.strategies.base import StrategyFn

F = TypeVar("F", bound=StrategyFn)


class StrategySpec(BaseModel):
    """A named signal generator plus its default parameters."""

    name: str
    signal_fn: SkipValidation[StrategyFn]
    description: str = ""
    defaults: dict[str, Any] = {}

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("strategy name must not be empty")
        return normalized


registry: Registry[StrategySpec] = Registry("strategy")

#: ``name -> signal_fn`` live projection of :data:`registry`.
STRATEGIES: DerivedView[StrategySpec, StrategyFn] = DerivedView(
    registry, lambda spec: spec.signal_fn
)


def strategy(
    name: str,
    *,
    description: str = "",
    defaults: dict[str, Any] | None = None,
    **meta: Any,
) -> Callable[[F], F]:
    """Register a ``(bars, **params) -> Signals`` strategy."""

    def _wrap(value: F) -> F:
        registry.add(
            name,
            StrategySpec(
                name=name,
                signal_fn=value,
                description=description,
                defaults=dict(defaults or {}),
            ),
            **meta,
        )
        return value

    return _wrap


def discover_plugins() -> None:
    """Import every plugin module so its ``@strategy`` decorator fires."""
    from obt.strategies.plugins import (  # noqa: F401
        buy_open,
        ema_cross,
        orb,
    )


def get_strategy(name: str) -> StrategySpec:
    discover_plugins()
    return registry.get(name)


def strategy_names() -> list[str]:
    discover_plugins()
    return sorted(registry.names())
