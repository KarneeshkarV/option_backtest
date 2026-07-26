"""Vol-model registry and the ``@vol_model`` decorator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, SkipValidation
from screener._registry import Registry

from obt._view import DerivedView
from obt.vol.base import VolModel

F = TypeVar("F", bound=Callable[..., VolModel])


class VolSpec(BaseModel):
    name: str
    factory: SkipValidation[Callable[..., VolModel]]
    description: str = ""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


registry: Registry[VolSpec] = Registry("vol model")

VOL_MODELS: DerivedView[VolSpec, Callable[..., VolModel]] = DerivedView(
    registry, lambda spec: spec.factory
)


def vol_model(name: str, *, description: str = "", **meta: Any) -> Callable[[F], F]:
    def _wrap(value: F) -> F:
        registry.add(
            name, VolSpec(name=name, factory=value, description=description), **meta
        )
        return value

    return _wrap


def discover_plugins() -> None:
    """Import every plugin module so its ``@vol_model`` decorator fires."""
    from obt.vol.plugins import constant, gk_vrp  # noqa: F401


def get_vol_model(name: str, **kwargs: Any) -> VolModel:
    discover_plugins()
    return registry.get(name).factory(**kwargs)


def vol_model_names() -> list[str]:
    discover_plugins()
    return sorted(registry.names())
