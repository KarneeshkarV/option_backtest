"""Live read-only projection over a :class:`screener._registry.Registry`.

Ported from ``screener.strategies.spec.DerivedView``, which is hard-bound to
that module's strategy registry. This variant takes the registry as an
argument so every seam in this package can share one implementation.

This is deliberately *not* a stored dict: every lookup and iteration re-reads
the underlying registry, so there is no second copy of the table that can drift
out of sync when a plugin registers late.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

from screener._registry import Registry


# Parameterized on both the spec type and the projected value type. Inheriting
# `Mapping[str, V]` alone would leave the class generic in V only, making every
# `DerivedView[SomeSpec, SomeValue]` annotation an error.
class DerivedView[S, V](Mapping[str, V]):
    """``name -> project(spec)`` view, excluding entries where it returns None."""

    def __init__(self, registry: Registry[S], project: Callable[[S], V | None]) -> None:
        self._registry = registry
        self._project = project

    def __getitem__(self, key: str) -> V:
        value = self._project(self._registry.get(key))
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return (
            name
            for name, spec in self._registry.items()
            if self._project(spec) is not None
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)
