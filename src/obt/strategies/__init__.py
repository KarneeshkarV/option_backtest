"""Pluggable option strategies."""

from obt.strategies.base import Signals, StrategyFn
from obt.strategies.spec import (
    STRATEGIES,
    StrategySpec,
    get_strategy,
    strategy,
    strategy_names,
)

__all__ = [
    "STRATEGIES",
    "Signals",
    "StrategyFn",
    "StrategySpec",
    "get_strategy",
    "strategy",
    "strategy_names",
]
