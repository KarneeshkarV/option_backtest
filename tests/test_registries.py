"""The plugin seams: registration, lookup, and the drift the pattern invites.

Explicit-import ``discover_plugins()`` has exactly one failure mode -- you add
``plugins/my_thing.py``, forget the import line, and the plugin is simply
invisible with no error anywhere. :func:`test_every_plugin_module_is_discovered`
globs each plugins directory and diffs it against the registry, so that bug
fails a test instead of being discovered later by confusion.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from obt import datasource, strategies, vol
from obt.datasource import spec as source_spec
from obt.strategies import spec as strategy_spec
from obt.vol import spec as vol_spec

SEAMS = [
    pytest.param(source_spec, "obt.datasource.plugins", id="datasource"),
    pytest.param(strategy_spec, "obt.strategies.plugins", id="strategy"),
    pytest.param(vol_spec, "obt.vol.plugins", id="vol"),
]


@pytest.mark.parametrize(("module", "package"), SEAMS)
def test_every_plugin_module_is_discovered(module, package):
    """Every file in plugins/ must be reachable through discover_plugins()."""
    module.discover_plugins()
    plugin_dir = Path(importlib.import_module(package).__file__).parent
    on_disk = {
        path.stem for path in plugin_dir.glob("*.py") if not path.stem.startswith("_")
    }
    imported = {
        name.rsplit(".", 1)[-1]
        for name in list(importlib.sys.modules)
        if name.startswith(f"{package}.")
    }
    missing = on_disk - imported
    assert not missing, (
        f"{sorted(missing)} exist in {plugin_dir} but are not imported by "
        f"{module.__name__}.discover_plugins() -- they are invisible at runtime"
    )


@pytest.mark.parametrize(("module", "package"), SEAMS)
def test_duplicate_registration_is_rejected(module, package):
    """Silently overwriting a registered name would be the worst outcome."""
    existing = sorted(module.registry.names())[0]
    with pytest.raises(Exception) as excinfo:
        module.registry.add(existing, module.registry.get(existing))
    assert existing in str(excinfo.value)


@pytest.mark.parametrize(("module", "package"), SEAMS)
def test_unknown_name_lists_the_known_ones(module, package):
    with pytest.raises(KeyError) as excinfo:
        module.registry.get("no-such-plugin")
    message = str(excinfo.value)
    assert all(name in message for name in module.registry.names())


def test_reference_plugins_are_registered():
    assert {"orb", "buy_open", "ema_cross"} <= set(strategies.strategy_names())
    assert {"nifty_csv", "parquet_spot"} <= set(datasource.source_names())
    assert {"nifty_atm_options_csv"} <= set(datasource.option_source_names())
    assert {"gk_vrp", "constant"} <= set(vol.vol_model_names())


def test_derived_view_projects_live():
    """The view must read through to the registry, not snapshot it."""
    strategy_spec.discover_plugins()
    assert set(strategies.STRATEGIES) == set(strategy_spec.registry.names())
    assert strategies.STRATEGIES["orb"] is strategy_spec.registry.get("orb").signal_fn


def test_strategy_defaults_are_not_shared_between_specs():
    """A mutable default leaking across specs would couple unrelated plugins."""
    strategy_spec.discover_plugins()
    first = strategy_spec.registry.get("orb").defaults
    second = strategy_spec.registry.get("buy_open").defaults
    assert first is not second


def test_empty_strategy_name_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        strategy_spec.StrategySpec(name="  ", signal_fn=lambda bars: None)
