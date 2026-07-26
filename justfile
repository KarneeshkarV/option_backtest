set positional-arguments

python := "uv run python"

# List available recipes.
default:
    @just --list

# Install/refresh the environment (pulls screener from ../screener).
sync:
    uv sync

# Lint and check formatting.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Apply formatting and safe lint fixes.
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Static types.
typecheck:
    uv run mypy

# Tests. Calibration tests skip when the observed-chain CSVs are absent.
test *args:
    uv run pytest "$@"

# Everything CI would run.
check: lint typecheck test

# Full ORB report: banner, coverage, sanity checks, stats, sensitivity grid.
run-orb *args:
    {{python}} scripts/run_orb.py orb "$@"

# The decay canary -- a long ATM call held all day. It should lose money.
# If this ever shows a profit, the pricer or the cost model is broken.
run-canary *args:
    {{python}} scripts/run_orb.py buy_open --no-sensitivity "$@"

# Measure the synthetic vol model against real observed option quotes.
calibrate:
    {{python}} scripts/calibrate.py

# Prove the seams: swap the data source, the vol model, and the strategy.
verify-seams:
    {{python}} scripts/verify_seams.py

# List everything registered in each plugin registry.
plugins:
    @{{python}} -c "from obt.datasource import source_names; \
      from obt.strategies import strategy_names; from obt.vol import vol_model_names; \
      print('sources   :', source_names()); \
      print('strategies:', strategy_names()); \
      print('vol models:', vol_model_names())"
