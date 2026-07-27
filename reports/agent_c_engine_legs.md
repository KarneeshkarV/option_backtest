# Agent C — per-leg direction and size in the engine

Owned files: `src/obt/engine.py` (all of it EXCEPT lines ~176-192, the
signal-shift call site owned by Agent B), and the new test file
`tests/test_engine_legs.py`. No other files were edited; where another file
would ideally change, that's called out below instead of touched.

---

## Finding 2 — direction/size collapsed to a portfolio-wide scalar

### What was wrong

```python
is_short = is_short or leg.direction == "short"          # engine.py:213 (old)
...
reference_leg = groups[0][0]                              # engine.py:219
effective_lot = lot_size if lot_from_history else reference_leg.lot_size
quantity = reference_leg.lots * effective_lot
...
portfolio = vbt.Portfolio.from_signals(
    ...,
    size=float(quantity),                                  # one number, every column
    direction="shortonly" if is_short else "longonly",      # one flag, every column
    ...
)
```

`_resolve_legs` produces one vectorbt column per distinct `LegSpec`, but
`direction` and `size` were each reduced to a single scalar for the whole
portfolio: `is_short` was `True` if *any* leg was short, and `quantity` was
taken only from `groups[0][0]`. A strategy long a call one day and short a
put the next would have **both columns traded `shortonly`** (the long call
sold short) and **both sized off the first leg's lot count**.
`LegSpec.signed_quantity` (chain.py:80-83) existed but was never called
anywhere — dead code that would have made the direction/size coupling worse,
not better, if used naively.

### Fix chosen: option 1/2 — genuine per-column direction and size

I verified empirically (not from docs) that the installed vectorbt (1.0.0)
supports this. Probe script and output:

```python
# probe_percol.py
import pandas as pd, vectorbt as vbt

idx = pd.date_range("2024-01-01", periods=6, freq="1min")
close = pd.DataFrame({"A": [10,11,12,12,12,12], "B": [12,12,12,20,21,22]}, index=idx, dtype=float)
entries = pd.DataFrame({"A": [True,False,False,False,False,False],
                        "B": [False,False,False,True,False,False]}, index=idx)
exits   = pd.DataFrame({"A": [False,False,True,False,False,False],
                        "B": [False,False,False,False,False,True]}, index=idx)

# First attempt: a pd.Series keyed by column name -- FAILS.
direction = pd.Series({"A": "longonly", "B": "shortonly"})
size = pd.Series({"A": 1.0, "B": 2.0})
vbt.Portfolio.from_signals(close, entries, exits, size=size, size_type="amount",
                            direction=direction, init_cash=1e6, group_by=True,
                            cash_sharing=True, freq="1min")
# -> ValueError: shape mismatch: objects cannot be broadcast to a single
#    shape. Mismatch is between arg 0 with shape (2, 1) and arg 23 with
#    shape (6, 2).

# Second attempt: a plain Python list, positional against close.columns.
# This is the pattern vectorbt's own docstring uses
# (`direction = ['longonly', 'shortonly']  # per column`, base.py:88).
direction = ["longonly", "shortonly"]
size = [1.0, 2.0]
pf = vbt.Portfolio.from_signals(close, entries, exits, size=size, size_type="amount",
                                 direction=direction, init_cash=1e6, group_by=True,
                                 cash_sharing=True, freq="1min")
print(pf.trades.records_readable[["Column","Size","Direction","Avg Entry Price","Avg Exit Price","PnL"]])
```

Output of the working (list) version:

```
  Column  Size Direction  Avg Entry Price  Avg Exit Price  PnL
0      A   1.0      Long             10.0            12.0  2.0
1      B   2.0     Short             20.0            22.0 -4.0
```

Column A (long, size 1) bought at 10 and sold at 12 for +2; column B (short,
size 2) sold at 20 and covered at 22 for -4 (a short losing on a rally) — both
correct, independently, in the same grouped/cash-shared portfolio the engine
already uses (`group_by=True, cash_sharing=True`). **A `pd.Series` indexed by
column name does not broadcast the same way and raises**, so `direction` and
`size` must stay plain Python lists in `close.columns` order.

I chose **unsigned per-leg quantity (`leg.lots * effective_lot`) + a separate
per-column `direction` list**, not `LegSpec.signed_quantity`. Using
`signed_quantity` would encode sign twice (once in the negative/positive size,
once in the `direction` flag), which is redundant and riskier to reason about;
the unsigned-quantity-plus-explicit-direction shape is exactly what the old
scalar code already did, just repeated per column instead of computed once
from `groups[0][0]`. This is the smallest correct change, and it means the
homogeneous (single quantity, single direction) case reduces to *literally*
the pre-fix computation, one term at a time.

A second, related bug surfaced while wiring this up: **`leg.label` collides**
whenever two distinct `LegSpec`s share `right`/`direction`/`strike_rule` but
differ only in `lots` (e.g. 1-lot vs 2-lot long ATM calls) — both hash to
different dict keys in `_resolve_legs`'s grouping (so `groups` correctly has
two entries) but to the *same* string in the `premium_columns` /
`entry_columns` / `exit_columns` dicts keyed by `leg.label`, so the second
leg's entries/exits silently overwrote the first's. Fixed by disambiguating
repeated names with a `#2`, `#3`, ... suffix (first occurrence keeps the
plain label, so single-leg and orb/buy_open naming is unchanged).

I did **not** fall back to raising (option 4): per-column direction/size is
correct and empirically supported, so raising on the heterogeneous case would
have thrown away a case the seam is designed to serve, unnecessarily.

### Exact diff

```diff
diff --git a/src/obt/engine.py b/src/obt/engine.py
--- a/src/obt/engine.py
+++ b/src/obt/engine.py
@@ -197,7 +197,8 @@ def run(
     entry_columns: dict[str, np.ndarray] = {}
     exit_columns: dict[str, np.ndarray] = {}
     leg_frames: dict[str, pd.DataFrame] = {}
-    is_short = False
+    column_legs: list[LegSpec] = []
+    used_names: dict[str, int] = {}
 
     for leg, leg_open in groups:
         priced = pinned_leg(
@@ -205,28 +206,52 @@ def run(
         )
         # Close only the trades this leg actually opened.
         leg_close = close_mask & _closes_for_opens(leg_open, open_mask, close_mask)
+        # `leg.label` collides whenever two distinct LegSpecs describe the same
+        # column name (e.g. same right/direction/strike_rule but different lot
+        # counts) -- disambiguate rather than silently let the second group's
+        # entries/exits overwrite the first's in these dicts.
         name = leg.label
+        used_names[name] = used_names.get(name, 0) + 1
+        if used_names[name] > 1:
+            name = f"{name} #{used_names[name]}"
         premium_columns[name] = priced["premium"]
         entry_columns[name] = leg_open
         exit_columns[name] = leg_close
         leg_frames[name] = priced
-        is_short = is_short or leg.direction == "short"
+        column_legs.append(leg)
 
     close = pd.DataFrame(premium_columns, index=bars.index)
     entry_df = pd.DataFrame(entry_columns, index=bars.index)
     exit_df = pd.DataFrame(exit_columns, index=bars.index)
 
-    reference_leg = groups[0][0]
-    effective_lot = lot_size if lot_from_history else reference_leg.lot_size
-    quantity = reference_leg.lots * effective_lot
+    # Direction and size are resolved PER COLUMN, never collapsed to a
+    # portfolio-wide scalar: a strategy long calls on some days and short puts
+    # on others must not have one leg's direction or size bleed into the
+    # other's. Empirically verified against the installed vectorbt (1.0.0):
+    # `from_signals`'s `direction` and `size` accept a plain Python list
+    # broadcast positionally against `close`'s columns (this is the pattern
+    # vectorbt's own docstring uses, e.g. `direction = ['longonly',
+    # 'shortonly']`). A `pd.Series` keyed by column name does NOT broadcast
+    # the same way -- it raises a shape-mismatch error -- so these must stay
+    # plain lists in `close.columns` order, which is exactly the order
+    # `column_legs` was built in above.
+    is_short = any(leg.direction == "short" for leg in column_legs)
+    quantities = [
+        leg.lots * (lot_size if lot_from_history else leg.lot_size)
+        for leg in column_legs
+    ]
+    directions = [
+        "shortonly" if leg.direction == "short" else "longonly" for leg in column_legs
+    ]
 
     portfolio = vbt.Portfolio.from_signals(
         close,
         entry_df,
         exit_df,
-        size=float(quantity),
+        size=[float(q) for q in quantities],
         size_type="amount",
-        direction="shortonly" if is_short else "longonly",
+        direction=directions,
         init_cash=float(init_cash),
         fees=vbt_fees(costs),
         slippage=float(slippage.premium_pct),
@@ -273,9 +302,15 @@ def run(
             )
 
     if not lot_from_history:
+        # Report the flat lot size(s) actually used per leg (usually one
+        # shared value, but heterogeneous legs may assume different flats).
+        distinct_lots = sorted({leg.lot_size for leg in column_legs})
+        lot_desc = (
+            str(distinct_lots[0]) if len(distinct_lots) == 1 else str(distinct_lots)
+        )
         warnings.append(
             f"No point-in-time lot history found; assumed a flat lot size of "
-            f"{quantity // max(groups[0][0].lots, 1)}. NIFTY's lot size changed "
+            f"{lot_desc}. NIFTY's lot size changed "
             "during this sample, so early-period notionals are approximate."
         )
     if is_short:
@@ -284,6 +319,17 @@ def run(
             "return-on-capital is optimistic."
         )
 
+    # A single scalar quantity is only meaningful when every leg trades the
+    # same amount, which is the only case the pre-fix engine ever produced
+    # correctly (and is bit-identical to it here). Heterogeneous per-leg sizes
+    # -- newly correct, not previously reachable -- are reported per column
+    # instead of forcing a misleading single number.
+    quantity_per_trade: int | dict[str, int] = (
+        quantities[0]
+        if len(set(quantities)) == 1
+        else dict(zip(close.columns, quantities, strict=True))
+    )
+
     assumptions = {
         "strategy": strategy_name,
         "params": merged,
@@ -293,7 +339,7 @@ def run(
         "stt_rate": costs.stt_rate,
         "init_cash": init_cash,
         "lot_from_history": lot_from_history,
-        "quantity_per_trade": quantity,
+        "quantity_per_trade": quantity_per_trade,
         "time_basis": "calendar" if calendar_time else "trading",
     }
```

(The `import` line and the `entries = shift_signals(...)` block visible in a
plain `git diff` on this file belong to Agent B's concurrent edit to the
signal-shift call site; not authored by me, shown here only because it's the
same file.)

### The short-margin warning, `quantity_per_trade`, and the lot-size warning

- **`is_short` warning (engine.py, "Short legs present..."):** now
  `any(leg.direction == "short" for leg in column_legs)` — still fires the
  moment *any* resolved leg is short, exactly as before, just computed from
  the per-column leg list instead of a fold during the pricing loop.
- **`quantity_per_trade`:** scalar (identical type/value to before) when
  every leg's computed quantity is equal — the only case the pre-fix code
  ever got right, and the only case the shipped strategies produce. When
  legs differ, it's a `{column_name: quantity}` dict instead of a misleading
  single number — a shape that literally could not occur before the fix, so
  there's no backward compatibility to preserve there.
- **Lot-size warning ("assumed a flat lot size of ..."):** now reports the
  distinct `leg.lot_size` value(s) actually used across `column_legs`
  (`{75}` -> `"75"` for the homogeneous case, same text as before since
  `quantity // groups[0][0].lots` was just a roundabout way of computing
  `reference_leg.lot_size`; a `[75, 100]`-style list if legs genuinely differ).

---

## Tests (`tests/test_engine_legs.py`) — the first end-to-end `engine.run` tests

A test-only strategy, `engine_legs_test_strategy`, is registered inside the
test module (per the task's own suggestion) rather than as a plugin file: it
fires at a fixed bar-of-day on however many of the fixture's leading days
it's given a `LegSpec` for, using the same per-bar `legs` shape `orb` uses.

All four tests use a trivial constant-IV `VolModel` (`_ConstantVol`, defined
in the test file) instead of the real `gk_vrp` model. `obt.vol.base.VolModel`
is a `Protocol` precisely so callers can substitute one; `gk_vrp.py` was
under **concurrent, independent edit by another agent** for the entire
duration of this task (its realized-vol warmup semantics visibly changed
under me mid-session — see the Measurement section), and these tests exist to
pin down `obt.engine`'s direction/size handling, not `gk_vrp`'s warmup
requirements. Using the real model would have made these tests intermittently
fail depending on unrelated, in-flight work in a file I don't own.

1. **`test_mixed_long_call_short_put_executes_each_leg_correctly`** — the
   exact failure in the finding: a long call opens day 1, a short put opens
   day 2. Asserts, from `portfolio.trades.records_readable`: each column's
   `Direction` is right (`Long`/`Short`, not both `Short`), each column's
   `Size` is right (not both sized off the call), each trade's entry/exit
   land on the correct, single day, the short-margin warning still fires,
   and `quantity_per_trade` stays a scalar (sizes are equal even though
   direction differs). **Verified it fails on the pre-fix code**: rather than
   `git stash`-ing the shared `engine.py` (risky while Agent B is concurrently
   editing lines 176-192 of the same file), I reimplemented the exact pre-fix
   scalar computation
   (`direction="shortonly" if any(leg.direction=="short" for leg in groups) else "longonly"`,
   `size=float(groups[0][0].lots * effective_lot)`) in a standalone script
   against the same priced/entries/exits this test builds, and ran it: the
   long-call column comes back `Direction: Short` (confirmed output:
   `Column='long atm call', Size=75.0, Direction='Short'`) -- the exact bug,
   reproduced.
2. **`test_mixed_size_same_direction_uses_correct_per_leg_amount`** — two
   long ATM calls, 1 lot vs 2 lots. `small.label == big.label` is asserted
   directly (documenting the collision), then asserts both columns survive
   as distinct trades with the correct, different sizes, `leg_frame` carries
   two distinct top-level columns, and `quantity_per_trade` becomes a dict
   once sizes genuinely differ. **Verified it fails on the pre-fix code** the
   same way (standalone script, no shared-file edits): `_resolve_legs`
   correctly resolves 2 groups, but building `premium_columns`/`entry_columns`
   keyed by the un-disambiguated `leg.label` collapses them to **1** dict
   entry (`num groups resolved: 2`, `num distinct column names: 1`) --
   the second leg's entries silently overwrite the first's, so only one
   trade would ever reach the portfolio, sized off whichever leg won the
   dict-write race. (Separately, the scalar `size` would also have sized
   whatever survived off `groups[0][0]`'s amount only.)
3. **`test_buy_open_matches_prefix_scalar_portfolio`** — the no-regression
   guard. Rebuilds the pre-fix call by hand (same `pinned_leg`/`resolve_trades`
   pieces `engine.run` uses internally, single premium column, scalar
   `size=float(quantity)` / `direction="...only"`) and asserts the resulting
   portfolio is bit-identical to `engine.run`'s (`total_return`, the entire
   `.value()` series via `pd.testing.assert_series_equal`, and every trade's
   `PnL`/`Size`) to the actual fixed engine's output for `buy_open`. Both
   sides use the identical premium/entry/exit input, so this isolates
   exactly the scalar-vs-list broadcasting change and nothing else.
4. **`test_strike_and_expiry_frozen_between_a_fill_pair`** — proves the
   premium column reaching the portfolio came from a frozen strike: same-day
   entry/exit, then inspects `result.leg_frame[leg.label]` over the window
   between them and asserts `strike` and `expiry` are each constant
   (`nunique() == 1`), even though `pinned_leg` carries the *previous*
   trade's values forward outside that window.

All four pass on current `HEAD` + my diff; ruff check/format clean for both
owned files:

```
uv run ruff check src/obt/engine.py tests/test_engine_legs.py     -> All checks passed!
uv run ruff format --check src/obt/engine.py tests/test_engine_legs.py -> 2 files already formatted
uv run pytest tests/test_engine_legs.py -q                        -> 4 passed
```

Ran the suite twice in a row to confirm it isn't flaky against the
concurrently-changing `gk_vrp.py` (it isn't, by construction — see above).

**Overlap note for Agent E** (also writing end-to-end engine tests): this
file is scoped to direction/size/legs mechanics only — mixed direction,
mixed size, the `leg.label` collision, the homogeneous no-regression guard,
and strike/expiry pinning across a fill pair. It does not touch censored-sample
detection, block-edge pricing, lot-history warnings' triggering conditions
(only their *content* once triggered), or any strategy other than `buy_open`
and a test-only stub — plenty of engine surface left for E to cover without
duplicating this.

---

## Measurement: homogeneous path, and cross-agent contamination

### What I can prove directly: bit-identical under a stable vol model

`test_buy_open_matches_prefix_scalar_portfolio` is the rigorous version of
this claim: same premium data in, pre-fix scalar `from_signals` call vs
post-fix per-column-list call, and the two portfolios are asserted
bit-identical (return, full value curve, every trade's PnL and size). This
holds regardless of what `gk_vrp.py` is doing this minute, because both
branches consume the identical `close` DataFrame.

### What `scripts/run_orb.py` shows on the real data right now, and why

Baseline (pristine, pre-any-agent-change, captured in
`reports/baseline_pristine_orb.txt` / `baseline_pristine_canary.txt`):

```
orb:       -27.68% total return, daily Sharpe -2.25, max DD -28.05%, 871 trades, 31.46% win rate
buy_open:  -45.94%, 877 trades
```

Running `uv run python scripts/run_orb.py orb --no-sensitivity` against the
current working tree (my `engine.py` fix + Agent B's signal-shift edits +
Agent A's in-flight `gk_vrp.py` edits, all uncommitted and concurrent) gives:

```
orb:       -35.67% total return, daily Sharpe -3.18, max DD -35.76%, 792 trades, 29.67% win rate
                (79 of 871 intended trades CENSORED -- "cash ran out")
buy_open:  -51.29%, 797 trades
                (80 of 877 intended trades CENSORED)
```

Both runs' own embedded sanity checks **fail outright** ("median nan% of
spot", "IV range nan-nan") — the premiums are NaN for a real slice of the
sample right now. That is not something my diff touches; `engine.py` doesn't
compute IV. I traced it: `obt/vol/plugins/gk_vrp.py` is mid-edit by Agent A
for the entire duration of this task (I observed its `git diff` appear,
disappear, and reappear between my own tool calls as they iterated), and its
new `atm_iv` requires `seed_days` (default 10) sessions of prior in-block
history before returning a non-NaN value, with no `bfill` fallback for the
sessions that don't have it — by design, to avoid a lookahead leak, per its
own docstring. Directly measured: `atm_iv(bars).isna().mean()` over the full
dataset is currently **9.1%** (the first ~10 sessions of each of the 8 data
blocks). A full-series `.min()`/`.max()` without `skipna` then turns any NaN
into a NaN aggregate, which is why *every* sanity check line reads "nan"
even though ~91% of the surface is fine.

To isolate my change from theirs, I monkeypatched `GkVrpVolModel.atm_iv` at
runtime (no file edits) to `.bfill()` its output -- i.e., undo just the
warmup-NaN behaviour -- and reran both strategies through the *current*,
already-patched `engine.run`:

```
orb,       bfill-patched vol:  871 trades (exactly the pristine baseline count), -36.34% total return
buy_open,  bfill-patched vol:  877 trades (exactly the pristine baseline count), -58.62% total return
```

Trade counts return to **exactly** the pristine baseline (871 / 877) the
moment the warmup-NaN censoring is neutralized -- confirming the trade-count
drift (871->792, 877->797) is caused by Agent A's in-flight warmup change,
not my fix. The **total return** figures still don't match the pristine
baseline exactly (-36.34% vs -27.68%, -58.62% vs -45.94%): that remainder is
Agent A's actual vol-formula change (EWMA smoothing moved into variance units
before the square root, `sqrt(E[V])` instead of `E[sqrt(V)]`, per their
in-flight docstring) legitimately repricing every premium, plus whatever
residual effect Agent B's session-boundary-aware `shift_signals`/`shift_legs`
has on which signals survive a shift. Both are real, intentional, in-scope
changes by agents who own those files -- not something `engine.py`'s
direction/size fix could cause or should compensate for.

**Conclusion: my fix moves the homogeneous-path baseline by exactly zero**,
demonstrated directly (the bit-identical unit test, immune to vol-model
churn) and indirectly (trade counts return to the exact pristine figures once
the *other* agents' concurrent, unrelated changes are neutralized one at a
time). The residual drift in the full-repo run belongs to Agents A and B's
own in-flight work and should be judged against their own before/after
measurements, not mine.

---

## What I could not do (files I don't own)

- Could not touch `src/obt/vol/plugins/gk_vrp.py` even to note that the new
  `atm_iv` warmup-NaN behaviour will, once it lands, need `report.py`'s
  sanity checks (`obt/report.py`) to use `skipna` aggregates, or every
  sanity check will read "nan" for any run whose sample includes a data
  block's first `seed_days` sessions -- currently every real run, since the
  dataset has 8 blocks. Flagging this for whoever owns `report.py` / is
  reviewing Agent A's change.
- Could not touch `src/obt/chain.py` to wire up `LegSpec.signed_quantity`
  now that a caller for it was ready to exist -- decided against calling it
  at all (see "Fix chosen" above), so it remains unused. Worth a follow-up
  decision on whether to remove it or find it a real caller; not this
  finding's scope either way.
- Left the `entries = shift_signals(...)` / `exits = shift_signals(...)` /
  `shifted_legs = ...` block (engine.py ~176-192) untouched per Agent B's
  ownership; my test file calls the new two-argument `shift_signals(mask,
  dates)` / `shift_legs(legs, dates)` signature Agent B introduced there,
  confirmed by reading their current call site rather than assuming the old
  one-argument signature.
