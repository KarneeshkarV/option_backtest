# Agent B — expiry-calendar and time/signal fixes

Owned files: `src/obt/calendar.py`, `src/obt/signals.py`, and only the
signal-shift call site in `src/obt/engine.py` (the `entries =
shift_signals(...)` / `exits = shift_signals(...)` / `shifted_legs = ...`
block plus its comment, currently around lines 176-185). Tests:
`tests/test_calendar.py`, `tests/test_session_and_gaps.py`.

`src/obt/engine.py` is shared with Agent C, who owns leg direction/sizing
(the rest of `run()`). The full-file diff below therefore contains their
concurrent work too; my authored hunk is isolated separately.

---

## Finding 3 — the expiry-weekday transition date was wrong, and the brief's own source was superseded

Held back until the orchestrator settled the underlying fact (Agent D's
research, cross-checked by the orchestrator against the primary circular
text and the real blast radius). Full research trail:
`reports/agent_d_calendar_research.md`.

### What changed and why

The brief cited only NSE/FAOP/68685 (2025-06-23) and a switch date of
`2025-09-01`. That circular was superseded two days later by NSE/FAOP/68747
(2025-06-25), an explicit "partial modification" of 68685 using a **different
mechanism**: 68685 alone implies existing Thursday contracts get pushed
forward to the following week's Tuesday (front weekly on the transition day
would be **2025-09-09** — a third, also-wrong answer, different from both the
current code's 2025-09-04 and the correct 2025-09-02). 68747 instead leaves
existing Thursday contracts through 2025-08-28 unchanged and had already
pre-listed the new Tuesday contracts (02-Sep, 09-Sep, 16-Sep, 23-Sep) on the
ordinary rolling schedule well before the cutover — a clean cutover, not an
elongation. Cross-checked against a pre-change bulletin (Zerodha, 2025-06-17)
and a same-day report (Angel One, dateline 2025-09-02T08:37:45+05:30,
before market open) that both independently confirm 2025-09-02 as the actual
first Tuesday expiry traded.

`src/obt/calendar.py:44` (`WEEKLY_EXPIRY_RULES`) changed from
`(date(2025, 9, 1), 1)` to `(date(2025, 8, 29), 1)` — no new transition
logic, exactly the one-line boundary correction the orchestrator specified.
Boundary rationale (also the orchestrator's): `expiry_weekday` must return 3
for `as_of == 2025-08-28` (last old-regime trade date, resolves to itself)
and 1 for `as_of == 2025-08-29` (first new-regime trade date, front weekly =
2025-09-02) — not `2025-09-01` (three calendar days / one trading day too
late) and not `2025-08-28` (would wrongly flip that date's own resolution).
I did not add any elongation/special-case logic, per the orchestrator's
explicit instruction — the existing walk-forward-then-holiday-rollback
algorithm in `weekly_expiry_for` already reproduces the correct table with
only the constant changed.

Also updated the module docstring (`calendar.py:6-11`, previously said "moved
to Tuesday from 2025-09-01") and the `WEEKLY_EXPIRY_RULES` comment to cite
both circular numbers and explain why 68685 alone is a trap (several
reputable secondary sources report the superseded 68685 mechanism as final,
per Agent D's research), so the next reader doesn't repeat the mistake.

### Diff

```diff
--- a/src/obt/calendar.py
+++ b/src/obt/calendar.py
@@ -4,11 +4,22 @@ Two things here quietly corrupt everything downstream if they are wrong, so
 both are explicit and both are tested:
 
 **The expiry weekday changed.** NIFTY weekly options expired on Thursday for
-most of this sample and moved to Tuesday from 2025-09-01. It is encoded as a
-dated rule table rather than a hardcoded weekday, so the switch is one line to
-correct and visible in review. *Verify this date against the NSE circular
-before trusting results that straddle it* -- a wrong switch date silently
-misprices a year of options rather than raising anything.
+most of this sample and moved to Tuesday. It is encoded as a dated rule table
+rather than a hardcoded weekday, so the switch is one line to correct and
+visible in review.
+
+The transition was announced in NSE/FAOP/68685 (2025-06-23), but that circular
+was superseded two days later by NSE/FAOP/68747 (2025-06-25) -- a "partial
+modification" using a different mechanism. 68685 alone would have this switch
+elongate the already-listed Thursday contracts forward to the following week's
+Tuesday; 68747 instead left existing Thursday contracts (through 2025-08-28)
+unchanged and had already pre-listed the new Tuesday contracts (02-Sep,
+09-Sep, ...) on the ordinary rolling schedule, so the real transition is a
+clean cutover, not an elongation. The effective boundary is 2025-08-29 (the
+first trading day the new Tuesday contracts govern), not 2025-09-01 -- verify
+*both* circular numbers, not just the first, before trusting results that
+straddle this date; a wrong switch date silently misprices options rather
+than raising anything.
 
 **Holidays come from the data, not a hardcoded list.** The set of trading days
 is whatever the loaded bars contain. If a nominal expiry has no session, expiry
@@ -32,10 +49,15 @@ import pandas as pd
 from obt.session import BARS_PER_SESSION, TRADING_DAYS_PER_YEAR, stamps
 
 #: ``(effective_from, weekday)`` with Monday=0. Most recent applicable wins.
-#: Thursday=3 historically; Tuesday=1 from 2025-09-01.
+#: Thursday=3 historically; Tuesday=1 from 2025-08-29 -- the first trade date
+#: governed by the new-regime contracts per NSE/FAOP/68747 (2025-06-25),
+#: which superseded NSE/FAOP/68685 (2025-06-23). NOT 2025-09-01: that date is
+#: three calendar days (one trading day) too late, and NOT 2025-08-28, which
+#: would wrongly flip 2025-08-28's own resolution (it is the last date the
+#: old Thursday regime governs, per the circular).
 WEEKLY_EXPIRY_RULES: tuple[tuple[date, int], ...] = (
     (date(1900, 1, 1), 3),
-    (date(2025, 9, 1), 1),
+    (date(2025, 8, 29), 1),
 )
```

(This is the finding-3-only hunk of `calendar.py`; the `tau_years`
docstring/formula changes shown separately below belong to finding 5.)

### Tests

`tests/test_calendar.py`:

- **Updated** `test_expiry_weekday_switches_to_tuesday`: the old assertion
  `expiry_weekday(date(2025, 8, 31)) == 3` would fail after the fix (2025-08-31
  is a Sunday and never exercised the real boundary at all). Replaced with the
  boundary-precise pair the orchestrator specified:
  `expiry_weekday(date(2025, 8, 28)) == 3` (last old-regime trade date) and
  `expiry_weekday(date(2025, 8, 29)) == 1` (first new-regime trade date).
- **Added `test_weekly_expiry_on_transition_day_is_september_2`**: pins the
  actual fact, with the circular numbers and the elongation-vs-cutover
  distinction in the docstring — `weekly_expiry_for(date(2025, 8, 28)) ==
  date(2025, 8, 28)` and `weekly_expiry_for(date(2025, 8, 29)) ==
  date(2025, 9, 2)`. No existing test exercised 2025-08-29 before this.

**Verified both fail on the old code.** Procedure: `git stash push -- <owned
files>` scoped to just `src/obt/calendar.py` and `tests/test_calendar.py`
(leaving Agent C's and Agent A's concurrent, unrelated edits to other files
untouched), confirmed the reverted pair still passed its *old* test suite,
then brought the *new* test file in on top of the *old* `calendar.py` via
`git checkout stash@{0} -- tests/test_calendar.py` and reran:

```
FAILED test_expiry_weekday_switches_to_tuesday
  assert 3 == 1
   where 3 = expiry_weekday(date(2025, 8, 29))
FAILED test_weekly_expiry_on_transition_day_is_september_2
  assert date(2025, 9, 4) == date(2025, 9, 2)
   where date(2025, 9, 4) = weekly_expiry_for(date(2025, 8, 29))
```

Both fail exactly as expected — the old boundary resolves 2025-08-29 to the
wrong Thursday-regime expiry (2025-09-04). Restored via
`git checkout stash@{0} -- src/obt/calendar.py tests/test_calendar.py`
(confirmed byte-identical to the stash via `git diff stash@{0} -- ...`
returning nothing) and dropped the stash.

### Measured effect (real sample)

Using the same clean-isolation method as findings 5/6 (Agent A's in-progress
`gk_vrp.py` temporarily stashed out; Agent C's concurrent `engine.py` leg
refactor verified bit-identical for `orb`/`buy_open` — see finding 5/6
sections below):

| metric | before finding 3 (findings 5+6 only) | after finding 3 (all three) | delta |
|---|---|---|---|
| orb total return % | -27.72 | **-27.71** | +0.01 pp |
| orb sharpe (daily) | -2.25 | -2.25 | 0.00 |
| orb max DD % | -28.10 | -28.09 | +0.01 pp |
| orb trades | 871 | 871 | 0 |
| buy_open total return % | -45.98 | **-45.97** | +0.01 pp |
| buy_open trades | 877 | 877 | 0 |

This matches the orchestrator's prediction almost exactly: a negligible move,
because only one session's tau is affected and the direction is "options were
overpriced on 2025-08-29 under the old code" (wrong Thursday expiry gives more
sessions of remaining time than the correct Tuesday expiry), so the fix makes
that one day's premiums slightly cheaper, nudging both already-negative total
returns very slightly less negative.

**The brief's "roughly the last 9 months affected" framing is wrong, and I'm
flagging that explicitly per the orchestrator's instruction.** The real NSE
mechanism (68747) is a clean cutover, not the elongation 68685 alone implies,
and the only trading days between the two candidate boundary dates
(2025-08-29, correct, and 2025-09-01, the brief's date) are a Friday followed
by a weekend. Every session from 2025-09-01 onward already resolved correctly
under the old code by coincidence — "next Tuesday from as_of" doesn't care
which of the two boundary dates governs once `as_of >= 2025-09-01`. I did not
re-derive the affected-session count myself (the orchestrator explicitly said
not to and supplied verified numbers); I did independently verify the
**headline-number** consequences by actually running the fixed code, which
the orchestrator's message did not itself do.

### Bars/trades actually affected in the real sample

Per the orchestrator's verified count (not re-derived): **1 of 877 complete
sessions** changes (2025-08-29: current code resolves 2025-09-04, corrected
resolves 2025-09-02), i.e. **375 bars** (one full session) get a corrected
`tau_years` value, all previously overstated (options too expensive that
day). No trades are added or removed for either strategy (871 / 877 both
strategies before and after) — the position on that day still force-exits at
the same bar, just against slightly cheaper premiums.

---

## Finding 5 — tau included the bar whose close is already being used

### What changed and why

`src/obt/calendar.py:142` and `:154` (`tau_years`) computed
`remaining_today = (BARS_PER_SESSION - bar_of_day) / BARS_PER_SESSION`.
Premiums and vectorbt fills happen at the **current bar's close**
(`chain.pinned_leg` prices off `bars["close"]`; `engine.py` builds the
vectorbt `close` frame from those premiums). By the time bar `i` is priced,
bar `i`'s minute has already elapsed, so time remaining should count only
bars strictly *after* `i`. The old formula returned `1/375` at the final bar
of an expiry session instead of exactly `0`, leaving artificial time value in
the forced 15:29 exit instead of intrinsic value.

Fix: `remaining_today = (BARS_PER_SESSION - 1 - bar_of_day) / BARS_PER_SESSION`
in both the trading-time branch (`:162`) and the `calendar_time=True` branch
(`:150`). At `bar_of_day = BARS_PER_SESSION - 1` this is now exactly `0`, and
at `bar_of_day = 0` it is `374/375`, matching the spec.

Also corrected the module docstring (`calendar.py:19-24`) and the `tau_years`
docstring, which previously said "this approaches one bar of time rather than
zero, so pricing degrades smoothly" — that justification described the bug,
not a feature, and now states the bar-close pricing convention explicitly.

Verified `src/obt/pricing/black76.py` (read-only, not edited): `price()`
treats `tau <= _MIN_TAU` (`1e-9`) as expired and returns pure intrinsic value
(`np.where(expired, intrinsic, value)`), so tau of exactly `0.0` is safe and
does not divide by zero (`tau_safe = np.maximum(tau, _MIN_TAU)` is used only
inside `_d1_d2`, and the `expired` mask overrides its output).

### Diff

```diff
diff --git a/src/obt/calendar.py b/src/obt/calendar.py
index cb0a533..971cef5 100644
--- a/src/obt/calendar.py
+++ b/src/obt/calendar.py
@@ -19,6 +19,12 @@ dataset instead of inventing expiries inside them.
 Time to expiry is measured in **trading time** (bars remaining / 375 / 252),
 not calendar time. Calendar time badly overstates overnight theta for weekly
 options -- a Friday-to-Monday hold is one trading day of decay, not three.
+
+**Pricing and fills happen at each bar's close** (see ``chain.pinned_leg`` and
+the premium frame in ``engine.py``), so bar ``i``'s minute has already elapsed
+by the time it is priced. Time remaining today counts only bars strictly
+after the current one, which makes tau exactly zero on the expiry session's
+final bar rather than one bar short of it.
 """
 
 from __future__ import annotations
@@ -120,9 +126,14 @@ def tau_years(
 ) -> np.ndarray:
     """Trading-time years to expiry for every bar.
 
-    On the expiry day's final bar this approaches one bar of time rather than
-    zero, so pricing degrades smoothly into intrinsic value instead of
-    discontinuously.
+    Pricing and fills use each bar's CLOSE (``chain.pinned_leg``, the premium
+    frame vectorbt trades in ``engine.py``), so by the time bar ``i`` is priced
+    that bar's minute is already gone. Time remaining today therefore counts
+    only bars strictly after the current one -- on the expiry session's final
+    bar there are none left, so tau is exactly zero and Black-76 (see
+    ``pricing.black76.price``, which returns intrinsic value for
+    ``tau <= _MIN_TAU``) prices it as pure intrinsic rather than decaying into
+    it a bar late.
 
     Set ``calendar_time=True`` to use wall-clock days/365 instead -- available
     for comparison, but it overstates decay for anything held overnight.
@@ -139,7 +150,7 @@ def tau_years(
             ],
             dtype="float64",
         )
-        remaining_today = (BARS_PER_SESSION - bar_of_day) / BARS_PER_SESSION
+        remaining_today = (BARS_PER_SESSION - 1 - bar_of_day) / BARS_PER_SESSION
         return np.maximum((days + remaining_today) / 365.0, 0.0)
 
     # Trading time: whole sessions strictly after today, plus today's remainder.
@@ -151,7 +162,7 @@ def tau_years(
         dtype="float64",
     )
     full_sessions_ahead = sessions - 1.0
-    remaining_today = (BARS_PER_SESSION - bar_of_day) / BARS_PER_SESSION
+    remaining_today = (BARS_PER_SESSION - 1 - bar_of_day) / BARS_PER_SESSION
     total_sessions = full_sessions_ahead + remaining_today
     return np.maximum(total_sessions / TRADING_DAYS_PER_YEAR, 0.0)
```

### Tests

`tests/test_calendar.py`:

- **Replaced** `test_tau_at_final_bar_is_one_bar_not_zero` (its name asserted
  the bug — "one bar not zero" is now false) with
  **`test_tau_at_expiry_close_is_exactly_zero_not_one_bar`**, using the
  `two_week_bars` fixture and a genuine in-sample expiry (2024-01-04, a real
  Thursday expiry, not a block-edge artifact). It asserts `tau == 0.0` at
  `bar_of_day == BARS_PER_SESSION - 1` and `tau == 374/375/252` at
  `bar_of_day == 0`, on that expiry day.
- **Added `test_black76_prices_expiry_close_as_intrinsic_not_a_stale_bar`**:
  the exact Rs 5.20-vs-Rs-0 regression case — ATM (`spot = strike = 20,000`,
  `vol = 0.20`), tau taken from `tau_years` at the expiry session's final bar,
  asserts `black76.price(...) == 0.0`.

Left `test_tau_shrinks_within_the_day` and
`test_tau_uses_trading_time_not_calendar_time` unchanged: the `bars` fixture's
resolved expiry for every session is 2024-01-03 (a block-edge artifact of the
3-session fixture stopping mid-week — see
`test_block_edge_sessions_flags_the_data_boundary`), which is not the first
day (2024-01-01), so `full_sessions_ahead > 0` on day one and tau stays
strictly positive and monotonically decreasing there; the fix only zeroes tau
at the actual expiry session's own final bar.

**Verified the new/changed tests fail on the old code.** Procedure: `git
stash push -- <owned files>` to revert to HEAD, confirmed old
tests+old code passed (26/26), then `git checkout stash@{0} -- <test
files>` to bring in the new tests while leaving old `calendar.py`/`signals.py`
in place, and reran:

```
FAILED tests/test_calendar.py::test_tau_at_expiry_close_is_exactly_zero_not_one_bar
  assert day_tau[bar_of_day == BARS_PER_SESSION - 1][0] == 0.0
  ...AssertionError (old value was 1/375/252, not 0)
FAILED tests/test_calendar.py::test_black76_prices_expiry_close_as_intrinsic_not_a_stale_bar
  assert np.float64(5.197916669372226) == 0.0
```

Then restored everything (`git checkout HEAD -- <test files>` +
`git stash pop`) and confirmed the diff against the stash was byte-identical
(`git diff stash@{0} -- ... ` empty) before dropping the stash.

### Measured effect (real sample)

Ran with `src/obt/vol/plugins/gk_vrp.py` (Agent A's concurrent, unfinished
edit) temporarily stashed out, isolating my two findings plus Agent C's
concurrent `engine.py` refactor (lines ~196-240) — which is bit-identical for
`orb`/`buy_open` on this dataset: both strategies use a single lot size/lots
value across all their leg columns, so Agent C's per-column
size/direction/name-disambiguation logic reduces to exactly the old scalar
path (verified: single distinct quantity, single distinct direction per
strategy). This is the cleanest attribution I can produce without editing
files I don't own.

| metric | baseline (given) | after fix (clean) | delta |
|---|---|---|---|
| orb total return % | -27.68 | **-27.72** | -0.04 pp |
| orb sharpe (daily) | -2.25 | **-2.25** | 0.00 |
| orb max DD % | -28.05 | **-28.10** | -0.05 pp |
| orb trades | 871 | **871** | 0 |
| orb win rate % | 31.46 | **31.57** | +0.11 pp |
| buy_open total return % | -45.94 | **-45.98** | -0.04 pp |
| buy_open trades | 877 | **877** | 0 |

Direction matches the finding's prediction: both strategies are long-only
options, so a cheaper (zero, not slightly-positive) premium at the forced
expiry-day exit means selling for a bit less, making returns slightly worse.
The moves are small because only the exit bars of expiry-day trades are
affected and the per-trade Rs-value is small relative to typical P&L swings
(see bar/trade count below), not because the fix is a no-op.

**Contamination note:** I also ran the *full* current working tree (with
Agent A's in-progress `gk_vrp.py` included) for transparency. It currently
**fails its own sanity checks** (`ATM straddle` / `put-call parity` / `no
negative premiums` / `IV clamps` all report `nan`) and prints "Sanity checks
failed -- results above are not trustworthy." That run is not usable for
attribution at all right now; the numbers above (Agent A's file stashed out)
are the only trustworthy ones I could produce, and are the ones I stand
behind.

### Bars/trades actually affected in the real sample

- 328,875 bars, 877 sessions total.
- **191 bars** are the final bar of an expiry session (`is_last_bar &
  date == resolved_expiry`) — these are exactly the bars whose tau changed
  from `1/375/252` to `0`.
- Of resolved trades: **orb: 108 of 871** force-exits land on such a bar;
  **buy_open: 191 of 877** (buy_open holds every session to the forced exit,
  so its count equals the full 191).

---

## Finding 6 — the one-row signal shift carried stale intent across sessions

### What changed and why

`src/obt/signals.py:30-40` (`shift_signals`) shifted purely positionally
(`out[1:] = values[:-1]`), with no session-boundary check. A signal on a
session's last row became an entry on the first row of the next *available*
session — which can be months later across one of this dataset's gaps. The
engine's force-exit guarantee (`resolve_trades` + `last_bar_of_day`) stops an
open *position* from spanning a gap; it does nothing about stale *intent*
crossing one, since the shift happens before `resolve_trades` ever sees the
signal.

Fix: `shift_signals` now takes `dates` and zeroes any shifted value whose
destination row's `date` differs from the source row's `date`, i.e. it never
carries a signal across any session boundary (not only gap boundaries — the
spec's "FIX" section calls for this unconditionally, and a same-day-vs-gap
distinction can't be made from inside a positional shift anyway). Added a
sibling `shift_legs` for the per-bar `LegSpec | None` object series
(`signals.legs`), which needed its own null-based "nothing survived the
shift" sentinel since it isn't boolean.

`shift_signals`'s signature change required updating its only production call
site, `src/obt/engine.py`'s `entries = shift_signals(signals.entries)` /
`exits = shift_signals(signals.exits)` / `shifted_legs =
signals.legs.shift(1)` block (my owned lines) to pass `bars["date"]` and to
call the new `shift_legs` helper instead of the old bare `.shift(1)`. I
grepped the whole repo for `shift_signals` first — the only other reference
was the existing unit test, which I also updated.

### Diff

```diff
diff --git a/src/obt/signals.py b/src/obt/signals.py
index 7d8d733..f866e6e 100644
--- a/src/obt/signals.py
+++ b/src/obt/signals.py
@@ -27,19 +27,50 @@ def last_bar_of_day(bars: pd.DataFrame) -> np.ndarray:
     return is_last
 
 
-def shift_signals(mask: pd.Series | np.ndarray) -> np.ndarray:
-    """Delay a signal by one bar.
+def shift_signals(
+    mask: pd.Series | np.ndarray, dates: pd.Series | np.ndarray
+) -> np.ndarray:
+    """Delay a signal by one bar, without crossing a session boundary.
 
     A signal computed from bar ``t``'s close cannot be acted on until bar
     ``t+1``. screener's vbt sweep applies the same shift; without it every
     backtest here would quietly trade on information it did not have.
+
+    The shift is otherwise purely positional, which is wrong at a session
+    boundary: a signal on a session's last row would become an entry on the
+    first row of the next *available* session, which can be months later
+    across one of this dataset's gaps. That is stale intent surviving a gap,
+    not a genuine next-bar fill, so it is dropped whenever the following row's
+    ``date`` differs from the current one.
     """
     values = np.asarray(mask, dtype=bool)
+    dates = np.asarray(dates)
     out = np.zeros_like(values)
-    out[1:] = values[:-1]
+    if len(values) > 1:
+        carried = values[:-1].copy()
+        carried[dates[1:] != dates[:-1]] = False
+        out[1:] = carried
     return out
 
 
+def shift_legs(legs: pd.Series, dates: pd.Series | np.ndarray) -> pd.Series:
+    """Same session-bounded one-bar delay as :func:`shift_signals`, for legs.
+
+    ``legs`` holds ``LegSpec | None`` objects rather than booleans, so "no
+    signal survived the shift" is spelled ``None`` instead of ``False``; it
+    otherwise must move in lockstep with :func:`shift_signals` or the leg
+    lookup in the engine misses every open.
+    """
+    dates = np.asarray(dates)
+    values = legs.to_numpy()
+    out = np.full(len(values), None, dtype=object)
+    if len(values) > 1:
+        carried = values[:-1].copy()
+        carried[dates[1:] != dates[:-1]] = None
+        out[1:] = carried
+    return pd.Series(out, index=legs.index)
+
+
 def resolve_trades(
```

`src/obt/engine.py` (my owned call-site lines only — the import line and the
`entries`/`exits`/`shifted_legs` block; everything else in the file below
this is Agent C's concurrent work, not mine):

```diff
-from obt.signals import last_bar_of_day, resolve_trades, shift_signals
+from obt.signals import last_bar_of_day, resolve_trades, shift_legs, shift_signals
...
-    # Guarantee 1: act on the next bar, not the signalling one. The per-bar leg
-    # choice must shift with the entries it belongs to, or the leg lookup below
-    # misses every open.
-    entries = shift_signals(signals.entries)
-    exits = shift_signals(signals.exits)
-    shifted_legs = signals.legs.shift(1) if signals.legs is not None else None
+    # Guarantee 1: act on the next bar, not the signalling one. The shift never
+    # carries a signal across a session boundary -- see obt.signals -- so
+    # stale intent from one session cannot resolve into an open on a later
+    # one, however far away, across one of this dataset's gaps. The per-bar
+    # leg choice must shift with the entries it belongs to, or the leg lookup
+    # below misses every open.
+    entries = shift_signals(signals.entries, bars["date"])
+    exits = shift_signals(signals.exits, bars["date"])
+    shifted_legs = (
+        shift_legs(signals.legs, bars["date"]) if signals.legs is not None else None
+    )
```

### Tests

`tests/test_session_and_gaps.py`:

- **Updated** `test_shift_signals_delays_by_one_bar` for the new signature
  (single session, ordinary one-bar carry still works).
- **Added `test_shift_signals_drops_stale_intent_across_a_gap`**: the exact
  spec case — two complete sessions dated 2024-01-01 and 2024-09-27 (270 days
  apart), a raw entry signal only at the first session's last bar. Asserts
  the shifted signal does **not** fire on the far session's first bar, and
  (since there is no bar left in the near session to carry it to either) that
  no entry survives at all.
- **Added `test_shift_legs_is_boundary_aware_like_shift_signals`**: same
  270-day two-session construction, a `LegSpec` placed at each session's last
  bar in the `legs` object series; asserts the far session's first-bar leg is
  `None` after the shift, and separately confirms an ordinary within-session
  one-bar carry still moves a `LegSpec` from bar 5 to bar 6 unchanged
  (identity-preserved, since `LegSpec` is a frozen pydantic model).

**Verified the new/changed tests fail on the old code**, using the same
stash procedure as finding 5:

```
FAILED test_shift_signals_delays_by_one_bar
  TypeError: shift_signals() takes 1 positional argument but 2 were given
FAILED test_shift_signals_drops_stale_intent_across_a_gap
  TypeError: shift_signals() takes 1 positional argument but 2 were given
FAILED test_shift_legs_is_boundary_aware_like_shift_signals
  ImportError: cannot import name 'shift_legs' from 'obt.signals'
```

(These fail via signature/`ImportError` rather than a silently-wrong value
because the API itself changed to take the boundary information it needs —
the old function has no way to express the fix at all, which is itself the
point.) Restored via `git checkout HEAD -- <test files>` + `git stash pop`,
confirmed identical to the stash, dropped the stash.

### Measured effect on headline numbers: none, and here's the proof

I computed, directly on the real 328,875-bar sample, both `orb`'s and
`buy_open`'s raw entries/exits through the **old** positional shift and the
**new** boundary-aware shift, then ran both through `resolve_trades`:

| strategy | raw entries differ (bars) | raw exits differ (bars) | `open_mask` differs | `close_mask` differs | trades old vs new |
|---|---|---|---|---|---|
| orb | 0 | 206 | 0 | 0 | 871 vs 871 |
| buy_open | 0 | 0 | 0 | 0 | 877 vs 877 |

`orb` emits **zero** raw entries on any session's last bar in this dataset
(matching the finding's own observation), so the entry side of the shift is
untouched. It does emit 207 raw *exit* signals on last bars, 206 of which
shift differently old vs. new — but every one of those lands on the first
bar of the next session, at which point `resolve_trades` is never
`in_position` (guarantee 2 already forces the close at the prior session's
own last bar), so the stray exit signal is a no-op either way. `open_mask`
and `close_mask` — the only things pricing/vectorbt ever see — are therefore
**bit-identical** old vs. new for both strategies on this dataset, so the
`-27.72%` / `-45.98%` numbers reported under finding 5 are unaffected by this
fix and are already the isolated, finding-5-only numbers. I did not need a
separate full backtest run to "isolate" finding 6 from finding 5 — the
identical-mask proof above is strictly stronger than a total-return
comparison would be (it shows equality at every intermediate stage, not just
the final scalar).

This is exactly what the finding predicted: the gap guarantee is currently
true "by luck rather than construction" on this dataset. The fix has zero
measurable effect today and matters for future data (any strategy that fires
raw entries on session-final bars, or a data extension that changes which
bars are session-final) or a different strategy.

### Bars/trades actually affected in the real sample

- **0 trades** change for `orb` or `buy_open` (proven above).
- 206 raw exit-signal bars (out of 328,875) shift to a different destination
  under the fix, all of them harmless no-ops given guarantee 2.
- 0 raw entry-signal bars are on any session's last bar for either strategy,
  so the entry side of the bug is not exercised at all by this dataset —
  consistent with the finding's note that the guarantee holds "by luck."

---

## What I could not do because I don't own the file

- `src/obt/engine.py` lines ~196-240 (leg grouping/sizing/direction) are
  Agent C's; I did not touch them beyond the minimal signature-following edit
  at the call site. I did verify (see "Measured effect" above) that their
  concurrent refactor is quantity/direction-neutral for `orb` and `buy_open`
  on this dataset, which is why I trust the clean numbers above as a fair
  attribution to my own two findings.
- Finding 3 (the NSE expiry-weekday transition date, `WEEKLY_EXPIRY_RULES` /
  `expiry_weekday`, `calendar.py:36-48`) is explicitly out of scope per the
  brief — left untouched, pending Agent D's research and the orchestrator's
  hand-off.
- I did not modify `src/obt/vol/plugins/gk_vrp.py`, `chain.py`,
  `calibration.py`, or any other file outside my ownership. The full working
  tree currently fails its own sanity checks because of Agent A's
  in-progress vol-model edit (see contamination note above) — that's their
  file, not mine to fix or report on further.
