# All 9 findings — fixes, tests, and attributed movement

Baseline re-measured from a pristine `git worktree` at HEAD (`5cbea17`) with the
gitignored CSVs symlinked in, *not* from the working tree, so the "before"
column is uncontaminated by in-flight edits. It reproduced every stated number
exactly.

`obt.calibration.chain_available()` → **True** throughout, so the calibration
work is real and finding 7 was never silently skipped.

## Headline movement

| metric | baseline | final | net |
|---|---|---|---|
| orb total return | −27.68% | **−30.97%** | −3.29 pp |
| orb daily Sharpe | −2.25 | **−2.75** | −0.50 |
| orb max drawdown | −28.05% | **−31.15%** | −3.10 pp |
| orb trades | 871 | **792** | −79 |
| orb win rate | 31.46% | **30.43%** | −1.03 pp |
| buy_open canary | −45.94% | **−44.59%** | +1.35 pp |
| canary trades | 877 | **797** | −80 |
| ATM straddle median | 1.185% | **1.191%** | +0.006 pp |
| spot buy & hold | +50.12% | +50.12% | unchanged (untouched control) |

### Attribution, in the order the fixes landed

Order matters — these are not independent, and finding 7 explicitly had to
follow finding 1.

| step | orb return | trades | cause |
|---|---|---|---|
| baseline | −27.68% | 871 | |
| + findings 5, 6, 3 | −27.71% | 871 | tau/signals/calendar: **−0.03 pp** |
| + findings 1, 4, 8 | −35.67% | 792 | vol estimator: **−7.96 pp**, −79 trades |
| + finding 7 refit | **−30.97%** | 792 | `vrp_mult` 1.31→1.24: **+4.70 pp** |

The vol estimator moves the number most, and roughly 60% of that move is given
back by the refit — which is precisely the double-counting the brief warned
about. Had `vrp_mult` been left at 1.31, the reported loss would have been
−35.67% and about 4.7 points of it would have been the same correction counted
twice.

---

## 1. HIGH — EWMA smoothed volatility, not variance ✅

**Fix.** `_garman_klass_daily_variance` keeps daily GK in variance units;
`atm_iv` EWMAs the variance and calls `_annualize_vol` once at the end. Public
`garman_klass_daily` keeps its "annualized vol" contract byte-identically (it
had no external callers, but the docstring was a public promise).

**Test.** `test_ewma_smooths_variance_not_vol` — hand-calculates GK variance
independently, asserts output equals `sqrt(EWMA(variance))` and explicitly
*not* `EWMA(sqrt(variance))`. Verified to fail pre-fix.

**Effect.** Smoothed realized vol rose ~7% (median new/old **1.069**). Median
realized vol over the observed window 10.8% → 11.5%. Drove the −7.96 pp step
above (jointly with 4).

## 2. HIGH — one short leg turned every leg short; all legs inherited leg 0's size ✅

**Fix.** Direction and size are resolved **per column**. Verified empirically
that vectorbt 1.0 broadcasts plain Python lists positionally against
`close.columns` (a name-keyed `pd.Series` does *not*, and raises). Agent C also
found and fixed a related latent bug: `leg.label` collides when two `LegSpec`s
differ only in `lots`, so one leg's entries could silently overwrite another's;
columns are now disambiguated.

**Test.** `tests/test_engine_legs.py` — mixed long-call/short-put (pre-fix, the
call column comes back `Direction: Short`), mixed size (pre-fix collapses to one
column), plus a bit-identical no-regression guard.

**Effect.** **Zero** on headline numbers, as required — both shipped strategies
are single-leg long, and the homogeneous path is unchanged.

## 3. HIGH — expiry-weekday switch date ✅ CONFIRMED, but the brief was wrong twice

The brief cited only **NSE/FAOP/68685**. That circular was superseded two days
later by **NSE/FAOP/68747**, which explicitly "partially modifies" it and uses a
different mechanism. 68685 alone implies a front-weekly expiry of **2025-09-09**
— a third wrong answer. 68747 left existing Thursday contracts through
2025-08-28 alone and pre-listed the new Tuesday contracts, making it a clean
cutover. Cross-checked against a same-day report datelined
`2025-09-02T08:37:45+05:30` confirming the first-ever Tuesday expiry.

**Fix.** One line: boundary `2025-09-01` → **`2025-08-29`**. No transition logic
needed — the existing walk-forward + holiday-rollback algorithm reproduces the
whole 2025-08-20..09-15 table once the constant is right.

**Test.** `test_weekly_expiry_on_transition_day_is_september_2` pins
`weekly_expiry_for(2025-08-29) == 2025-09-02`. The old
`expiry_weekday(2025-08-31) == 3` assertion was replaced with the
boundary-precise pair — note 2025-08-31 is a **Sunday** and never exercised the
real case at all.

**Effect.** The brief's "roughly the last 9 months affected" is **not true**.
Measured directly: **1 of 877 sessions** (2025-08-29, 375 bars). Every session
from 2025-09-01 on already computed correctly by coincidence. Headline move
≈ +0.01 pp.

## 4. MEDIUM — warmup ignored `seed_days`, looked ahead on day one, carried state across gaps ✅

**Fix.** `seed_days` now drives `min_periods`; the lookahead `bfill()` is gone;
EWMA resets per `covered_periods` block. Unwarmed bars emit **NaN**.

**Tests.** `seed_days=1` vs `10` now differ (byte-identical before); causal
first-day output; gap-reset across a >7-day boundary.

**Effect.** 80 of 877 sessions (30,000 bars) are unwarmed → **−79 orb trades,
−80 canary trades**. This required two integration fixes the vol agent correctly
declined to make in files it did not own:

- `report.sanity_checks` was sampling NaN bars and failing **all five** checks.
  Unwarmed bars carry no surface by construction, so they are now excluded and
  the exclusion is printed. All five PASS again.
- `engine.run` let vectorbt silently drop 79 NaN-priced orders, which then
  resurfaced as a **"CENSORED SAMPLE … almost certainly because cash ran out"**
  warning. That was a misdiagnosis. Entry intent is now dropped on unwarmed
  bars and the warning states the real cause.
- `scripts/verify_seams.py` asserted "swapping the vol model never changes the
  schedule", which is no longer strictly true: a model that declines to price a
  bar legitimately removes it. Restated precisely — gk_vrp's trades must be a
  **subset** of `constant`'s on **identical timestamps**. This is a stronger
  check than before, and it passes (792 of 871).

## 5. MEDIUM — tau included the bar already being priced ✅

**Fix.** `remaining_today = (BARS_PER_SESSION - 1 - bar_of_day)/BARS_PER_SESSION`
in both the trading-time and calendar-time branches. The docstring that
justified the bug as deliberate smoothing was wrong and is corrected to state
the bar-close convention.

**Test.** tau is now exactly `0.0` at an expiry session's final bar. The old
`test_tau_at_final_bar_is_one_bar_not_zero` asserted the defect and was
rewritten — its name was itself a false statement.

**Effect.** Verified on real data: **191** expiry-session final bars now price at
exactly zero tau, no negative tau anywhere, first bar of session = 374/375. The
motivating case (spot = strike = 20,000) now prices to **Rs 0.00**, not Rs 5.20.
This is why the canary *improved* 1.35 pp — a long ATM call no longer sells
phantom time value at the forced exit.

## 6. MEDIUM — one-row shift carried stale intent across sessions and gaps ✅

**Fix.** `shift_signals(mask, dates)` zeroes any shift crossing a session
boundary; new `shift_legs` does the same for the object-dtype `LegSpec | None`
series (a bool-only helper would have silently mishandled it).

**Test.** Two sessions 270 days apart, signal only on the first session's last
bar — asserts no open on the far session.

**Effect.** **Zero trades changed** (0/871 orb, 0/877 canary), confirming the
finding's own diagnosis that the guarantee was "true by luck rather than
construction". It is now true by construction.

## 7. MEDIUM — VRP fit treated autocorrelated minutes as independent draws ✅

**Fix.** The estimator is now a **median of per-session medians** (one vote per
day). Added session/expiry cluster keys, a session-level **block bootstrap**,
sensitivity by right and tau bucket, and a **held-out** tail of 5 weekly cycles.
`VolFit.summary()` now leads with sessions and expiries and states outright that
the bar count is not the sample size.

**Refit.** **1.31 → 1.24.** Agent A reported the refit "barely moves it (1.3126)";
that was measured against a stale file state and is wrong. Against the corrected
estimator the fit is **1.226 bar-weighted / 1.244 equal-day**, because raising
realized vol by 7% necessarily lowers the multiplier by about the same.

```
vrp_mult = 1.244  [95% CI 1.201-1.260]
  sample     : 23 sessions / 13 weekly expiries -- NOT 7,100 independent bars
  reweighted : bar-weighted 1.226, per-expiry 1.257
  by right   : call 1.182, put 1.252
  by tau     : 2d 1.331, 3d 1.237, 4d 1.198, 5d 1.221
  in/out     : 1.254 on the first 14 sessions vs 1.199 on 9 held-out sessions
```

Reweightings disagree by **more than the CI is wide**, and held-out cycles drift
to 1.199 — so `gk_vrp.py` and the README now say to read it as "about 1.2, one
regime, three months", not three significant figures.

**Effect.** Call residual +4.26% → **+1.36%**; put +1.01% → −3.33% (put remains
contaminated by unvalidated skew, as documented). Headline +4.70 pp.

## 8. LOW — inverted IV clamp bounds silently flattened the surface ✅

**Fix.** Pydantic `model_validator(mode="after")` on `VolParams` rejects
`iv_cap < iv_floor`. Equality still accepted.

**Test.** Equality accepted; `GkVrpVolModel(iv_floor=0.80, iv_cap=0.06)` now
raises instead of returning a flat 0.06 surface.

**Effect.** None on headline numbers — the shipped config was already valid.

## 9. LOW — tautological tests, no load-bearing integration coverage ✅

See `reports/agent_e_tests.md`. Tests **65 → 93**. The dead `or True` is gone,
the circular golden check is replaced by a synthetic fixture that recovers an
injected `vrp = 1.5` to 1e-3, and `engine.run` now has real end-to-end coverage.

**A real bug surfaced here:** `chain.py` justified the Rs 0.05 tick floor as
"conservative … a long that should have died worthless still pays to close". A
long *sells* to close. The floor is **anti-conservative for longs** (+Rs 3.75
per affected trade at a 75-unit lot) and conservative for shorts. Both shipped
strategies are long, so it flatters results — 156 of 792 orb trades exit at the
floor, ≈ Rs 585, 0.03% of capital. The comment is corrected; the floor itself is
unchanged, since vectorbt rejects a zero price and the brief lists it as
confirmed-correct.

---

## Final state

- `just check` → ruff clean, ruff format clean, mypy clean on 31 source files,
  **93 tests pass**
- `scripts/verify_seams.py` → **all PASS**

## Process notes worth keeping

- **Never `git stash` on a shared working tree.** Two agents collided on the
  single global stash stack and silently reverted each other's files to HEAD.
  Verification of "does this test fail pre-fix" should copy through a scratch
  directory instead.
- `pyproject.toml` gained `extend-exclude = ["reports"]`: ruff formats Python
  snippets inside Markdown, which would have rewritten quoted circular text and
  before/after diffs held as evidence.
