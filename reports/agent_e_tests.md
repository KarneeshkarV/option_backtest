# Finding 9 — tautological tests, and no load-bearing integration coverage

Agent E was terminated mid-task by a session limit before it wrote its own
report. Its code landed and is sound; this write-up was reconstructed and
completed by the orchestrator, who also verified every claim below and closed
the gaps E did not reach.

## (a) The unfailable assertion

`tests/test_calibration.py:64` was `assert not chain.index.has_duplicates or True`.
The `or True` made it unfailable for any input whatsoever.

Replaced in `test_observed_chain_is_shaped_as_expected` with the real
invariant: every timestamp appears **exactly twice**, and the two rows are one
call and one put — never a lone leg, never a triplicate.

Catches: a vendor file that starts dropping one side's quote for a minute, or
duplicating a row. Both would silently bias the fit; neither could fail before.

## (b) The circular golden check

`test_fitted_vrp_matches_the_configured_default` compared a default derived
from this dataset against a refit on the same dataset. It could only fail if
someone edited the constant — never because the estimator was wrong.

- Renamed to `test_configured_default_vrp_has_not_gone_stale` and its docstring
  now says plainly that it is a **staleness guard, not a validation**.
- Its sample-size assertion was `fit.n_atm_bars > 1_000`. That re-imported the
  exact error finding 7 exists to retire, so it now asserts on the cluster
  counts instead: `n_sessions >= 20`, `n_expiries >= 10`.
- **The real test** is `test_synthetic_injected_vrp_is_recovered`: it builds a
  synthetic chain priced at an independently chosen `vrp = 1.5` — deliberately
  far from 1.24 and outside the ±0.10 staleness band — and asserts the
  estimator recovers it to `abs=1e-3`. It does, for both `vrp_mult` and
  `vrp_bar_weighted`. Nothing in this test is read off the real data, so it can
  fail because the estimator is wrong.
- `test_equal_day_estimator_resists_a_bar_heavy_session` is the test finding 7
  most needed and nothing covered: one session contributes far more retained
  minutes at a different ratio. The equal-day estimator holds at **1.2**; the
  bar-weighted one is dragged to **2.0**. That is the entire argument for
  clustering, now executable.

## (c) Tick-floor terminal value — A REAL FINDING

The old test asserted observed quotes >= tick size, which is a property of the
input data, not of our code.

`tests/test_terminal_value.py` (new, 6 tests) tests the thing that matters:
whether flooring every **modeled** mark at Rs 0.05 is financially correct on
both sides. **It is not, and `chain.py`'s stated justification was backwards.**

`chain.py` claimed the floor "points the conservative way -- a long that should
have died worthless still pays to close." A long does not pay to close; it
*sells*. Verified numerically at a 75-unit lot, entry Rs 50, true exit 0:

| side | true P&L | floored P&L | bias |
|---|---|---|---|
| long  | −3750.00 | −3746.25 | **+3.75 — floor HELPS the long** |
| short | +3750.00 | +3746.25 | **−3.75 — floor HURTS the short** |

So the floor is **anti-conservative for longs** and conservative for shorts,
exactly opposite to the comment. Both shipped strategies are long, so the bias
flatters the headline numbers. Measured footprint: **156 of the 792 `orb`
trades** exit exactly at the floor → about Rs 585, or 0.03% of starting
capital. Small, but documented backwards, which is how small errors survive.

The `chain.py` comment has been corrected to state the asymmetry, the
direction, and this measured magnitude. The floor itself was **not** changed —
it remains load-bearing (vectorbt rejects a zero price outright), and the
task's "confirmed correct" list says so.

## (d) End-to-end `engine.run` coverage

Nothing previously ran `engine.run` end to end. Now, in
`tests/test_engine_end_to_end.py` (6 tests), complementing rather than
duplicating Agent C's `tests/test_engine_legs.py` (direction/size/legs):

- `test_entry_and_exit_timestamps_match_the_shifted_signal` — entries land on
  the bar *after* the signal (guarantee 1), asserted against real vectorbt
  trade records.
- `test_same_day_closure_holds_across_every_trade_in_the_run` — guarantee 2.
- `test_position_never_spans_a_data_gap` — on a fixture with a multi-month hole.
- `test_force_exit_fires_on_the_sessions_final_bar`.
- `test_strike_and_expiry_frozen_within_every_trade_and_re_pinned_across_them` —
  proves via `leg_frame` that the premium column reaching the portfolio came
  from a frozen strike (guarantee 3).
- `test_unwarmed_sessions_produce_no_trades_and_are_reported` — the finding-4
  integration behaviour, previously untested.

## Result

`just check`: ruff clean, ruff format clean, mypy clean on 31 source files,
**93 tests pass** (was 65). `scripts/verify_seams.py`: all PASS.
