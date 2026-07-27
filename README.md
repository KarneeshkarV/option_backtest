# obt — modular NIFTY option backtester

Intraday NIFTY option backtesting on vectorbt, with pluggable data sources,
strategies and volatility models.

## Read this first

**The premiums are synthetic.** The five-year dataset is NIFTY *index spot*
1-minute OHLC — no strikes, no expiries, no option prices. Every premium in a
backtest is Black-76 output computed from real spot and a modelled volatility.

The sibling `screener` project states the rule this deliberately breaks:

> P&L always uses observed chain prices, never model prices.

That inversion is the whole premise here, so it is surfaced rather than buried:
every report prints an assumption banner, sanity checks and a sensitivity table
over the assumptions that drive the result.

**What is real:** the spot path. Every P&L number traces to a NIFTY print that
actually happened. The missing months stay missing — nothing is simulated.

**What is measured:** the volatility *level*. `vrp_mult = 1.24` was fitted to
real observed NIFTY ATM quotes, not guessed — though to about one decimal
place, not three. See [Calibration](#calibration).

**What is still assumed:** skew, the term structure, and the absence of a
margin model. Named individually below so none of them can hide.

## Quick start

```bash
just sync          # installs, incl. screener from ../screener as a path dep
just check         # lint + typecheck + tests
just run-orb       # full report for the reference strategy
just calibrate     # synthetic premiums vs real observed quotes
just verify-seams  # prove the plugin architecture actually swaps
just plugins       # list everything registered
```

`just run-orb --quick` restricts to the last contiguous block; the sensitivity
grid is 16 full backtests, so `--no-sensitivity` is the fast path.

## The data

| File | What it is | Coverage |
|---|---|---|
| `NIFTY_1MIN_5YEAR (1).csv` | Index spot 1-min OHLC | 2021-06-17 → 2026-06-15, 891 days |
| `NIFTY_index_1min_2026-*.csv` | Index spot, second vendor | 2026-04-22 → 2026-07-21, 62 days |
| `NIFTY_ATM_{CE,PE}_1min_2026-*.csv` | **Real** ATM option quotes | 2026-04-22 → 2026-07-21, 60 days |
| `NIFTY_ATM_options_1min_2026-*.csv` | `concat(CE, PE)` (same rows, often reordered) — do not load alongside CE/PE | — |

**Observed-premium backtests.** By default premiums are still Black-76. To price
fills from the CE/PE CSVs instead:

```bash
just run-orb-observed          # ORB on traded quotes (2026 index spot + ATM options)
just run-canary-observed       # buy_open canary on the same path
```

Or in code: `engine.run(bars, "orb", option_source="nifty_atm_options_csv")`.
Spot should come from `nifty_index_csv` so timestamps align with the option window.

Cleaning keeps only complete 375-bar sessions (09:15–15:29), leaving 877 days
and 328,875 bars. Dropped: ~14 partial days, and 181 after-hours bars on three
dates — real Diwali Muhurat sessions, outside the session every strategy here
assumes.

**The sample has holes**, including all of Jan–Sep 2022. Reports print the eight
contiguous blocks rather than a min/max that would imply five continuous years.
No position can span a gap: the engine forces an exit on the last bar of every
session, so this is structural rather than something each strategy must
remember.

The two spot feeds overlap by 37 sessions and agree exactly on 96.9% of bars,
within 0.5 points on 99.0% — asserted in `tests/test_calibration.py`, which is
what makes the data-source seam a checked fact rather than a design claim.

## Architecture

```
src/obt/
  session.py       session shaping, gap detection      <- one definition of "a bar"
  datasource/      SpotSource Protocol + plugins/      <- seam 1
  vol/             VolModel Protocol + plugins/        <- seam 2
  pricing/         vectorized Black-76
  calendar.py      dated expiry rules, holiday rollback
  chain.py         strike selection, pinned_leg        <- the dangerous file
  costs.py         Indian F&O statutory stack
  signals.py       one-bar shift, EOD exit, trade resolution
  strategies/      registry + plugins/                 <- seam 3
  engine.py        vectorbt wiring
  calibration.py   synthetic vs real observed quotes
  report.py        banner, sanity checks, sensitivity
```

Each seam is a `typing.Protocol` plus a `Registry` of frozen pydantic specs and
an explicit-import `discover_plugins()` — the pattern `screener` already uses.
Adding a plugin is one file plus one import line, and
`tests/test_registries.py` globs each `plugins/` directory to catch the one
real failure mode: adding a file and forgetting the import, which otherwise
fails silently.

Reused from `screener` rather than reimplemented: `_registry.Registry`,
`options.greeks` (as the test oracle for the vectorized pricer),
`options.lot_history`, and the `backtester.costs` fee machinery.

### Three guarantees the engine owns

Strategies are not trusted with these, because a strategy that forgets one
still produces results that look fine:

1. **Signals shift one bar.** A rule computed from bar *t*'s close fills at
   *t+1*.
2. **Every position closes at its session's last bar.** This is what makes
   spanning a data gap structurally impossible.
3. **Strikes pin to resolved position opens**, never to raw entry signals.

### `pinned_leg` vs `rolling_atm`

The single most dangerous distinction in the package. `rolling_atm` re-picks the
strike every bar — useful for studying premium behaviour, **meaningless for
P&L**, because the strike silently rolls mid-position and the resulting chart
looks entirely plausible. `pinned_leg` freezes strike and expiry at the opening
bar. Only `pinned_leg` may reach a portfolio.

When a real option chain arrives, `pinned_leg` is the one function to replace.

## Calibration

`just calibrate` prices the exact contracts in the observed files — same strike,
same expiry, same right — and compares to what they actually traded at:

```
vrp_mult = 1.244  [95% CI 1.201-1.260]
  estimator  : median of per-session medians (one vote per day)
  sample     : 23 sessions / 13 weekly expiries -- NOT 7,100 independent bars
  reweighted : bar-weighted 1.226, per-expiry 1.257
  by right   : call 1.182, put 1.252
  by tau     : 2d 1.331, 3d 1.237, 4d 1.198, 5d 1.221
  in/out     : 1.254 on the first 14 sessions vs 1.199 on 9 held-out later sessions

  call : median  +1.36% | MAE Rs 13.1 | corr 0.9909
  put  : median  -3.33% | MAE Rs 12.9 | corr 0.9882
```

This moved `vrp_mult` from a textbook 1.15 to a measured 1.24. `vrp_mult` alone
decides whether premium-selling looks profitable, so the honest reading of that
number matters more than the number: **the retained rows are minutes, not
observations.** They come from 23 sessions across 13 weekly expiries, share one
realized-vol denominator per day, and autocorrelate heavily within a session —
so the fit clusters at the session level and reports a session-bootstrap band.
The reweightings disagree by more than that band is wide, and the held-out
cycles drift to 1.199. Read it as "about 1.2, one regime, three months".

An earlier fit of 1.31 was measured while the vol model smoothed volatility
instead of variance, which understated realized vol by about 7%; the multiplier
had silently absorbed that bias, so correcting the estimator required
re-fitting it down rather than keeping both corrections.

**Skew is not calibrated and cannot be from this data.** The files carry one
strike per weekly cycle, so the moneyness spread comes from spot drifting away
from a pinned strike over days, not from a smile across strikes at an instant.
Regressing implied vol on log-moneyness therefore measures the leverage effect
through *time* and returns a positive slope — the wrong sign for an equity
index. `fit_vol_params` refuses to report a skew estimate for that reason, and
the residual −5% put error is the size of what remains unmeasured.

Two further limits worth stating: the observed strike is ATM at the *start* of
each weekly cycle (spot drifts a median 144 points from it by the end, max 525),
and three months is one volatility regime.

## Strategies

- **`orb`** — opening range breakout. Bars 0–14 set the range; a break up buys
  an ATM call, a break down an ATM put; square off at 15:29. The reference
  strategy, chosen because it exercises every seam.
- **`buy_open`** — long ATM call from bar 5 to the close. A deliberate control:
  pure long premium, long decay. It *should* lose money. If it ever shows a
  profit, the pricer or the cost model is broken.

Neither is a recommendation. Both currently lose to spot buy-and-hold, which is
the honest result for long-premium intraday strategies paying 0.75% slippage.

## Known limitations

- **No margin model.** vectorbt has no notion of SPAN/ELM, so return-on-capital
  for any short-premium strategy is optimistic. Flagged in the run warnings.
- **No point-in-time lot history.** NIFTY's lot size changed (25 → 50 → 75)
  during the sample. `screener` refuses to fabricate that history and so does
  this; runs assume a flat 75 and say so.
- **Single-leg only.** Spreads need a custom order function. `LegSpec` is
  shaped to accept them later.
- **Censored samples are detected, not prevented.** A losing run can exhaust
  the account, after which vectorbt silently rejects orders and the backtest
  "completes" over a truncated period. The engine compares intended against
  executed trades and prints a loud warning.
- **Block-edge expiries.** Holidays are derived from data presence, which is
  right for real holidays and wrong at the trailing edge of a covered block:
  the calendar cannot tell "the exchange was shut" from "our file ends here",
  so it rolls the expiry back onto the block's last session. Those bars get a
  shortened time to expiry and therefore under-priced options. It affects 17 of
  877 sessions here; `block_edge_sessions()` finds them and the engine warns
  with the count of trades involved.
- **Tick floor.** Synthetic premiums are floored at ₹0.05, since Black-76
  prices a far-OTM expiring option at exactly zero, which is both untradeable
  and rejected by vectorbt. The real quotes bottom out at exactly ₹0.05 too.
