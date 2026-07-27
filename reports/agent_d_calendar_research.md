# Finding 3 — NIFTY weekly expiry weekday switch (calendar.py:36-48)

## VERDICT: CONFIRMED

The front NIFTY weekly contract trading on Friday 2025-08-29 expired **Tuesday 2025-09-02**, not
Thursday 2025-09-04 as `calendar.py`'s current `WEEKLY_EXPIRY_RULES` boundary (`2025-09-01`)
computes. The claim in finding 3 is correct. However, getting to that verdict required going past
the single circular named in the finding — that circular was itself superseded two days later, and
using it alone would have produced a *third*, also-wrong answer. Details below.

---

## 1. Primary source(s) actually retrieved

`WebFetch` on the circular URL timed out. Fallback per instructions: `curl` with a browser
`User-Agent` succeeded (HTTP 200) against `nsearchives.nseindia.com`; text extracted locally with
`pypdf` (installed to a scratch dir via `pip install --target`, not added to the repo's
`pyproject.toml`). Files kept under
`/tmp/claude-0/-root-screneer-main-option-backtest/1305486c-c017-4b84-80e9-128549aa357f/scratchpad/`
(`FAOP68685.pdf`, `FAOP68747.pdf`).

### Source A — NSE/FAOP/68685, dated June 23, 2025 (the circular named in the finding)

> "Revision in Expiry Day of Index and Stock Derivatives Contracts... NIFTY weekly contracts:
> Current Expiry Day: Thursday of the week. Revised Expiry Day: Tuesday of the week."
>
> "1. The circular shall come into effect from August 28, 2025 (EOD) i.e. Expiry day for all
> existing contracts will be revised to "New Expiry Day" on August 28, 2025 (EOD).
> 2. Any new contract created for trading on / after effective date shall be created as per the
> revised expiry day...
> 3. Revised expiry date of all existing derivatives contracts shall be available in the contract
> file which shall be generated on August 28,2025 (EOD) end of the day which shall be applicable
> for trading on August 29,2025."

Its Annexure gives a worked table for NIFTY Weekly Expiry — **existing** (Thursday) date →
**revised** (Tuesday) date:

```
04-Sep-25 -> 09-Sep-25
11-Sep-25 -> 16-Sep-25
18-Sep-25 -> 23-Sep-25
01-Oct-25 (since 02-Oct-25 is holiday) -> 07-Oct-25
```

Read in isolation, 68685 says NSE would **push each already-listed Thursday contract forward five
calendar days to the following Tuesday** (Thu 04-Sep → Tue 09-Sep, not the Tue in the same week).
Under this mechanism the front weekly on 2025-08-29 would be the elongated Sep-4-contract-turned-
Sep-9, i.e. **2025-09-09**, not 2025-09-02. This is a different wrong answer from the current code
(which says 2025-09-04) and also different from the claim under test (which says 2025-09-02).

### Source B — NSE/FAOP/68747, dated June 25, 2025 — **"Update", explicitly amends 68685**

This circular's own words: "This is in reference and **partial modification to Exchange circular
reference 68685 dated June 23, 2025**..."

> "1. The expiry of already introduced i.e. existing index and stock derivatives contracts with
> expiry falling on / before August 31, 2025, will remain unchanged except for long dated option
> contracts in NIFTY & BANKNIFTY (i.e. September 2025 expiry and onwards).
> 2. Expiry day of existing long dated option contracts in NIFTY & BANKNIFTY (i.e. September 2025
> expiry and onwards) shall be modified to re-align with revised expiry day (i.e. Tuesday) on July
> 31, 2025 EOD...
> 3. Newly generated contracts with expiry falling on/after September 01, 2025 shall be introduced
> with revised expiry day (i.e. Tuesday) effective from June 26, 2025 EOD onwards..."

Its Annexure, row "1 NIFTY Weekly Contracts":

```
03-Jul-25 / 10-Jul-25 / 17-Jul-25 / 24-Jul-25: No Revision
07-Aug-25: To be generated with Thursday expiry on July 03, 2025 (EOD)
14-Aug-25: To be generated with Thursday expiry on July 10, 2025 (EOD)
21-Aug-25: To be generated with Thursday expiry on July 17, 2025 (EOD)
-> 02-Sep-25 (New weekly contract): To be generated with Tuesday expiry on July 24, 2025 (EOD)
-> 09-Sep-25 (New weekly contract): To be generated with Tuesday expiry on August 07, 2025 (EOD)
-> 16-Sep-25 (New weekly contract): To be generated with Tuesday expiry on August 14, 2025 (EOD)
-> 23-Sep-25 (New weekly contract): To be generated with Tuesday expiry on August 21, 2025 (EOD)
```

This is a **materially different mechanism**: existing Thursday contracts through 2025-08-28 are
left alone (not pushed forward); instead, brand-new Tuesday-expiry contracts (02-Sep, 09-Sep,
16-Sep, 23-Sep) were generated in advance on the ordinary rolling schedule (24-Jul, 07-Aug, 14-Aug,
21-Aug EOD respectively), so they are already listed and trading well before 2025-08-28. Once the
last old-regime Thursday contract (2025-08-28) expires, the nearest not-yet-expired contract on
2025-08-29 is the already-listed 2025-09-02 Tuesday contract — a clean cutover, no elongation.

**68747 is dated later than 68685 and explicitly amends it, with no further amendment found** (see
cross-checks below), so 68747's mechanism is the operative one, not 68685's.

## 2. Independent cross-checks

- **Zerodha bulletin**, 17 Jun 2025 (predates both circulars but matches 68747's eventual
  mechanism): "Contracts expiring on or before August 31, 2025 are unaffected. Starting September
  1, 2025: NSE: Expiry moves from Thursday to the following Tuesday. **The first Nifty weekly
  contract expires on September 2, 2025, Tuesday.**" —
  https://zerodha.com/marketintel/bulletin/417370/revision-in-expiry-day-of-index-and-stock-derivatives-contracts
- **Angel One, published live on the day itself** — dateline `2025-09-02T08:37:45+05:30` (8:37 AM
  IST, before market open on the day in question): *"Today marks a historic day for traders as the
  National Stock Exchange (NSE) holds its first-ever Tuesday expiry for all futures and options
  (F&O) contracts. After 25 years of Thursday being known as 'expiry day'... The change officially
  took effect after trading ended on August 28, 2025..."* —
  https://www.angelone.in/news/market-updates/nifty-weekly-expiry-today-see-what-changed-on-sept-1-2025
  This is same-day, real-time confirmation that 2025-09-02 was in fact the first Tuesday expiry
  actually traded, not a subsequently-abandoned plan.
- **Caveat found during cross-checking**: several reputable secondary sources (BusinessToday,
  ICICI Direct, Angel One's own separate June article, others) report the *68685* Sep4→Sep9
  mechanism as if it were final — they were evidently written on/around June 23 and never picked
  up the June 25 amendment. Anyone verifying this from a news search alone has a good chance of
  landing on the superseded version. This is exactly the trap the original finding partially fell
  into by citing only 68685.

Net: two independent sources anchored specifically in time around the actual event (a pre-change
bulletin describing the amended mechanism, and a live day-of report) agree: **2025-09-02, Tuesday,
is the correct first new-regime NIFTY weekly expiry**, confirming the claim under test.

## 3. Corrected (trade date → expected expiry) table, 2025-08-20 .. 2025-09-15

Built from the repo's own actual trading-day calendar (`NIFTY_1MIN_5YEAR (1).csv`, which already
reflects the 2025-08-27 Ganesh Chaturthi holiday — that date has zero bars and is correctly absent
below) plus the 68747-confirmed mechanism, which — unlike 68685's — is a plain cutover requiring
**no special-case elongation logic**: it is exactly the same "walk forward to the nominal weekday,
roll back on holiday" algorithm `ExpiryCalendar.weekly_expiry_for` already implements, with only
the switch date corrected.

| Trade date (as_of) | Weekday | Expiry (current code) | Expiry (correct) | Regime |
|---|---|---|---|---|
| 2025-08-20 | Wed | 2025-08-21 | 2025-08-21 | old (unaffected) |
| 2025-08-21 | Thu | 2025-08-21 | 2025-08-21 | old (unaffected) |
| 2025-08-22 | Fri | 2025-08-28 | 2025-08-28 | old (unaffected) |
| 2025-08-25 | Mon | 2025-08-28 | 2025-08-28 | old (unaffected) |
| 2025-08-26 | Tue | 2025-08-28 | 2025-08-28 | old (unaffected) |
| *2025-08-27* | *(holiday — no trading day)* | | | |
| 2025-08-28 | Thu | 2025-08-28 | 2025-08-28 | old — **last day old regime governs** |
| **2025-08-29** | **Fri** | **2025-09-04 (WRONG)** | **2025-09-02** | **new — first day new regime governs** |
| 2025-09-01 | Mon | 2025-09-02 | 2025-09-02 | new (already correct in current code) |
| 2025-09-02 | Tue | 2025-09-02 | 2025-09-02 | new (already correct) |
| 2025-09-03 | Wed | 2025-09-09 | 2025-09-09 | new (already correct) |
| 2025-09-04 | Thu | 2025-09-09 | 2025-09-09 | new (already correct) |
| 2025-09-05 | Fri | 2025-09-09 | 2025-09-09 | new (already correct) |
| 2025-09-08 | Mon | 2025-09-09 | 2025-09-09 | new (already correct) |
| 2025-09-09 | Tue | 2025-09-09 | 2025-09-09 | new (already correct) |
| 2025-09-10 | Wed | 2025-09-16 | 2025-09-16 | new (already correct) |
| 2025-09-11 | Thu | 2025-09-16 | 2025-09-16 | new (already correct) |
| 2025-09-12 | Fri | 2025-09-16 | 2025-09-16 | new (already correct) |
| 2025-09-15 | Mon | 2025-09-16 | 2025-09-16 | new (already correct) |

**The bug is narrower than the finding's framing suggested.** Because the actual NSE mechanism
(68747) is a clean cutover rather than a push-forward, and because the only trading days that fall
between the current wrong boundary (2025-09-01) and the correct boundary (2025-08-29) are a Friday
followed by a weekend, **exactly one trading session in the entire dataset resolves incorrectly:
2025-08-29 itself.** Every session from 2025-09-01 onward already happens to compute the right
answer under the current code, purely because "next Tuesday from as_of" is insensitive to which of
the two boundary dates is used once as_of ≥ 2025-09-01. There is no multi-month swath of mispriced
sessions — the earlier suspicion that "roughly the last 9 months of the sample are affected" is not
borne out; the transition-week worked example the finding leads with (2025-08-29 09:15) is in fact
the *only* trading day this bug touches.

## 4. Precise re-expression of the rule

The fix is a **one-line boundary change**, no new transition logic needed:

```python
WEEKLY_EXPIRY_RULES: tuple[tuple[date, int], ...] = (
    (date(1900, 1, 1), 3),
    (date(2025, 8, 29), 1),   # was date(2025, 9, 1) — wrong by 3 calendar days / 1 trading day
)
```

Rationale for the exact boundary value: `expiry_weekday(as_of)` must return **3 (Thursday)** for
`as_of == 2025-08-28` (the last old-regime trade date, which must resolve to itself) and **1
(Tuesday)** for `as_of == 2025-08-29` (the first new-regime trade date, front weekly = 2025-09-02).
`2025-08-29` is therefore the correct `effective_from` — not `2025-09-01` (too late by the one
trading day that matters) and not `2025-08-28` (would incorrectly flip Aug 28's own resolution to
Tuesday, which the circular explicitly does not do — Aug 28 is the "existing contract, unchanged"
side of the cutover).

No change is needed to `weekly_expiry_for`, `_next_week_expiry`, or the holiday-rollback logic —
the existing "walk forward to nominal weekday, roll back on holiday" algorithm already reproduces
the full table above once the single date constant is corrected, because the real-world transition
(68747) turned out to be a simple cutover rather than the contract-elongation transition the more
prominently-cited circular (68685) had originally proposed.

## 5. What `tests/test_calendar.py:35-36` must become

Current (locks in the wrong boundary):

```python
def test_expiry_weekday_switches_to_tuesday():
    assert expiry_weekday(date(2024, 6, 1)) == 3  # Thursday
    assert expiry_weekday(date(2025, 8, 31)) == 3
    assert expiry_weekday(date(2025, 9, 1)) == 1  # Tuesday
    assert expiry_weekday(date(2026, 1, 5)) == 1
```

Line 35 (`date(2025, 8, 31) == 3`) will **fail** once the boundary is corrected — 2025-08-31 ≥
2025-08-29, so `expiry_weekday` will (correctly) return 1. That assertion needs to flip to `== 1`,
but more importantly it should be replaced/supplemented with assertions that actually pin the real
boundary (2025-08-31 is a Sunday, not even a trading day, so it never exercised the interesting
case):

```python
def test_expiry_weekday_switches_to_tuesday():
    assert expiry_weekday(date(2024, 6, 1)) == 3          # Thursday, deep in old regime
    assert expiry_weekday(date(2025, 8, 28)) == 3          # last old-regime trade date
    assert expiry_weekday(date(2025, 8, 29)) == 1          # first new-regime trade date
    assert expiry_weekday(date(2026, 1, 5)) == 1           # Tuesday, deep in new regime
```

And a `weekly_expiry_for`-level test should be added (currently absent — no existing test exercises
2025-08-29 at all) asserting the specific fact this whole finding is about:

```python
def test_weekly_expiry_on_transition_day_is_september_2():
    # NSE circular NSE/FAOP/68747 (June 25, 2025), amending NSE/FAOP/68685 (June 23, 2025):
    # existing Thursday contracts through 2025-08-28 unchanged; new Tuesday contract 2025-09-02
    # already listed since 2025-07-24 EOD becomes the front weekly from 2025-08-29 trading.
    assert calendar.weekly_expiry_for(date(2025, 8, 28)) == date(2025, 8, 28)  # last old expiry
    assert calendar.weekly_expiry_for(date(2025, 8, 29)) == date(2025, 9, 2)   # first new expiry
```

Its passing today is not evidence for or against the boundary date — it only tests that
`expiry_weekday` is a monotone step function around whatever constant is in the code, not that the
constant is right. That is exactly why this had to be settled against the circular rather than the
test suite.

## 6. Affected-sessions count in the 5-year sample

Checked directly against the repo's own bar file (`NIFTY_1MIN_5YEAR (1).csv`, 2021-06-17 to
2026-06-15, 891 distinct trading days): only trade date **2025-08-29** resolves to a different
front-weekly expiry under the corrected boundary. That session has exactly **375 one-minute bars**
(`BARS_PER_SESSION` = 375, confirmed a full, non-truncated session). So:

- **1 of 891 sessions (0.11%)** in the entire sample has its resolved expiry field wrong under the
  current code.
- **375 bars** total get a wrong `tau_years` value (all of them priced against a 2025-09-04
  Thursday expiry instead of the correct 2025-09-02 Tuesday expiry — i.e. current code overstates
  tau for that day, consistent with the finding's directional claim that the current code
  overprices options on 2025-08-29).
- No other date in Sep 2025 – Jun 2026 is affected: the current code's per-week resolution from
  2025-09-01 onward already coincides with the corrected boundary's output, purely because the only
  trading day sitting between the two candidate boundary dates (2025-08-29 and 2025-09-01) is
  2025-08-29 itself (Aug 30–31 are a weekend). This directly contradicts the "roughly the last 9
  months of the sample are affected" framing in the task brief — that framing does not hold up
  against the corrected mechanism; the true footprint of this specific bug is one trading day.

## Files consulted / produced (scratchpad, not committed)

- `.../scratchpad/FAOP68685.pdf`, `.../scratchpad/FAOP68747.pdf` — downloaded circulars
- `.../scratchpad/zerodha_bulletin.html`, `.../scratchpad/angelone.html` — downloaded cross-check pages
- `.../scratchpad/pylibs/` — local `pypdf` install (`pip install --target`), not added to repo `pyproject.toml`
