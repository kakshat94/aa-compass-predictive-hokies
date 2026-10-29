# Vacation allocation and flex assignment

**Quick start:** set the rules in `vacation_flex_config.yaml`, then
`run_vacation(PHASE2_DIR, PHASE3_DIR, ROOT, FAIR_SECONDS, PHASE1_DIR=...)` followed by
`run_flex(PHASE2_DIR, PHASE3_DIR, ROOT, PHASE1_DIR)`, or `run_vacation_and_flex(...)`. All three are in `executable.py`.

## Flex assignment (`flex_pipeline.py`, `flex_assignment.py`)

A flex day turns an **off-day** into a workday. An off-day is a date the dispatcher is not scheduled for in
`Supply_Regular_qualified.csv`, is not training (`all_training_data.csv`), and has no pre-assigned flex
(`Flex_assignment_qualified.csv`). A dispatcher's remaining flex budget is 4 minus their pre-assigned flex days. By
default they may flex only onto their own `WORK_CENTER_ID` (never a secondary qualification), only on their own
shift, and at most once per date. Set `flex.home_work_center_only: false` to also allow qualified secondary desks.

The assignment is an exact min-cost flow, ranked:
1. Cover as much shortage as possible.
2. Prefer the dispatcher's own desk (and shift, if other shifts are allowed).
3. Spread flex days evenly.

It uses the same candidate rules as the vacation step's flex estimate. So the flex step covers exactly what the
vacation step expected, and the run fails loudly if the two ever disagree.

**How pre-assigned flex affects vacations.** Pre-assigned flex days on real off-days add a person to that desk, so
vacations can use them as cover. Every pre-assigned row, valid or not, uses up flex budget. Rows that land on a
scheduled workday or a training day add no cover and are listed in `preassigned_flex_conflicts.csv`. In the sample
files 114 of 230 rows are such conflicts (the file is from a different run). Training days are never workdays in the
sample, so they only block flex.

Outputs, in `flex_output_dir` (same names as the previous flex step): `flex_days_new.csv`, `flex_added_supply_rows.csv`,
`Flex_assignment_combined.csv`, `supply_after_vacation.csv`, `supply_after_vacation_and_flex.csv`,
`staffing_coverage_before_after_flex.csv`, `flex_by_dispatcher.csv`, `preassigned_flex_conflicts.csv`,
`flex_summary.json`.

## Configuration file (`vacation_flex_config.yaml`)

| Section | Keys | Meaning |
|---|---|---|
| `rule1_daily_total` | `enabled`, `min`, `max`, `tolerance_below`, `tolerance_above` | daily totals within [min − below, max + above]; off = any total |
| `rule2_fairness` | `slack_days`, `weight` | counts within L−slack … L+1+slack; weight 1000 = even split, ~1 = use the band freely |
| `rule3_inverse_volume` | `enabled`, `tolerance` | daily total = round(K / volume); tolerance on the proportional fit |
| `base_per_day` | `enabled`, `value`, `tolerance` | annual total within (value ± tolerance) × dates |
| `flex` | `max_days_per_dispatcher`, `home_work_center_only`, `same_shift_only`, `*_cost` | flex rules and preferences |

Input files and output folders are not in the YAML; `executable.py` sets them in the `run_*` functions:
`PHASE2_DIR/Supply_Regular_qualified.csv`, `PHASE2_DIR/all_training_data.csv`,
`PHASE1_DIR/Flex_assignment_qualified.csv`, `ROOT/inputs/annual_desk_needs_reference.csv`,
`ROOT/inputs/demand_volume_per_day_reference.csv`; outputs go to `PHASE3_DIR` (vacations) and
`PHASE3_DIR/flex_maxflow` (flex).

Unknown or misspelt keys are rejected.


Turns scheduled dispatcher workdays into vacation days. **The annual number of vacations is not fixed**: the model
chooses it. Produces

* `vacation_initial.csv`: the supply rows that became vacation
* `vacation_outputs_updatedFormat.csv`: all other supply rows (same columns, `SHIFT_DATE` as `YYYY-MM-DD`)

plus the files the flex step and reviewers need: `demand_imputation_audit.csv`, `vacation_initial_summary.json`,
`vacation_daily_targets.csv`, `vacation_fairness_by_dispatcher.csv`, `vacation_shortage_by_bucket.csv`, and
`vacation_scale_search.csv` (every candidate total the model tried). A full-year run takes about 30 seconds.

## Run

```python
# from executable.py (unchanged signature; FAIR_SECONDS is ignored)
run_vacation(PHASE2_DIR, PHASE3_DIR, ROOT, FAIR_SECONDS)
run_vacation(PHASE2_DIR, PHASE3_DIR, ROOT, config=AllocationConfig(fairness_slack_days=3, fairness_weight=300))
```
```
python -m pipelines.phase2_vacation_flex.vacation_allocation \
    --supply Supply_Regular_qualified.csv --demand annual_desk_needs_reference.csv \
    --volume demand_volume_per_day_reference.csv --out-dir out/ [--training all_training_data.csv] -v
python fairness_tradeoff.py ...      # shortage vs. per-dispatcher spread, as a table
python shortage_goal.py ... --out-dir out/   # relax rules in order until a shortage goal is met
pytest -q tests                      # 13 tests, ~3 s
```
Needs `pandas`, `numpy` and `ortools`, nothing else.

## Priority order

| # | Rule | How it is enforced |
|---|---|---|
| R1 | total vacations on each `SHIFT_DATE` in 25..29 | hard |
| R2 | minimal deviation between dispatchers | hard: every dispatcher gets **L or L+1** days; the model chooses L |
| R3 | daily total inversely proportional to that day's volume | hard: daily total = floor or ceiling of `clip(K / volume_d, 25, 29)` for **one scale K chosen by the model** |
| 4 | minimise shortage | desk demand left uncovered after the best possible flex assignment |
| 5 | maximise vacations | every vacation that adds no shortage is taken |

**Why R3 needs a scale, and a tolerance.** "Inversely proportional" fixes the *shape* of the daily totals but not
their level, and the level K is exactly what decides the annual total. Volume varies by a factor of 2.4 between the
busiest and quietest days, but R1 only allows a factor of 1.16 (25 to 29), so every K pushes some days onto a bound.
The fit error `E(K) = Σ |clip(K/v_d) − K/v_d| / (K/v_d)` measures that distortion. K is *admissible* if
`E(K) ≤ (1 + proportionality_tolerance) × best E`. Without a tolerance, shortage minimisation would choose K so small
that every day sits at 25, and proportionality would be lost.

**How the total is chosen.** For each admissible K (41 grid points) and each fairness level L, an exact min-cost
flow returns the allocation with the least shortage and, among those, the most vacations. A max-flow then scores it
by shortage left after best-case flex. The best candidate wins, ranked by residual shortage, then raw shortage, then
more vacations, then better proportional fit. The three best candidates are re-priced once more by flex
coverability. Every candidate is listed in `vacation_scale_search.csv`.

**Min-cost flow.** One unit of flow is one vacation: `dispatcher → (date, work centre, shift) → date`.
Dispatchers send L forced units plus one optional unit (R2). Dates take floor(target) forced units plus up to
ceiling(target) (R1/R3). A bucket's first `supply − demand` vacations are free and further ones carry a shortage
price. Each optional vacation earns a reward smaller than any shortage price, which implements "maximise
vacations". Capacities are integral and costs convex, so the solution is exact.

**Flex awareness (estimate only, nothing is assigned).** A dispatcher can cover a desk on a date they are not
scheduled or in training, if qualified for that work centre (qualification columns), normally on that shift, up to
`max_flex_days` (4) in total. The max-flow over those candidates gives the best-case cover. Its residual graph shows
which desks can still absorb another short unit, and those shortages are priced lower (`cost_coverable`) than ones
flex cannot fix (`cost_uncoverable`).

## Varying vacation days per dispatcher

By default every dispatcher gets L or L+1 days. Two settings widen that:

* `fairness_slack_days = k` is a hard limit: counts may range from L−k to L+1+k.
* `fairness_weight = w` is the price of being outside the {L, L+1} pair, in the same units as shortage
  (an uncoverable shortage unit costs 1,000, a flex-coverable one 300). A dispatcher's first day outside the pair
  costs 2w, the second 4w, the third 6w, and so on (a squared penalty). A day is moved only if it removes more
  shortage than that price, or adds a vacation at no shortage cost.

At the default `w = 1000` no move ever pays off, so `k` has no effect until `w` is lowered.

Example on the corrected supply (proportionality tolerance 0.05; the L+1 row is 20):

| Setting | Tight desks (ATC, CHF, OC, ROC, Domestic) | Slack desks (Europe, Latin, Pacific) | Total | Shortage | After flex |
|---|---|---|---|---|---|
| strict (k=0) | 19 (Domestic 19–20) | 20 | 9,456 | 3,502 | 1,558 |
| k=3, w=300 | 19 | 21 | 9,550 | 3,437 | 1,493 |
| k=3, w=100 | 17–18 | 22–23 | 9,456 | 3,059 | 1,115 |

With `w = 300`, a 21st day for a Europe dispatcher costs 2 × 300 = 600. That buys back a Domestic dispatcher's
20th day, which would have caused an uncoverable shortage worth 1,000, so the swap is made. A 22nd day would cost
4 × 300 = 1,200 more, so it is not.

## Configuration (`AllocationConfig`)

| Field | Default | Meaning |
|---|---|---|
| `daily_min`, `daily_max` | 25, 29 | R1 |
| `fairness_slack_days`, `fairness_weight` | 0, 1000 | R2 (see above) |
| `proportionality_tolerance` | 0.05 | R3: how much worse than the best proportional fit a scale may be. 0 = best fit only |
| `enforce_proportionality` | `True` | `False` switches R3 off: any daily total within R1 |
| `scale_grid_points` | 41 | scales tried across the admissible band |
| `max_flex_days`, `flex_same_shift_only` | 4, `True` | flex rule used for the estimate |
| `flex_pricing_iterations`, `pricing_candidates` | 4, 3 | flex-aware re-pricing |
| `cost_uncoverable`, `cost_coverable` | 1000, 300 | shortage prices |

## Results on the corrected supply (92,036 rows, 486 dispatchers, no training calendar)

* 9,456 vacations (26.05 a day; K = 88,077). Every date is 25–29; every dispatcher gets 19 or 20 days (264 × 19, 222 × 20).
* Vacations vs volume: Pearson −0.90. The quietest fifth of days average 28.4; the busiest two fifths sit at 25.
* Desk shortage is 325 before vacations and 3,502 after. Best-case flex covers 1,944, which is all 486 × 4 flex
  days, so 1,558 remain. The supply is tight: about 1 vacation in 3 creates a shortage, mostly at Domestic, OC and ATC.
* The proportionality tolerance moves the total only a little:

| `proportionality_tolerance` | Total | Shortage | After flex |
|---|---|---|---|
| 0 (best fit only) | 9,524 | 3,569 | 1,625 |
| 0.02 | 9,473 | 3,519 | 1,575 |
| 0.05 (default) | 9,456 | 3,502 | 1,558 |
| 0.10 | 9,441 | 3,487 | 1,543 |
| 0.20 | 9,403 | 3,449 | 1,505 |

## Reaching a shortage goal (`shortage_goal.py`)

Goal: shortage after vacations ≤ 1,500 and nothing left that best-case flex cannot absorb. Rules are relaxed in
this order, and the first setting that meets the goal is written out (`shortage_goal_ladder.csv` lists every step):
R3 (tolerance 0.5, then off), then R2 (band ±1 … ±10 days, using the full band), then R1 (daily minimum 24, 23, …).

On the corrected supply:

| Step | Vacations | Days per dispatcher | Shortage | After best-case flex |
|---|---|---|---|---|
| all rules | 9,456 | 19–20 | 3,502 | 1,558 |
| R3 off | 9,095 | 18–19 | 3,298 | 1,354 |
| + band ±4 | 9,095 | 16–25 | 2,453 | 541 |
| + band ±8 | 9,095 | 11–28 | 1,876 | 16 |
| + band ±9 | 9,095 | 10–29 | 1,718 | **0** |
| + band ±10 | 9,095 | 10–31 | 1,519 | **0** |
| + daily minimum 24 | 8,742 | 10–30 | **1,346** | **0** ← goal met |

Shortage cannot drop below 1,519 while every day keeps at least 25 vacations, even with R2 and R3 fully off. 325 of
that exists before any vacations, and 25 × 363 days forces at least 9,075 vacations. With R3 off, the minimisation
puts almost every day at the minimum: in the goal run, 353 of 363 days have exactly 24.

What the relaxed fairness looks like: counts end up equal within each desk and differ between desks according to
spare staff. Roughly, ATC, OC and ROC dispatchers get 10 days, Domestic 13–16, and Europe, Latin and Pacific 29–31.

Alternatives checked (none did better):
* Optimising vacations and flex jointly (an LP with both) gives exactly the same residual as this method at every
  fairness band tried, so the method is not the limit.
* Making flex-coverable shortage almost free does not narrow the spread needed.
* Keeping R2 strict and relaxing only R1 needs a daily minimum of about 15 for shortage of about 1,200 (residual 10),
  which means about 5,700 vacations a year instead of about 9,100.

## Fixing the base at 27 per day (`executable_base27.py`)

Set `base_daily=27` and `base_tolerance=t` to hold the annual total within (27 ± t) × dates. `t = 0` fixes it at
27 × 363 = 9,801. Inside that range the model still minimises shortage, then maximises vacations, so it settles at
the low end of the range. `executable_base27.py` has the same `run_vacation` arguments as `executable.py` and keeps
these settings as constants at the top of the file.

On the corrected supply (daily totals 25–29 throughout):

| Rules | Base | Vacations | Days per dispatcher | Shortage | After best-case flex |
|---|---|---|---|---|---|
| all rules kept | 27 ± 0 | 9,801 | 20–21 | 3,765 | 1,821 |
| all rules kept | 27 ± 0.5 | 9,620 | 19–20 | 3,665 | 1,721 |
| R3 on, fairness ±10 | 27 ± 0 | 9,801 | 10–31 | 2,108 | 187 |
| R3 on, fairness ±10 | 27 ± 0.5 | 9,620 | 10–31 | 1,927 | 44 |
| R3 off, fairness ±10 (file default) | 27 ± 0 | 9,801 | 11–32 | 1,950 | 58 |
| R3 off, fairness ±10 | 27 ± 1 | 9,438 | 10–31 | 1,745 | 0 |

A tolerance of 1 or more no longer binds with all rules kept: the model's own choice (9,456) is already within it.

## Assumptions to confirm

1. **Training calendar** (`all_training_data.csv`) was not available, so results are provisional on training days.
2. **Flex rule.** The rule above is inferred. If the flex module allows other shifts or has other limits, change
   `FlexCapacity` / `flex_same_shift_only`. Flex days already used upstream (`Flex_assignment.csv`) are not deducted.
3. Demand with no scheduled supply at all (15 units) cannot be changed by vacations and is reported separately.
