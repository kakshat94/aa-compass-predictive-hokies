# Flex-Day Scheduling — American Airlines Workforce Planning

## Overview

This module schedules **flex days after the base bid-line schedule, training, and vacations have been determined**. Its objective is to use eligible additional workdays to reduce remaining desk-staffing shortages, without changing already scheduled vacations or the underlying bid lines.

The standard bid-line cycle is **4 days on / 3 off / 4 on / 4 off (4344)**. A flex day can extend a work block—for example, changing the pattern to **5 on / 2 off / 4 on / 4 off (5244)**. Each dispatcher can receive **at most four flex days over the planning year**.

## Optimization approach

### 1. Greedy heuristic: initial baseline

We first implemented a **coverage-priority greedy heuristic**. It identifies work-center/shift/date combinations with shortages and assigns an eligible dispatcher on a day off. It prioritizes larger shortages and, when possible, distributes flex assignments among dispatchers with fewer already assigned flex days.

Greedy scheduling is fast and produces a feasible baseline, but it makes local decisions: assigning one dispatcher to a particular day might prevent that dispatcher from covering a more difficult shortage later.

### 2. Exact Maximum Flow: current preferred method

We subsequently modeled flex-day allocation as a **maximum-flow network**, using OR-Tools. It optimizes all eligible flex assignments together rather than deciding one at a time.

Conceptually, the network consists of:

- **Source → dispatcher:** capacity of 4, enforcing the annual flex-day limit.
- **Dispatcher → eligible understaffed date/shift:** capacity of 1 for each valid additional workday.
- **Understaffed date/shift → sink:** capacity equal to that date/shift's remaining staffing shortage.

The maximum flow is the maximum number of shortages that can be covered using valid flex-day assignments **under the restrictions currently represented in the model**. Unlike the greedy heuristic, Maximum Flow certifies optimality for that model.

We considered CP-SAT as well. CP-SAT may be preferable if later rules introduce dependencies across dates, such as mandatory spacing between flex days, work-rest requirements, or more complex qualification constraints. For the current independent eligibility and annual-cap structure, Maximum Flow is simpler and exact.

## Constraints and modeling decisions

- **Four flex days maximum per dispatcher per year.** This is an upper bound, not a requirement to allocate four to everyone.
- **Flex days are placed on original scheduled days off.** We do not overwrite a base workday, vacation, or training assignment.
- **Same work center and shift.** The current version infers one work-center/shift assignment per dispatcher from reference supply data; full qualifications are not yet integrated.
- **Coverage-first objective.** A flex day is used only if it fills an existing modeled shortage. Assignments that would simply increase overcoverage are not selected.
- **No four-consecutive-workday cap.** We removed an earlier assumption that prohibited five-day work blocks. The current setting `max_consecutive=0` means *no consecutive-day restriction is imposed*, not zero consecutive days. Additional work-rest rules must still be validated.
- **No minimum spacing between flex assignments** is currently imposed (`min_spacing=0`). The exact Maximum Flow method does not support optional cross-date spacing/consecutive-day restrictions; those would require an extended model.

## Current results

The following results use **515 dispatchers**, the provisional reference bid-line supply, the vacation module's output, and demand data supplemented by **24 previously imputed demand buckets**. Training assignments have **not** yet been included.

| Metric | Greedy baseline | Exact Maximum Flow |
|---|---:|---:|
| Flex days assigned | 609 | **609** |
| Dispatchers receiving flex days | 191 | **184** |
| Staffing shortage before flex | 1,503 | 1,503 |
| Staffing shortage after flex | 894 | **894** |
| Shortage reduction | 609 | **609** |
| Share of shortage eliminated | 40.5% | **40.5%** |
| Optimality guarantee | None | **Yes, for current constraints** |

**Interpretation.** Both methods cover 609 previously uncovered desk-shifts, reducing shortage by **40.5%**. Maximum Flow did not beat greedy on shortage reduction, but proved that **609 is optimal given the fixed inputs and modeled eligibility rules**. The two methods select somewhat different dispatchers, which explains the difference between 191 and 184 people receiving flex assignments.

The **894 remaining shortages** do *not* mean the overall workforce plan is globally optimal: they are only unavoidable within the current flex model with the current vacation assignments and reference bid lines held fixed. Different upstream bid lines, vacations, training, or additional qualification flexibility may change the outcome.

The run also reports **8,532 units of overcoverage**, aggregated across date/work-center/shift buckets. Overcoverage in one bucket cannot automatically fill shortages in another.

## Runtime and computational considerations

The original greedy algorithm is computationally inexpensive but does not provide a proof of optimality. Maximum Flow uses an exact network optimization algorithm and does **not** require a user-selected CP-SAT time limit or Adaptive Large Neighborhood Search (LNS). The script reports solver optimality, but **we have not recorded a reproducible wall-clock benchmark** for the full-data run, so no numerical runtime claim is made here.

The vacation module uses CP-SAT and Adaptive LNS because its large scheduling/fairness problem has a different structure. Flex-day allocation can currently be handled more directly with Maximum Flow.

## Running the module

Install dependencies:

```bash
pip install pandas ortools
```

From the project directory, after generating vacation assignments:

```bash
python flex_day_maxflow.py --output-dir outputs/flex_maxflow
```

For the earlier greedy baseline:

```bash
python flex_day_maxflow.py --solver greedy --output-dir outputs/flex_greedy
```

The default inputs are `data/initial_supply_data_reference.csv`, `data/annual_desk_needs_reference.csv`, and `outputs/vacation_initial.csv`. The script also incorporates the existing `outputs/demand_imputation_audit.csv` when needed. To include a training calendar, supply a compatible file with dispatcher IDs and dates using `--training path/to/training.csv`.

### Output files

| File | Contents |
|---|---|
| `flex_days.csv` | Final dispatcher/date/work-center/shift flex assignments |
| `staffing_coverage_before_after_flex.csv` | Bucket-level staffing and shortage before and after flex days |
| `flex_by_dispatcher.csv` | Individual flex-day counts and reference assignments |
| `flex_summary.json` | Aggregate results, solver method, and optimization comparison |

## Current status and next steps

**Status: working, provisional flex-day module.** The exact flow model has been run successfully on the current reference-based data, and both algorithms achieve the same coverage improvement. Before final competition submission, we need to integrate the team's optimized bid lines and training schedule, verify the official qualification/work-rest/flex rules, and rerun the analysis. If those rules introduce dependencies between flex assignments on different days, we may extend the model with CP-SAT.
