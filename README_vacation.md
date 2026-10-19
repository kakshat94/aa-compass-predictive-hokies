# Vacation Scheduling Optimization — American Airlines Case Competition

## Overview

This module assigns vacation days to flight dispatchers across the annual planning horizon while prioritizing staffing coverage and improving fairness across employees. It is one part of our team's workforce-planning pipeline: **bid-line schedules → training assignments → vacation scheduling → flex-day placement**.

Our approach combines **Constraint Programming (CP-SAT)** with **Adaptive Large Neighborhood Search (Adaptive LNS)** and a final **targeted fairness-repair heuristic**. The current results use competition reference supply data; final integration with the team's optimized bid lines and training assignments is still pending.

## 1. Why CP-SAT?

We use Google OR-Tools **CP-SAT** because vacation assignment is a discrete scheduling problem involving many binary decisions and interacting constraints. For each dispatcher and eligible date, the model decides whether that dispatcher takes vacation.

The model respects vacation eligibility and exact daily vacation quotas, while accounting for how absences affect staffing by date, work center, and shift. CP-SAT provides a convenient way to express these constraints and to distinguish a feasible solution from a **provably optimal** one.

**Coverage comes first.** The primary objective minimizes the total modeled shortage of staffed desk-shifts. Only after this coverage objective is proven optimal does the optional full-year fairness stage run, keeping the minimum shortage fixed so that fairness cannot worsen coverage.

In our current reference-data run, the coverage stage returned **OPTIMAL**, with **1,503 modeled shortage desk-shifts**. This is optimal for the *current model and assumptions*; it is not yet a final operational staffing assessment.

## 2. How we define fairness

Let \(V_i\) denote the number of vacation days assigned to dispatcher \(i\), and let \(\bar V\) be the average across all dispatchers.

With 9,803 vacation assignments and 515 dispatchers, the current mean is approximately **19.03 days per dispatcher**.

We evaluate fairness using several complementary measures:

- **Total absolute deviation:** \(\sum_i |V_i-\bar V|\), measuring total imbalance across all employees.
- **Minimax (maximum absolute deviation):** \(\max_i |V_i-\bar V|\), measuring the worst individual imbalance.
- **Standard deviation:** \(\sqrt{\frac{1}{N}\sum_i (V_i-\bar V)^2}\), measuring overall spread. A lower value means vacation allocations are more concentrated around the mean.
- **Zero-vacation count:** the number of dispatchers assigned no vacation days, a particularly important practical fairness indicator.

**Important distinction:** Standard deviation is a *reported evaluation metric*, **not** the objective directly minimized by our solver. The full-year fairness CP-SAT stage minimizes **total absolute deviation**. Adaptive LNS uses **minimax first**, with total absolute deviation as the second priority.

### Why minimax?

Minimizing total imbalance alone can still leave a small number of people with very poor allocations. A **minimax** objective prioritizes the worst-off dispatcher, discouraging extreme allocations before improving aggregate fairness. We use a lexicographic priority: **(1) reduce the largest individual deviation, then (2) reduce total absolute deviation**.

## 3. Why Adaptive Large Neighborhood Search?

The full-year fairness problem covers **515 dispatchers over 363 days**. Solving all fairness decisions simultaneously can be computationally demanding: a short time limit previously led to CP-SAT reporting `UNKNOWN` (no fairness solution found within that run's limit).

Rather than relying entirely on a long, global solve, we use **Adaptive LNS** to improve an existing feasible schedule in smaller pieces:

1. Select a contiguous window of dates, generally starting at **21 days**.
2. Hold assignments outside that window fixed and let CP-SAT reoptimize vacation decisions inside it.
3. Enforce the exact daily quotas and prevent an increase in modeled staffing shortage.
4. Accept a candidate only if it improves the annual **minimax-first fairness score**.
5. Adapt the neighborhood: focus on imbalanced employees, occasionally explore different windows, and enlarge the window after stagnation (up to **60 days**).

This hybrid design preserves the already-achieved coverage while searching efficiently for more equitable allocations. It does **not** prove that the final fairness solution is globally optimal.

### Targeted zero-vacation repair

After LNS, we apply a small additional heuristic aimed at dispatchers with zero or very few vacation days. It transfers a vacation day from a better-allocated dispatcher to an underallocated dispatcher **on the same date, in the same work center and shift**. Such swaps preserve daily vacation totals and bucket-level coverage. Swaps are accepted only when they improve the fairness score.

This repair can reduce zero-vacation cases that LNS leaves behind, but it does not guarantee that every dispatcher receives vacation.

## 4. Runtime settings and our decisions

Our baseline configuration is:

| Stage | Time budget / setting | Rationale |
|---|---|---|
| Coverage CP-SAT | **90 seconds** | Prioritize the staffing objective |
| Full-year fairness CP-SAT | **Configurable** (`--fair-seconds`) | Improve total absolute deviation without worsening optimal coverage |
| Adaptive LNS | **10 iterations**, **10 seconds per window** | Provide repeated, localized fairness improvements |
| Initial LNS window | **21 days** | Balance flexibility against solve difficulty |
| Maximum LNS window | **60 days** | Allow larger changes after stagnation |
| LNS random seed | **42** | Make the neighborhood selection more reproducible |
| Zero-vacation repair | **Up to 3 passes** | Further improve allocations through coverage-preserving swaps |

The full-year fairness stage is **optional**, not prohibited. Setting `--fair-seconds 0` skips it; fairness improvement still occurs through Adaptive LNS and targeted repair. Increasing the budget (for example to **120 or 300 seconds**) may produce a better starting assignment, but longer runtimes do not guarantee a better final result.

A representative command is:

```bash
python vacation_cp_sat_zero_targeted.py --fair-seconds 120
```

Use `--fair-seconds 300` to test a longer full-year fairness solve. The total runtime also includes coverage optimization, LNS, and repair; it is **not** just the fairness time limit. CP-SAT reports `FEASIBLE` when it finds a valid solution without proving optimality, `OPTIMAL` when it proves the objective optimum, and `UNKNOWN` when it cannot establish a feasible solution or a proof within the allotted search.

## 5. Current results and improvements

The most recent run with the full-year fairness stage enabled produced the following progression:

| Metric | After full-year CP-SAT | After Adaptive LNS | After targeted repair |
|---|---:|---:|---:|
| Minimum vacation days | 0 | 0 | 0 |
| Maximum vacation days | 57 | 34 | **34** |
| Mean vacation days | 19.03 | 19.03 | **19.03** |
| Standard deviation | 14.50 | 7.81 | **7.73** |
| Total absolute deviation | 6,527.48 | 2,740.44 | **2,716.58** |
| Dispatchers with zero vacations | 81 | 29 | **13** |
| Dispatchers with over 40 vacations | 29 | 0 | **0** |

**Key observations:**

- Coverage remained at its proven minimum of **1,503 modeled shortage desk-shifts**.
- All **9,803** required vacation assignments were scheduled under the current daily-quota assumptions.
- Adaptive LNS produced most of the fairness improvement, reducing the standard deviation from **14.50 to 7.81** and the maximum vacation count from **57 to 34**.
- The targeted repair further reduced zero-vacation dispatchers from **29 to 13**, with a final standard deviation of **7.73**.
- In this run, **9 of 10 LNS iterations** were accepted; targeted repair completed **21 swaps**.
- The full-year fairness solver returned **FEASIBLE**, not `OPTIMAL`. Thus, coverage optimality is proven but global fairness optimality is **not**.

Compared with an earlier run that skipped the full-year fairness stage, the latest configuration improved the final maximum vacation count (**37 → 34**) and final standard deviation (approximately **8.1 → 7.73**), while the remaining zero-vacation count was **13** in both cases. These are experimental results, not proof that more solver time always helps.

## 6. Current status and limitations

**The vacation optimizer is a working prototype suitable for team review and integration**, but the reference-data output is **not yet our final competition schedule**. The model currently uses an assumed volume-based daily vacation quota, and it has not yet incorporated the team's final optimized bid-line schedule or the finalized training and qualification assignments. Individual vacation entitlements and any official requirements for consecutive vacation blocks must also be validated against the competition rules.

Our next priorities are to integrate team outputs, validate all scheduling requirements, and implement the **flex-day placement** stage. Further fairness tuning can follow once the complete pipeline is functioning.
