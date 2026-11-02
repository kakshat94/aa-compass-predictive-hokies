
## Objective
Minimize staffing shortages while improving vacation allocation fairness across dispatcher-specific annual bid lines. In particular, minimize the maximum and average deviations in annual vacation days between dispatchers.

## Methodology
- **OR-Tools CP-SAT:** Solves local vacation reassignment subproblems subject to staffing and scheduling constraints.
- **Adaptive Large Neighborhood Search (LNS):** Alternates vacation swaps and count-changing moves; dynamically adjusts neighborhood size based on progress.
- **Minimax fairness:** Penalizes the largest absolute deviation from the mean annual vacation-day allocation, along with average absolute deviation. Each `TECH_ID` is evaluated separately as a dispatcher-specific annual bid-line allocation.
- **Coverage-first approach:** Assigns vacations only where staffing exceeds the required coverage, preserving the minimum shortage attainable from the input supply.

## Constraints
- **Staffing coverage:** Vacation assignments cannot increase staffing shortages beyond the shortage already present in the available schedule.
- **Training conflicts:** No vacation is assigned on a recorded training day.
- **Daily vacation limits:** Maximum of **29** vacations per day; **25** is a soft daily target, not a mandatory minimum. Volume-based daily caps use an approximate rank-based policy.
- **Workday eligibility:** Vacations can be assigned only on eligible scheduled workdays.

## Parameters
| Parameter | Value |
|---|---:|
| Daily vacation hard maximum | 29 |
| Daily vacation soft target | 25 |
| Annual worst-deviation weight | 1.0 |
| Annual average-deviation weight | 1.0 |
| Soft daily-target penalty weight | 2.0 |
| Adaptive LNS iterations | 80 |
| CP-SAT time limit per LNS iteration | 6 seconds |
| Initial / maximum neighborhood size | 6 / 12 buckets |
| CP-SAT workers | 4 |
| Random seed | 42 |

## Results
| Metric | Initial | Final |
|---|---:|---:|
| Staffing shortage | 63* | **63** |
| Total vacation days assigned | 8,800 | **8,800** |
| Mean absolute annual deviation per dispatcher | 5.030 days | **5.023 days** |
| Maximum absolute annual deviation | 12.391 days | **12.391 days** |
| Standard deviation of annual vacation days | 6.522 days | **6.527 days** |
| Annual vacation days per dispatcher (min–max) | 5–26 | **5–28** |

