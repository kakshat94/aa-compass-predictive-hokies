Flex-Day Optimization

## Objective
Minimize remaining staffing shortages after vacation scheduling by assigning eligible flex days to dispatchers while respecting workforce scheduling constraints.

## Methodology
We use **OR-Tools SimpleMaxFlow** to formulate flex-day allocation as an **exact bipartite maximum-flow problem**. Dispatcher nodes are connected to eligible understaffed date/work-center/shift assignments, with capacities representing each dispatcher's remaining flex allowance and each staffing bucket's shortage. Maximizing flow assigns as many useful flex days as possible without creating redundant coverage. The solution is optimal under the modeled constraints and supplied inputs.

## Constraints and Parameters
- **Staffing coverage:** Flex days are allocated only to date/work-center/shift combinations with staffing shortages.
- **Scheduling conflicts:** No flex day may overlap an existing workday, vacation day, training day, or preassigned flex day.
- **Work assignment:** Each dispatcher is assigned only to their associated work center and shift type.
- **Annual flex limit:** At most **4 flex days per dispatcher**, including any preassigned days.
- **Planning scope:** **January 11, 2026 is excluded** to match the vacation scheduling horizon.
- **Inputs:** Original dispatcher supply, post-vacation supply, vacation assignments, training calendar, preassigned flex assignments, and daily staffing demand.

## Results

| Metric | Result |
|---|---:|
| Staffing shortage after vacations | 313 |
| Newly assigned flex days | 312 |
| Dispatchers receiving flex days | 150 |
| Preassigned flex days | 0 |
| Remaining staffing shortage | **1** |
| Reduction in staffing shortage | **312 (99.7%)** |
| Supply rows after vacations | 85,648 |
| Supply rows after flex allocation | 85,960 |

The maximum-flow solution covers **312 of 313** staffing shortage units, leaving **1** unresolved shortage under the modeled eligibility and capacity constraints.

