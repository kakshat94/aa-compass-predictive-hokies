"""How much shortage does a perfectly even vacation split cost?

Runs the allocation several times with a progressively cheaper price on uneven
vacation counts and prints (and saves) the resulting frontier, so the fairness
vs. coverage decision can be made on numbers.

    python fairness_tradeoff.py --supply Supply_Regular_qualified.csv \
        --demand annual_desk_needs_reference.csv \
        --volume demand_volume_per_day_reference.csv --out tradeoff.csv

Row "strict" is the production setting (every dispatcher gets floor or ceil of
the average).  Other rows allow counts to stray in exchange for fewer
uncovered desk-demand units after best-case flex.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from vacation_allocation import AllocationConfig, data_io, run_allocation


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--supply", required=True)
    p.add_argument("--demand", required=True)
    p.add_argument("--volume", required=True)
    p.add_argument("--training", default=None)
    p.add_argument(
        "--preassigned-flex",
        default=None,
        help="pre-assigned flex file (Flex_assignment_qualified.csv)",
    )
    p.add_argument("--out", default="fairness_tradeoff.csv")
    p.add_argument(
        "--slack", type=int, default=6, help="fairness_slack_days for the relaxed rows"
    )
    p.add_argument("--weights", type=int, nargs="+", default=[300, 200, 100, 60, 30, 10],
                   help="fairness_weight values to try (lower = uneven counts are cheaper)")  # fmt: skip
    a = p.parse_args()
    logging.basicConfig(level=logging.ERROR)

    supply = data_io.read_csv(a.supply)
    demand = data_io.read_csv(a.demand)
    volume = data_io.read_csv(a.volume)
    training = data_io.load_training(a.training) if a.training else set()
    preflex = data_io.read_csv(a.preassigned_flex) if a.preassigned_flex else None

    runs = [("strict", AllocationConfig())] + [
        (
            f"weight={w}",
            AllocationConfig(fairness_slack_days=a.slack, fairness_weight=w),
        )
        for w in a.weights
    ]
    rows = []
    for label, cfg in runs:
        r = run_allocation(supply, demand, volume, training, cfg, preflex).stats
        q, c = r["requirements"], r["coverage"]
        rows.append(
            {
                "setting": label,
                "min_days_per_dispatcher": q["per_dispatcher_min"],
                "max_days_per_dispatcher": q["per_dispatcher_max"],
                "std_days": round(q["per_dispatcher_std"], 2),
                "total_vacations": q["total_vacations"],
                "shortage_after_vacation": c["shortage_after_vacation"],
                "flex_coverable": c["best_case_flex_coverable"],
                "residual_after_flex": c["residual_shortage_after_best_case_flex"],
            }
        )
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
