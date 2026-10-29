"""Relax the vacation rules, in a fixed order, until a shortage goal is met.

Goal (defaults): shortage after vacations <= 1500 AND shortage that best-case
flex cannot absorb <= 0.  Rules are loosened one step at a time, in this order,
and the first setting that meets the goal is written out:

    1. R3 inverse proportionality   tolerance 0.05 -> 0.5 -> off
    2. R2 dispatcher deviation      band +-1, +-2, ... +-max_band days around an even split
    3. R1 25..29 per day            daily_min 24, 23, ... down to min_daily_min

    python shortage_goal.py --supply Supply_Regular_qualified.csv \
        --demand annual_desk_needs_reference.csv \
        --volume demand_volume_per_day_reference.csv --out-dir out/

Every step tried is saved to <out-dir>/shortage_goal_ladder.csv.  Use
--keep-proportionality to skip step 1 (relax R2, then R1, with R3 kept).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import pandas as pd

from vacation_allocation import AllocationConfig, data_io, run_allocation
from vacation_allocation.outputs import write_all


def ladder(
    keep_proportionality: bool,
    max_band: int,
    min_daily_min: int,
    base: AllocationConfig,
):
    """Yield (step, description, config) from least to most relaxed."""
    yield "none", "all rules", base
    r3 = {} if keep_proportionality else {"enforce_proportionality": False}
    if not keep_proportionality:
        yield "R3", "R3 tolerance 0.5", dataclasses.replace(
            base, proportionality_tolerance=0.5
        )
        yield "R3", "R3 off", dataclasses.replace(base, **r3)
    for k in range(1, max_band + 1):
        yield "R2", f"band +-{k} days", dataclasses.replace(
            base, fairness_slack_days=k, fairness_weight=1, **r3
        )
    for m in range(base.daily_min - 1, min_daily_min - 1, -1):
        yield "R1", f"daily_min {m}", dataclasses.replace(
            base, daily_min=m, fairness_slack_days=max_band, fairness_weight=1, **r3
        )


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
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--max-shortage", type=int, default=1500, help="raw shortage after vacations"
    )
    p.add_argument(
        "--max-residual",
        type=int,
        default=0,
        help="shortage best-case flex cannot absorb",
    )
    p.add_argument(
        "--max-band",
        type=int,
        default=10,
        help="largest R2 band (days) before touching R1",
    )
    p.add_argument(
        "--min-daily-min",
        type=int,
        default=15,
        help="lowest daily minimum allowed for R1",
    )
    p.add_argument("--keep-proportionality", action="store_true", help="never relax R3")
    a = p.parse_args()
    logging.basicConfig(level=logging.ERROR)

    supply = data_io.read_csv(a.supply)
    demand = data_io.read_csv(a.demand)
    volume = data_io.read_csv(a.volume)
    training = data_io.load_training(a.training) if a.training else set()
    preflex = data_io.read_csv(a.preassigned_flex) if a.preassigned_flex else None
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, chosen = [], None
    for step, label, cfg in ladder(
        a.keep_proportionality, a.max_band, a.min_daily_min, AllocationConfig()
    ):
        res = run_allocation(supply, demand, volume, training, cfg, preflex)
        q, c = res.stats["requirements"], res.stats["coverage"]
        met = (
            c["shortage_after_vacation"] <= a.max_shortage
            and c["residual_shortage_after_best_case_flex"] <= a.max_residual
        )
        rows.append(
            {
                "relaxed_rule": step,
                "setting": label,
                "total_vacations": q["total_vacations"],
                "daily_min": q["daily_min"],
                "daily_max": q["daily_max"],
                "days_per_dispatcher_min": q["per_dispatcher_min"],
                "days_per_dispatcher_max": q["per_dispatcher_max"],
                "corr_with_volume": round(
                    q.get("pearson_vacations_vs_volume", float("nan")), 2
                ),
                "shortage_after_vacation": c["shortage_after_vacation"],
                "best_case_flex_coverable": c["best_case_flex_coverable"],
                "residual_after_flex": c["residual_shortage_after_best_case_flex"],
                "goal_met": met,
            }
        )
        print(rows[-1], flush=True)
        if met:
            chosen = (label, cfg, res)
            break

    pd.DataFrame(rows).to_csv(out / "shortage_goal_ladder.csv", index=False)
    if chosen is None:
        print(
            "\nGoal not reached within the allowed relaxations; nothing written except the ladder."
        )
        return
    label, cfg, res = chosen
    res.stats["goal_search"] = {
        "chosen_setting": label,
        "config": cfg.to_dict(),
        "ladder": rows,
    }
    write_all(res, out)
    print(f"\nGoal met with: {label}.  Outputs written to {out}")


if __name__ == "__main__":
    main()
