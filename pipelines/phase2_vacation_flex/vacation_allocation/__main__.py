"""Command line:  python -m vacation_allocation --supply ... --demand ... --volume ... --out-dir ..."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import AllocationConfig
from .pipeline import run_from_files


def main(argv=None) -> int:
    d = AllocationConfig()
    p = argparse.ArgumentParser(
        description="Allocate vacations out of scheduled dispatcher supply"
    )
    p.add_argument("--supply", required=True, help="Supply_Regular_qualified.csv")
    p.add_argument("--demand", required=True, help="annual_desk_needs_reference.csv")
    p.add_argument(
        "--volume", required=True, help="demand_volume_per_day_reference.csv"
    )
    p.add_argument(
        "--training",
        default=None,
        help="optional training calendar (TECH_ID, SHIFT_DATE)",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--daily-min", type=int, default=d.daily_min)
    p.add_argument("--daily-max", type=int, default=d.daily_max)
    p.add_argument("--fairness-slack-days", type=int, default=d.fairness_slack_days,
                   help="0 = even split (default); k>0 lets counts stray k days to cut shortage")  # fmt: skip
    p.add_argument("--fairness-weight", type=int, default=d.fairness_weight)
    p.add_argument("--proportionality-tolerance", type=float, default=d.proportionality_tolerance,
                   help="0 = only the best-fitting scale; 0.05 = fit may be up to 5%% worse")  # fmt: skip
    p.add_argument(
        "--base-daily",
        type=int,
        default=None,
        help="average vacations per day, e.g. 27",
    )
    p.add_argument(
        "--base-tolerance",
        type=float,
        default=0.0,
        help="+- vacations per day around the base",
    )
    p.add_argument("--no-proportionality", action="store_true", help="switch R3 off")
    p.add_argument("--max-flex-days", type=int, default=d.max_flex_days)
    p.add_argument(
        "--flex-any-shift",
        action="store_true",
        help="flex candidates may come from any shift",
    )
    p.add_argument(
        "--flex-pricing-iterations", type=int, default=d.flex_pricing_iterations
    )
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if a.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    cfg = AllocationConfig(
        daily_min=a.daily_min,
        daily_max=a.daily_max,
        fairness_slack_days=a.fairness_slack_days,
        fairness_weight=a.fairness_weight,
        proportionality_tolerance=a.proportionality_tolerance,
        enforce_proportionality=not a.no_proportionality,
        base_daily=a.base_daily,
        base_tolerance=a.base_tolerance,
        max_flex_days=a.max_flex_days,
        flex_same_shift_only=not a.flex_any_shift,
        flex_pricing_iterations=a.flex_pricing_iterations,
    )
    result, paths = run_from_files(
        a.supply, a.demand, a.volume, a.out_dir, a.training, cfg
    )
    c = result.stats["coverage"]
    r = result.stats["requirements"]
    print(f"vacations: {r['total_vacations']:,}  daily [{r['daily_min']}, {r['daily_max']}]  "
          f"per-dispatcher [{r['per_dispatcher_min']}, {r['per_dispatcher_max']}]")  # fmt: skip
    print(f"shortage after vacation: {c['shortage_after_vacation']:,}  "
          f"flex-coverable: {c['best_case_flex_coverable']:,}  residual: "
          f"{c['residual_shortage_after_best_case_flex']:,}")  # fmt: skip
    print("outputs:", ", ".join(str(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
