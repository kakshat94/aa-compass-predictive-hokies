"""Orchestration: inputs -> targets -> priced min-cost-flow loop -> outputs."""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_io
from .allocator import InfeasibleAllocation, allocate, capacity_ok, level_range
from .config import AllocationConfig
from .flex_capacity import FlexCapacity
from .outputs import write_all
from .problem import Problem, build_problem, clean_preflex
from .validation import validate_allocation, volume_relationship
from .volume_targets import (
    admissible_scales,
    daily_bounds_for_scale,
    fit_error,
    impute_volume,
)

log = logging.getLogger(__name__)

ASSUMPTIONS = [
    "Vacation days are taken from existing workdays; each dispatcher works one work center/shift.",
    "The annual total is not fixed. Priority: R1 daily total in [daily_min, daily_max] (hard); "
    "R2 every dispatcher L or L+1 days, L chosen by the model (hard); R3 daily total = floor or ceil of "
    "clip(K/volume) for one scale K whose proportional fit is within proportionality_tolerance of the "
    "best (hard); then minimise shortage left after best-case flex; then maximise vacations.",
    "Flex headroom: a dispatcher can cover a desk on a day they are neither scheduled nor in training, "
    "if qualified for that work center (qualification columns) and, by default, normally on that shift; "
    "max_flex_days per dispatcher; no flex days used before this step.",
    "Flex days are estimated for pricing and reporting only; none are assigned.",
]


@dataclass
class AllocationResult:
    assignments: pd.DataFrame
    remaining: pd.DataFrame
    daily: pd.DataFrame
    fairness: pd.DataFrame
    shortage_by_bucket: pd.DataFrame
    demand_audit: pd.DataFrame
    stats: dict
    search_log: pd.DataFrame | None = None


# --------------------------------------------------------------------------
def _price_loop(problem: Problem, cand: dict, flex: FlexCapacity, cfg, rounds: int):
    """Re-price shortages by flex coverability; keep the best allocation found.

    Round 0 charges every shortage-creating vacation the same price.  After
    each solve the exact flex max-flow tells which buckets can still absorb a
    further short unit; those get the cheaper price next round, steering new
    shortages to where flex can fix them.  The best round by (residual, raw
    shortage) is kept, so a round can never make the result worse.
    """
    price = np.full(problem.n_bucket, cfg.cost_uncoverable, dtype=np.int64)
    best, history, stale, prev = None, [], 0, None
    for it in range(rounds + 1):
        t0 = time.time()
        sol = allocate(problem, cand["day_lo"], cand["day_hi"], cand["level"], price, cfg,
                       total_range=cand.get("total_range"))  # fmt: skip
        short = problem.shortage(sol.per_bucket)
        fx = flex.evaluate(short)
        score = (fx.residual, fx.raw_shortage, -int(sol.selected.sum()))
        history.append(
            {
                "round": it,
                "vacations": int(sol.selected.sum()),
                "raw_shortage": fx.raw_shortage,
                "flex_covered": fx.covered,
                "residual_after_flex": fx.residual,
                "flow_objective": sol.objective,
                "seconds": round(time.time() - t0, 2),
            }
        )
        if best is None or score < best[0]:
            best, stale = (score, sol, fx, it), 0
        else:
            stale += 1
        if prev is not None and np.array_equal(prev, sol.selected):
            break  # fixed point
        prev = sol.selected
        if stale >= 2 or it == rounds:
            break
        target_price = np.where(
            fx.coverable_at_margin, cfg.cost_coverable, cfg.cost_uncoverable
        )
        price = np.rint(
            cfg.pricing_damping * price + (1 - cfg.pricing_damping) * target_price
        ).astype(np.int64)
    for h in history:
        h["selected"] = h["round"] == best[3]
    return best[0], best[1], best[2], history


def _search(problem: Problem, volume: np.ndarray, flex: FlexCapacity, cfg):
    """Try every admissible (scale K, fairness level L); return the best and a log.

    Ranking: residual shortage after best-case flex, then raw shortage, then
    more vacations, then closer proportional fit.
    """
    total_range = None
    if cfg.base_daily is not None:  # base +- tolerance per day on average
        nd = len(volume)
        total_range = (
            int(np.ceil((cfg.base_daily - cfg.base_tolerance) * nd - 1e-9)),
            int(np.floor((cfg.base_daily + cfg.base_tolerance) * nd + 1e-9)),
        )
    scales, k_star, e_star = admissible_scales(
        volume, cfg.daily_min, cfg.daily_max, cfg.proportionality_tolerance,
        cfg.scale_grid_points, total_range,
    )  # fmt: skip
    options = []
    if cfg.enforce_proportionality:
        if scales.size == 0:
            raise InfeasibleAllocation(
                f"no scale K within the proportionality tolerance can reach the base range "
                f"{total_range}; raise proportionality_tolerance or base_tolerance"
            )
        for k in scales:
            ideal, lo_, hi_ = daily_bounds_for_scale(
                k, volume, cfg.daily_min, cfg.daily_max
            )
            options.append(
                (k, ideal, lo_, hi_, fit_error(k, volume, cfg.daily_min, cfg.daily_max))
            )
    else:  # R3 off: any daily total within R1
        n = len(volume)
        options.append((float("nan"), np.full(n, np.nan), np.full(n, cfg.daily_min, dtype=np.int64),
                        np.full(n, cfg.daily_max, dtype=np.int64), float("nan")))  # fmt: skip
        scales = np.array([np.nan])
    n, k_sl = problem.n_tech, cfg.fairness_slack_days
    cands, seen, log_rows = [], set(), []
    for k, ideal, lo_, hi_, err in options:
        for level in level_range(problem, cfg, lo_, hi_):
            key = (lo_.tobytes(), hi_.tobytes(), level)
            if key in seen:
                continue
            seen.add(key)
            row = {"scale_K": float(k), "fit_error": err, "fairness_level_L": level,
                   "min_total": int(lo_.sum()), "max_total": int(hi_.sum())}  # fmt: skip
            reason = capacity_ok(problem, cfg, lo_, level)
            if reason is None and total_range is not None:
                t_lo = max(total_range[0], int(lo_.sum()), n * max(0, level - k_sl))
                t_hi = min(total_range[1], int(hi_.sum()), n * (level + 1 + k_sl))
                if t_lo > t_hi:
                    reason = f"cannot reach the base range {total_range} with this scale/level"
            if reason:
                log_rows.append({**row, "status": "infeasible: " + reason})
                continue
            cands.append({"K": float(k), "err": err, "ideal": ideal, "day_lo": lo_, "day_hi": hi_,
                          "level": level, "row": row, "total_range": total_range})  # fmt: skip
    if not cands:
        raise InfeasibleAllocation(
            "no admissible scale/fairness level satisfies R1-R3: "
            + "; ".join(r["status"] for r in log_rows[:3])
        )

    # round 0 (uniform shortage price) for every candidate
    scored = []
    for c in cands:
        try:
            score, sol, fx, hist = _price_loop(problem, c, flex, cfg, rounds=0)
        except InfeasibleAllocation as exc:
            log_rows.append({**c["row"], "status": f"infeasible: {exc}"})
            continue
        scored.append((score + (c["err"],), c, sol, fx, hist))
        log_rows.append({**c["row"], "status": "solved", "vacations": -score[2],
                         "shortage_after_vacation": fx.raw_shortage,
                         "residual_after_flex": fx.residual})  # fmt: skip
    if not scored:
        raise InfeasibleAllocation("every admissible candidate was infeasible")
    scored.sort(key=lambda x: x[0])

    # flex-aware re-pricing for the most promising candidates
    best = scored[0]
    if cfg.flex_pricing_iterations > 0:
        for _, c, _, _, _ in scored[: cfg.pricing_candidates]:
            score, sol, fx, hist = _price_loop(
                problem, c, flex, cfg, cfg.flex_pricing_iterations
            )
            if score + (c["err"],) < best[0]:
                best = (score + (c["err"],), c, sol, fx, hist)
    _, c, sol, fx, hist = best
    info = {
        "best_fit_scale_K": k_star,
        "best_fit_error": e_star,
        "admissible_scale_range": (
            [float(np.nanmin(scales)), float(np.nanmax(scales))]
            if cfg.enforce_proportionality
            else None
        ),
        "candidates_tried": len(log_rows),
        "base_total_range": list(total_range) if total_range else None,
        "chosen_scale_K": c["K"],
        "chosen_fit_error": c["err"],
        "chosen_fairness_level_L": c["level"],
        "rounds": hist,
    }
    return c, sol, fx, info, pd.DataFrame(log_rows)


def _group_table(
    problem: Problem, flex: FlexCapacity, per_bucket: np.ndarray, cfg
) -> list:
    short = problem.shortage(per_bucket)
    base = problem.shortage(np.zeros_like(per_bucket))
    rows = {}
    for b, (key, wc, sh) in enumerate(
        zip(problem.bucket_keys, problem.bucket_wc, problem.bucket_shift)
    ):
        r = rows.setdefault(
            (key[1], sh),
            {"work_center": key[1], "shift": sh, "dispatchers": 0, "shortage_before_vacation": 0,
             "shortage_after_vacation": 0, "_wc": wc},
        )  # fmt: skip
        r["shortage_before_vacation"] += int(base[b])
        r["shortage_after_vacation"] += int(short[b])
    labels = np.array([k[1] for k in problem.bucket_keys], dtype=object)[
        problem.row_bucket
    ]
    shifts = np.array(problem.bucket_shift, dtype=object)[problem.row_bucket]
    rostered = (
        pd.DataFrame({"l": labels, "s": shifts, "t": problem.row_tech})
        .groupby(["l", "s"])["t"]
        .nunique()
    )
    out = []
    for (label, sh), r in sorted(rows.items()):
        r["dispatchers"] = int(rostered.get((label, sh), 0))
        pool = flex.pools.get((r.pop("_wc"), sh), np.zeros(0))
        r["flex_pool_dispatchers"] = int(pool.size)
        r["flex_pool_budget_if_exclusive"] = int(pool.size * cfg.max_flex_days)
        out.append(r)
    return out


def run_allocation(
    supply_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    training: set | None = None,
    cfg: AllocationConfig | None = None,
    preflex_df: pd.DataFrame | None = None,
) -> AllocationResult:
    cfg = cfg or AllocationConfig()
    cfg.validate()
    training = training or set()
    t_start = time.time()

    supply = data_io.clean_supply(supply_df)
    demand = data_io.clean_demand(demand_df)
    volume = data_io.clean_volume(volume_df)
    dates = [pd.Timestamp(d) for d in sorted(supply["SHIFT_DATE"].unique())]

    # --- volume per date (missing dates imputed from the same weekday)
    vol_by_date, vol_audit = impute_volume(volume, dates)
    vol = vol_by_date.to_numpy(dtype=float)

    # --- desk demand (imputation is audited; the flex module reads the audit file)
    demand, demand_audit = data_io.fill_missing_demand(demand, supply, dates)

    problem = build_problem(supply, demand, dates, training, clean_preflex(preflex_df))
    if not cfg.enforce_daily_bounds:  # R1 off: 0 .. eligible workdays per day
        cap = int(
            np.bincount(
                problem.row_day[problem.row_eligible], minlength=problem.n_day
            ).max()
        )
        cfg = dataclasses.replace(cfg, daily_min=0, daily_max=cap)
    flex = FlexCapacity(problem, cfg)
    log.info(
        "%d dispatchers, %d dates, %d buckets, %d flex candidate arcs",
        problem.n_tech, problem.n_day, problem.n_bucket, flex.n_candidate_arcs,
    )  # fmt: skip

    cand, sol, fx, search_info, search_log = _search(problem, vol, flex, cfg)
    total = int(sol.selected.sum())
    rng = search_info.get("base_total_range")
    if rng is not None:
        assert rng[0] <= total <= rng[1], f"total {total} outside the base range {rng}"
    log.info(
        "chosen K=%.0f L=%d: %d vacations, shortage %d, flex-coverable %d, residual %d",
        cand["K"], cand["level"], total, fx.raw_shortage, fx.covered, fx.residual,
    )  # fmt: skip

    # --- result tables (supply rows untouched: same columns/values as the input)
    sel = sol.selected
    assignments = problem.supply.loc[sel].copy()
    remaining = problem.supply.loc[~sel].copy()

    checks = validate_allocation(
        problem.supply, assignments, remaining, cfg, training, total,
        day_lo=cand["day_lo"], day_hi=cand["day_hi"],
    )  # fmt: skip
    daily_counts = checks.pop("daily_totals")
    tech_counts = checks.pop("dispatcher_counts")

    surplus = np.bincount(
        problem.bucket_day,
        weights=np.maximum(0, problem.bucket_slack),
        minlength=problem.n_day,
    )
    daily = pd.DataFrame(
        {
            "SHIFT_DATE": dates,
            "VOLUME": vol,
            "VOLUME_IMPUTED": [d in set(vol_audit["DATE"]) for d in dates],
            "IDEAL_TARGET": cand["ideal"],
            "ALLOWED_MIN": cand["day_lo"],
            "ALLOWED_MAX": cand["day_hi"],
        }
    )
    daily["VACATION_TARGET"] = daily_counts.to_numpy()
    daily["SCHEDULED_ROWS"] = np.bincount(problem.row_day, minlength=problem.n_day)
    daily["SURPLUS_ROWS_OVER_DEMAND"] = surplus.astype(int)

    fairness = pd.DataFrame({"TECH_ID": problem.tech_ids})
    fairness["VACATION_DAYS"] = tech_counts.reindex(problem.tech_ids).to_numpy()
    fairness["SCHEDULED_WORKDAYS"] = problem.tech_workdays
    fairness["ELIGIBLE_VACATION_DAYS"] = problem.tech_eligible_days
    fairness["DEVIATION_FROM_AVERAGE"] = (
        fairness["VACATION_DAYS"] - total / problem.n_tech
    )

    short_after = problem.shortage(sol.per_bucket)
    short_before = problem.shortage(np.zeros_like(sol.per_bucket))
    mask = (short_after > 0) | (short_before > 0)
    b = np.flatnonzero(mask)
    shortage_tbl = pd.DataFrame(
        {
            "SHIFT_DATE": [problem.bucket_keys[i][0] for i in b],
            "WORK_CENTER": [problem.bucket_keys[i][1] for i in b],
            "SHIFT_TYPE": [problem.bucket_keys[i][2] for i in b],
            "DEMAND": problem.bucket_demand[b],
            "SCHEDULED_SUPPLY": problem.bucket_total[b],
            "TRAINING": problem.bucket_training[b],
            "VACATIONS": sol.per_bucket[b],
            "SUPPLY_AFTER_VACATION": problem.bucket_total[b] - problem.bucket_training[b] - sol.per_bucket[b],
            "SHORTAGE_BEFORE_VACATION": short_before[b],
            "SHORTAGE_AFTER_VACATION": short_after[b],
        }
    )  # fmt: skip

    ideal_gap = float(np.abs(daily["VACATION_TARGET"] - daily["IDEAL_TARGET"]).mean())
    stats = {
        "module": "vacation_allocation (exact min-cost flow + flex-aware pricing)",
        "config": cfg.to_dict(),
        "assumptions": ASSUMPTIONS,
        "inputs": {
            "supply_rows": int(len(problem.supply)),
            "dispatchers": problem.n_tech,
            "dates": problem.n_day,
            "first_date": dates[0],
            "last_date": dates[-1],
            "buckets": problem.n_bucket,
            "training_days_supplied": len(training),
            "training_validated": bool(training),
            "volume_dates_imputed": [d.date().isoformat() for d in vol_audit["DATE"]],
            "desk_demand_buckets_imputed": int(len(demand_audit)),
        },
        "requirements": {
            **checks,
            "R1_daily_bounds_ok": True,
            "R2_fairness_slack_days": cfg.fairness_slack_days,
            "R3_scale_K": cand["K"],
            "R3_fit_error": cand["err"],
            "R3_best_possible_fit_error": search_info["best_fit_error"],
            "R3_days_at_lower_bound": int(
                (daily["IDEAL_TARGET"] <= cfg.daily_min + 1e-9).sum()
            ),
            "R3_days_at_upper_bound": int(
                (daily["IDEAL_TARGET"] >= cfg.daily_max - 1e-9).sum()
            ),
            "R3_mean_abs_gap_to_ideal_target": ideal_gap,
            **volume_relationship(daily_counts, vol_by_date),
            "average_daily_vacations": float(daily_counts.mean()),
        },
        "coverage": {
            "shortage_before_vacation": int(short_before.sum()),
            "shortage_after_vacation": fx.raw_shortage,
            "vacation_induced_shortage": fx.raw_shortage - int(short_before.sum()),
            "best_case_flex_coverable": fx.covered,
            "residual_shortage_after_best_case_flex": fx.residual,
            "demand_without_any_scheduled_supply": int(
                problem.bucket_demand[problem.bucket_total == 0].sum()
            ),
            "preassigned_flex_rows": int(problem.tech_preflex.sum()),
            "preassigned_flex_adding_cover": int(problem.bucket_extra.sum()),
            "preassigned_flex_conflicts": int(len(problem.preflex_conflicts)),
            "by_work_center_shift": _group_table(problem, flex, sol.per_bucket, cfg),
        },  # fmt: skip
        "solver": {**search_info, "total_seconds": round(time.time() - t_start, 1)},
    }
    return AllocationResult(
        assignments,
        remaining,
        daily,
        fairness,
        shortage_tbl,
        demand_audit,
        stats,
        search_log,
    )


# --------------------------------------------------------------------------
def run_from_files(
    supply_path,
    demand_path,
    volume_path,
    output_dir,
    training_path=None,
    cfg: AllocationConfig | None = None,
    preflex_path=None,
) -> tuple[AllocationResult, dict[str, Path]]:
    training = data_io.load_training(training_path) if training_path else set()
    preflex = data_io.read_csv(preflex_path) if preflex_path else None
    if not training:
        log.warning(
            "No training calendar supplied: results are PROVISIONAL w.r.t. training days."
        )
    result = run_allocation(
        data_io.read_csv(supply_path),
        data_io.read_csv(demand_path),
        data_io.read_csv(volume_path),
        training,
        cfg,
        preflex,
    )
    paths = write_all(result, output_dir)
    log.info("wrote %s", ", ".join(p.name for p in paths.values()))
    return result, paths
