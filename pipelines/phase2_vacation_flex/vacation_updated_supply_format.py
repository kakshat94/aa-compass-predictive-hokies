"""AA COMPASS vacation scheduling: CP-SAT, adaptive LNS, targeted swap repair.

Run from a project directory containing inputs/ and outputs/:
    python vacation_cp_sat.py

Requires: pandas, numpy, ortools (pip install -r requirements.txt)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
import sys
import logging
import numpy as np
import pandas as pd

try:
    from ortools.sat.python import cp_model
except ImportError as exc:
    raise SystemExit("OR-Tools is required: pip install ortools pandas numpy") from exc


# ============================================================
# INPUT CONFIGURATION -- UPDATE THESE TWO TEAMMATE DEPENDENCIES
# ============================================================
# Ashkat's optimized dispatcher-level annual bid-line output.
DEFAULT_SUPPLY = "outputs/phase2_training/Supply_Regular_qualified.csv"

# PERMANENT reference input: daily operational volume (NOT desk demand).
DEFAULT_VOLUME = "inputs/demand_volume_per_day_reference.csv"

# PERMANENT reference input: desk demand per date/work center/shift.
DEFAULT_DEMAND = "inputs/annual_desk_needs_reference.csv"

# Daniel's employee-level training calendar.
# Expected normalized columns: TECH_ID, SHIFT_DATE (one training day per row).
# Until available, training is NOT checked; output is PROVISIONAL.
DEFAULT_TRAINING = "outputs/phase2_training/all_training_data.csv"

DEFAULT_OUTPUT = "outputs/phase2_vacation_flex/vacation_outputs_updatedFormat.csv"
DEFAULT_VACATION_ASSIGNMENTS = "outputs/phase2_vacation_flex/vacation_initial.csv"

# MODELING CHOICES -- NOT specified as exact competition formulas:
# Rank days by operational volume, then allocate daily quotas between 25 and 29.
# 5 equal-sized rank groups -> 29 (quietest), 28, 27, 26, 25 (busiest).
# Quotas must be treated as configurable assumptions, not official AA targets.
QUOTA_MIN = 25
QUOTA_MAX = 29

# Coverage shortage is prioritized over individual fairness using a
# lexicographic two-pass solve (within the time budgets provided).


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p.resolve()}")
    df = pd.read_csv(p, low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df


def require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(
            f"{name}: missing required columns {missing}; available: {list(df.columns)}"
        )


def parse_dates(df: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    if df[column].isna().any():
        raise ValueError(
            f"{name}: invalid/missing {column} values in {df[column].isna().sum()} rows"
        )
    return df


def clean_supply(df: pd.DataFrame) -> pd.DataFrame:
    require(df, ["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"], "supply")
    df = parse_dates(df, "SHIFT_DATE", "supply")
    df = df.copy()
    df["TECH_ID"] = df["TECH_ID"].astype("string").str.strip()
    # df["WORK_CENTER_ID"] = df["WORK_CENTER_ID"].astype("string").str.strip().str.upper()
    df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype("string").str.strip().str.upper()
    if df[["TECH_ID", "WORK_CENTER_ID", "SHIFT_TYPE"]].isna().any().any():
        raise ValueError("Supply has missing identifier, work center or shift")
    if df.duplicated(["TECH_ID", "SHIFT_DATE"]).any():
        bad = df.loc[
            df.duplicated(["TECH_ID", "SHIFT_DATE"], keep=False),
            ["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"],
        ].head(8)
        raise ValueError(
            "Duplicate employee-date rows in supply; cannot model one vacation/day:\n"
            + str(bad)
        )
    return df.sort_values(["SHIFT_DATE", "TECH_ID"]).reset_index(drop=True)


def clean_demand(df: pd.DataFrame) -> pd.DataFrame:
    require(df, ["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "DEMAND"], "desk demand")
    df = parse_dates(df, "SHIFT_DATE", "desk demand")
    df = df.copy()
    # df["WORK_CENTER"] = df["WORK_CENTER"].astype(str).str.strip().str.upper()
    df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype(str).str.strip().str.upper()
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="raise")
    if (df["DEMAND"] < 0).any() or not (df["DEMAND"] % 1 == 0).all():
        raise ValueError("Desk demand values must be nonnegative integers")
    if df.duplicated(["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE"]).any():
        raise ValueError("Desk demand contains duplicate date/work-center/shift keys")
    return df


def clean_volume(df: pd.DataFrame) -> pd.DataFrame:
    require(df, ["DATE", "DEMAND"], "daily operational volume")
    df = parse_dates(df, "DATE", "daily operational volume")
    df = df.copy()
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="raise")
    if df["DEMAND"].isna().any() or df.duplicated(["DATE"]).any():
        raise ValueError("Daily volume has missing values or duplicated dates")
    return df


def make_quota(
    volume: pd.DataFrame, dates: list[pd.Timestamp], fixed_quota: int | None
) -> dict:
    """Generate quotas using volume quintiles, with auditable volume imputation.

    The quintile rule is our model assumption, not an AA-mandated formula.
    """
    if fixed_quota is not None:
        if fixed_quota < 0:
            raise ValueError("fixed quota must be nonnegative")
        return {d: fixed_quota for d in dates}

    vol = volume.set_index("DATE")["DEMAND"].copy()
    missing = sorted(set(dates) - set(vol.index))
    for d in missing:
        # Prefer the same weekday in the previous week (seasonally closer).
        prior = d - pd.Timedelta(days=7)
        following = d + pd.Timedelta(days=7)
        if prior in vol.index:
            donor = prior
        elif following in vol.index:
            donor = following
        else:
            raise ValueError(
                f"No same-weekday observed volume for {d.date()} (+/- 7 days). "
                "Provide the missing data or explicitly choose another policy."
            )
        vol.loc[d] = vol.loc[donor]
        logging.warning(
            f"WARNING: Imputed operational volume on {d.date()} from "
            f"{donor.date()} ({vol.loc[donor]:g}); unobserved estimate."
        )

    ranked = pd.DataFrame({"date": dates, "volume": [vol.loc[d] for d in dates]})
    ranked = ranked.sort_values(["volume", "date"]).reset_index(drop=True)
    n = len(ranked)
    if n == 0:
        raise ValueError("No schedule dates provided")
    ranked["group"] = [min(4, 5 * i // n) for i in range(n)]
    ranked["quota"] = QUOTA_MAX - ranked["group"]
    return dict(zip(ranked["date"], ranked["quota"]))


def fill_missing_demand(
    demand: pd.DataFrame, supply: pd.DataFrame, dates: list[pd.Timestamp]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate absent desk-demand records from same bucket, previous/next week.

    Values are *imputed*, not validated actual demand. Return an audit table.
    Never interpret missing required demand as zero.
    """
    demand = demand.copy()
    known = {
        (r.SHIFT_DATE, r.WORK_CENTER, r.SHIFT_TYPE): int(r.DEMAND)
        for r in demand.itertuples(index=False)
    }
    schedule_buckets = set(
        zip(supply["SHIFT_DATE"], supply["WORK_CENTER_ID"], supply["SHIFT_TYPE"])
    )
    missing = sorted(
        (d, c, sh)
        for d, c, sh in schedule_buckets
        if d in dates and (d, c, sh) not in known
    )
    audit = []
    for d, c, sh in missing:
        # Same weekday and same work-center/shift preferred.
        donor = next(
            (
                candidate
                for candidate in [d - pd.Timedelta(days=7), d + pd.Timedelta(days=7)]
                if (candidate, c, sh) in known
            ),
            None,
        )
        if donor is None:
            # Last-resort: closest observed day for this exact work-center/shift.
            candidates = [dd for (dd, cc, ss) in known if cc == c and ss == sh]
            if not candidates:
                raise ValueError(
                    f"Cannot estimate desk demand for {d.date()}, "
                    f"{c}, {sh}: no donor exists for this bucket"
                )
            donor = min(candidates, key=lambda dd: (abs((dd - d).days), dd))
        value = known[(donor, c, sh)]
        known[(d, c, sh)] = value
        audit.append(
            {
                "SHIFT_DATE": d,
                "WORK_CENTER": c,
                "SHIFT_TYPE": sh,
                "IMPUTED_DEMAND": value,
                "DONOR_DATE": donor,
                "METHOD": (
                    "same_weekday" if abs((donor - d).days) == 7 else "nearest_date"
                ),
            }
        )
        demand.loc[len(demand)] = {
            "SHIFT_DATE": d,
            "WORK_CENTER": c,
            "SHIFT_TYPE": sh,
            "DEMAND": value,
        }
    if audit:
        logging.warning(
            f"WARNING: Estimated {len(audit)} missing desk-demand buckets. "
            "See demand_imputation_audit.csv; results are provisional."
        )
    return demand, pd.DataFrame(
        audit,
        columns=[
            "SHIFT_DATE",
            "WORK_CENTER",
            "SHIFT_TYPE",
            "IMPUTED_DEMAND",
            "DONOR_DATE",
            "METHOD",
        ],
    )


def load_training(path: str | None) -> set[tuple[str, pd.Timestamp]]:
    if path is None:
        return set()
    df = read_csv(path)
    require(df, ["TECH_ID", "SHIFT_DATE"], "Daniel training")
    df = parse_dates(df, "SHIFT_DATE", "Daniel training")
    return set(zip(df["TECH_ID"].astype("string").str.strip(), df["SHIFT_DATE"]))


def improve_with_lns(
    allowed: pd.DataFrame,
    demand_map: dict,
    original_bucket: dict,
    training_bucket: dict,
    quota: dict,
    chosen: list[int],
    n_emp: int,
    iterations: int,
    window_days: int,
    seconds: float,
    workers: int,
    max_window_days: int = 60,
    seed: int = 42,
    all_employee_ids=None,
) -> tuple[list[int], dict]:
    """Adaptive CP-SAT LNS: minimize worst individual imbalance first,
    then the total absolute deviation; enlarge windows after stagnation.

    Prioritize dates on which the most over-/under-assigned employees have
    vacation choices. A seeded random exploration component prevents every
    iteration from inspecting the same neighborhood. Fixed assignments outside
    the selected date window ensure the whole annual schedule stays feasible.
    Each candidate is independently checked for exact quotas and no increase
    in shortage before acceptance; the reported fairness objective is annual
    total absolute deviation (scaled to use integers).
    """
    if iterations < 0 or window_days < 1 or seconds <= 0:
        raise ValueError(
            "Iterations must be nonnegative, window_days >= 1, seconds > 0"
        )
    if max_window_days < window_days:
        raise ValueError("max_window_days must be >= window_days")
    rng = np.random.default_rng(seed)
    dates = sorted(quota)
    rows = list(allowed.itertuples(index=False))
    by_date = defaultdict(list)
    by_emp = defaultdict(list)
    by_bucket = defaultdict(list)
    for j, r in enumerate(rows):
        by_date[r.SHIFT_DATE].append(j)
        by_emp[str(r.TECH_ID)].append(j)
        by_bucket[(r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))].append(j)
    # Include every employee in the fairness calculation, including those
    # with no vacation-eligible shifts after training filtering.
    all_emps = sorted(set(str(x) for x in all_employee_ids))
    total_vac = sum(quota.values())
    target = total_vac / n_emp
    # Upper bound on total absolute deviation, used to make minimax fairness
    # the strict first priority and total deviation the second priority.
    # With annual quotas, sum(abs(n_emp*V_i-total_vac)) <= 2*n_emp*total_vac.
    FAIRNESS_WEIGHT = 2 * n_emp * total_vac + 1

    def employee_totals(selection):
        totals = defaultdict(int)
        for j, val in enumerate(selection):
            if val:
                totals[str(rows[j].TECH_ID)] += 1
        return totals

    def fairness_score(totals):
        deviations = [abs(n_emp * totals[e] - total_vac) for e in all_emps]
        return (max(deviations), sum(deviations))

    def objective_score(score):
        return score[0] * FAIRNESS_WEIGHT + score[1]

    def shortage(b, number_vacations):
        return max(
            0,
            demand_map[b]
            - original_bucket.get(b, 0)
            + training_bucket.get(b, 0)
            + number_vacations,
        )

    current_window = min(window_days, len(dates))
    accepted = 0
    stagnation = 0
    logs = []
    incumbent_totals = employee_totals(chosen)
    incumbent_score = fairness_score(incumbent_totals)
    for it in range(iterations):
        # Over-assigned and under-assigned dispatchers both contribute to
        # potential fairness improvements. Prefer dates on which the former
        # currently vacation and the latter have eligible workdays.
        if it % 4 == 3:  # exploration, deterministically reproducible
            start = int(rng.integers(0, max(1, len(dates) - current_window + 1)))
            selection_method = "exploration"
        else:
            daily_priority = []
            for d in dates:
                over = sum(
                    max(0, incumbent_totals[str(rows[j].TECH_ID)] - target)
                    for j in by_date[d]
                    if chosen[j]
                )
                under = sum(
                    max(0, target - incumbent_totals[str(rows[j].TECH_ID)])
                    for j in by_date[d]
                    if not chosen[j]
                )
                daily_priority.append(float(over + under))
            rolling = np.convolve(
                np.asarray(daily_priority), np.ones(current_window), mode="valid"
            )
            best = np.flatnonzero(rolling >= rolling.max() - 1e-8)
            # Seeded tie-breaking diversifies equally promising windows.
            start = int(rng.choice(best))
            selection_method = "imbalance_targeted"
        window_dates = dates[start : start + current_window]
        local_indices = [j for d in window_dates for j in by_date[d]]
        local_set = set(local_indices)
        model = cp_model.CpModel()
        xs = {j: model.NewBoolVar(f"adaptive_{it}_{j}") for j in local_indices}
        for d in window_dates:
            model.Add(sum(xs[j] for j in by_date[d]) == int(quota[d]))

        window_set = set(window_dates)
        window_buckets = [b for b in demand_map if b[0] in window_set]
        incumbent_vac = defaultdict(int)
        for j in local_indices:
            if chosen[j]:
                r = rows[j]
                incumbent_vac[
                    (r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
                ] += 1
        old_shortage = sum(shortage(b, incumbent_vac[b]) for b in window_buckets)
        short_vars = []
        for b in window_buckets:
            required = int(demand_map[b])
            base = original_bucket.get(b, 0) - training_bucket.get(b, 0)
            sv = model.NewIntVar(0, required, f"short_{it}_{len(short_vars)}")
            model.Add(sv >= required - base + sum(xs[j] for j in by_bucket[b]))
            short_vars.append(sv)
        model.Add(sum(short_vars) <= old_shortage)

        devs = []
        for emp in all_emps:
            emp_in = [j for j in by_emp[emp] if j in local_set]
            outside = incumbent_totals[emp] - sum(chosen[j] for j in emp_in)
            new_total = outside + sum(xs[j] for j in emp_in)
            dev = model.NewIntVar(0, n_emp * len(dates) + total_vac, f"dev_{it}_{emp}")
            model.AddAbsEquality(dev, n_emp * new_total - total_vac)
            devs.append(dev)
        max_dev = model.NewIntVar(
            0, n_emp * len(dates) + total_vac, f"max_deviation_{it}"
        )
        model.AddMaxEquality(max_dev, devs)
        model.Minimize(FAIRNESS_WEIGHT * max_dev + sum(devs))
        for j in local_indices:
            model.AddHint(xs[j], chosen[j])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = seed + it
        status = solver.Solve(model)
        name = solver.StatusName(status)
        improved = False
        score_before = incumbent_score
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            proposal = chosen.copy()
            for j in local_indices:
                proposal[j] = int(solver.Value(xs[j]))
            new_totals = employee_totals(proposal)
            new_score = fairness_score(new_totals)
            new_vac = defaultdict(int)
            for j in local_indices:
                if proposal[j]:
                    r = rows[j]
                    new_vac[
                        (r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
                    ] += 1
            new_shortage = sum(shortage(b, new_vac[b]) for b in window_buckets)
            # Verify instead of trusting an inexact/time-limited optimization.
            valid_quotas = all(
                sum(proposal[j] for j in by_date[d]) == quota[d] for d in window_dates
            )
            if (
                valid_quotas
                and new_shortage <= old_shortage
                and new_score < incumbent_score
            ):
                chosen = proposal
                incumbent_totals = new_totals
                incumbent_score = new_score
                accepted += 1
                improved = True
        logs.append(
            {
                "iteration": it + 1,
                "method": selection_method,
                "start_date": str(window_dates[0].date()),
                "window_days": len(window_dates),
                "status": name,
                "accepted": improved,
                "max_deviation_before_days": score_before[0] / n_emp,
                "max_deviation_after_days": incumbent_score[0] / n_emp,
                "total_absolute_deviation_before": score_before[1] / n_emp,
                "total_absolute_deviation_after": incumbent_score[1] / n_emp,
                "fairness_score_before": objective_score(score_before),
                "fairness_score_after": objective_score(incumbent_score),
                "solver_seconds": round(solver.WallTime(), 3),
            }
        )
        if improved:
            stagnation = 0
            current_window = max(window_days, current_window - 7)
        else:
            stagnation += 1
            if stagnation >= 2:
                current_window = min(max_window_days, len(dates), current_window + 7)
                stagnation = 0
    return chosen, {
        "fairness_objective": "minimize_max_absolute_deviation_then_total_absolute_deviation",
        "lns_accepted": accepted,
        "lns_iterations_run": len(logs),
        "lns_statuses": [entry["status"] for entry in logs],
        "lns_mode": "adaptive_minimax_then_absolute_deviation",
        "lns_history": logs,
    }


def repair_low_vacation_dispatchers(
    allowed: pd.DataFrame,
    chosen: list[int],
    all_employee_ids: list[str],
    passes: int = 3,
) -> tuple[list[int], dict]:
    """Cross-year, zero/low-vacation-targeted swap neighborhood.

    Replace a selected vacation with another eligible dispatcher on the SAME
    date, work center, and shift. Therefore daily quotas and every coverage
    bucket remain exactly unchanged (unlike cross-bucket transfers).

    Prefer reducing the worst deviation, then total deviation, then the count
    of zero-vacation dispatchers. No claim of globally optimal fairness.
    This small heuristic complements rather than replaces CP-SAT adaptive LNS.
    """
    if passes < 0:
        raise ValueError("zero-repair-passes must be nonnegative")
    ids = sorted(str(e) for e in all_employee_ids)
    n = len(ids)
    rows = list(allowed.itertuples(index=False))
    total = int(sum(chosen))
    counts = dict.fromkeys(ids, 0)
    in_bucket = defaultdict(list)
    by_person = defaultdict(list)
    for j, r in enumerate(rows):
        emp = str(r.TECH_ID)
        counts[emp] += int(chosen[j])
        by_person[emp].append(j)
        in_bucket[(r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))].append(j)

    def dev(value):
        return abs(n * value - total)

    def metric():
        values = [counts[e] for e in ids]
        return (
            max(dev(v) for v in values),
            sum(dev(v) for v in values),
            sum(v == 0 for v in values),
        )

    before = metric()
    swaps = []
    for iteration in range(passes):
        moved = 0
        # Zero-vacation employees first, then those furthest below the mean.
        recipients = sorted(ids, key=lambda e: (counts[e], e))
        for recipient in recipients:
            if counts[recipient] >= total / max(1, n):
                continue
            # Search every eligible date, not only a fixed 21-day window.
            best = None
            old_metric = metric()
            for j in by_person[recipient]:
                if chosen[j]:
                    continue
                r = rows[j]
                bucket = (r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
                for k in in_bucket[bucket]:
                    if not chosen[k]:
                        continue
                    donor = str(rows[k].TECH_ID)
                    if donor == recipient or counts[donor] <= 1:
                        continue
                    # Only consider moves from someone with more vacation days.
                    if counts[donor] <= counts[recipient] + 1:
                        continue
                    # Change two employee totals by one, evaluate lexicographic
                    # fairness, and revert unless an improvement is found.
                    counts[recipient] += 1
                    counts[donor] -= 1
                    candidate = metric()
                    counts[recipient] -= 1
                    counts[donor] += 1
                    if candidate < old_metric and (best is None or candidate < best[0]):
                        best = (candidate, j, k, donor)
            if best is None:
                continue
            _, j, k, donor = best
            chosen[j], chosen[k] = 1, 0
            counts[recipient] += 1
            counts[donor] -= 1
            swaps.append(
                {
                    "pass": iteration + 1,
                    "recipient": recipient,
                    "donor": donor,
                    "shift_date": str(rows[j].SHIFT_DATE.date()),
                    "work_center": str(rows[j].WORK_CENTER_ID),
                    "shift_type": str(rows[j].SHIFT_TYPE),
                }
            )
            moved += 1
        if not moved:
            break
    after = metric()
    return chosen, {
        "zero_repair_swaps": len(swaps),
        "zero_repair_before_zero_count": before[2],
        "zero_repair_after_zero_count": after[2],
        "zero_repair_before_max_deviation_days": before[0] / n,
        "zero_repair_after_max_deviation_days": after[0] / n,
        "zero_repair_before_abs_deviation": before[1] / n,
        "zero_repair_after_abs_deviation": after[1] / n,
        "zero_repair_history": swaps,
        "zero_repair_method": "same_day_same_work_center_shift_coverage_preserving_swaps",
    }


# Column order of the supplied Snowflake SUPPLY_DATA_AFTER_VACATION_SHRINKAGE_REFERENCE view.
REFERENCE_SUPPLY_COLUMNS = [
    "INDEX",
    "MAJOR_WORK_TYPE",
    "START_TIME",
    "EMPLOYEE_TITLE",
    "WORK_CENTER_ID",
    "CONTRACTOR_FLAG",
    "CLUSTER",
    "EMPLOYEE_WORK_CENTER_ID",
    "TECH_ID",
    "END_TIME",
    "SHIFT_DATE",
    "SHIFT_LENGTH",
    "SHIFT_TYPE",
    "FINAL_QUALS",
    "PRIMARY_QUAL",
    "DOMESTIC",
    "ATC",
    "ROC",
    "CHF",
    "OC",
    "LATIN",
    "EUROPE",
    "PACIFIC",
    "QUAL_COUNT",
    "SF_LOAD_TMS",
]


def export_supply_after_vacation(
    supply: pd.DataFrame, assignments: pd.DataFrame, destination: Path
) -> tuple[int, list[str]]:
    """Remove vacation-shift rows; retain all source supply fields and values.

    Do not manufacture missing qualification columns: such fields must come
    from the actual upstream supply file. Preserve any additional source fields.
    """
    key_cols = ["TECH_ID", "SHIFT_DATE"]
    if supply.duplicated(key_cols).any():
        raise ValueError("Cannot export supply: duplicate TECH_ID/SHIFT_DATE keys")
    vac_keys = pd.MultiIndex.from_frame(assignments[key_cols])
    supply_keys = pd.MultiIndex.from_frame(supply[key_cols])
    if not vac_keys.is_unique or not vac_keys.isin(supply_keys).all():
        raise ValueError("Vacation assignments are duplicated or missing from supply")
    remaining = supply.loc[~supply_keys.isin(vac_keys)].copy()
    # Reference ordering when possible; leave additional upstream columns intact.
    missing_reference_columns = [
        c for c in REFERENCE_SUPPLY_COLUMNS if c not in supply.columns
    ]
    ordered = [c for c in REFERENCE_SUPPLY_COLUMNS if c in remaining.columns]
    ordered += [c for c in remaining.columns if c not in ordered]
    remaining = remaining[ordered]
    assert len(remaining) + len(assignments) == len(supply)
    remaining.to_csv(destination, index=False, date_format="%Y-%m-%d")
    return len(remaining), missing_reference_columns


def build_and_solve(
    supply: pd.DataFrame,
    demand: pd.DataFrame,
    quota: dict,
    training: set,
    seconds: float,
    workers: int,
    fair_seconds: float,
    lns_iterations: int = 0,
    lns_window_days: int = 21,
    lns_seconds: float = 10.0,
    lns_max_window_days: int = 60,
    lns_seed: int = 42,
    zero_repair_passes: int = 3,
):
    dates = sorted(quota)
    allowed = supply.loc[supply["SHIFT_DATE"].isin(dates)].copy()
    if allowed.empty:
        raise ValueError("No supply records match quota dates")
    # Filter training assignments BEFORE creating vacation candidate variables.
    excluded = allowed.apply(
        lambda r: (r["TECH_ID"], r["SHIFT_DATE"]) in training, axis=1
    )
    allowed = allowed.loc[~excluded].reset_index(drop=True)
    counts = allowed.groupby("SHIFT_DATE").size()
    insufficient = [
        (str(d.date()), quota[d], int(counts.get(d, 0)))
        for d in dates
        if int(counts.get(d, 0)) < quota[d]
    ]
    if insufficient:
        raise ValueError(
            f"Daily vacation quotas cannot be met from eligible staff: {insufficient[:8]}"
        )

    model = cp_model.CpModel()
    vac = [model.NewBoolVar(f"vac_{j}") for j in range(len(allowed))]
    by_day = defaultdict(list)
    by_employee = defaultdict(list)
    by_bucket = defaultdict(list)
    for j, r in enumerate(allowed.itertuples(index=False)):
        d = r.SHIFT_DATE
        emp = str(r.TECH_ID)
        b = (d, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
        by_day[d].append(vac[j])
        by_employee[emp].append(vac[j])
        by_bucket[b].append(vac[j])
    for d in dates:
        model.Add(sum(by_day[d]) == int(quota[d]))

    # Every scheduled work row counts as one base unit of coverage unless it
    # is occupied by training or vacation. Training is already excluded from
    # vacation candidacy, but must ALSO reduce productive supply in coverage.
    original_bucket = (
        supply.groupby(["SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"]).size().to_dict()
    )
    training_bucket = defaultdict(int)
    for r in supply.itertuples(index=False):
        if (str(r.TECH_ID), r.SHIFT_DATE) in training:
            training_bucket[
                (r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
            ] += 1

    needed = demand.loc[demand["SHIFT_DATE"].isin(dates)].copy()
    demand_map = {
        (r.SHIFT_DATE, str(r.WORK_CENTER), str(r.SHIFT_TYPE)): int(r.DEMAND)
        for r in needed.itertuples(index=False)
    }
    missing_bucket_keys = set(original_bucket).intersection(
        {(d, c, s) for d, c, s in original_bucket if d in quota}
    ) - set(demand_map)
    if missing_bucket_keys:
        raise ValueError(
            f"Missing desk demand for {len(missing_bucket_keys)} "
            f"work-center/shift/date combinations. Example: {sorted(missing_bucket_keys)[:5]}"
        )

    # For each known demand bucket, shortage >= required - productive supply.
    # Existing undercoverage is captured even when no dispatcher can vacation.
    shortage_vars = []
    for b, required in demand_map.items():
        base = original_bucket.get(b, 0) - training_bucket.get(b, 0)
        sv = model.NewIntVar(0, required, f"short_{len(shortage_vars)}")
        model.Add(sv >= required - base + sum(by_bucket[b]))
        shortage_vars.append(sv)

    total_shortage = model.NewIntVar(0, sum(demand_map.values()), "total_shortage")
    model.Add(total_shortage == sum(shortage_vars))
    model.Minimize(total_shortage)

    def solve(limit: float):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = limit
        solver.parameters.num_search_workers = workers
        status = solver.Solve(model)
        return solver, status

    first_solver, first_status = solve(seconds)
    if first_status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        raise RuntimeError(
            "Coverage stage has no feasible schedule: "
            + first_solver.StatusName(first_status)
        )
    first_best = int(first_solver.Value(total_shortage))
    chosen = [int(first_solver.Value(v)) for v in vac]
    first_status_name = first_solver.StatusName(first_status)
    second_status_name = "SKIPPED"
    # Lexicographic fairness only after proving the primary shortage optimum.
    # Otherwise fixing a merely feasible shortage value would falsely lock in
    # a possibly poor coverage result. Save time by skipping if not proven.
    if first_status == cp_model.OPTIMAL and fair_seconds > 0:
        model.Add(total_shortage == first_best)
        total_vac = sum(quota.values())
        employee_ids = sorted(supply["TECH_ID"].astype(str).unique())
        n_emp = len(employee_ids)
        # Minimize sum |n_emp*V_i - total_vac|, entirely integer.
        deviation = []
        for emp in employee_ids:
            possible = len(by_employee[emp])
            k = model.NewIntVar(0, possible, f"employee_vac_{emp}")
            model.Add(k == sum(by_employee[emp]))
            dev = model.NewIntVar(0, n_emp * len(dates) + total_vac, f"dev_{emp}")
            model.AddAbsEquality(dev, n_emp * k - total_vac)
            deviation.append(dev)
        model.Minimize(sum(deviation))
        for j, v in enumerate(vac):
            model.AddHint(v, chosen[j])
        second_solver, second_status = solve(fair_seconds)
        second_status_name = second_solver.StatusName(second_status)
        if second_status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            chosen = [int(second_solver.Value(v)) for v in vac]
        else:
            # CP-SAT UNKNOWN commonly means the time limit was reached without
            # a solution to the harder fairness model. Keep valid stage-1 solution.
            logging.warning(
                f"WARNING: Fairness stage {second_status_name}; retaining "
                "the feasible coverage-optimal vacation assignment."
            )

    # Optional small-window LNS: preserves daily quotas and never increases
    # modeled uncovered desk demand within any reoptimized window.
    before_lns = chosen.copy()
    if lns_iterations > 0:
        chosen, lns_stats = improve_with_lns(
            allowed,
            demand_map,
            original_bucket,
            training_bucket,
            quota,
            chosen,
            len(supply["TECH_ID"].unique()),
            iterations=lns_iterations,
            window_days=lns_window_days,
            seconds=lns_seconds,
            workers=workers,
            max_window_days=lns_max_window_days,
            seed=lns_seed,
            all_employee_ids=sorted(supply["TECH_ID"].astype(str).unique()),
        )
    else:
        lns_stats = {"lns_accepted": 0, "lns_iterations_run": 0, "lns_statuses": []}
    # Additional zero-/low-vacation-targeted neighborhood across all 363 days.
    # A swap stays in the same day/work-center/shift bucket, so neither the
    # daily vacation quota nor the coverage total can change.
    fairness_after_adaptive_lns_selection = chosen.copy()
    chosen, zero_stats = repair_low_vacation_dispatchers(
        allowed,
        chosen,
        sorted(supply["TECH_ID"].astype(str).unique()),
        passes=zero_repair_passes,
    )
    employee_ids = sorted(supply["TECH_ID"].astype(str).unique())

    def fairness_numbers(selection):
        totals = dict.fromkeys(employee_ids, 0)
        for j, selected in enumerate(selection):
            if selected:
                totals[str(allowed.iloc[j]["TECH_ID"])] += 1
        values = np.array(list(totals.values()), dtype=float)
        return {
            "min": int(values.min()),
            "max": int(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "total_absolute_deviation": float(np.abs(values - values.mean()).sum()),
            "maximum_absolute_deviation": float(np.abs(values - values.mean()).max()),
            "zero_vacation_dispatchers": int((values == 0).sum()),
            "over_40_vacation_dispatchers": int((values > 40).sum()),
        }

    fairness_before = fairness_numbers(before_lns)
    fairness_after_adaptive_lns = fairness_numbers(
        fairness_after_adaptive_lns_selection
    )
    fairness_after = fairness_numbers(chosen)
    result = allowed.loc[[j for j, selected in enumerate(chosen) if selected]].copy()
    result = result.sort_values(["SHIFT_DATE", "TECH_ID"])
    return result, {
        "coverage_stage_status": first_status_name,
        "fairness_stage_status": second_status_name,
        "initial_total_shortage": first_best,
        "training_validated": bool(training),
        "assigned_vacations": int(sum(chosen)),
        "total_quota": int(sum(quota.values())),
        "dispatchers": int(supply["TECH_ID"].nunique()),
        "num_dates": len(dates),
        "fairness_before_lns": fairness_before,
        "fairness_after_adaptive_lns": fairness_after_adaptive_lns,
        "fairness_after_lns": fairness_after,
        **lns_stats,
        **zero_stats,
    }


def main():
    p = argparse.ArgumentParser(
        description="Initial AA vacation scheduling with OR-Tools CP-SAT"
    )
    p.add_argument("--supply", default=DEFAULT_SUPPLY)
    p.add_argument("--volume", default=DEFAULT_VOLUME)
    p.add_argument("--demand", default=DEFAULT_DEMAND)
    p.add_argument("--training", default=DEFAULT_TRAINING)
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="complete supply after removing vacation-assigned shifts",
    )
    p.add_argument(
        "--vacation-output",
        default=DEFAULT_VACATION_ASSIGNMENTS,
        help="separate four-column vacation assignment CSV for flex scheduling",
    )
    p.add_argument(
        "--fixed-quota",
        type=int,
        default=None,
        help="testing only: set same vacation quota on every day",
    )
    p.add_argument(
        "--seconds", type=float, default=90.0, help="CP-SAT coverage time budget"
    )
    p.add_argument(
        "--fair-seconds",
        type=float,
        default=30.0,
        help="fairness stage only if coverage optimal",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--lns-iterations",
        type=int,
        default=10,
        help="number of LNS windows; 0 disables LNS",
    )
    p.add_argument("--lns-window-days", type=int, default=21)
    p.add_argument(
        "--lns-seconds",
        type=float,
        default=10.0,
        help="solver time budget per LNS window",
    )
    p.add_argument("--lns-max-window-days", type=int, default=60)
    p.add_argument("--lns-seed", type=int, default=42)
    p.add_argument(
        "--zero-repair-passes",
        type=int,
        default=3,
        help="full-year coverage-preserving swaps targeting under-assigned dispatchers",
    )
    args = p.parse_args()

    supply = clean_supply(read_csv(args.supply))
    demand = clean_demand(read_csv(args.demand))
    volume = clean_volume(read_csv(args.volume)) if args.fixed_quota is None else None
    # We model all the dates present in the initial schedule, not just a
    # particular week. Note: AA dataset may contain different date horizons;
    # impute and disclose any missing operational-volume dates.
    dates = [pd.Timestamp(d) for d in sorted(supply["SHIFT_DATE"].unique())]
    quota = make_quota(volume, dates, args.fixed_quota)
    # COMPETITION REFERENCE DATA GAPS: use flagged, auditable estimation,
    # never silently fill missing desk demand with zero.
    demand, demand_audit = fill_missing_demand(demand, supply, dates)
    training = load_training(args.training)
    assignments, stats = build_and_solve(
        supply,
        demand,
        quota,
        training,
        args.seconds,
        args.workers,
        args.fair_seconds,
        args.lns_iterations,
        args.lns_window_days,
        args.lns_seconds,
        args.lns_max_window_days,
        args.lns_seed,
        args.zero_repair_passes,
    )
    fields = ["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    demand_audit.to_csv(
        output.parent / "demand_imputation_audit.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    stats["imputed_demand_buckets"] = len(demand_audit)
    if stats.get("zero_repair_history"):
        pd.DataFrame(stats["zero_repair_history"]).to_csv(
            output.parent / "vacation_zero_repair_history.csv", index=False
        )
    if stats.get("lns_history"):
        pd.DataFrame(stats["lns_history"]).to_csv(
            output.parent / "vacation_lns_history.csv", index=False
        )
    stats["demand_imputation_policy"] = (
        "previous_week_or_next_week_then_nearest_same_bucket"
    )
    vacation_output = Path(args.vacation_output)
    vacation_output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == vacation_output.resolve():
        raise ValueError("--output and --vacation-output must be different paths")
    # Keep the original vacation assignment format for the existing flex model.
    assignments[fields].to_csv(vacation_output, index=False, date_format="%Y-%m-%d")
    # The main output has the SAME COLUMNS as the upstream dispatcher-level supply,
    # but omits rows corresponding to vacations (supply after vacation shrinkage).
    remaining_rows, missing_cols = export_supply_after_vacation(
        supply, assignments, output
    )
    stats["supply_rows_before_vacation"] = int(len(supply))
    stats["supply_rows_after_vacation"] = remaining_rows
    stats["supply_rows_removed_for_vacation"] = int(len(assignments))
    stats["vacation_assignments_file"] = str(vacation_output)
    stats["missing_columns_from_screenshot_schema"] = missing_cols
    if missing_cols:
        logging.warning(
            "WARNING: Upstream supply does not contain these screenshot columns; "
            "they were NOT fabricated:",
            missing_cols,
        )
    # Vacation fairness report includes dispatchers receiving zero vacation days.
    all_ids = sorted(supply["TECH_ID"].astype(str).unique())
    assigned_counts = assignments["TECH_ID"].astype(str).value_counts()
    fairness_report = pd.DataFrame({"TECH_ID": all_ids})
    fairness_report["VACATION_DAYS"] = (
        fairness_report["TECH_ID"].map(assigned_counts).fillna(0).astype(int)
    )
    # Diagnostic: distinguish solver unfairness from missing workday eligibility.
    # TODO: Recompute from Ashkat's final supply and Daniel's training calendar.
    eligible = supply.loc[
        ~supply.apply(
            lambda r: (str(r["TECH_ID"]), r["SHIFT_DATE"]) in training, axis=1
        )
    ]
    eligible_counts = eligible["TECH_ID"].astype(str).value_counts()
    work_counts = supply["TECH_ID"].astype(str).value_counts()
    fairness_report["SCHEDULED_WORKDAYS"] = (
        fairness_report["TECH_ID"].map(work_counts).fillna(0).astype(int)
    )
    fairness_report["ELIGIBLE_VACATION_DAYS"] = (
        fairness_report["TECH_ID"].map(eligible_counts).fillna(0).astype(int)
    )
    fairness_report["DEVIATION_FROM_AVERAGE"] = (
        fairness_report["VACATION_DAYS"]
        - stats["assigned_vacations"] / stats["dispatchers"]
    )
    fairness_report.to_csv(
        output.parent / "vacation_fairness_by_dispatcher.csv", index=False
    )
    # Validate daily quota in exported result.
    observed = assignments.groupby("SHIFT_DATE").size().to_dict()
    assert all(
        observed.get(d, 0) == quota[d] for d in dates
    ), "Vacation quotas violated"
    assert not any(
        (str(r.TECH_ID), r.SHIFT_DATE) in training
        for r in assignments.itertuples(index=False)
    )
    stats["training_file_supplied"] = bool(args.training)
    stats["daily_quota_method"] = (
        "fixed_test_quota"
        if args.fixed_quota is not None
        else "volume_rank_quintiles_ASSUMPTION"
    )
    (output.parent / (output.stem + "_summary.json")).write_text(
        json.dumps(stats, indent=2)
    )
    pd.DataFrame(
        {"SHIFT_DATE": dates, "VACATION_TARGET": [quota[d] for d in dates]}
    ).to_csv(
        output.parent / "vacation_daily_targets.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    logging.info("Saved supply after vacation", output)
    logging.info("Saved vacation assignments", vacation_output)
    logging.info(json.dumps(stats, indent=2))
    logging.info("Zero-vacation targeted swaps:", stats["zero_repair_swaps"])
    logging.info(
        "Zero-vacation dispatchers after repair:", stats["zero_repair_after_zero_count"]
    )
    if not args.training:
        logging.warning(
            "WARNING: Daniel training input is absent. Results are PROVISIONAL."
        )


if __name__ == "__main__":
    main()
