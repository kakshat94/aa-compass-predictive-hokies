"""Exact max-flow or greedy flex-day placement after bidlines, training, and vacations.

Uses existing dispatcher-day schedule as fixed baseline. Additional flex days
are allocated on otherwise off days to reduce desk staffing shortages.

This is a provisional Phase-II heuristic, NOT a certified FAA/CBA compliance
checker. The optional consecutive-day cap is disabled by default because flex may extend a 4-day block to 5 days. It is not a substitute for
full rest/qualification rules.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import pandas as pd
import logging


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing input: {p.resolve()}")
    df = pd.read_csv(p, dtype={"TECH_ID": str}, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    return df


def normalize_dates(df, field, name):
    if field not in df:
        raise ValueError(f"{name} must have {field}")
    df = df.copy()
    original = df[field].copy()
    # Demand and audit CSVs can mix YYYY-MM-DD with timestamp strings.
    df[field] = pd.to_datetime(
        df[field], format="mixed", errors="coerce"
    ).dt.normalize()
    if df[field].isna().any():
        examples = original[df[field].isna()].head(5).tolist()
        raise ValueError(
            f"{name} contains {df[field].isna().sum()} invalid {field} values; examples: {examples}"
        )
    return df


def normalized_people(df, name):
    df = normalize_dates(df, "SHIFT_DATE", name)
    if "TECH_ID" not in df:
        raise ValueError(f"{name} must have TECH_ID")
    df["TECH_ID"] = df["TECH_ID"].astype("string").str.strip()
    if df["TECH_ID"].isna().any() or (df["TECH_ID"] == "").any():
        raise ValueError(f"{name}: missing TECH_ID")
    return df


def normalize_supply(df):
    df = normalized_people(df, "supply")
    for col in ("WORK_CENTER_ID", "SHIFT_TYPE"):
        if col not in df:
            raise ValueError(f"supply needs {col}")
        df[col] = df[col].astype("string").str.strip().str.upper()
        if df[col].isna().any():
            raise ValueError(f"supply has null {col}")
    if df.duplicated(["TECH_ID", "SHIFT_DATE"]).any():
        raise ValueError("supply has duplicate dispatcher-day rows")
    return df


def normalize_demand(df):
    df = normalize_dates(df, "SHIFT_DATE", "demand")
    required = ("WORK_CENTER", "SHIFT_TYPE", "DEMAND")
    if any(c not in df for c in required):
        raise ValueError(f"demand requires {required}")
    for col in ("WORK_CENTER", "SHIFT_TYPE"):
        df[col] = df[col].astype("string").str.strip().str.upper()
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="raise")
    if (
        df["DEMAND"].isna().any()
        or (df["DEMAND"] < 0).any()
        or (df["DEMAND"] % 1 != 0).any()
    ):
        raise ValueError("demand must be nonnegative integers")
    if df.duplicated(["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE"]).any():
        raise ValueError("duplicate demand bucket (date, center, shift)")
    return df


def max_run_if_added(workdays: set, date: pd.Timestamp, max_days: int) -> bool:
    """True if adding date would NOT create a work streak over max_days."""
    left, right = 0, 0
    day = date - pd.Timedelta(days=1)
    while day in workdays:
        left += 1
        day -= pd.Timedelta(days=1)
    day = date + pd.Timedelta(days=1)
    while day in workdays:
        right += 1
        day += pd.Timedelta(days=1)
    return left + 1 + right <= max_days


def schedule_flex(
    supply,
    demand,
    vacation,
    training,
    *,
    max_flex=4,
    max_consecutive=0,
    min_spacing=0,
    start=None,
    end=None,
    solver="maxflow",
):
    if solver not in ("maxflow", "greedy"):
        raise ValueError("solver must be maxflow or greedy")
    if solver == "maxflow" and (max_consecutive or min_spacing):
        raise ValueError(
            "Exact max-flow handles the confirmed four-flex-day cap, but not optional spacing or consecutive-day rules. Use --solver greedy for those, or a CP-SAT extension."
        )
    supply = normalize_supply(supply)
    demand = normalize_demand(demand)
    vacation = normalized_people(vacation, "vacation")
    training = normalized_people(training, "training")
    if max_flex < 0 or min_spacing < 0 or max_consecutive < 0:
        raise ValueError(
            "max-flex, min-spacing, and max-consecutive cannot be negative"
        )
    if not len(supply):
        raise ValueError("Empty base schedule")

    dates = sorted(supply["SHIFT_DATE"].unique())
    begin = pd.Timestamp(start).normalize() if start else pd.Timestamp(dates[0])
    finish = pd.Timestamp(end).normalize() if end else pd.Timestamp(dates[-1])
    if finish < begin:
        raise ValueError("end date precedes start")
    horizon = set(pd.date_range(begin, finish, freq="D"))
    original_work = defaultdict(set)
    buckets = defaultdict(Counter)
    for r in supply.itertuples(index=False):
        original_work[str(r.TECH_ID)].add(r.SHIFT_DATE)
        buckets[str(r.TECH_ID)][(str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))] += 1
    people = sorted(original_work)
    assignment = {p: bucket.most_common(1)[0][0] for p, bucket in buckets.items()}
    ambiguous = {p: list(bucket) for p, bucket in buckets.items() if len(bucket) > 1}
    if ambiguous:
        raise ValueError(
            f"{len(ambiguous)} dispatchers work multiple center/shift combinations; "
            f"cannot infer unique flex qualifications. Example: {list(ambiguous.items())[:3]}"
        )

    vacation_pairs = set(zip(vacation["TECH_ID"].astype(str), vacation["SHIFT_DATE"]))
    training_pairs = set(zip(training["TECH_ID"].astype(str), training["SHIFT_DATE"]))
    valid_pairs = {
        (str(r.TECH_ID), r.SHIFT_DATE) for r in supply.itertuples(index=False)
    }
    invalid_v = vacation_pairs - valid_pairs
    if invalid_v:
        raise ValueError(
            f"{len(invalid_v)} vacation assignments are not on base scheduled workdays; example {next(iter(invalid_v))}"
        )
    unknown_training = {p for p, _ in training_pairs if p not in original_work}
    if unknown_training:
        raise ValueError(
            f"Training contains unknown dispatchers: {sorted(unknown_training)[:5]}"
        )
    if training_pairs & vacation_pairs:
        raise ValueError(
            "Training and vacation overlap; resolve before flex scheduling"
        )

    # Each existing scheduled day counts as availability except vacation or training.
    base_active = Counter()
    for r in supply.itertuples(index=False):
        pair = (str(r.TECH_ID), r.SHIFT_DATE)
        if pair not in vacation_pairs and pair not in training_pairs:
            base_active[(r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))] += 1

    required = {}
    for r in demand.itertuples(index=False):
        if r.SHIFT_DATE in horizon:
            required[(r.SHIFT_DATE, str(r.WORK_CENTER), str(r.SHIFT_TYPE))] = int(
                r.DEMAND
            )
    # Require demand for every represented work bucket, never silently assume zero.
    represented = {
        (r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
        for r in supply.itertuples(index=False)
        if r.SHIFT_DATE in horizon
    }
    missing = represented - required.keys()
    if missing:
        raise ValueError(
            f"Missing demand for {len(missing)} scheduled buckets; example {next(iter(missing))}. "
            "Use the vacation module's demand-imputation audit before flex allocation."
        )

    # Validate the fixed flex assignments from the Phase 1 MIP. They must be on
    # original days off and must not collide with vacation or training.
    if preassigned is None:
        preassigned = pd.DataFrame(columns=["TECH_ID", "SHIFT_DATE"])
    preassigned = normalized_people(preassigned, "preassigned flex")
    if preassigned.duplicated(["TECH_ID", "SHIFT_DATE"]).any():
        bad = preassigned.loc[
            preassigned.duplicated(["TECH_ID", "SHIFT_DATE"], keep=False),
            ["TECH_ID", "SHIFT_DATE"],
        ].head(5)
        raise ValueError(f"preassigned flex has duplicate dispatcher-day rows:\n{bad}")
    for col in ("WORK_CENTER_ID", "SHIFT_TYPE"):
        if col in preassigned:
            preassigned[col] = preassigned[col].astype("string").str.strip().str.upper()
    if "ASSIGNMENT_TYPE" not in preassigned:
        preassigned["ASSIGNMENT_TYPE"] = "FLEX_PREASSIGNED"
    preassigned["ASSIGNMENT_TYPE"] = (
        preassigned["ASSIGNMENT_TYPE"].fillna("FLEX_PREASSIGNED").astype(str)
    )

    pre_records = []
    pre_counts = Counter()
    pre_by_bucket = Counter()
    for r in preassigned.itertuples(index=False):
        person = str(r.TECH_ID)
        date = r.SHIFT_DATE
        if person not in original_work:
            raise ValueError(f"preassigned flex uses unknown dispatcher {person}")
        if date not in horizon:
            raise ValueError(
                f"preassigned flex date {date.date()} is outside the scheduling horizon"
            )
        if date in original_work[person]:
            raise ValueError(
                f"preassigned flex for {person} on {date.date()} is an original working day"
            )
        if (person, date) in vacation_pairs:
            raise ValueError(
                f"preassigned flex for {person} on {date.date()} overlaps vacation"
            )
        if (person, date) in training_pairs:
            raise ValueError(
                f"preassigned flex for {person} on {date.date()} overlaps training"
            )
        center = getattr(r, "WORK_CENTER_ID", None)
        shift = getattr(r, "SHIFT_TYPE", None)
        if pd.isna(center) or pd.isna(shift):
            center, shift = assignment[person]
        else:
            center, shift = str(center), str(shift)
            if (center, shift) != assignment[person]:
                raise ValueError(
                    f"preassigned flex for {person} on {date.date()} uses {center}/{shift}, "
                    f"expected {assignment[person][0]}/{assignment[person][1]}"
                )
        pre_counts[person] += 1
        if pre_counts[person] > max_flex:
            raise ValueError(
                f"preassigned flex gives {person} {pre_counts[person]} days, above the cap {max_flex}"
            )
        pre_records.append(
            {
                "TECH_ID": person,
                "SHIFT_DATE": date,
                "WORK_CENTER_ID": center,
                "SHIFT_TYPE": shift,
                "ASSIGNMENT_TYPE": str(r.ASSIGNMENT_TYPE),
                "IS_PREASSIGNED": True,
            }
        )
        pre_by_bucket[(date, center, shift)] += 1

    pre_allocations = pd.DataFrame(
        pre_records,
        columns=[
            "TECH_ID",
            "SHIFT_DATE",
            "WORK_CENTER_ID",
            "SHIFT_TYPE",
            "ASSIGNMENT_TYPE",
            "IS_PREASSIGNED",
        ],
    )

    active = Counter(base_active)
    counts = Counter(pre_counts)
    flex_dates = defaultdict(set)
    pre_dates = defaultdict(set)
    work_all = {p: set(days) for p, days in original_work.items()}
    for row in pre_records:
        pre_dates[row["TECH_ID"]].add(row["SHIFT_DATE"])
        key = (row["SHIFT_DATE"], row["WORK_CENTER_ID"], row["SHIFT_TYPE"])
        active[key] += 1
        flex_dates[row["TECH_ID"]].add(row["SHIFT_DATE"])
        work_all[row["TECH_ID"]].add(row["SHIFT_DATE"])

    after_preassigned_shortage = sum(
        max(0, dem - active[key]) for key, dem in required.items()
    )
    deficit = {key: max(0, dem - active[key]) for key, dem in required.items()}
    before_shortage = sum(
        max(0, dem - base_active[key]) for key, dem in required.items()
    )
    initial_deficit = dict(deficit)

    by_bucket = defaultdict(list)
    for p in people:
        by_bucket[assignment[p]].append(p)
    chosen = []
    added_by_bucket = Counter()

    # At each step fill a high-shortage bucket, using the least-used eligible
    # dispatcher. A global greedy heuristic avoids CP-SAT runtime here.
    remaining = set(k for k, v in deficit.items() if v > 0)
    while remaining:
        # Prioritize larger deficits, then earlier dates and stable bucket names.
        key = min(remaining, key=lambda k: (-deficit[k], k[0], k[1], k[2]))
        date, center, shift = key
        eligible = []
        for p in by_bucket.get((center, shift), ()):
            if counts[p] >= max_flex or date in work_all[p]:
                continue
            if (p, date) in vacation_pairs or (p, date) in training_pairs:
                continue
            if min_spacing and any(
                abs((date - d).days) < min_spacing for d in flex_dates[p]
            ):
                continue
            if max_consecutive and not max_run_if_added(
                work_all[p], date, max_consecutive
            ):
                continue
            # Prefer few assigned flex days, then avoid placing flex next
            # to an already-working day, then deterministic employee ID.
            adjacent = int(date - pd.Timedelta(days=1) in work_all[p]) + int(
                date + pd.Timedelta(days=1) in work_all[p]
            )
            eligible.append((counts[p], adjacent, p))
        if not eligible:
            remaining.remove(key)
            continue
        _, _, person = min(eligible)
        chosen.append(
            {
                "TECH_ID": person,
                "SHIFT_DATE": date,
                "WORK_CENTER_ID": center,
                "SHIFT_TYPE": shift,
                "ASSIGNMENT_TYPE": "FLEX_FIXED",
            }
        )
        counts[person] += 1
        added_by_bucket[key] += 1
        flex_dates[person].add(date)
        work_all[person].add(date)
        active[key] += 1
        deficit[key] -= 1
        if deficit[key] == 0:
            remaining.remove(key)

    greedy_count = len(chosen)
    if solver == "maxflow":
        try:
            from ortools.graph.python import max_flow
        except ImportError as exc:
            raise ImportError(
                "OR-Tools is needed for --solver maxflow. Install with: pip install ortools"
            ) from exc
        # Bipartite network: source -> dispatcher (annual flex cap) ->
        # date/work-center/shift with shortage (one flex for this person/day)
        # -> sink (the shortage capacity). Max-flow proves maximal coverage
        # with the currently specified independent eligibility conditions.
        graph = max_flow.SimpleMaxFlow()
        source, sink = 0, 1
        next_node = 2
        person_nodes = {}
        bucket_nodes = {}
        for p in people:
            person_nodes[p] = next_node
            next_node += 1
        for key, need in sorted(initial_deficit.items()):
            if need > 0:
                bucket_nodes[key] = next_node
                next_node += 1
        for p in people:
            remaining_capacity = max_flex - pre_counts[p]
            if remaining_capacity > 0:
                graph.add_arc_with_capacity(source, person_nodes[p], remaining_capacity)
        for key, node in bucket_nodes.items():
            graph.add_arc_with_capacity(node, sink, initial_deficit[key])
        assignment_arcs = []
        for key, node in sorted(bucket_nodes.items()):
            date, center, shift = key
            for p in by_bucket.get((center, shift), ()):
                if (
                    date in original_work[p]
                    or (p, date) in vacation_pairs
                    or (p, date) in training_pairs
                ):
                    continue
                arc = graph.add_arc_with_capacity(person_nodes[p], node, 1)
                assignment_arcs.append((arc, p, key))
        status = graph.solve(source, sink)
        if status != graph.OPTIMAL:
            raise RuntimeError(
                f"Max-flow did not return OPTIMAL; solver status: {status}"
            )
        chosen = []
        added_by_bucket = Counter()
        counts = Counter(pre_counts)
        active = Counter(base_active)
        for key, count in pre_by_bucket.items():
            active[key] += count
        for arc, p, key in assignment_arcs:
            if graph.flow(arc) == 1:
                date, center, shift = key
                chosen.append(
                    {
                        "TECH_ID": p,
                        "SHIFT_DATE": date,
                        "WORK_CENTER_ID": center,
                        "SHIFT_TYPE": shift,
                        "ASSIGNMENT_TYPE": "FLEX_FIXED",
                    }
                )
                counts[p] += 1
                added_by_bucket[key] += 1
                active[key] += 1
        assert len(chosen) == graph.optimal_flow(), "Flow extraction mismatch"
        if len(chosen) < greedy_count:
            raise AssertionError(
                f"Exact maximum flow {len(chosen)} < greedy baseline {greedy_count}; check modeling mismatch"
            )

    allocations = pd.DataFrame(
        chosen,
        columns=[
            "TECH_ID",
            "SHIFT_DATE",
            "WORK_CENTER_ID",
            "SHIFT_TYPE",
            "ASSIGNMENT_TYPE",
        ],
    )
    cover_rows = []
    for key in sorted(required):
        date, center, shift = key
        dem = required[key]
        post = active[key]
        flex_count = added_by_bucket[key]
        pre = post - flex_count
        cover_rows.append(
            {
                "SHIFT_DATE": date,
                "WORK_CENTER": center,
                "SHIFT_TYPE": shift,
                "DEMAND": dem,
                "AVAILABLE_BEFORE_FLEX": pre,
                "AVAILABLE_AFTER_FLEX": post,
                "SHORTAGE_BEFORE_FLEX": max(0, dem - pre),
                "SHORTAGE_AFTER_FLEX": max(0, dem - post),
                "FLEX_ASSIGNED": flex_count,
                "OVERCOVERAGE_AFTER_FLEX": max(0, post - dem),
            }
        )
    coverage = pd.DataFrame(cover_rows)
    employee_report = pd.DataFrame({"TECH_ID": people})
    employee_report["FLEX_DAYS"] = (
        employee_report["TECH_ID"].map(counts).fillna(0).astype(int)
    )
    employee_report["BASE_WORKDAYS"] = employee_report["TECH_ID"].map(
        lambda p: len(original_work[p])
    )
    employee_report["WORK_CENTER_ID"] = employee_report["TECH_ID"].map(
        lambda p: assignment[p][0]
    )
    employee_report["SHIFT_TYPE"] = employee_report["TECH_ID"].map(
        lambda p: assignment[p][1]
    )
    summary = {
        "dispatchers": len(people),
        "max_flex_per_dispatcher": max_flex,
        "max_consecutive_workdays_assumption": max_consecutive,
        "min_spacing_between_flex_days": min_spacing,
        "flex_days_added": len(chosen),
        "dispatchers_receiving_flex": int((employee_report["FLEX_DAYS"] > 0).sum()),
        "total_shortage_before_flex": int(before_shortage),
        "total_shortage_after_flex": int(coverage["SHORTAGE_AFTER_FLEX"].sum()),
        "shortage_reduction": int(
            before_shortage - coverage["SHORTAGE_AFTER_FLEX"].sum()
        ),
        "total_overcoverage_after_flex": int(coverage["OVERCOVERAGE_AFTER_FLEX"].sum()),
        "training_file_supplied": bool(len(training)),
        "solver": (
            "exact_bipartite_max_flow"
            if solver == "maxflow"
            else "coverage_priority_greedy_heuristic"
        ),
        "greedy_baseline_flex_days": greedy_count,
        "improvement_over_greedy": len(chosen) - greedy_count,
        "proven_optimal_under_current_flex_constraints": solver == "maxflow",
        "caveat": "Preliminary: not certified for FAA/CBA, qualifications or mandatory four-day entitlements",
    }
    assert len(chosen) <= len(people) * max_flex
    assert employee_report["FLEX_DAYS"].max() <= max_flex
    assert int(coverage["FLEX_ASSIGNED"].sum()) == total_flex
    assert int(coverage["FLEX_NEW"].sum()) == len(chosen)
    assert (
        summary["shortage_reduction"]
        == summary["shortage_reduction_from_preassigned"]
        + summary["shortage_reduction_from_new_flex"]
    )
    return allocations, coverage, employee_report, summary


def main():
    parser = argparse.ArgumentParser(
        description="Flex-day scheduling: exact max-flow or greedy baseline"
    )
    parser.add_argument(
        "--solver",
        choices=("maxflow", "greedy"),
        default="maxflow",
        help="maxflow is exact under four-day cap and eligibility; greedy allows optional spacing restrictions",
    )
    parser.add_argument(
        "--supply",
        default="outputs/exact_gurobi/Supply_Regular_qualified.csv",
        help="Base bidline employee-day work schedule (later replace with teammate output)",
    )
    parser.add_argument("--vacation", default="outputs/vacation_initial.csv")
    parser.add_argument(
        "--demand",
        default="inputs/annual_desk_needs_reference.csv",
        help="Complete or previously-imputed desk demand",
    )
    parser.add_argument(
        "--demand-audit",
        default="outputs/demand_imputation_audit.csv",
        help="Vacation module demand imputation audit; needed for missing desk-demand buckets",
    )
    parser.add_argument(
        "--training",
        default="outputs/exact_gurobi/all_training_data.csv",
        help="Optional TECH_ID,SHIFT_DATE training schedule",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-flex", type=int, default=4)
    parser.add_argument(
        "--max-consecutive",
        type=int,
        default=0,
        help="Optional consecutive-workday cap; 0 disables (default). Apply only if competition rules require it",
    )
    parser.add_argument(
        "--min-spacing",
        type=int,
        default=0,
        help="Minimum calendar days between flex assignments per dispatcher",
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    tr = (
        load_csv(args.training)
        if args.training
        else pd.DataFrame(columns=["TECH_ID", "SHIFT_DATE"])
    )
    demand = normalize_dates(load_csv(args.demand), "SHIFT_DATE", "demand")
    if args.demand_audit and Path(args.demand_audit).is_file():
        audit = normalize_dates(
            load_csv(args.demand_audit), "SHIFT_DATE", "demand audit"
        )
        required_audit = {"SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "IMPUTED_DEMAND"}
        if not required_audit.issubset(audit.columns):
            raise ValueError(f"Demand audit must contain {sorted(required_audit)}")
        supplement = audit[
            ["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "IMPUTED_DEMAND"]
        ].rename(columns={"IMPUTED_DEMAND": "DEMAND"})
        existing = set(
            zip(
                demand["SHIFT_DATE"],
                demand["WORK_CENTER"].astype(str).str.strip().str.upper(),
                demand["SHIFT_TYPE"].astype(str).str.strip().str.upper(),
            )
        )
        supplement = supplement.loc[
            [
                key not in existing
                for key in zip(
                    supplement["SHIFT_DATE"],
                    supplement["WORK_CENTER"].astype(str).str.strip().str.upper(),
                    supplement["SHIFT_TYPE"].astype(str).str.strip().str.upper(),
                )
            ]
        ]
        if len(supplement):
            logging.warning(
                f"WARNING: Using {len(supplement)} previously imputed demand buckets from {args.demand_audit}"
            )
            demand = pd.concat([demand, supplement], ignore_index=True)
    a, c, e, s = schedule_flex(
        load_csv(args.supply),
        demand,
        load_csv(args.vacation),
        tr,
        max_flex=args.max_flex,
        max_consecutive=args.max_consecutive,
        min_spacing=args.min_spacing,
        start=args.start,
        end=args.end,
        solver=args.solver,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    a.to_csv(out / "flex_days.csv", index=False, date_format="%Y-%m-%d")
    c.to_csv(
        out / "staffing_coverage_before_after_flex.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    e.to_csv(out / "flex_by_dispatcher.csv", index=False)
    (out / "flex_summary.json").write_text(json.dumps(s, indent=2))
    logging.info(json.dumps(s, indent=2))
    logging.info(f"Saved flex results to {out.resolve()}")
    if not args.training:
        logging.warning("WARNING: Training is missing. Results are provisional.")


if __name__ == "__main__":
    main()
