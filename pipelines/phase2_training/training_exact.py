"""Exact training assignment (COMPASS Phase II) -- brainstorming "T1" model.

Purpose
-------
Replace the random training placement (``members/daniel/training_assigner``) with
an exact mixed-integer program that *chooses where* to place the mandatory
training days so the resulting shrinkage does the least damage to coverage. It is
a drop-in for :func:`members.daniel.training_assigner.assign_training`: same call
signature shape and the same :class:`TrainingResult` return type, so the rest of
the pipeline (``write_training_outputs``, ``verify_training``, ``run_vacation``,
``coverage_report``) is unchanged.

Model (per shift type -- the 12-seat class cap couples only within a shift)
--------------------------------------------------------------------------
Decision ``y[r,d] in {0,1}``: requirement ``r = (line, skill, month)`` is trained
on working day ``d``. Let ``z[w,d] = sum_r y[r,d]`` be the number of trainings
removed from slot ``(work center w, date d)``, ``base[w,d]`` the roster supply and
``demand[w,d]`` the desk need. Then

    min   100 * sum(under) + 10 * sum(over)
    s.t.  sum_d y[r,d] = 1                         (one training per requirement)
          sum_{r of line l, same month} y[r,d] <= 1 (<=1 training per line per day)
          sum_{r, shift(r)=s} y[r,d] <= class_size (shared class cap per shift/day)
          z[w,d] = sum_r y[r,d]
          under[w,d] >= demand[w,d] - (base[w,d] - z[w,d])
          over[w,d]  >= (base[w,d] - z[w,d]) - demand[w,d]
          y binary, under/over >= 0

The ``under``/``over`` terms are convex piecewise-linear in the integer ``z``, so
the program is exact and small (~2.1k requirements x ~15 eligible days).

Decomposition
-------------
The model is solved **exactly** as one MIP per ``(shift, month)``. Training months
never share calendar dates, so nothing couples two months: the class cap is a
per-date constraint, the per-line-per-day rule is per-date, and coverage is
per-date. The union of the per-month optima is therefore the global optimum. It
also keeps every model under the FICO **Community licence** ceiling of 5000
rows+columns (largest group ~3636), so the identical code runs on both Gurobi and
Xpress without a heuristic repair.

Inputs / outputs
----------------
* ``schedule``: the full working-day roster (regular + the two training files
  concatenated, qualification columns populated) -- one row per bid-line worker-day.
* ``training_months``: ``{skill: [month ints]}`` (see ``params.yaml``).
* ``demand`` / ``demand_path``: desk demand keyed by
  ``(work center, shift, M/D/YYYY)``; defaults to ``inputs/annual_desk_needs_reference.csv``.
* Returns a :class:`~members.daniel.training_assigner.TrainingResult` whose
  ``report`` attribute holds per-model solver status, objective, runtime and mode.

Assumptions
-----------
* Training falls only on a scheduled working day and not on a flex/vacation day
  (the latter two sets are optional and, in the current pipeline, empty).
* Reproducibility: the model is deterministic given identical input.

References
----------
* Gurobi Optimizer Reference Manual (Python API).
* FICO Xpress Python API reference and Community licence limits.
* Bechtold & Jacobs (1990) -- implicit modelling of flexible intra-tour choices.
* Özgüven & Sungur (2013) -- integer models with excess off-days / idle time.
"""

from __future__ import annotations

import argparse
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataclasses import dataclass, field  # noqa: E402
from typing import Dict, List, Optional, Set, Tuple  # noqa: E402

import pandas as pd  # noqa: E402

from pipelines.phase2_training.qualification_mapper import (
    parse_final_quals,
)  # noqa: E402
from pipelines.phase2_training.solvers import (  # noqa: E402
    FEASIBLE,
    OPTIMAL,
    BaseBackend,
    make_backend,
)
from pipelines.phase2_training.training_assigner import (  # noqa: E402
    CLASS_SIZE_DEFAULT,
    RANDOM_SEED,
    TrainingResult,
    _format_date,
    build_result_from_marks,
    load_training_months,
)

_GOOD = {OPTIMAL, FEASIBLE}


@dataclass
class _Context:
    """Shared, immutable facts the model builders need."""

    tech_days: Dict[object, List[Tuple[str, int]]]
    tech_shift: Dict[object, str]
    tech_wc: Dict[object, str]
    blocked: Set[Tuple[object, str]]
    base_supply: Dict[Tuple[str, str, str], int]
    demand: Dict[Tuple[str, str, str], float]
    date_month: Dict[str, int] = field(default_factory=dict)

    def eligible_days(self, tech: object, month: int) -> List[str]:
        return [
            d
            for d, m in self.tech_days.get(tech, [])
            if m == month and (str(tech), d) not in self.blocked
        ]


def _prepare(schedule: pd.DataFrame) -> pd.DataFrame:
    """Add non-padded ``_DATE`` and integer ``_MONTH`` helper columns."""
    df = schedule.copy().reset_index(drop=True)
    df["_DT"] = pd.to_datetime(df["SHIFT_DATE"])
    df["_DATE"] = df["_DT"].map(_format_date)
    df["_MONTH"] = df["_DT"].dt.month
    return df


def _load_demand(demand_path: str) -> Dict[Tuple[str, str, str], float]:
    """Desk demand keyed by ``(WORK_CENTER, SHIFT_TYPE, M/D/YYYY)``."""
    d = pd.read_csv(demand_path)
    dates = pd.to_datetime(d["SHIFT_DATE"]).map(_format_date)
    keys = list(zip(d["WORK_CENTER"].astype(str), d["SHIFT_TYPE"].astype(str), dates))
    return dict(zip(keys, d["DEMAND"].astype(float)))


def _solve_group(
    backend: BaseBackend,
    reqs: List[Tuple[object, str, int]],
    ctx: _Context,
    shift: str,
    class_size: int,
    under_penalty: float,
    over_penalty: float,
    time_limit: Optional[float],
    mip_gap: Optional[float],
    daily_cap: Optional[Dict[str, int]] = None,
) -> dict:
    """Build and solve one training MIP over a set of requirements that share a shift.

    ``daily_cap`` optionally overrides the per-date class cap (used by the
    decomposed mode to hand each work center its share of the shift's seats).
    Returns ``{chosen, status, objective, seconds, failures, requirements}``.
    """
    y: Dict[Tuple[object, str, int, str], object] = {}
    req_elig: Dict[Tuple[object, str, int], List[str]] = {}
    failures: List[Tuple[object, str, int]] = []

    for req in reqs:
        tech, skill, month = req
        days = ctx.eligible_days(tech, month)
        req_elig[req] = days
        if not days:
            failures.append(req)
        for d in days:
            y[(tech, skill, month, d)] = backend.add_binary(
                f"y_{tech}_{skill}_{month}_{d}"
            )

    # C1: exactly one training per requirement.
    for req in reqs:
        terms = [(1.0, y[(req[0], req[1], req[2], d)]) for d in req_elig[req]]
        if terms:
            backend.add_constraint(terms, "==", 1.0, name="one_per_requirement")

    # C2: <=1 training per line per day. Only same-line same-month requirements can
    # collide on a calendar day, so scope the rows there to keep the model small.
    by_tech_month: Dict[Tuple[object, int], List[Tuple[object, str, int]]] = {}
    for req in reqs:
        by_tech_month.setdefault((req[0], req[2]), []).append(req)
    for (tech, month), group in by_tech_month.items():
        if len(group) < 2:
            continue
        days_union = sorted({d for r in group for d in req_elig[r]})
        for d in days_union:
            terms = [(1.0, y[(r[0], r[1], r[2], d)]) for r in group if d in req_elig[r]]
            if terms:
                backend.add_constraint(terms, "<=", 1.0, name="one_per_line_day")

    # C3: shared class cap per (date, shift); or the injected per-date budget.
    by_date: Dict[str, List[object]] = {}
    for (_tech, _skill, _month, d), var in y.items():
        by_date.setdefault(d, []).append(var)
    for d, vars_ in by_date.items():
        cap = class_size if daily_cap is None else daily_cap.get(d, class_size)
        backend.add_constraint(
            [(1.0, v) for v in vars_], "<=", float(cap), name="class_cap"
        )

    # Coverage: link training counts to per-slot under/over-coverage.
    slot_vars: Dict[Tuple[str, str], List[object]] = {}
    for (tech, _skill, _month, d), var in y.items():
        slot_vars.setdefault((ctx.tech_wc[tech], d), []).append(var)

    obj_terms: List[Tuple[float, object]] = []
    for (wc, d), vars_ in slot_vars.items():
        z = backend.add_continuous(f"z_{wc}_{d}", lb=0.0)
        backend.add_constraint(
            [(1.0, v) for v in vars_] + [(-1.0, z)], "==", 0.0, name="z_def"
        )
        dem = float(ctx.demand.get((wc, shift, d), 0.0))
        base = float(ctx.base_supply.get((wc, shift, d), 0))
        under = backend.add_continuous(f"u_{wc}_{d}", lb=0.0)
        over = backend.add_continuous(f"o_{wc}_{d}", lb=0.0)
        backend.add_constraint(
            [(1.0, under), (-1.0, z)], ">=", dem - base, name="under_def"
        )
        backend.add_constraint(
            [(1.0, over), (1.0, z)], ">=", base - dem, name="over_def"
        )
        obj_terms.append((under_penalty, under))
        obj_terms.append((over_penalty, over))

    if not obj_terms:  # degenerate group with no requirements
        obj_terms.append((0.0, backend.add_continuous("const0", lb=0.0)))
    backend.set_min_objective(obj_terms)

    status = backend.optimize(time_limit=time_limit, mip_gap=mip_gap)
    chosen: Dict[Tuple[object, str], str] = {}
    if status in _GOOD:
        for (tech, skill, _month, d), var in y.items():
            if backend.value(var) > 0.5:
                chosen[(tech, d)] = skill
    return {
        "chosen": chosen,
        "status": status,
        "objective": backend.obj_value() if status in _GOOD else None,
        "seconds": round(backend.solve_time, 3),
        "failures": failures,
        "requirements": len(reqs),
    }


def assign_training_exact(
    schedule: pd.DataFrame,
    training_months: Dict[str, List[int]],
    class_size: int = CLASS_SIZE_DEFAULT,
    solver: str = "gurobi",
    demand: Optional[Dict[Tuple[str, str, str], float]] = None,
    demand_path: str = "inputs/annual_desk_needs_reference.csv",
    flex_days: Optional[Set[Tuple[object, str]]] = None,
    vacation_days: Optional[Set[Tuple[object, str]]] = None,
    seed: int = RANDOM_SEED,
    time_limit: Optional[float] = 120.0,
    mip_gap: Optional[float] = None,
    log_path: Optional[str] = None,
    under_penalty: float = 100.0,
    over_penalty: float = 10.0,
    frozen: Optional[Dict[Tuple[object, str], str]] = None,
) -> TrainingResult:
    """Drop-in exact replacement for :func:`assign_training` (see module docstring).

    The model is solved exactly in one MIP per ``(shift, month)`` pair. This is
    lossless: training months never share calendar dates, so neither the class
    cap (a per-date constraint) nor the per-line-per-day rule couples two months,
    and the union of the per-month optima is the global optimum. It also keeps
    every model well under the FICO Community licence ceiling of 5000 rows+columns.

    ``frozen`` optionally pins ``(tech, date) -> skill`` assignments from Phase I.
    Those trainings stay on their dates, consume class seats, and are subtracted
    from the supply baseline; the MIP then places only the remaining
    secondary-qualification requirements.
    """
    del seed  # deterministic MIP; kept for signature parity with the random path
    frozen = dict(frozen or {})
    blocked = {
        (str(tech), date)
        for tech, date in set(flex_days or set()) | set(vacation_days or set())
    }

    df = _prepare(schedule)
    df["_WC"] = df["WORK_CENTER_ID"].astype(str)

    tech_days: Dict[object, List[Tuple[str, int]]] = {}
    tech_shift: Dict[object, str] = {}
    tech_wc: Dict[object, str] = {}
    for tech, grp in df.groupby("TECH_ID", sort=False):
        tech_days[tech] = list(zip(grp["_DATE"], grp["_MONTH"]))
        tech_shift[tech] = str(grp["SHIFT_TYPE"].iloc[0])
        tech_wc[tech] = str(grp["_WC"].iloc[0])

    tech_quals: Dict[object, List[str]] = (
        df.drop_duplicates("TECH_ID")
        .set_index("TECH_ID")["FINAL_QUALS"]
        .map(parse_final_quals)
        .to_dict()
    )

    date_month = dict(zip(df["_DATE"], df["_MONTH"]))
    base_supply = df.groupby(["_WC", "SHIFT_TYPE", "_DATE"]).size().to_dict()

    # Pin the Phase I schedule first: validate it, block its days, consume its
    # class seats, and remove it from the supply baseline the MIP sees.
    schedule_rows = set(zip(df["TECH_ID"], df["_DATE"]))
    frozen_requirements: Set[Tuple[object, str, int]] = set()
    frozen_seats: Dict[Tuple[str, str], int] = {}
    for (tech, date_str), skill in frozen.items():
        if (tech, date_str) not in schedule_rows:
            raise ValueError(
                f"frozen training for TECH_ID {tech} on {date_str} is not a working day"
            )
        if (str(tech), date_str) in blocked:
            raise ValueError(
                f"frozen training for TECH_ID {tech} on {date_str} overlaps a flex or vacation day"
            )
        if skill not in tech_quals.get(tech, []):
            raise ValueError(
                f"frozen training for TECH_ID {tech} uses unheld skill {skill!r}"
            )
        month = date_month[date_str]
        if month not in training_months.get(skill, []):
            raise ValueError(
                f"frozen training for TECH_ID {tech} uses skill {skill!r} "
                f"in month {month}, which is not an eligible training month"
            )
        frozen_requirements.add((tech, skill, month))
        blocked.add((str(tech), date_str))
        shift = tech_shift[tech]
        frozen_seats[(date_str, shift)] = frozen_seats.get((date_str, shift), 0) + 1
        base_key = (tech_wc[tech], shift, date_str)
        base_supply[base_key] = base_supply.get(base_key, 0) - 1

    requirements: List[Tuple[object, str, int]] = []
    for tech, quals in tech_quals.items():
        for skill in quals:
            for month in training_months.get(skill, []):
                req = (tech, skill, month)
                if req not in frozen_requirements:
                    requirements.append(req)

    if demand is None:
        demand = _load_demand(demand_path)

    ctx = _Context(
        tech_days, tech_shift, tech_wc, blocked, base_supply, demand, date_month
    )

    chosen: Dict[Tuple[object, str], str] = dict(frozen)
    failures: List[Tuple[object, str, int]] = []
    report: Dict[str, dict] = {}

    # Group requirements by (shift, month): the exact, licence-fitting decomposition.
    req_by_group: Dict[Tuple[str, int], List[Tuple[object, str, int]]] = {}
    for req in requirements:
        req_by_group.setdefault((tech_shift.get(req[0], "UNKNOWN"), req[2]), []).append(
            req
        )

    for (shift, month), reqs in sorted(
        req_by_group.items(), key=lambda kv: -len(kv[1])
    ):
        backend = make_backend(solver, log_path=log_path)
        daily_cap = {
            date_str: class_size - seats
            for (date_str, seat_shift), seats in frozen_seats.items()
            if seat_shift == shift
        }
        res = _solve_group(
            backend,
            reqs,
            ctx,
            shift,
            class_size,
            under_penalty,
            over_penalty,
            time_limit,
            mip_gap,
            daily_cap=daily_cap,
        )
        chosen.update(res["chosen"])
        failures.extend(res["failures"])
        report[f"{shift}/M{month:02d}"] = {
            "solver": backend.name,
            "mode": "shift-month-frozen" if frozen else "shift-month",
            "status": res["status"],
            "objective": res["objective"],
            "seconds": res["seconds"],
            "requirements": res["requirements"],
        }

    df["_TRAIN_SKILL"] = [
        chosen.get((t, d)) for t, d in zip(df["TECH_ID"], df["_DATE"])
    ]
    result = build_result_from_marks(df, training_months, failures)
    result.report = report
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Exact (MIP) training assignment -- standalone runner (members/daniel)."
    )
    parser.add_argument(
        "--input-data-dir",
        default="data",
        help="Directory with the competition inputs (annual_desk_needs_reference.csv, Supply_sample.csv).",
    )
    parser.add_argument(
        "--output-data-dir",
        default="outputs/exact_gurobi",
        help="Directory for the generated training files (default outputs/exact_gurobi).",
    )
    parser.add_argument(
        "--regular",
        default=None,
        help="Phase I regular roster (default outputs/Supply_Regular.csv).",
    )
    parser.add_argument(
        "--standard",
        default=None,
        help="Phase I standard-training roster (default outputs/Supply_Standard_Training.csv).",
    )
    parser.add_argument(
        "--special",
        default=None,
        help="Phase I special-training roster (default outputs/Supply_Special_Training.csv).",
    )
    parser.add_argument(
        "--qual-data",
        default=None,
        help="Qualification reference (default <input-data-dir>/Supply_sample.csv).",
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--demand",
        default=None,
        help="Desk demand (default <input-data-dir>/annual_desk_needs_reference.csv).",
    )
    parser.add_argument("--solver", choices=["gurobi", "xpress"], default="gurobi")
    parser.add_argument("--class-size", type=int, default=CLASS_SIZE_DEFAULT)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument(
        "--freeze-phase1-training",
        action="store_true",
        help="Keep the Phase I training days fixed and schedule only "
        "the secondary-qualification trainings around them.",
    )
    args = parser.parse_args()

    in_dir = args.input_data_dir
    out_dir = args.output_data_dir
    regular_path = args.regular or "outputs/Supply_Regular.csv"
    standard_path = args.standard or "outputs/Supply_Standard_Training.csv"
    special_path = args.special or "outputs/Supply_Special_Training.csv"
    qual_data_path = args.qual_data or os.path.join(in_dir, "Supply_sample.csv")
    demand_path = args.demand or os.path.join(in_dir, "annual_desk_needs_reference.csv")

    from pipelines.phase2_training.qualification_mapper import (  # local import keeps CLI light
        DEFAULT_PRIORITY_LIST,
        RANDOM_SEED as _QUAL_SEED,
        build_prop_table,
        map_qualifications_to_supply,
        sprinkle_qualifications,
    )
    from pipelines.phase2_training.training_assigner import (
        load_frozen_assignments,
        write_training_outputs,
    )
    from pipelines.phase2_training.training_config import (
        load_secondary_proportion_scales,
    )

    frames = [pd.read_csv(p) for p in (regular_path, standard_path, special_path)]
    base_columns = list(frames[0].columns)
    cohort = pd.concat(frames, ignore_index=True)
    qual_data = pd.read_csv(qual_data_path)

    merged = map_qualifications_to_supply(cohort, qual_data, DEFAULT_PRIORITY_LIST)
    merged = sprinkle_qualifications(
        merged,
        build_prop_table(qual_data, DEFAULT_PRIORITY_LIST),
        DEFAULT_PRIORITY_LIST,
        seed=_QUAL_SEED,
        prop_scales=load_secondary_proportion_scales(
            args.params, DEFAULT_PRIORITY_LIST
        ),
    )
    merged = merged[base_columns]

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"training_{args.solver}.log")
    training_months = load_training_months(args.params)
    frozen = None
    if args.freeze_phase1_training:
        frozen = load_frozen_assignments(standard_path, special_path, training_months)
    result = assign_training_exact(
        merged,
        training_months,
        class_size=args.class_size,
        solver=args.solver,
        demand_path=demand_path,
        time_limit=args.time_limit,
        log_path=log_path,
        frozen=frozen,
    )
    write_training_outputs(result, out_dir)

    logging.info("\n============== EXACT TRAINING (T1) ==============")
    logging.info(f"solver       : {args.solver}")
    logging.info(f"failures     : {len(result.failures)}")
    logging.info(f"regular rows : {len(result.regular):,}")
    logging.info(f"standard     : {len(result.standard_data):,}")
    logging.info(f"intl         : {len(result.intl_data):,}")
    for key, info in result.report.items():
        logging.info(f"  [{key}] {info}")
    logging.info("================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
