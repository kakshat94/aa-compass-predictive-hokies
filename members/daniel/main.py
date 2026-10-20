"""members/daniel pipeline: secondary qualifications -> training -> vacation.

Run from the repo root:

    python3 members/daniel/main.py

Stage 1 (Phase 1-3): sprinkle primary/secondary qualifications onto the three
supply files produced by ``members/akshat/main.py`` (see ``qualification_mapper``).

Stage 2: build a random training solution honouring ``params.yaml``
``training_months`` (see ``training_assigner``).

Stage 3: scale ``demand_volume_per_day_reference.csv`` to [-1, 1], derive the
25..29/day vacation quota and assign which bid lines take vacation
(see ``vacation_assigner``), then report the two balance metrics
(see ``balance_metrics``).

Outputs (in ``--output-data-dir``, default ``outputs``):
    Supply_Regular_qualified.csv
    standard_training_data.csv / standard_training_scheduled_df.csv
    intl_special_training_data.csv / intl_special_training_scheduled_df.csv
    vacation_data.csv
    supply_data_after_vacation_shrinkage.csv
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd  # noqa: E402

from members.daniel.balance_metrics import (  # noqa: E402
    compute_balance_metrics,
    print_balance_report,
)
from members.daniel.qualification_mapper import (  # noqa: E402
    DEFAULT_PRIORITY_LIST,
    RANDOM_SEED,
    build_prop_table,
    map_qualifications_to_supply,
    sprinkle_qualifications,
    validate_quals,
)
from members.daniel.training_assigner import (  # noqa: E402
    CLASS_SIZE_DEFAULT,
    assign_training,
    load_frozen_assignments,
    load_training_months,
    write_training_outputs,
)
from members.daniel.training_exact import assign_training_exact  # noqa: E402
from members.daniel.vacation_assigner import (  # noqa: E402
    BASE_VACATION_DEFAULT,
    MAX_VARIABLE_DEFAULT,
    VACATION_SEED_DEFAULT,
    run_vacation,
)
from members.daniel.verify_training import verify_training  # noqa: E402


def _write_training_log(
    output_dir: str,
    method: str,
    solver: str,
    training,
    frozen_count: int = 0,
) -> str:
    """Write a descriptive run log (solver status, objective, runtime)."""
    path = os.path.join(output_dir, "training_run_log.txt")
    lines = [
        f"method={method}",
        f"solver={solver if method == 'exact-mip' else 'random'}",
        f"frozen_phase1_rows={frozen_count}",
        f"regular_rows={len(training.regular)}",
        f"standard_rows={len(training.standard_data)}",
        f"intl_rows={len(training.intl_data)}",
        f"failures={len(training.failures)}",
    ]
    for shift, info in (getattr(training, "report", None) or {}).items():
        lines.append(f"{shift}: {info}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def run_pipeline(
    regular_path: str,
    standard_training_path: str,
    special_training_path: str,
    qual_data_path: str = "data/Supply_sample.csv",
    params_path: str = "members/akshat/params.yaml",
    volume_path: str = "data/demand_volume_per_day_reference.csv",
    demand_path: str = "data/Demand.csv",
    method: str = "random",
    solver: str = "gurobi",
    time_limit: float = 120.0,
    log: bool = False,
    priority_list=None,
    seed: int = RANDOM_SEED,
    class_size: int = CLASS_SIZE_DEFAULT,
    vac_base: int = BASE_VACATION_DEFAULT,
    vac_variable: int = MAX_VARIABLE_DEFAULT,
    vac_seed: int = VACATION_SEED_DEFAULT,
    output_dir: str = "outputs",
    dedupe: bool = False,
    freeze_phase1_training: bool = False,
    run_vacation_stage: bool = True,
    write: bool = True,
):
    """Run qualifications (Phase 1-3), training assignment, then vacation.

    ``method`` selects the training layer: ``"random"`` (the original
    :func:`assign_training`) or ``"exact-mip"`` (the T1 model in
    :mod:`members.daniel.training_exact`, solved with ``solver``).

    ``freeze_phase1_training`` keeps the training days in the two Phase I
    training files exactly where they are and only schedules the missing
    secondary-qualification trainings around them.

    ``run_vacation_stage=False`` stops after training (and writing the training
    files) so an external vacation/flex pipeline can take over.
    """
    priority_list = priority_list or DEFAULT_PRIORITY_LIST

    inputs = [regular_path, standard_training_path, special_training_path]
    frames = [pd.read_csv(p) for p in inputs]
    base_columns = list(frames[0].columns)
    cohort = pd.concat(frames, ignore_index=True)

    qual_data = pd.read_csv(qual_data_path)

    # --- Stage 1: qualifications (Phase 1-3) ---
    merged = map_qualifications_to_supply(cohort, qual_data, priority_list, dedupe=dedupe)
    prop_table = build_prop_table(qual_data, priority_list)
    merged = sprinkle_qualifications(merged, prop_table, priority_list, seed=seed)
    merged = merged[base_columns]  # drop MAPPED_QUAL_ID / _HIST_* helper columns

    # --- Stage 2: training assignment (random or exact MIP) ---
    training_months = load_training_months(params_path)
    frozen = None
    if freeze_phase1_training:
        frozen = load_frozen_assignments(
            standard_training_path, special_training_path, training_months
        )
    if method == "exact-mip":
        solver_log = os.path.join(output_dir, f"training_{solver}.log") if log else None
        if solver_log and write:
            os.makedirs(output_dir, exist_ok=True)
        training = assign_training_exact(
            merged,
            training_months,
            class_size=class_size,
            solver=solver,
            demand_path=demand_path,
            time_limit=time_limit,
            log_path=solver_log,
            frozen=frozen,
        )
    elif method == "random":
        training = assign_training(
            merged, training_months, class_size=class_size, seed=seed, frozen=frozen
        )
    else:
        raise ValueError(f"unknown method {method!r}; choose 'random' or 'exact-mip'")

    if write:
        write_training_outputs(training, output_dir)
        if log:
            _write_training_log(
                output_dir, method, solver, training, frozen_count=len(frozen or {})
            )

    # --- Stage 3: vacation (after training files are produced) ---
    if run_vacation_stage:
        vacation = run_vacation(
            training.regular,
            volume_path=volume_path,
            base_vacation=vac_base,
            max_variable=vac_variable,
            seed=vac_seed,
            output_dir=output_dir,
            write=write,
        )
        metrics = compute_balance_metrics(
            vacation["vacation_data"], training.regular, priority_list
        )
    else:
        vacation = None
        metrics = None

    return {
        "training": training,
        "training_months": training_months,
        "vacation": vacation,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualification + training + vacation pipeline (members/daniel).")
    parser.add_argument("--input-data-dir", default="data",
                        help="Directory with the competition inputs (Demand.csv, "
                             "Supply_sample.csv, demand_volume_per_day_reference.csv).")
    parser.add_argument("--output-data-dir", default="outputs",
                        help="Directory for the files this pipeline generates "
                             "(default outputs; use e.g. outputs/exact_gurobi for a run).")
    parser.add_argument("--regular", default=None,
                        help="Phase I regular roster (default outputs/Supply_Regular.csv).")
    parser.add_argument("--standard", default=None,
                        help="Phase I standard-training roster (default outputs/Supply_Standard_Training.csv).")
    parser.add_argument("--special", default=None,
                        help="Phase I special-training roster (default outputs/Supply_Special_Training.csv).")
    parser.add_argument("--qual-data", default=None,
                        help="Qualification reference (default <input-data-dir>/Supply_sample.csv).")
    parser.add_argument("--params", default="members/akshat/params.yaml")
    parser.add_argument("--volume", default=None,
                        help="Operational-volume reference (default <input-data-dir>/demand_volume_per_day_reference.csv).")
    parser.add_argument("--demand", default=None,
                        help="Desk demand (default <input-data-dir>/Demand.csv).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--class-size", type=int, default=CLASS_SIZE_DEFAULT)
    parser.add_argument("--vac-base", type=int, default=BASE_VACATION_DEFAULT)
    parser.add_argument("--vac-variable", type=int, default=MAX_VARIABLE_DEFAULT)
    parser.add_argument("--vac-seed", type=int, default=VACATION_SEED_DEFAULT)
    parser.add_argument("--score", action="store_true", help="Score the effective supply against demand.")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile the shrinkage files.")
    parser.add_argument("--no-write", action="store_true", help="Compute only; do not write files.")
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--random", dest="method", action="store_const", const="random",
        help="Training layer: random placement (default, original behaviour).",
    )
    method_group.add_argument(
        "--exact-mip", dest="method", action="store_const", const="exact-mip",
        help="Training layer: exact T1 MIP solved with --solver.",
    )
    parser.set_defaults(method="random")
    parser.add_argument("--solver", choices=["gurobi", "xpress"], default="gurobi",
                        help="Solver for --exact-mip (default gurobi).")
    parser.add_argument("--time-limit", type=float, default=120.0,
                        help="Per-shift solve time limit in seconds for --exact-mip.")
    parser.add_argument("--log", action="store_true",
                        help="Write a run log (solver status/objective/runtime) into --output-data-dir.")
    parser.add_argument("--freeze-phase1-training", action="store_true",
                        help="Keep the training days from the Phase I training files fixed and "
                             "schedule only the secondary-qualification trainings around them.")
    args = parser.parse_args()

    in_dir = args.input_data_dir
    out_dir = args.output_data_dir
    # The Phase I roster lives in outputs/ (it is written by members/akshat/main.py).
    regular_path = args.regular or "outputs/Supply_Regular.csv"
    standard_path = args.standard or "outputs/Supply_Standard_Training.csv"
    special_path = args.special or "outputs/Supply_Special_Training.csv"
    qual_data_path = args.qual_data or os.path.join(in_dir, "Supply_sample.csv")
    volume_path = args.volume or os.path.join(in_dir, "demand_volume_per_day_reference.csv")
    demand_path = args.demand or os.path.join(in_dir, "Demand.csv")

    out = run_pipeline(
        regular_path=regular_path,
        standard_training_path=standard_path,
        special_training_path=special_path,
        qual_data_path=qual_data_path,
        params_path=args.params,
        volume_path=volume_path,
        demand_path=demand_path,
        method=args.method,
        solver=args.solver,
        time_limit=args.time_limit,
        log=args.log,
        seed=args.seed,
        class_size=args.class_size,
        vac_base=args.vac_base,
        vac_variable=args.vac_variable,
        vac_seed=args.vac_seed,
        output_dir=out_dir,
        freeze_phase1_training=args.freeze_phase1_training,
        write=not args.no_write,
    )

    training = out["training"]
    volume = out["vacation"]["scaled_volume"]
    vac = out["vacation"]["vacation_data"]

    failures = 0
    print("\n================ DANIEL PIPELINE ================")
    solver_label = args.solver if args.method == "exact-mip" else "random"
    print(f"training method      : {args.method} (solver={solver_label})")
    print(f"regular rows         : {len(training.regular):,}")
    print(f"standard training    : {len(training.standard_data):,}")
    print(f"intl special training: {len(training.intl_data):,}")
    print(f"training failures    : {len(training.failures)}")
    for shift, info in (getattr(training, "report", None) or {}).items():
        print(f"  [{shift}] status={info['status']} obj={info['objective']} "
              f"t={info['seconds']}s reqs={info['requirements']}")

    qual_problems = validate_quals(training.regular)
    print(f"qualification check  : {'OK' if not qual_problems else qual_problems}")
    if qual_problems:
        failures += 1

    if not args.no_write:
        train_problems = verify_training(
            os.path.join(out_dir, "Supply_Regular_qualified.csv"),
            os.path.join(out_dir, "standard_training_data.csv"),
            os.path.join(out_dir, "intl_special_training_data.csv"),
            os.path.join(out_dir, "standard_training_scheduled_df.csv"),
            os.path.join(out_dir, "intl_special_training_scheduled_df.csv"),
            out["training_months"],
            args.class_size,
        )
        print(f"training check       : {'OK' if not train_problems else train_problems}")
        if train_problems:
            failures += 1

    # --- volume scaling report ---
    print("\n---- vacation volume scaling (demand_volume_per_day_reference.csv) ----")
    print(volume.to_string(index=False, max_rows=8))
    print("...")
    print("VARIABLE_VACATION distribution:",
          volume["VARIABLE_VACATION"].value_counts().sort_index().to_dict())
    print("TOTAL_VACATION/day distribution:",
          volume["TOTAL_VACATION"].value_counts().sort_index().to_dict())
    corr = volume["DEMAND"].corr(volume["TOTAL_VACATION"])
    print(f"corr(volume, total_vacation) = {corr:.3f}  (negative = quieter days carry more)")

    daily = vac.groupby(pd.to_datetime(vac["SHIFT_DATE"]).dt.date).size()
    print(f"assigned vacation rows: {len(vac):,} on {daily.size} days; "
          f"per-day min/max/mean = {daily.min()}/{daily.max()}/{daily.mean():.1f}")

    # --- balance metrics ---
    print_balance_report(out["metrics"])

    # --- optional: effective coverage + shrinkage reconciliation ---
    if args.score or args.reconcile:
        if args.no_write:
            print("coverage/reconciliation requested but --no-write was set; skipped.\n")
        else:
            from members.daniel.coverage_report import run as run_coverage
            _, coverage_problems = run_coverage(
                demand_path=demand_path,
                regular_path=os.path.join(out_dir, "Supply_Regular_qualified.csv"),
                standard_training_path=os.path.join(out_dir, "standard_training_data.csv"),
                intl_training_path=os.path.join(out_dir, "intl_special_training_data.csv"),
                vacation_path=os.path.join(out_dir, "vacation_data.csv"),
                effective_path=os.path.join(out_dir, "supply_data_after_vacation_shrinkage.csv"),
                name="Effective supply (members/daniel)",
                do_score=args.score,
                do_reconcile=args.reconcile,
            )
            if coverage_problems:
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
