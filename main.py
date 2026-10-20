#!/usr/bin/env python3
"""Combined Phase II pipeline.

Runs, in order:
  1. exact training placement with the Phase I training days frozen;
  2. ``all_training_data.csv`` (standard + international training) in the run folder;
  3. the CP-SAT vacation module;
  4. the max-flow flex-day module.

Run from the repo root:

    .venv/bin/python main.py

or use the wrapper:

    ./run_pipeline.sh
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from members.daniel.main import run_pipeline  # noqa: E402
from members.daniel.training_assigner import (  # noqa: E402
    CLASS_SIZE_DEFAULT,
    load_training_months,
)
from members.daniel.verify_training import verify_training  # noqa: E402


def _training_keys(frame: pd.DataFrame) -> set:
    dates = pd.to_datetime(frame["SHIFT_DATE"]).dt.strftime("%Y-%m-%d")
    return set(zip(frame["TECH_ID"], dates))


def _check_frozen_preserved(
    standard_path: str,
    special_path: str,
    output_dir: Path,
) -> None:
    """Fail if the frozen Phase I training rows moved or disappeared."""
    phase1_standard = pd.read_csv(standard_path)
    phase1_special = pd.read_csv(special_path)
    out_standard = pd.read_csv(output_dir / "standard_training_data.csv")
    out_intl = pd.read_csv(output_dir / "intl_special_training_data.csv")

    missing_standard = _training_keys(phase1_standard) - _training_keys(out_standard)
    missing_special = _training_keys(phase1_special) - _training_keys(out_intl)
    if missing_standard or missing_special:
        raise RuntimeError(
            "frozen Phase I training was not preserved: "
            f"{len(missing_standard)} standard and {len(missing_special)} special rows missing"
        )
    print(
        "frozen training check  : OK "
        f"({len(phase1_standard):,} standard + {len(phase1_special):,} special rows preserved)"
    )


def _write_all_training(output_dir: Path) -> Path:
    """Concatenate standard + international training, keeping the union schema."""
    standard = pd.read_csv(output_dir / "standard_training_data.csv")
    intl = pd.read_csv(output_dir / "intl_special_training_data.csv")
    standard = standard.loc[:, ~standard.columns.str.startswith("Unnamed:")]
    intl = intl.loc[:, ~intl.columns.str.startswith("Unnamed:")]

    columns = list(standard.columns) + [c for c in intl.columns if c not in standard.columns]
    combined = pd.concat(
        [standard.reindex(columns=columns), intl.reindex(columns=columns)],
        ignore_index=True,
    )
    duplicates = int(combined.duplicated(["TECH_ID", "SHIFT_DATE"]).sum())
    if duplicates:
        raise RuntimeError(f"all_training_data.csv has {duplicates} duplicate (tech, date) rows")

    path = output_dir / "all_training_data.csv"
    combined.to_csv(path, index=False)
    print(
        f"all_training_data      : {len(combined):,} rows "
        f"({len(standard):,} standard + {len(intl):,} international) -> {path}"
    )
    return path


def _run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / script), *args]
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Combined frozen-training + vacation + flex pipeline.")
    parser.add_argument("--regular", default="outputs/Supply_Regular.csv")
    parser.add_argument("--standard", default="outputs/Supply_Standard_Training.csv")
    parser.add_argument("--special", default="outputs/Supply_Special_Training.csv")
    parser.add_argument("--qual-data", default="data/Supply_sample.csv")
    parser.add_argument("--params", default="members/akshat/params.yaml")
    parser.add_argument("--demand", default="data/Demand.csv")
    parser.add_argument("--volume", default="data/demand_volume_per_day_reference.csv")
    parser.add_argument("--output-dir", default="outputs/exact_gurobi_frozen")
    parser.add_argument("--solver", choices=["gurobi", "xpress"], default="gurobi")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--class-size", type=int, default=CLASS_SIZE_DEFAULT)
    parser.add_argument("--vacation-output", default="outputs/vacation_initial.csv")
    parser.add_argument("--fair-seconds", type=float, default=120.0)
    parser.add_argument("--flex-output-dir", default="outputs/flex_maxflow")
    parser.add_argument("--skip-vacation", action="store_true")
    parser.add_argument("--skip-flex", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n================ 1. FROZEN TRAINING ================")
    result = run_pipeline(
        regular_path=args.regular,
        standard_training_path=args.standard,
        special_training_path=args.special,
        qual_data_path=args.qual_data,
        params_path=args.params,
        volume_path=args.volume,
        demand_path=args.demand,
        method="exact-mip",
        solver=args.solver,
        time_limit=args.time_limit,
        log=True,
        class_size=args.class_size,
        output_dir=str(output_dir),
        freeze_phase1_training=True,
        run_vacation_stage=False,
        write=True,
    )

    training = result["training"]
    print(f"regular rows           : {len(training.regular):,}")
    print(f"standard training      : {len(training.standard_data):,}")
    print(f"intl special training  : {len(training.intl_data):,}")
    print(f"training failures      : {len(training.failures)}")
    for key, info in (getattr(training, "report", None) or {}).items():
        print(
            f"  [{key}] status={info['status']} obj={info['objective']} "
            f"t={info['seconds']}s reqs={info['requirements']}"
        )

    _check_frozen_preserved(args.standard, args.special, output_dir)
    problems = verify_training(
        str(output_dir / "Supply_Regular_qualified.csv"),
        str(output_dir / "standard_training_data.csv"),
        str(output_dir / "intl_special_training_data.csv"),
        str(output_dir / "standard_training_scheduled_df.csv"),
        str(output_dir / "intl_special_training_scheduled_df.csv"),
        load_training_months(args.params),
        args.class_size,
    )
    if problems:
        raise RuntimeError("training verification failed: " + "; ".join(problems))
    print("training check         : OK")

    print("\n================ 2. ALL TRAINING ===================")
    all_training = _write_all_training(output_dir)

    if args.skip_vacation:
        print("\nvacation pipeline      : skipped")
        return 0

    print("\n================ 3. VACATION =======================")
    _run(
        "vacation_cp_sat.py",
        "--supply", str(output_dir / "Supply_Regular_qualified.csv"),
        "--training", str(all_training),
        "--demand", args.demand,
        "--volume", args.volume,
        "--output", args.vacation_output,
        "--fair-seconds", str(args.fair_seconds),
    )

    if args.skip_flex:
        print("\nflex pipeline          : skipped")
        return 0

    print("\n================ 4. FLEX DAYS ======================")
    vacation_path = Path(args.vacation_output)
    _run(
        "flex_day_maxflow.py",
        "--supply", str(output_dir / "Supply_Regular_qualified.csv"),
        "--training", str(all_training),
        "--vacation", str(vacation_path),
        "--demand", args.demand,
        "--demand-audit", str(vacation_path.parent / "demand_imputation_audit.csv"),
        "--output-dir", args.flex_output_dir,
    )

    print("\nCombined pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
