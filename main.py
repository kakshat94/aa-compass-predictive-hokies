"""AA COMPASS full pipeline entry point.

Run from the repo root:
    python main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import logging
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bidline_check import BidlinePipelineValidator
from common.coverage_report import CoverageReporter
import pipelines.phase1.executable as phase1
import pipelines.phase2_training.executable as phase2
import pipelines.phase2_vacation_flex.executable as phase3

PARAMS_PATH = ROOT / "params.yaml"
PHASE1_DIR = ROOT / "outputs" / "phase1"
PHASE2_DIR = ROOT / "outputs" / "phase2_training"
PHASE3_DIR = ROOT / "outputs" / "phase2_vacation_flex"

# ── solver / time settings (edit here, no CLI needed) ────────────────────────
SOLVER = "xpress"
FAIR_SECONDS = 120.0

SCHEMA_MAIN = {
    "MAJOR_WORK_TYPE": "major_work_type",
    "START_TIME": "start_time",
    "EMPLOYEE_TITLE": "employee_title",
    "WORK_CENTER_ID": "work_center_id",
    "CONTRACTOR_FLAG": "contractor_flag",
    "CLUSTER": "cluster",
    "EMPLOYEE_WORK_CENTER_ID": "employee_work_center_id",
    "TECH_ID": "tech_id",
    "END_TIME": "end_time",
    "SHIFT_DATE": "shift_date",
    "SHIFT_LENGTH": "shift_length",
    "SHIFT_TYPE": "shift_type",
    "FINAL_QUALS": "Final_Quals",
    "PRIMARY_QUAL": "Primary_Qual",
    "DOMESTIC": "Domestic",
    "ATC": "ATC",
    "ROC": "ROC",
    "CHF": "CHF",
    "OC": "OC",
    "LATIN": "Latin",
    "EUROPE": "Europe",
    "PACIFIC": "Pacific",
    "QUAL_COUNT": "Qual_Count",
}

SCHEMA_DF = {
    "EMP_ID": "emp_id",
    "SKILL": "skill",
    "TRAINING_SCHEDULE_DATE": "training_schedule_date",
    "PERIOD": "period",
    "FROM_SHIFT_TYPE": "from_shift_type",
    "FROM_WORK_CENTER_ID": "from_work_center_id"
}

def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _section(title: str) -> None:
    bar = "=" * 56
    logging.info(f"\n{bar}\n  {title}\n{bar}")


def capitalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure string columns are capitalized."""

    return df


def export_outputs(file_in, file_out):
    """Export all outputs to the outputs directory."""
    export_dir = ROOT / "outputs" / "solutions"
    export_dir.mkdir(exist_ok=True)

    # Copy the final supply file
    if file_in.exists():
        export_path = export_dir / file_out
        df = pd.read_csv(file_in)

        # Enforce the schema (reorders columns and adds missing ones as NaN)
        if file_out in [
            "standard_training_scheduled_df.csv",
            "intl_special_training_scheduled_df.csv",
        ]:
            df.rename(columns=SCHEMA_DF, inplace=True)
        elif file_out=="intl_special_training_data.csv":
            schema_local = SCHEMA_MAIN.copy()
            schema_local["MONTH"] = "Month"
            df = df.reindex(columns=schema_local.keys())
            df.rename(columns=schema_local, inplace=True)
        else:
            df = df.reindex(columns=SCHEMA_MAIN.keys())
            df.rename(columns=SCHEMA_MAIN, inplace=True)
            

        # Save out the formatted file
        
        df.to_csv(export_path, index=False)
        
        logging.info(f"Exported {file_in} to {export_path}\{file_out}")
    else:
        logging.warning("Final supply file not found, skipping export.")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,  # Minimum level of messages to capture
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("predictive_hokies.log"),  # Saves to file
            logging.StreamHandler(),  # Prints to console
        ],
    )

    timings: dict[str, float] = {}
    pipeline_start = time.time()

    # ── Phase 1 — Bidline Creation ────────────────────────────────────────────
    _section("PHASE 1 — Bidline Creation")
    t0 = time.time()
    bidlines = phase1.main(PHASE1_DIR, ROOT)
    timings["phase1"] = time.time() - t0
    logging.info(f"  elapsed: {_hms(timings['phase1'])}")

    # ── Check 1 — Phase 1 integrity ───────────────────────────────────────────
    _section("CHECK 1 — Phase 1 bidline integrity")

    reg = pd.read_csv(PHASE1_DIR / "Supply_Regular.csv")
    std = pd.read_csv(PHASE1_DIR / "Supply_Standard_Training.csv")
    spc = pd.read_csv(PHASE1_DIR / "Supply_Special_Training.csv")
    flx = pd.read_csv(PHASE1_DIR / "Flex_assignment.csv")

    validator = BidlinePipelineValidator(reg, std, spc)
    all_initial_tech_ids = {
        str(value)
        for value in set(reg["TECH_ID"]).union(std["TECH_ID"], spc["TECH_ID"])
    }
    unmapped_tech_ids = all_initial_tech_ids - set(validator.tech_id_patterns.keys())

    if unmapped_tech_ids:
        logging.info(
            f"Stage 1 Structural Error: The following TECH_IDs violate the 4-3-4-4 structure: {list(unmapped_tech_ids)}"
        )

    stage1_flex_results = validator.stage4_flex_checks(flx)
    if (
        stage1_flex_results["invalid_flex_days"]
        or stage1_flex_results["max_count_violations"]
    ):
        logging.info("Stage 1 (Baseline) Errors Found:")
        logging.info(
            f" - Initial Flex days assigned on original WORKING days: {stage1_flex_results['invalid_flex_days']}"
        )
        logging.info(
            f" - Initial TECH_IDs exceeding 4 flex days: {stage1_flex_results['max_count_violations']}"
        )

    report, supply = CoverageReporter.evaluate(
        str(ROOT / "inputs" / "annual_desk_needs_reference.csv"),
        str(PHASE1_DIR / "Supply_all.csv"),
    )
    CoverageReporter.print_summary(report, stage="Phase 1 — Bidline Creation")
    timings["check1"] = time.time() - t0
    logging.info(f"  elapsed: {_hms(timings['check1'])}")

    # ── Phase 2 — Qualification + Training ───────────────────────────────────
    _section("PHASE 2 — Qualification + Training")
    phase2.main(PHASE1_DIR, PHASE2_DIR, ROOT, PARAMS_PATH, SOLVER)
    timings["phase2"] = time.time() - t0
    logging.info(f"  elapsed: {_hms(timings['phase2'])}")

    # ── Check 2 — Phase 2 integrity ───────────────────────────────────────────
    _section("CHECK 2 — Phase 2 training integrity")

    std2 = pd.read_csv(PHASE2_DIR / "standard_training_data.csv")
    spc2 = pd.read_csv(PHASE2_DIR / "intl_special_training_data.csv")

    stage2_results = validator.stage2_training_checks(std2, spc2, flx)

    stage2_errors = (
        stage2_results["invalid_training_days"]
        or stage2_results["duplicate_trainings"]
        or stage2_results["invalid_flex_days"]
        or stage2_results["training_flex_overlap"]
    )
    if stage2_errors:
        logging.info("Stage 2 Errors Found:")
        logging.info(
            f" - Invalid training days (assigned on OFF day): {stage2_results['invalid_training_days']}"
        )
        logging.info(
            f" - Duplicate trainings on same day: {stage2_results['duplicate_trainings']}"
        )
        logging.info(
            f" - Flex days on original working days: {stage2_results['invalid_flex_days']}"
        )
        logging.info(
            f" - Training on pre-assigned flex days: {stage2_results['training_flex_overlap'][:10]}"
        )
        raise RuntimeError("Stage 2 integrity checks failed")

    report2, supply2 = CoverageReporter.evaluate(
        str(ROOT / "inputs" / "annual_desk_needs_reference.csv"),
        str(PHASE2_DIR / "Supply_Regular_qualified.csv"),
    )
    CoverageReporter.print_summary(report2, stage="Phase 2 — Qualified Regular Supply")
    timings["check2"] = time.time() - t0
    logging.info(f"  elapsed: {_hms(timings['check2'])}")

    # ── Phase 3a — Vacation Scheduling ───────────────────────────────────────
    _section("PHASE 3a — Vacation Scheduling")
    phase3.run_vacation(PHASE2_DIR, PHASE3_DIR, ROOT, FAIR_SECONDS, PHASE1_DIR)
    timings["phase3_vacation"] = time.time() - t0
    logging.info(f"  elapsed: {_hms(timings['phase3_vacation'])}")

    report_v, supply_v = CoverageReporter.evaluate(
        str(ROOT / "inputs" / "annual_desk_needs_reference.csv"),
        str(PHASE3_DIR / "vacation_outputs_updatedFormat.csv"),
    )
    CoverageReporter.print_summary(report_v, stage="Phase 3 — Vacation")

    # ── Check 3a — Phase 3a integrity ───────────────────────────────────────────

    vacation_df = pd.read_csv("outputs/phase2_vacation_flex/vacation_initial.csv")
    supply_regular_qualified = pd.read_csv(
        "outputs/phase2_training/Supply_Regular_qualified.csv"
    )

    # Run Stage 3 checks
    stage3_results = validator.stage3_vacation_checks(
        vacation_df, supply_regular_qualified
    )

    if (
        stage3_results["invalid_vacation_source"]
        or stage3_results["daily_count_violations"]
    ):
        logging.info("Stage 3 Errors Found:")
        logging.info(
            f" - Vacations not mapped to a qualified working day: {stage3_results['invalid_vacation_source']}"
        )
        logging.info(
            f" - Dates failing 25-29 vacation count rule: {stage3_results['daily_count_violations']}"
        )

    # ── Phase 3b — Flex Day Scheduling ───────────────────────────────────────
    _section("PHASE 3b — Flex Day Scheduling")
    phase3.run_flex(PHASE2_DIR, PHASE3_DIR, ROOT, PHASE1_DIR)
    timings["phase3_flex"] = time.time() - t0
    logging.info(f"  elapsed: {_hms(timings['phase3_flex'])}")

    # ── Check 3b — Phase 3b integrity ───────────────────────────────────────────

    final_flex = pd.read_csv(
        "outputs/phase2_vacation_flex/flex_maxflow/flex_added_supply_rows.csv"
    )

    # Run Stage 4 checks
    stage4_results = validator.stage4_flex_checks(final_flex)

    if stage4_results["invalid_flex_days"] or stage4_results["max_count_violations"]:
        logging.info("Stage 4 Errors Found:")
        logging.info(
            f" - Flex days assigned on original WORKING days: {stage4_results['invalid_flex_days']}"
        )
        logging.info(
            f" - TECH_IDs exceeding 4 flex days: {stage4_results['max_count_violations']}"
        )
    logging.info(f"  flex assignments checked: {len(final_flex):,}")

    report_final, supply_final = CoverageReporter.evaluate(
        str(ROOT / "inputs" / "annual_desk_needs_reference.csv"),
        str(PHASE3_DIR / "flex_maxflow" / "supply_after_vacation_and_flex.csv"),
    )
    CoverageReporter.print_summary(
        report_final, stage="Final — Qualified Regular Supply"
    )

    # ── Timing summary ────────────────────────────────────────────────────────
    _section("TIMING SUMMARY")
    for step, secs in timings.items():
        logging.info(f"  {step:<20} {_hms(secs)}")
    logging.info(f"  {'TOTAL':<20} {_hms(time.time() - pipeline_start)}")

    # ── Begin Export ────────────────────────────────────────────────────────

    df_regular = pd.read_csv(PHASE1_DIR / "Supply_Regular.csv")
    df_qualified = pd.read_csv(PHASE2_DIR / "Supply_Regular_qualified.csv")

    # Define the join keys and the specific columns to pull over
    join_keys = ["TECH_ID", "SHIFT_TYPE"]
    target_columns = [
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
    ]

    # Filter the qualified dataset to only include the keys and the required columns
    df_qualified_subset = df_qualified[join_keys + target_columns]

    df_qualified_unique = df_qualified_subset.drop_duplicates(subset=join_keys)

    # Perform a left join on TECH_ID and SHIFT_TYPE
    df_regular = pd.concat([df_regular, pd.read_csv(PHASE1_DIR / "Supply_Standard_Training.csv"), pd.read_csv(PHASE1_DIR / "Supply_Special_Training.csv")])
    df_merged = pd.merge(
        df_regular[
            [
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
            ]
        ],
        df_qualified_unique,
        on=join_keys,
        how="left",
    )

    # Save the updated dataset to a new CSV file
    df_merged.to_csv(PHASE1_DIR / "Supply_Regular_Merged.csv", index=False)

    output_files = [
        (
            "outputs/phase1/Supply_Regular_Merged.csv",
            "initial_supply_(bid_lines)_data_reference.csv",
        ),
        (
            "outputs/phase2_vacation_flex/flex_maxflow/Flex_assignment_combined.csv",
            "flex_days_data.csv",
        ),
        ("outputs/phase2_vacation_flex/vacation_initial.csv", "vacation_data.csv"),
        (
            "outputs/phase2_vacation_flex/flex_maxflow/supply_after_vacation_and_flex.csv",
            "supply_data_after_vacation_shrinkage.csv",
        ),
        (
            "outputs/phase2_training/intl_special_training_data.csv",
            "intl_special_training_data.csv",
        ),
        (
            "outputs/phase2_training/intl_special_training_scheduled_df.csv",
            "intl_special_training_scheduled_df.csv",
        ),
        (
            "outputs/phase2_training/standard_training_data.csv",
            "standard_training_data.csv",
        ),
        (
            "outputs/phase2_training/standard_training_scheduled_df.csv",
            "standard_training_scheduled_df.csv",
        ),
    ]

    for file_in, file_out in output_files:
        export_outputs(Path(file_in), file_out)

    surplus_actuals = report_final["OVER_COVERAGE"].sum()
    supply_actuals = report_final["SUPPLY"].sum()
    supply_allpresent = len(
        pd.read_csv("outputs/solutions/supply_data_after_vacation_shrinkage.csv")
    )

    metrics = {
        "Runtime": str(time.time() - t0),
        "Number of bidlines": str(bidlines),
        "Total supply (not including workcenter-shift combination with no demand)": str(
            supply_actuals
        ),
        "Total Shortages": str(report_final["UNDER_COVERAGE"].sum()),
        "Total Surplus (not including workcenter-shift combination with no demand)": str(
            surplus_actuals
        ),
        "Total Vacation days assigned": str(
            len(pd.read_csv("outputs/solutions/vacation_data.csv"))
        ),
        "Total Flex days assigned": str(
            len(pd.read_csv("outputs/solutions/flex_days_data.csv"))
        ),
        "Total supply (including workcenter-shift combination with no demand)": str(
            supply_allpresent
        ),
        "Total surplus (including workcenter-shift combination with no demand)": str(
            surplus_actuals + supply_allpresent - supply_actuals
        ),
    }

    with open(ROOT / "outputs" / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
