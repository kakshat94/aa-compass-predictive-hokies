"""AA COMPASS full pipeline entry point.

Run from the repo root:
    python main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bidline_check import BidlinePipelineValidator
from common.coverage_report import CoverageReporter
import pipelines.phase1.executable as phase1
import pipelines.phase2_training.executable as phase2
import pipelines.phase2_vacation_flex.executable as phase3

PARAMS_PATH  = ROOT / "pipelines" / "phase1" / "params.yaml"
PHASE1_DIR   = ROOT / "outputs" / "phase1"
PHASE2_DIR   = ROOT / "outputs" / "phase2_training"
PHASE3_DIR   = ROOT / "outputs" / "phase2_vacation_flex"

# ── solver / time settings (edit here, no CLI needed) ────────────────────────
SOLVER       = "xpress"
FAIR_SECONDS = 120.0


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _section(title: str) -> None:
    bar = "=" * 56
    print(f"\n{bar}\n  {title}\n{bar}")


if __name__ == "__main__":
    timings: dict[str, float] = {}
    pipeline_start = time.time()

    # ── Phase 1 — Bidline Creation ────────────────────────────────────────────
    _section("PHASE 1 — Bidline Creation")
    t0 = time.time()
    phase1.main(PHASE1_DIR, ROOT)
    timings["phase1"] = time.time() - t0
    print(f"  elapsed: {_hms(timings['phase1'])}")

    # ── Check 1 — Phase 1 integrity ───────────────────────────────────────────
    _section("CHECK 1 — Phase 1 bidline integrity")
    t0 = time.time()

    reg = pd.read_csv(PHASE1_DIR / "Supply_Regular.csv")
    std = pd.read_csv(PHASE1_DIR / "Supply_Standard_Training.csv")
    spc = pd.read_csv(PHASE1_DIR / "Supply_Special_Training.csv")
    flx = pd.read_csv(PHASE1_DIR / "Flex_assignment.csv")

    validator = BidlinePipelineValidator(reg, std, spc)
    all_initial_tech_ids = {str(value) for value in set(reg['TECH_ID']).union(std['TECH_ID'], spc['TECH_ID'])}
    unmapped_tech_ids = all_initial_tech_ids - set(validator.tech_id_patterns.keys())

    if unmapped_tech_ids:
        print(f"Stage 1 Structural Error: The following TECH_IDs violate the 4-3-4-4 structure: {list(unmapped_tech_ids)}")
    
    stage1_flex_results = validator.stage4_flex_checks(flx)
    if stage1_flex_results['invalid_flex_days'] or stage1_flex_results['max_count_violations']:
        print("Stage 1 (Baseline) Errors Found:")
        print(f" - Initial Flex days assigned on original WORKING days: {stage1_flex_results['invalid_flex_days']}")
        print(f" - Initial TECH_IDs exceeding 4 flex days: {stage1_flex_results['max_count_violations']}")


    report = CoverageReporter.evaluate(
        str(ROOT / "data" / "Demand.csv"),
        str(PHASE1_DIR / "Supply_all.csv"),
    )
    CoverageReporter.print_summary(report, stage="Phase 1 — Bidline Creation")
    timings["check1"] = time.time() - t0
    print(f"  elapsed: {_hms(timings['check1'])}")

    # ── Phase 2 — Qualification + Training ───────────────────────────────────
    _section("PHASE 2 — Qualification + Training")
    t0 = time.time()
    phase2.main(PHASE1_DIR, PHASE2_DIR, ROOT, PARAMS_PATH, SOLVER)
    timings["phase2"] = time.time() - t0
    print(f"  elapsed: {_hms(timings['phase2'])}")

    # ── Check 2 — Phase 2 integrity ───────────────────────────────────────────
    _section("CHECK 2 — Phase 2 training integrity")
    t0 = time.time()

    std2 = pd.read_csv(PHASE2_DIR / "standard_training_data.csv")
    spc2 = pd.read_csv(PHASE2_DIR / "intl_special_training_data.csv")

    stage2_results = validator.stage2_training_checks(std2, spc2, flx)

    stage2_errors = (
        stage2_results['invalid_training_days']
        or stage2_results['duplicate_trainings']
        or stage2_results['invalid_flex_days']
        or stage2_results['training_flex_overlap']
    )
    if stage2_errors:
        print("Stage 2 Errors Found:")
        print(f" - Invalid training days (assigned on OFF day): {stage2_results['invalid_training_days']}")
        print(f" - Duplicate trainings on same day: {stage2_results['duplicate_trainings']}")
        print(f" - Flex days on original working days: {stage2_results['invalid_flex_days']}")
        print(f" - Training on pre-assigned flex days: {stage2_results['training_flex_overlap'][:10]}")
        raise RuntimeError("Stage 2 integrity checks failed")

    report2 = CoverageReporter.evaluate(
        str(ROOT / "data" / "Demand.csv"),
        str(PHASE2_DIR / "Supply_Regular_qualified.csv"),
    )
    CoverageReporter.print_summary(report2, stage="Phase 2 — Qualified Regular Supply")
    timings["check2"] = time.time() - t0
    print(f"  elapsed: {_hms(timings['check2'])}")

    # ── Phase 3a — Vacation Scheduling ───────────────────────────────────────
    _section("PHASE 3a — Vacation Scheduling")
    phase3.run_vacation(PHASE2_DIR, PHASE3_DIR, ROOT, FAIR_SECONDS)
    timings["phase3_vacation"] = time.time() - t0
    print(f"  elapsed: {_hms(timings['phase3_vacation'])}")

    # ── Check 3a — Phase 3a integrity ───────────────────────────────────────────

    # vacation_df = pd.read_csv('Vacation_assignment.csv')
    # supply_regular_qualified = pd.read_csv('Supply_Regular_postvacations.csv')

    # # Run Stage 3 checks
    # stage3_results = validator.stage3_vacation_checks(vacation_df, supply_regular_qualified)

    # if stage3_results['invalid_vacation_source'] or stage3_results['daily_count_violations']:
    #     print("Stage 3 Errors Found:")
    #     print(f" - Vacations not mapped to a qualified working day: {stage3_results['invalid_vacation_source']}")
    #     print(f" - Dates failing 25-29 vacation count rule: {stage3_results['daily_count_violations']}")

    # ── Phase 3b — Flex Day Scheduling ───────────────────────────────────────
    _section("PHASE 3b — Flex Day Scheduling")
    qualified_flex = PHASE2_DIR / "Flex_assignment_qualified.csv"
    if not qualified_flex.is_file():
        qualified_flex = PHASE1_DIR / "Flex_assignment.csv"
    phase3.run_flex(PHASE2_DIR, PHASE3_DIR, ROOT, qualified_flex)
    timings["phase3_flex"] = time.time() - t0
    print(f"  elapsed: {_hms(timings['phase3_flex'])}")

    # ── Check 3b — Phase 3b integrity ───────────────────────────────────────────
    _section("CHECK 3b — Phase 3b flex integrity")
    final_flex = pd.read_csv(PHASE3_DIR / "Flex_assignment_final.csv")
    stage4_results = validator.stage4_flex_checks(final_flex)
    if stage4_results['invalid_flex_days'] or stage4_results['max_count_violations']:
        print("Stage 4 Errors Found:")
        print(f" - Flex days assigned on original WORKING days: {stage4_results['invalid_flex_days']}")
        print(f" - TECH_IDs exceeding 4 flex days: {stage4_results['max_count_violations']}")
        raise RuntimeError("Stage 3b integrity checks failed")
    print(f"  flex assignments checked: {len(final_flex):,}")


    # ── Timing summary ────────────────────────────────────────────────────────
    _section("TIMING SUMMARY")
    for step, secs in timings.items():
        print(f"  {step:<20} {_hms(secs)}")
    print(f"  {'TOTAL':<20} {_hms(time.time() - pipeline_start)}")
