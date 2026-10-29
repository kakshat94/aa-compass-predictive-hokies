from pathlib import Path
import logging
import os

# Settings live in this YAML file (rules 1-3, fairness, base, flex, file names).
DEFAULT_CONFIG = os.path.join("params.yaml")


def _settings(config_path):
    from pipelines.phase2_vacation_flex.vacation_allocation.settings import (
        load_settings,
    )

    return load_settings(config_path or DEFAULT_CONFIG)


def _training(PHASE2_DIR: Path):
    path = PHASE2_DIR / "all_training_data.csv"
    if path.is_file():
        return path
    logging.warning(
        f"training calendar not found at {path}; flex days are NOT checked against training"
    )
    return None


def _preassigned_flex(PHASE2_DIR: Path):
    if PHASE2_DIR is None:
        logging.warning(
            "PHASE2_DIR not given: no pre-assigned flex days are taken into account"
        )
        return None
    return PHASE2_DIR / "Flex_assignment_qualified.csv"


def run_vacation(
    PHASE2_DIR: Path,
    PHASE3_DIR: Path,
    ROOT: Path,
    FAIR_SECONDS: float = 0.0,
    PHASE1_DIR: Path = None,
    config_path: Path = None,
):
    """Allocate vacations.

    Inputs:
        PHASE2_DIR / Supply_Regular_qualified.csv
        PHASE2_DIR / all_training_data.csv              (used if present)
        PHASE1_DIR / Flex_assignment_qualified.csv      (if PHASE1_DIR is given)
        ROOT / inputs / annual_desk_needs_reference.csv
        ROOT / inputs / demand_volume_per_day_reference.csv
    Outputs in PHASE3_DIR:
        vacation_initial.csv, vacation_outputs_updatedFormat.csv,
        demand_imputation_audit.csv and the vacation_* reports.

    Pre-assigned flex days add coverage and use up flex budget, so pass
    PHASE1_DIR whenever that file exists.  FAIR_SECONDS is accepted but not used.
    """
    from pipelines.phase2_vacation_flex.vacation_allocation import run_from_files

    s = _settings(config_path)
    result, paths = run_from_files(
        supply_path=PHASE2_DIR / "Supply_Regular_qualified.csv",
        demand_path=ROOT / "inputs" / "annual_desk_needs_reference.csv",
        volume_path=ROOT / "inputs" / "demand_volume_per_day_reference.csv",
        output_dir=PHASE3_DIR,
        training_path=_training(PHASE2_DIR),
        cfg=s.vacation,
        preflex_path=_preassigned_flex(PHASE2_DIR),
    )
    req, cov = result.stats["requirements"], result.stats["coverage"]
    logging.info(
        f"vacation assigned      : {req['total_vacations']:,} rows -> {paths['vacation']}"
    )
    logging.info(
        f"vacation fairness      : {req['per_dispatcher_min']}-{req['per_dispatcher_max']} "
        f"days per dispatcher; daily totals {req['daily_min']}-{req['daily_max']}"
    )
    logging.info(
        f"shortage after vacation: {cov['shortage_after_vacation']:,} "
        f"(best-case flex can cover {cov['best_case_flex_coverable']:,}; "
        f"residual {cov['residual_shortage_after_best_case_flex']:,})"
    )
    return result


def run_flex(
    PHASE2_DIR: Path,
    PHASE3_DIR: Path,
    ROOT: Path,
    PHASE1_DIR: Path,
    config_path: Path = None,
):
    """Assign flex days on off-days after vacations.

    Inputs:
        PHASE2_DIR / Supply_Regular_qualified.csv
        PHASE2_DIR / all_training_data.csv              (used if present)
        PHASE1_DIR / Flex_assignment_qualified.csv
        PHASE3_DIR / vacation_initial.csv               (written by run_vacation)
        ROOT / inputs / annual_desk_needs_reference.csv
    Outputs in PHASE3_DIR / flex_maxflow:
        Flex_assignment_combined.csv, flex_days_new.csv, flex_added_supply_rows.csv,
        supply_after_vacation.csv, supply_after_vacation_and_flex.csv,
        staffing_coverage_before_after_flex.csv, flex_by_dispatcher.csv,
        preassigned_flex_conflicts.csv, flex_summary.json
    """
    from pipelines.phase2_vacation_flex.vacation_allocation.flex_pipeline import (
        run_flex_from_files,
    )

    s = _settings(config_path)
    out = PHASE3_DIR / "flex_maxflow"
    res, paths = run_flex_from_files(
        supply_path=PHASE2_DIR / "Supply_Regular_qualified.csv",
        vacation_path=PHASE3_DIR / "vacation_initial.csv",
        demand_path=ROOT / "inputs" / "annual_desk_needs_reference.csv",
        out_dir=out,
        training_path=_training(PHASE2_DIR),
        preflex_path=_preassigned_flex(PHASE2_DIR),
        cfg=s.vacation,
        fcfg=s.flex,
    )
    summary = res["summary"]
    logging.info(f"flex days added        : {summary['new_flex_days']:,} -> {out}")
    logging.info(
        f"shortage after flex    : {summary['shortage_after_flex']:,} "
        f"(was {summary['shortage_after_vacation']:,} after vacation)"
    )
    return res


def run_vacation_and_flex(
    PHASE1_DIR: Path,
    PHASE2_DIR: Path,
    PHASE3_DIR: Path,
    ROOT: Path,
    config_path: Path = None,
):
    """Both steps in order."""
    run_vacation(
        PHASE2_DIR, PHASE3_DIR, ROOT, PHASE1_DIR=PHASE1_DIR, config_path=config_path
    )
    return run_flex(PHASE2_DIR, PHASE3_DIR, ROOT, PHASE1_DIR, config_path=config_path)
