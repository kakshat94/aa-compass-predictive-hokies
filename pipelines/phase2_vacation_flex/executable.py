from pathlib import Path
import logging
import os
import yaml

# Settings live in this YAML file (rules 1-3, fairness, base, flex, file names).
DEFAULT_CONFIG = os.path.join("params.yaml")


# def _settings(config_path):
#     from pipelines.phase2_vacation_flex.vacation_allocation.settings import load_settings

#     return load_settings(config_path or DEFAULT_CONFIG)


# def _training(PHASE2_DIR: Path):
#     """Training days: these workdays can never become vacation."""
#     path = PHASE2_DIR / "all_training_data.csv"
#     if path.is_file():
#         return path
#     logging.warning(f"training calendar not found at {path}; training days are NOT protected from vacation")
#     return None


# def _merged_schedule(PHASE1_DIR: Path):
#     """All work and training days per dispatcher; no flex day goes on these."""
#     if PHASE1_DIR is None:
#         logging.warning("PHASE1_DIR not given: flex days are NOT checked against Supply_Regular_Merged.csv")
#         return None
#     return PHASE1_DIR / "Supply_Regular_Merged.csv"


# def _preassigned_flex(PHASE1_DIR: Path):
#     if PHASE1_DIR is None:
#         logging.warning("PHASE1_DIR not given: no pre-assigned flex days are taken into account")
#         return None
#     return PHASE1_DIR / "Flex_assignment_qualified.csv"


# def run_vacation(
#     PHASE2_DIR: Path,
#     PHASE3_DIR: Path,
#     ROOT: Path,
#     FAIR_SECONDS: float = 0.0,
#     PHASE1_DIR: Path = None,
#     config_path: Path = None,
# ):
#     """Allocate vacations.

#     Inputs:
#         PHASE2_DIR / Supply_Regular_qualified.csv       (vacations are taken from these workdays)
#         PHASE2_DIR / all_training_data.csv              (training days never become vacation)
#         PHASE1_DIR / Supply_Regular_Merged.csv          (if PHASE1_DIR is given; reporting only)
#         PHASE1_DIR / Flex_assignment_qualified.csv      (if PHASE1_DIR is given)
#         ROOT / inputs / annual_desk_needs_reference.csv
#         ROOT / inputs / demand_volume_per_day_reference.csv
#     Outputs in PHASE3_DIR:
#         vacation_initial.csv, vacation_outputs_updatedFormat.csv,
#         demand_imputation_audit.csv and the vacation_* reports.

#     Vacations minimise the desk demand they leave uncovered; flex days still to
#     be assigned are not counted on.  Pre-assigned flex days are existing supply
#     and do count as coverage, so pass PHASE1_DIR whenever that file exists.  FAIR_SECONDS is accepted but not used.
#     """
#     from pipelines.phase2_vacation_flex.vacation_allocation import run_from_files

#     s = _settings(config_path)
#     result, paths = run_from_files(
#         supply_path=PHASE2_DIR / "Supply_Regular_qualified.csv",
#         demand_path=ROOT / "inputs" / "annual_desk_needs_reference.csv",
#         volume_path=ROOT / "inputs" / "demand_volume_per_day_reference.csv",
#         output_dir=PHASE3_DIR,
#         training_path=_training(PHASE2_DIR),
#         cfg=s.vacation,
#         preflex_path=_preassigned_flex(PHASE2_DIR),
#         merged_schedule_path=_merged_schedule(PHASE1_DIR),
#     )
#     req, cov = result.stats["requirements"], result.stats["coverage"]
#     logging.info(f"vacation assigned      : {req['total_vacations']:,} rows -> {paths['vacation']}")
#     logging.info(
#         f"vacation fairness      : {req['per_dispatcher_min']}-{req['per_dispatcher_max']} "
#         f"days per dispatcher; daily totals {req['daily_min']}-{req['daily_max']}"
#     )
#     logging.info(
#         f"shortage after vacation: {cov['shortage_after_vacation']:,} "
#         f"(best-case flex can cover {cov['best_case_flex_coverable']:,}; "
#         f"residual {cov['residual_shortage_after_best_case_flex']:,})"
#     )
#     return result


# def run_flex(PHASE2_DIR: Path, PHASE3_DIR: Path, ROOT: Path, PHASE1_DIR: Path, config_path: Path = None):
#     """Assign flex days on off-days after vacations.

#     Inputs:
#         PHASE2_DIR / Supply_Regular_qualified.csv
#         PHASE1_DIR / Supply_Regular_Merged.csv          (work + training days: never flexed)
#         PHASE2_DIR / all_training_data.csv              (training workdays are not desk cover, as in run_vacation)
#         PHASE1_DIR / Flex_assignment_qualified.csv
#         PHASE3_DIR / vacation_initial.csv               (written by run_vacation)
#         ROOT / inputs / annual_desk_needs_reference.csv
#     Outputs in PHASE3_DIR / flex_maxflow:
#         Flex_assignment_combined.csv, flex_days_new.csv, flex_added_supply_rows.csv,
#         supply_after_vacation.csv, supply_after_vacation_and_flex.csv,
#         staffing_coverage_before_after_flex.csv, flex_by_dispatcher.csv,
#         preassigned_flex_conflicts.csv, flex_summary.json
#     """
#     from pipelines.phase2_vacation_flex.vacation_allocation.flex_pipeline import run_flex_from_files

#     s = _settings(config_path)
#     out = PHASE3_DIR / "flex_maxflow"
#     res, paths = run_flex_from_files(
#         supply_path=PHASE2_DIR / "Supply_Regular_qualified.csv",
#         vacation_path=PHASE3_DIR / "vacation_initial.csv",
#         demand_path=ROOT / "inputs" / "annual_desk_needs_reference.csv",
#         out_dir=out,
#         training_path=_training(PHASE2_DIR),  # same coverage count as run_vacation
#         preflex_path=_preassigned_flex(PHASE2_DIR),
#         cfg=s.vacation,
#         fcfg=s.flex,
#         merged_schedule_path=_merged_schedule(PHASE1_DIR),
#     )
#     summary = res["summary"]
#     logging.info(f"flex days added        : {summary['new_flex_days']:,} -> {out}")
#     logging.info(
#         f"shortage after flex    : {summary['shortage_after_flex']:,} "
#         f"(was {summary['shortage_after_vacation']:,} after vacation)"
#     )
#     return res


# def run_vacation_and_flex(PHASE1_DIR: Path, PHASE2_DIR: Path, PHASE3_DIR: Path, ROOT: Path, config_path: Path = None):
#     """Both steps in order."""
#     run_vacation(PHASE2_DIR, PHASE3_DIR, ROOT, PHASE1_DIR=PHASE1_DIR, config_path=config_path)
#     return run_flex(PHASE2_DIR, PHASE3_DIR, ROOT, PHASE1_DIR, config_path=config_path)

FLEX_KEYS = {"max_days_per_dispatcher", "solver", "exclude_date"}


def _settings(config_path):
    path = Path(config_path or DEFAULT_CONFIG)
    raw = yaml.safe_load(path.read_text()) or {}
    vacation = raw.get("vacation_lns") or {}
    flex = raw.get("flex_maxflow") or {}
    unknown = set(flex) - FLEX_KEYS
    if unknown:
        raise ValueError(f"unknown key(s) in 'flex_maxflow': {sorted(unknown)}; allowed: {sorted(FLEX_KEYS)}")
    return vacation, flex


def _training(PHASE2_DIR: Path):
    """Training days: these workdays can never become vacation or flex."""
    path = PHASE2_DIR / "all_training_data.csv"
    if path.is_file():
        return path
    logging.warning(f"training calendar not found at {path}; training days are NOT protected")
    return None


def _preassigned_flex(PHASE1_DIR: Path):
    if PHASE1_DIR is None:
        logging.warning("PHASE1_DIR not given: no pre-assigned flex days are taken into account")
        return None
    for name in ("Flex_assignment_qualified.csv", "Flex_assignment.csv"):
        if (PHASE1_DIR / name).is_file():
            return PHASE1_DIR / name
    logging.warning(f"no pre-assigned flex file in {PHASE1_DIR}; assuming none")
    return None


def _blocked_schedule(PHASE1_DIR: Path):
    """All work and training days per dispatcher; no flex day goes on these."""
    if PHASE1_DIR is None or not (PHASE1_DIR / "Supply_Regular_Merged.csv").is_file():
        logging.warning("Supply_Regular_Merged.csv not found: flex days are only checked against the supply and training files")
        return None
    return PHASE1_DIR / "Supply_Regular_Merged.csv"


def run_vacation(
    PHASE2_DIR: Path,
    PHASE3_DIR: Path,
    ROOT: Path,
    FAIR_SECONDS: float = 0.0,
    PHASE1_DIR: Path = None,
    config_path: Path = None,
):
    """Allocate vacations (teammate's shortage-first LNS model).

    Inputs:
        PHASE2_DIR / Supply_Regular_qualified.csv       (vacations are taken from these workdays)
        PHASE2_DIR / all_training_data.csv              (training days never become vacation)
        ROOT / inputs / annual_desk_needs_reference.csv
        ROOT / inputs / demand_volume_per_day_reference.csv
    Outputs in PHASE3_DIR:
        vacation_initial.csv, vacation_outputs_updatedFormat.csv,
        demand_imputation_audit.csv, vacation_summary.json and the other
        vacation_* / coverage_* reports.

    FAIR_SECONDS and PHASE1_DIR are accepted for compatibility and not used.
    """
    from pipelines.phase2_vacation_flex.vacation_Final_model_day_fairness_FIXED import args_from_settings, run

    vacation, _ = _settings(config_path)
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    args = args_from_settings(
        vacation,
        supply=PHASE2_DIR / "Supply_Regular_qualified.csv",
        demand=ROOT / "inputs" / "annual_desk_needs_reference.csv",
        volume=ROOT / "inputs" / "demand_volume_per_day_reference.csv",
        training=_training(PHASE2_DIR) or "none",
        output_dir=PHASE3_DIR,
    )
    summary = run(args)
    after = summary["after"]
    logging.info(f"vacation assigned      : {after['total_vacations']:,} rows -> {PHASE3_DIR / 'vacation_initial.csv'}")
    logging.info(
        f"vacation fairness      : {after['dispatcher_annual_vacation_day_min']}-"
        f"{after['dispatcher_annual_vacation_day_max']} days per dispatcher; "
        f"{after['days_below_daily_min']} dates below the daily minimum"
    )
    logging.info(
        f"shortage after vacation: {summary['final_coverage_shortage']:,} "
        f"({summary['shortage_before_vacation']:,} before vacations, {summary['shortage_added_by_vacation']:,} added)"
    )
    return summary


def run_flex(PHASE2_DIR: Path, PHASE3_DIR: Path, ROOT: Path, PHASE1_DIR: Path, config_path: Path = None):
    """Assign flex days after vacations (teammate's max-flow model).

    Inputs:
        PHASE2_DIR / Supply_Regular_qualified.csv       (supply before vacations)
        PHASE2_DIR / all_training_data.csv
        PHASE3_DIR / vacation_outputs_updatedFormat.csv (written by run_vacation)
        PHASE3_DIR / vacation_initial.csv               (written by run_vacation)
        PHASE3_DIR / demand_imputation_audit.csv        (written by run_vacation)
        PHASE1_DIR / Flex_assignment_qualified.csv      (or Flex_assignment.csv)
        PHASE1_DIR / Supply_Regular_Merged.csv          (work + training days: never flexed)
        ROOT / inputs / annual_desk_needs_reference.csv
    Outputs in PHASE3_DIR / flex_maxflow:
        Flex_assignment_combined.csv, flex_days_new.csv, flex_added_supply_rows.csv,
        supply_after_vacation_and_flex.csv, staffing_coverage_before_after_flex.csv,
        flex_by_dispatcher.csv, preassigned_flex_conflicts.csv, flex_summary.json
    """
    from pipelines.phase2_vacation_flex.flex_day_exclude_jan11 import run_from_files
    from pipelines.phase2_vacation_flex.vacation_Final_model_day_fairness_FIXED import exclusion_from_settings

    _, flex = _settings(config_path)
    summary, out = run_from_files(
        original_supply=PHASE2_DIR / "Supply_Regular_qualified.csv",
        after_vacation=PHASE3_DIR / "vacation_outputs_updatedFormat.csv",
        vacation=PHASE3_DIR / "vacation_initial.csv",
        training=_training(PHASE2_DIR),
        preassigned_flex=_preassigned_flex(PHASE1_DIR),
        demand=ROOT / "inputs" / "annual_desk_needs_reference.csv",
        demand_audit=PHASE3_DIR / "demand_imputation_audit.csv",
        output_dir=PHASE3_DIR / "flex_maxflow",
        max_flex=int(flex.get("max_days_per_dispatcher", 4)),
        solver=str(flex.get("solver", "maxflow")),
        exclude_date=exclusion_from_settings(flex.get("exclude_date")),
        blocked_schedule=_blocked_schedule(PHASE1_DIR),
    )
    logging.info(f"flex days added        : {summary['new_flex_days']:,} -> {out}")
    logging.info(
        f"shortage after flex    : {summary['shortage_after_all_flex']:,} "
        f"(was {summary['shortage_after_vacation_before_preassigned']:,} after vacation)"
    )
    return summary


def run_vacation_and_flex(PHASE1_DIR: Path, PHASE2_DIR: Path, PHASE3_DIR: Path, ROOT: Path, config_path: Path = None):
    """Both steps in order."""
    run_vacation(PHASE2_DIR, PHASE3_DIR, ROOT, PHASE1_DIR=PHASE1_DIR, config_path=config_path)
    return run_flex(PHASE2_DIR, PHASE3_DIR, ROOT, PHASE1_DIR, config_path=config_path)
