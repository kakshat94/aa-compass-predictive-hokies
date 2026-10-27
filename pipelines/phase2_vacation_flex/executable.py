import json
from pathlib import Path
import pandas as pd
import logging


def run_vacation(PHASE2_DIR: Path, PHASE3_DIR: Path, ROOT: Path, FAIR_SECONDS: float):
    from pipelines.phase2_vacation_flex.vacation_updated_supply_format import (
        build_and_solve,
        clean_demand,
        clean_supply,
        clean_volume,
        fill_missing_demand,
        load_training,
        make_quota,
        read_csv,
    )

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    supply = clean_supply(read_csv(str(PHASE2_DIR / "Supply_Regular_qualified.csv")))
    demand = clean_demand(
        read_csv(str(ROOT / "inputs" / "annual_desk_needs_reference.csv"))
    )
    volume = clean_volume(
        read_csv(str(ROOT / "inputs" / "demand_volume_per_day_reference.csv"))
    )
    training = load_training(str(PHASE2_DIR / "all_training_data.csv"))

    dates = [pd.Timestamp(d) for d in sorted(supply["SHIFT_DATE"].unique())]
    quota = make_quota(volume, dates, None)
    demand, demand_audit = fill_missing_demand(demand, supply, dates)

    assignments, stats = build_and_solve(
        supply,
        demand,
        quota,
        training,
        seconds=90.0,
        workers=8,
        fair_seconds=FAIR_SECONDS,
        lns_iterations=10,
        lns_window_days=21,
        lns_seconds=10.0,
        lns_max_window_days=60,
        lns_seed=42,
        zero_repair_passes=3,
    )

    vac_output = PHASE3_DIR / "vacation_initial.csv"
    demand_audit.to_csv(
        PHASE3_DIR / "demand_imputation_audit.csv", index=False, date_format="%Y-%m-%d"
    )
    assignments.to_csv(vac_output, index=False, date_format="%Y-%m-%d")

    # Export supply after vacation shrinkage
    from pipelines.phase2_vacation_flex.vacation_updated_supply_format import (
        export_supply_after_vacation,
    )

    export_supply_after_vacation(
        supply, assignments, PHASE3_DIR / "vacation_outputs_updatedFormat.csv"
    )

    if stats.get("zero_repair_history"):
        pd.DataFrame(stats["zero_repair_history"]).to_csv(
            PHASE3_DIR / "vacation_zero_repair_history.csv", index=False
        )
    if stats.get("lns_history"):
        pd.DataFrame(stats["lns_history"]).to_csv(
            PHASE3_DIR / "vacation_lns_history.csv", index=False
        )
    (PHASE3_DIR / "vacation_initial_summary.json").write_text(
        json.dumps(stats, indent=2)
    )
    pd.DataFrame(
        {"SHIFT_DATE": dates, "VACATION_TARGET": [quota[d] for d in dates]}
    ).to_csv(
        PHASE3_DIR / "vacation_daily_targets.csv", index=False, date_format="%Y-%m-%d"
    )

    all_ids = sorted(supply["TECH_ID"].astype(str).unique())
    assigned_counts = assignments["TECH_ID"].astype(str).value_counts()
    fairness = pd.DataFrame({"TECH_ID": all_ids})
    fairness["VACATION_DAYS"] = (
        fairness["TECH_ID"].map(assigned_counts).fillna(0).astype(int)
    )
    fairness.to_csv(PHASE3_DIR / "vacation_fairness_by_dispatcher.csv", index=False)
    logging.info(
        f"vacation assigned      : {stats['assigned_vacations']:,} rows -> {vac_output}"
    )


def run_flex(PHASE2_DIR: Path, PHASE3_DIR: Path, ROOT: Path, PHASE1_DIR: Path):
    from pipelines.phase2_vacation_flex.flex_day_updated_format_fixed import (
        load_csv,
        load_demand,
        run,
    )

    original = load_csv(str(PHASE2_DIR / "Supply_Regular_qualified.csv"))
    after_vac = load_csv(str(PHASE3_DIR / "vacation_outputs_updatedFormat.csv"))
    vacation = load_csv(str(PHASE3_DIR / "vacation_initial.csv"))
    training = load_csv(
        str(PHASE2_DIR / "all_training_data.csv"), optional=True, name="training"
    )
    pre_flex = load_csv(
        str(PHASE1_DIR / "Flex_assignment.csv"), optional=True, name="preassigned flex"
    )
    demand = load_demand(
        str(ROOT / "inputs" / "annual_desk_needs_reference.csv"),
        str(PHASE3_DIR / "demand_imputation_audit.csv"),
    )

    combined, new, added, final, cov, by_person, summary, after = run(
        original,
        after_vac,
        vacation,
        training,
        pre_flex,
        demand,
        max_flex=4,
        solver="maxflow",
        start=None,
        end=None,
    )

    out = PHASE3_DIR / "flex_maxflow"
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        out / "Flex_assignment_combined.csv", index=False, date_format="%Y-%m-%d"
    )
    new.to_csv(out / "flex_days_new.csv", index=False, date_format="%Y-%m-%d")
    added.to_csv(
        out / "flex_added_supply_rows.csv", index=False, date_format="%Y-%m-%d"
    )
    final.to_csv(
        out / "supply_after_vacation_and_flex.csv", index=False, date_format="%Y-%m-%d"
    )
    after.to_csv(out / "supply_after_vacation.csv", index=False, date_format="%Y-%m-%d")
    cov.to_csv(
        out / "staffing_coverage_before_after_flex.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    by_person.to_csv(out / "flex_by_dispatcher.csv", index=False)
    (out / "flex_summary.json").write_text(json.dumps(summary, indent=2))
    logging.info(f"flex days added        : {summary['new_flex_days']:,} -> {out}")
