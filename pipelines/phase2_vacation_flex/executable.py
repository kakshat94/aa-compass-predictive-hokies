import json
from pathlib import Path
import pandas as pd


def run_vacation(PHASE2_DIR: Path, PHASE3_DIR: Path, ROOT: Path, FAIR_SECONDS: float):
    from pipelines.phase2_vacation_flex.vacation_cp_sat import (
        build_and_solve, clean_demand, clean_supply, clean_volume,
        fill_missing_demand, load_training, make_quota, read_csv,
    )

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    supply   = clean_supply(read_csv(str(PHASE2_DIR / "Supply_Regular_qualified.csv")))
    demand   = clean_demand(read_csv(str(ROOT / "data" / "Demand.csv")))
    volume   = clean_volume(read_csv(str(ROOT / "data" / "demand_volume_per_day_reference.csv")))
    training = load_training(str(PHASE2_DIR / "all_training_data.csv"))

    dates  = [pd.Timestamp(d) for d in sorted(supply["SHIFT_DATE"].unique())]
    quota  = make_quota(volume, dates, None)
    demand, demand_audit = fill_missing_demand(demand, supply, dates)

    assignments, stats = build_and_solve(
        supply, demand, quota, training,
        seconds=90.0, workers=8, fair_seconds=FAIR_SECONDS,
        lns_iterations=10, lns_window_days=21, lns_seconds=10.0,
        lns_max_window_days=60, lns_seed=42, zero_repair_passes=3,
    )

    vac_output = PHASE3_DIR / "vacation_initial.csv"
    demand_audit.to_csv(PHASE3_DIR / "demand_imputation_audit.csv", index=False, date_format="%Y-%m-%d")
    assignments[["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"]].to_csv(
        vac_output, index=False, date_format="%Y-%m-%d")
    if stats.get("zero_repair_history"):
        pd.DataFrame(stats["zero_repair_history"]).to_csv(
            PHASE3_DIR / "vacation_zero_repair_history.csv", index=False)
    if stats.get("lns_history"):
        pd.DataFrame(stats["lns_history"]).to_csv(
            PHASE3_DIR / "vacation_lns_history.csv", index=False)
    (PHASE3_DIR / "vacation_initial_summary.json").write_text(json.dumps(stats, indent=2))
    pd.DataFrame({"SHIFT_DATE": dates, "VACATION_TARGET": [quota[d] for d in dates]}).to_csv(
        PHASE3_DIR / "vacation_daily_targets.csv", index=False, date_format="%Y-%m-%d")

    all_ids = sorted(supply["TECH_ID"].astype(str).unique())
    assigned_counts = assignments["TECH_ID"].astype(str).value_counts()
    fairness = pd.DataFrame({"TECH_ID": all_ids})
    fairness["VACATION_DAYS"] = fairness["TECH_ID"].map(assigned_counts).fillna(0).astype(int)
    fairness.to_csv(PHASE3_DIR / "vacation_fairness_by_dispatcher.csv", index=False)
    print(f"vacation assigned      : {stats['assigned_vacations']:,} rows -> {vac_output}")


def run_flex(PHASE2_DIR: Path, PHASE3_DIR: Path, ROOT: Path):
    from pipelines.phase2_vacation_flex.flex_day_maxflow import (
        load_csv, normalize_dates, schedule_flex,
    )

    supply   = load_csv(str(PHASE2_DIR / "Supply_Regular_qualified.csv"))
    training = load_csv(str(PHASE2_DIR / "all_training_data.csv"))
    vacation = load_csv(str(PHASE3_DIR / "vacation_initial.csv"))
    demand   = normalize_dates(load_csv(str(ROOT / "data" / "Demand.csv")), "SHIFT_DATE", "demand")

    audit_path = PHASE3_DIR / "demand_imputation_audit.csv"
    if audit_path.is_file():
        audit = normalize_dates(load_csv(str(audit_path)), "SHIFT_DATE", "demand audit")
        supplement = audit[["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "IMPUTED_DEMAND"]].rename(
            columns={"IMPUTED_DEMAND": "DEMAND"})
        existing = set(zip(
            demand["SHIFT_DATE"],
            demand["WORK_CENTER"].astype(str).str.strip().str.upper(),
            demand["SHIFT_TYPE"].astype(str).str.strip().str.upper(),
        ))
        supplement = supplement.loc[[
            (r.SHIFT_DATE, r.WORK_CENTER.strip().upper(), r.SHIFT_TYPE.strip().upper()) not in existing
            for r in supplement.itertuples(index=False)
        ]]
        if len(supplement):
            demand = pd.concat([demand, supplement], ignore_index=True)

    a, c, e, s = schedule_flex(supply, demand, vacation, training)
    a.to_csv(PHASE3_DIR / "flex_days.csv",                           index=False, date_format="%Y-%m-%d")
    c.to_csv(PHASE3_DIR / "staffing_coverage_before_after_flex.csv", index=False, date_format="%Y-%m-%d")
    e.to_csv(PHASE3_DIR / "flex_by_dispatcher.csv",                  index=False)
    (PHASE3_DIR / "flex_summary.json").write_text(json.dumps(s, indent=2))
    print(f"flex days added        : {s['flex_days_added']:,} -> {PHASE3_DIR}")
