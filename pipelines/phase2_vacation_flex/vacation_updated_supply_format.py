import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from ortools.sat.python import cp_model
import json


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df


def clean_supply(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SHIFT_DATE"] = pd.to_datetime(df["SHIFT_DATE"], errors="coerce").dt.normalize()
    df["TECH_ID"] = df["TECH_ID"].astype(str).str.strip()
    df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype(str).str.strip().str.upper()
    return df.sort_values(["SHIFT_DATE", "TECH_ID"]).reset_index(drop=True)


def clean_demand(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SHIFT_DATE"] = pd.to_datetime(df["SHIFT_DATE"], errors="coerce").dt.normalize()
    df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype(str).str.strip().str.upper()
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="raise")
    return df


def clean_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="raise")
    return df


def load_training(path: str | None) -> set:
    if not path or not Path(path).exists():
        return set()
    df = read_csv(path)
    df["SHIFT_DATE"] = pd.to_datetime(df["SHIFT_DATE"]).dt.normalize()
    return set(zip(df["TECH_ID"].astype(str).str.strip(), df["SHIFT_DATE"]))


def fill_missing_demand(
    demand: pd.DataFrame, supply: pd.DataFrame, dates: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = demand.copy()
    known = {
        (r.SHIFT_DATE, r.WORK_CENTER, r.SHIFT_TYPE): int(r.DEMAND)
        for r in demand.itertuples(index=False)
    }
    schedule_buckets = set(
        zip(supply["SHIFT_DATE"], supply["WORK_CENTER_ID"], supply["SHIFT_TYPE"])
    )
    missing = sorted(
        (d, c, sh)
        for d, c, sh in schedule_buckets
        if d in dates and (d, c, sh) not in known
    )
    audit = []
    for d, c, sh in missing:
        donor = d - pd.Timedelta(days=7)
        candidates = [dd for (dd, cc, ss) in known if cc == c and ss == sh]
        if donor not in candidates:
            donor = d + pd.Timedelta(days=7)
        if donor not in candidates and candidates:
            donor = min(candidates, key=lambda dd: abs((dd - d).days))

        val = known.get((donor, c, sh), 1)
        known[(d, c, sh)] = val
        audit.append(
            {
                "SHIFT_DATE": d,
                "WORK_CENTER": c,
                "SHIFT_TYPE": sh,
                "IMPUTED_DEMAND": val,
                "DONOR_DATE": donor,
                "METHOD": "nearest/week-offset",
            }
        )
        demand.loc[len(demand)] = {
            "SHIFT_DATE": d,
            "WORK_CENTER": c,
            "SHIFT_TYPE": sh,
            "DEMAND": val,
        }
    return demand, pd.DataFrame(audit)


def make_quota(volume: pd.DataFrame, dates: list, fixed_quota=None) -> dict:
    vol = volume.set_index("DATE")["DEMAND"].copy()
    missing = sorted(set(dates) - set(vol.index))
    for d in missing:
        prior = d - pd.Timedelta(days=7)
        following = d + pd.Timedelta(days=7)
        if prior in vol.index:
            vol.loc[d] = vol.loc[prior]
        elif following in vol.index:
            vol.loc[d] = vol.loc[following]
        else:
            vol.loc[d] = vol.mean()

    ranked = pd.DataFrame({"date": dates, "volume": [vol.loc[d] for d in dates]})
    ranked = ranked.sort_values("volume").reset_index(drop=True)
    n = len(ranked)
    if n > 0:
        ranked["group"] = [min(4, 5 * i // n) for i in range(n)]
        ranked["quota"] = 29 - ranked["group"]
    return dict(zip(ranked["date"], ranked["quota"]))


def build_and_solve(
    supply,
    demand,
    quota,
    training,
    seconds,
    workers,
    fair_seconds,
    lns_iterations,
    lns_window_days,
    lns_seconds,
    lns_max_window_days,
    lns_seed,
    zero_repair_passes,
):
    dates = sorted(quota.keys())
    allowed = supply.loc[supply["SHIFT_DATE"].isin(dates)].copy()
    excluded = allowed.apply(
        lambda r: (str(r["TECH_ID"]), r["SHIFT_DATE"]) in training, axis=1
    )
    allowed = allowed.loc[~excluded].reset_index(drop=True)

    all_techs = set(supply["TECH_ID"].astype(str).unique())
    supply_by_day = (
        allowed.groupby("SHIFT_DATE")["TECH_ID"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )
    flex_capacity_by_day = {
        d: len(all_techs - supply_by_day.get(d, set())) for d in dates
    }

    demand_map = {
        (r.SHIFT_DATE, str(r.WORK_CENTER), str(r.SHIFT_TYPE)): int(r.DEMAND)
        for r in demand.itertuples(index=False)
    }

    model = cp_model.CpModel()
    vac = {}
    by_day = defaultdict(list)
    by_emp = defaultdict(list)
    by_bucket = defaultdict(list)

    for j, r in allowed.iterrows():
        v = model.NewBoolVar(f"vac_{j}")
        vac[j] = v
        d = r.SHIFT_DATE
        emp = str(r.TECH_ID)
        b = (d, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))

        by_day[d].append(v)
        by_emp[emp].append(v)
        by_bucket[b].append(v)

    for d in dates:
        if d in by_day:
            model.Add(sum(by_day[d]) == int(quota[d]))

    shortage_vars = []
    for b, dmd in demand_map.items():
        if b not in by_bucket:
            continue
        d, wc, st = b
        sched = len(by_bucket[b])
        sv = model.NewIntVar(0, max(0, dmd), f"short_{d}_{wc}_{st}")
        model.Add(sv >= dmd - sched + sum(by_bucket[b]))

        flex_avail = flex_capacity_by_day.get(d, 0)
        weight = 100 if flex_avail < 2 else max(10, 100 - (flex_avail * 2))
        wsv = model.NewIntVar(0, max(0, dmd) * weight, f"wshort_{d}_{wc}_{st}")
        model.Add(wsv == sv * weight)
        shortage_vars.append(wsv)

    total_shortage = model.NewIntVar(0, 10000000, "total_shortage")
    model.Add(total_shortage == sum(shortage_vars))

    total_vac = sum(quota.values())
    mean_vac = total_vac // max(1, len(all_techs))
    deviations = []
    for emp in all_techs:
        dev = model.NewIntVar(0, 365, f"dev_{emp}")
        if emp in by_emp:
            model.AddAbsEquality(dev, sum(by_emp[emp]) - mean_vac)
        else:
            model.Add(dev == mean_vac)
        deviations.append(dev)

    max_dev = model.NewIntVar(0, 365, "max_dev")
    model.AddMaxEquality(max_dev, deviations)
    sum_dev = model.NewIntVar(0, 365 * len(all_techs), "sum_dev")
    model.Add(sum_dev == sum(deviations))

    model.Minimize(total_shortage * 1000 + max_dev * 100 + sum_dev)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180
    solver.parameters.num_search_workers = workers
    status = solver.Solve(model)

    stats = {
        "status": solver.StatusName(status),
        "assigned_vacations": 0,
        "dispatchers": len(all_techs),
    }

    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        chosen = [j for j, v in vac.items() if solver.Value(v)]
        result = allowed.loc[chosen].copy()
        stats["assigned_vacations"] = len(result)
    else:
        result = pd.DataFrame(columns=allowed.columns)

    return result.sort_values(["SHIFT_DATE", "TECH_ID"]), stats


def export_supply_after_vacation(supply, assignments, destination):
    key_cols = ["TECH_ID", "SHIFT_DATE"]
    vac_keys = pd.MultiIndex.from_frame(assignments[key_cols])
    supply_keys = pd.MultiIndex.from_frame(supply[key_cols])
    remaining = supply.loc[~supply_keys.isin(vac_keys)].copy()

    REFERENCE_SUPPLY_COLUMNS = [
        "INDEX",
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
        "SF_LOAD_TMS",
    ]
    ordered = [c for c in REFERENCE_SUPPLY_COLUMNS if c in remaining.columns]
    ordered += [c for c in remaining.columns if c not in ordered]
    remaining = remaining[ordered]

    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    remaining.to_csv(dst, index=False, date_format="%Y-%m-%d")
    return len(remaining), [
        c for c in REFERENCE_SUPPLY_COLUMNS if c not in supply.columns
    ]
