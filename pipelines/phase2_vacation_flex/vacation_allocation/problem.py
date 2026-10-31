"""Index structures shared by the vacation allocator, the flex estimator, the
flex assigner and the reports.

A *bucket* is one (date, work centre, shift) combination -- the unit at which
desk demand is expressed.  Buckets come from scheduled supply, from desk
demand on schedule dates, and from pre-assigned flex, so that demand with no
regular supply can still be covered by flex.

Supply in a bucket = regular workdays - training - vacations
                     + pre-assigned flex (valid ones) + new flex.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_io import demand_lookup, norm_wc


@dataclass
class Problem:
    supply: pd.DataFrame  # cleaned regular supply; row i of row_* arrays is supply row i
    dates: list
    tech_ids: list  # sorted dispatcher ids; index = tech code
    row_tech: np.ndarray
    row_day: np.ndarray
    row_bucket: np.ndarray
    row_eligible: np.ndarray  # False if the workday is also a training day

    bucket_keys: list  # (date, work-centre label, shift)
    bucket_wc: list  # normalised work centre per bucket
    bucket_shift: list
    bucket_day: np.ndarray
    bucket_total: np.ndarray  # scheduled regular rows
    bucket_training: np.ndarray  # scheduled rows lost to training
    bucket_extra: np.ndarray  # valid pre-assigned flex rows (extra people)
    bucket_eligible: np.ndarray  # rows that may become vacation
    bucket_demand: np.ndarray
    bucket_slack: np.ndarray  # total - training + extra - demand (can be negative)
    bucket_free: np.ndarray  # vacations absorbable without creating shortage

    tech_workdays: np.ndarray
    tech_eligible_days: np.ndarray
    tech_shifts: list  # set of shifts each dispatcher works
    tech_home_wc: list  # normalised home work centre
    tech_quals: dict  # normalised work centre -> bool array over dispatchers
    tech_preflex: np.ndarray  # pre-assigned flex days per dispatcher (all rows)
    busy: np.ndarray  # [tech, day] scheduled, training, merged-schedule shift or pre-assigned flex
    training_busy: np.ndarray  # [tech, day] training day

    wc_label: dict = field(default_factory=dict)  # normalised -> original label
    preflex_valid: pd.DataFrame = field(default_factory=pd.DataFrame)
    preflex_conflicts: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_tech(self) -> int:
        return len(self.tech_ids)

    @property
    def n_day(self) -> int:
        return len(self.dates)

    @property
    def n_bucket(self) -> int:
        return len(self.bucket_keys)

    def supply_after(self, vacations_per_bucket, flex_per_bucket=None) -> np.ndarray:
        s = self.bucket_total - self.bucket_training + self.bucket_extra - vacations_per_bucket
        return s if flex_per_bucket is None else s + flex_per_bucket

    def shortage(self, vacations_per_bucket, flex_per_bucket=None) -> np.ndarray:
        """Desk demand left uncovered for a given vacation (and optional flex) count."""
        return np.maximum(0, self.bucket_demand - self.supply_after(vacations_per_bucket, flex_per_bucket))


def clean_preflex(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Pre-assigned flex rows (same layout as supply)."""
    if df is None:
        return None
    from .data_io import _clean_ids, parse_dates, require

    require(df, ["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"], "pre-assigned flex")
    df = parse_dates(df, "SHIFT_DATE", "pre-assigned flex")
    df["TECH_ID"] = _clean_ids(df["TECH_ID"])
    df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)


def build_problem(
    supply: pd.DataFrame,
    demand: pd.DataFrame,
    dates: list[pd.Timestamp],
    training: set,
    preflex: pd.DataFrame | None = None,
    blocked: set | None = None,
) -> Problem:
    """``blocked``: (TECH_ID, date) pairs on which the dispatcher already has a
    shift (work or training, e.g. Supply_Regular_Merged.csv).  They only stop
    flex days; they do not change the regular supply or vacation eligibility."""
    sup = supply.reset_index(drop=True)
    tech_ids = sorted(sup["TECH_ID"].unique())
    tcode = {t: i for i, t in enumerate(tech_ids)}
    dcode = {d: i for i, d in enumerate(dates)}
    nt, nd = len(tech_ids), len(dates)
    row_tech = sup["TECH_ID"].map(tcode).to_numpy(dtype=np.int64)
    row_day = sup["SHIFT_DATE"].map(dcode).to_numpy(dtype=np.int64)
    row_wc = sup["WORK_CENTER_ID"].map(norm_wc).to_numpy()
    row_sh = sup["SHIFT_TYPE"].to_numpy()

    wc_label: dict[str, str] = {}
    for lab in sup["WORK_CENTER_ID"].unique():
        wc_label.setdefault(norm_wc(lab), lab)

    # --- training and scheduled days
    works = np.zeros((nt, nd), dtype=bool)
    works[row_tech, row_day] = True
    train_busy = np.zeros((nt, nd), dtype=bool)
    for t, d in training or ():
        if t in tcode and d in dcode:
            train_busy[tcode[t], dcode[d]] = True
    in_training = train_busy[row_tech, row_day]
    blocked_busy = np.zeros((nt, nd), dtype=bool)
    for t, d in blocked or ():
        if t in tcode and d in dcode:
            blocked_busy[tcode[t], dcode[d]] = True

    # --- pre-assigned flex: always uses budget; adds a person only on a real off-day
    pf_valid = pd.DataFrame(columns=["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"])
    pf_conf = pd.DataFrame(columns=["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE", "REASON"])
    tech_preflex = np.zeros(nt, dtype=np.int64)
    preflex_busy = np.zeros((nt, nd), dtype=bool)
    if preflex is not None and len(preflex):
        pf = preflex[["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"]].copy()
        known = pf["TECH_ID"].isin(tcode)
        for t, n in pf.loc[known, "TECH_ID"].value_counts().items():
            tech_preflex[tcode[t]] = n
        reason = np.where(~known, "dispatcher not in supply", "").astype(object)
        ti = pf["TECH_ID"].map(tcode)
        di = pf["SHIFT_DATE"].map(dcode)
        for i in np.flatnonzero(known.to_numpy()):
            if pd.isna(di.iloc[i]):
                reason[i] = "date outside schedule"
                continue
            t, d = int(ti.iloc[i]), int(di.iloc[i])
            if works[t, d]:
                reason[i] = "already a scheduled workday"
            elif train_busy[t, d]:
                reason[i] = "training day"
            elif blocked_busy[t, d]:
                reason[i] = "shift already assigned in merged schedule"
            preflex_busy[t, d] = True
        ok = reason == ""
        pf_valid = pf.loc[ok].reset_index(drop=True)
        pf_conf = pf.loc[~ok].assign(REASON=reason[~ok]).reset_index(drop=True)
        for lab in pf["WORK_CENTER_ID"].unique():
            wc_label.setdefault(norm_wc(lab), lab)

    # --- buckets: supply keys, demand keys on schedule dates, valid pre-flex keys
    dem = demand_lookup(demand)
    for lab in demand["WORK_CENTER"].unique():
        wc_label.setdefault(norm_wc(lab), lab)
    keys = set(zip(row_day.tolist(), row_wc.tolist(), row_sh.tolist()))
    keys |= {(dcode[d], wc, sh) for (d, wc, sh), v in dem.items() if d in dcode and v > 0}
    pf_keys = list(
        zip(
            pf_valid["SHIFT_DATE"].map(dcode).astype(int).tolist(),
            pf_valid["WORK_CENTER_ID"].map(norm_wc).tolist(),
            pf_valid["SHIFT_TYPE"].tolist(),
        )
    ) if len(pf_valid) else []  # fmt: skip
    keys |= set(pf_keys)
    keys = sorted(keys)
    bcode = {k: i for i, k in enumerate(keys)}
    nb = len(keys)
    row_bucket = np.array([bcode[k] for k in zip(row_day.tolist(), row_wc.tolist(), row_sh.tolist())],
                          dtype=np.int64)  # fmt: skip
    bucket_day = np.array([k[0] for k in keys], dtype=np.int64)
    bucket_wc = [k[1] for k in keys]
    bucket_shift = [k[2] for k in keys]
    bucket_keys = [(dates[k[0]], wc_label.get(k[1], k[1]), k[2]) for k in keys]

    bucket_total = np.bincount(row_bucket, minlength=nb).astype(np.int64)
    bucket_training = np.bincount(row_bucket[in_training], minlength=nb).astype(np.int64)
    bucket_eligible = bucket_total - bucket_training
    bucket_extra = np.zeros(nb, dtype=np.int64)
    for k in pf_keys:
        bucket_extra[bcode[k]] += 1

    missing = [bucket_keys[i] for i, k in enumerate(keys)
               if bucket_total[i] > 0 and (dates[k[0]], k[1], k[2]) not in dem]  # fmt: skip
    if missing:
        raise ValueError(
            f"Missing desk demand for {len(missing)} scheduled buckets, e.g. {missing[:3]}"
        )
    bucket_demand = np.array([dem.get((dates[k[0]], k[1], k[2]), 0) for k in keys], dtype=np.int64)
    bucket_slack = bucket_total - bucket_training + bucket_extra - bucket_demand
    bucket_free = np.minimum(bucket_eligible, np.maximum(0, bucket_slack))

    # --- dispatcher attributes
    row_eligible = ~in_training
    workdays = np.bincount(row_tech, minlength=nt).astype(np.int64)
    elig_days = np.bincount(row_tech[row_eligible], minlength=nt).astype(np.int64)
    shifts = [set() for _ in tech_ids]
    for t, sh in zip(row_tech, row_sh):
        shifts[t].add(sh)
    home = pd.Series(row_wc).groupby(row_tech).agg(lambda x: x.value_counts().index[0])
    tech_home_wc = [home.get(i, "") for i in range(nt)]

    quals: dict[str, np.ndarray] = {}
    for wc in sorted(set(bucket_wc)):
        flag = np.zeros(nt, dtype=bool)
        if wc in sup.columns:  # qualification columns are named after the work centre
            q = pd.to_numeric(sup[wc], errors="coerce").fillna(0).to_numpy() > 0
            flag[np.unique(row_tech[q])] = True
        flag[np.unique(row_tech[row_wc == wc])] = True  # always qualified for own desk
        quals[wc] = flag

    busy = works | train_busy | blocked_busy | preflex_busy

    return Problem(
        supply=sup, dates=list(dates), tech_ids=tech_ids,
        row_tech=row_tech, row_day=row_day, row_bucket=row_bucket, row_eligible=row_eligible,
        bucket_keys=bucket_keys, bucket_wc=bucket_wc, bucket_shift=bucket_shift,
        bucket_day=bucket_day, bucket_total=bucket_total, bucket_training=bucket_training,
        bucket_extra=bucket_extra, bucket_eligible=bucket_eligible, bucket_demand=bucket_demand,
        bucket_slack=bucket_slack, bucket_free=bucket_free,
        tech_workdays=workdays, tech_eligible_days=elig_days, tech_shifts=shifts,
        tech_home_wc=tech_home_wc, tech_quals=quals, tech_preflex=tech_preflex,
        busy=busy, training_busy=train_busy, wc_label=wc_label,
        preflex_valid=pf_valid, preflex_conflicts=pf_conf,
    )  # fmt: skip
