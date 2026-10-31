"""American Airlines dispatcher vacation optimizer: shortage-first + TECH_ID annual fairness.

Priorities (shortage.allow_added: true, the default):
  1. The rules: daily range (rule 1), fairness band in days (rule 2),
     inverse-volume caps/floors (rule 3) and the base, as configured.
  2. Minimize staffing shortage: every vacation placed where a desk/shift has
     no staff above demand adds one unit of shortage, priced by
     shortage.weight; vacations go to the spare buckets first.
  3. Balance ANNUAL vacation DAY COUNTS across dispatchers (worst and average
     absolute deviation from the mean, in days).
With shortage.allow_added: false (the original model) vacations are ONLY
placed in buckets with staff above demand, so they never add shortage; the
rules are then met by giving fewer vacations.

RULE 1 (daily total) uses the same settings as the other vacation module:
min, max, tolerance_below, tolerance_above, enabled. The allowed range per date
is [min - tolerance_below, max + tolerance_above]. The maximum is hard. The
minimum has top priority (a very large penalty per missing vacation), so it is
met on every date that has enough staff above demand; dates that do not are
filled as far as possible and reported. The annual total is a decision.

Parameters added on top of the original model (all optional; see
vacation_flex_lns_config.yaml):

  * FAIRNESS BAND (--slack-days k). Deviation is measured in vacation DAYS.
    With k set, every dispatcher's annual count must lie in [L-k, L+1+k] for
    some level L chosen by the model, i.e. max - min <= 2k + 1. k = 0 means
    everyone gets L or L+1 days. Unset = no hard band (original behaviour:
    only the weighted worst/average day deviation in the objective).
    Vacation RATE (vacations / eligible workdays) is reported as a diagnostic
    only; it is not optimized.
  * BASE PER DAY (--base-per-day 27 --base-tolerance t --base-weight w).
    Soft target for the annual total: (27 - t) x dates .. (27 + t) x dates.
    Every vacation outside that range costs w / dates in the objective, the
    same unit as the 25-per-day shortfall. Soft because vacations never add
    shortage: if the spare staff cannot reach the base, the model gets as
    close as it can and reports the gap.
  * INVERSE VOLUME (--volume-policy proportional --volume-tolerance t).
    Each day gets a target clip(K / volume, 25, 29), with K set so that the
    average target equals the base (or the midpoint of 25..29 when no base is
    set). The day's total is capped at ceil(target) + t (hard) and the soft
    floor becomes max(25, floor(target) - t). --volume-policy rank-cap keeps
    the original rank-based caps; none switches the rule off.

FAIRNESS UNIT: each TECH_ID. TECH_ID is NOT an official bid-line identifier.

Outputs include coverage_shortage_audit.csv so you can determine which
staffing buckets require new capacity/flex days to reach zero shortage.

Run: python vacation_Final_model_day_fairness_FIXED.py
Requirements: pip install pandas numpy ortools
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SUPPLY = "data/Supply_Regular_qualified.csv"
DEFAULT_DEMAND = "data/annual_desk_needs_reference.csv"
DEFAULT_VOLUME = "data/demand_volume_per_day_reference.csv"
DEFAULT_TRAINING = "data/all_training_data.csv"
DEFAULT_OUTPUT_DIR = "outputs/tech_id_annual_day_fairness"
DEFAULT_EXCLUDED_DATE = "2026-01-11"  # comma-separated list; "none" disables.

SEED_SCALE = 1_000_000  # integer price scale of the min-cost-flow seed
ROW, DATE = "SRC_ROW_ID", "SRC_SHIFT_DATE"  # helper columns (itertuples drops names starting with "_")


def csv_in(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p.resolve()}")
    df = pd.read_csv(p, low_memory=False, dtype={"TECH_ID": str})
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df


def require(df, names, label):
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing {missing}. Available columns: {list(df.columns)}")


def dates_in(df, name):
    df = df.copy()
    df[name] = pd.to_datetime(df[name], format="mixed", errors="coerce").dt.normalize()
    if df[name].isna().any():
        raise ValueError(f"Invalid {name} values")
    return df


def normalize_supply(df):
    require(df, ["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"], "supply")
    df = dates_in(df, "SHIFT_DATE")
    df["TECH_ID"] = df["TECH_ID"].astype("string").str.strip()
    for col in ("WORK_CENTER_ID", "SHIFT_TYPE"):
        df[col] = df[col].astype("string").str.strip().str.upper()
    if df[["TECH_ID", "WORK_CENTER_ID", "SHIFT_TYPE"]].isna().any().any():
        raise ValueError("Supply contains missing dispatcher, work-center, or shift IDs")
    for col in ("STATUS", "ASSIGNMENT_TYPE", "DAY_TYPE"):
        if col in df:
            excluded = {"OFF", "REST", "DAY OFF", "DAY_OFF", "VACATION", "FLEX", "TRAINING"}
            df = df.loc[~df[col].astype(str).str.strip().str.upper().isin(excluded)].copy()
    if df.duplicated(["TECH_ID", "SHIFT_DATE"]).any():
        raise ValueError("Multiple scheduled supply rows for a TECH_ID and SHIFT_DATE")
    return df.sort_values(["SHIFT_DATE", "TECH_ID"]).reset_index(drop=True)


def attach_dispatcher_groups(supply):
    """One TECH_ID = one dispatcher-specific annual fairness unit."""
    supply = supply.copy()
    supply["FAIR_GROUP"] = supply["TECH_ID"].astype(str)
    count = int(supply["TECH_ID"].nunique())
    print(f"Annual fairness groups: {count} TECH_ID dispatchers (one assigned bid line per dispatcher).")
    return supply, "TECH_ID_DISPATCHER_ANNUAL_LINE"


def normalize_demand(df):
    require(df, ["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "DEMAND"], "demand")
    df = dates_in(df, "SHIFT_DATE")
    df["WORK_CENTER"] = df.WORK_CENTER.astype("string").str.strip().str.upper()
    df["SHIFT_TYPE"] = df.SHIFT_TYPE.astype("string").str.strip().str.upper()
    df["DEMAND"] = pd.to_numeric(df.DEMAND, errors="raise")
    if df.DEMAND.isna().any() or (df.DEMAND < 0).any() or not (df.DEMAND % 1 == 0).all():
        raise ValueError("Demand must be nonnegative integers")
    if df.duplicated(["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE"]).any():
        raise ValueError("Duplicate desk demand bucket")
    return df


def demand_map_with_audit(df, supply, considered_dates, missing_mode):
    """Demand per bucket, imputing buckets that have supply but no demand row.

    `supply` should be the FULL supply (excluded dates included) so that the
    audit written for the flex module covers every scheduled bucket.
    """
    mapping = {
        (r.SHIFT_DATE, str(r.WORK_CENTER), str(r.SHIFT_TYPE)): int(r.DEMAND)
        for r in df.itertuples(index=False)
        if r.SHIFT_DATE in considered_dates
    }
    donors = {(r.SHIFT_DATE, str(r.WORK_CENTER), str(r.SHIFT_TYPE)): int(r.DEMAND) for r in df.itertuples(index=False)}
    missing = sorted(set(zip(supply.SHIFT_DATE, supply.WORK_CENTER_ID, supply.SHIFT_TYPE)) - set(mapping))
    if missing and missing_mode == "error":
        raise ValueError(f"{len(missing)} missing demand buckets; examples {missing[:4]}")
    records = []
    for date, center, shift in missing:
        donor_date = next(
            (z for z in (date - pd.Timedelta(days=7), date + pd.Timedelta(days=7)) if (z, center, shift) in donors),
            None,
        )
        if donor_date is None:
            possible = [k[0] for k in donors if k[1:] == (center, shift)]
            if not possible:
                raise ValueError(f"No demand donor for {date}, {center}, {shift}")
            donor_date = min(possible, key=lambda x: abs((x - date).days))
        demand = donors[(donor_date, center, shift)]
        mapping[(date, center, shift)] = demand
        records.append(
            dict(SHIFT_DATE=date, WORK_CENTER=center, SHIFT_TYPE=shift, IMPUTED_DEMAND=demand, DONOR_DATE=donor_date)
        )
    if records:
        print(f"WARNING: {len(records)} demand buckets imputed. Coverage numbers are PROVISIONAL.")
    return mapping, pd.DataFrame(
        records, columns=["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "IMPUTED_DEMAND", "DONOR_DATE"]
    )


def training_keys(path):
    if not path or str(path).lower() == "none" or not Path(path).is_file():
        print("WARNING: No training calendar; vacation eligibility is provisional.")
        return set(), False
    train = csv_in(path)
    require(train, ["TECH_ID", "SHIFT_DATE"], "training")
    train = dates_in(train, "SHIFT_DATE")
    return set(zip(train.TECH_ID.astype("string").str.strip(), train.SHIFT_DATE)), True


def load_volume(path, dates):
    """Operational volume per date. Missing dates take the same weekday a week
    earlier, else a week later (as in the original make_quota)."""
    table = csv_in(path)
    dcol = next((c for c in ("DATE", "SHIFT_DATE", "CALENDAR_DATE", "DAY") if c in table), None)
    vcol = next(
        (
            c
            for c in ("DEMAND_VOLUME_PER_DAY", "DEMAND_VOLUME", "DAILY_VOLUME", "OPERATIONAL_VOLUME", "VOLUME", "DEMAND")
            if c in table
        ),
        None,
    )
    if dcol is None or vcol is None:
        raise ValueError(f"Cannot detect volume date/value columns: {list(table.columns)}")
    table = dates_in(table, dcol)
    if table[dcol].duplicated().any():
        raise ValueError("Duplicate dates in volume CSV")
    table[vcol] = pd.to_numeric(table[vcol], errors="raise")
    if table[vcol].isna().any() or (table[vcol] <= 0).any():
        raise ValueError("Operational volume must be positive numbers")
    vol = dict(zip(table[dcol], table[vcol]))
    volumes, imputed = {}, []
    for d in dates:
        if d in vol:
            volumes[d] = float(vol[d])
            continue
        donor = next((z for z in (d - pd.Timedelta(days=7), d + pd.Timedelta(days=7)) if z in vol), None)
        if donor is None:
            raise ValueError(f"No volume for {d.date()} nor the same weekday one week before/after")
        volumes[d] = float(vol[donor])
        imputed.append(dict(SHIFT_DATE=d, IMPUTED_VOLUME=volumes[d], DONOR_DATE=donor))
    if imputed:
        print(f"WARNING: volume imputed for {len(imputed)} date(s) from the same weekday +-7 days.")
    return volumes, pd.DataFrame(imputed, columns=["SHIFT_DATE", "IMPUTED_VOLUME", "DONOR_DATE"])


def daily_limits(dates, volume_path, args):
    """Hard cap and soft floor per date (rule 1 + rule 3).

    Returns caps, soft floors, proportional targets (or None), volumes and the
    volume imputation audit.
    """
    caps = {d: args.day_hi for d in dates}
    soft = {d: args.day_lo for d in dates}
    target = {d: None for d in dates}
    volumes, vol_audit = {}, pd.DataFrame(columns=["SHIFT_DATE", "IMPUTED_VOLUME", "DONOR_DATE"])
    if args.volume_policy == "none":
        if volume_path and Path(volume_path).is_file():
            volumes, vol_audit = load_volume(volume_path, dates)
        return caps, soft, target, volumes, vol_audit
    volumes, vol_audit = load_volume(volume_path, dates)
    if args.volume_policy == "rank-cap":
        # Original rank-based policy: *upper* bounds only.
        ordered = sorted(dates, key=lambda d: (volumes[d], d))
        n_levels = min(5, args.day_hi + 1)
        for rank, date in enumerate(ordered):
            band = min(n_levels - 1, rank * n_levels // len(ordered))
            caps[date] = min(args.day_hi, max(0, args.day_hi - band + args.volume_cap_slack))
            soft[date] = min(soft[date], caps[date])
        print("NOTE: Volume rank caps are approximate policy choices, NOT AA-specified cutoffs.")
        return caps, soft, target, volumes, vol_audit

    # proportional: target_d = clip(K / volume_d, rule-1 range)
    lo, hi = args.day_lo, args.day_hi
    level = args.base_per_day if args.base_per_day is not None else (args.daily_min + args.daily_max) / 2
    level = min(max(level, lo), hi)
    v = np.array([volumes[d] for d in dates])

    def mean_target(k):
        return float(np.clip(k / v, lo, hi).mean())

    a, b = 0.0, 2.0 * max(level, 1.0) * v.max()
    for _ in range(200):  # mean_target is nondecreasing in K
        mid = (a + b) / 2
        if mean_target(mid) < level:
            a = mid
        else:
            b = mid
    k = b
    tol = args.volume_tolerance
    for d, vd in zip(dates, v):
        t = min(max(k / vd, lo), hi)
        target[d] = t
        caps[d] = min(hi, math.ceil(t - 1e-9) + tol)
        soft[d] = min(caps[d], max(lo, math.floor(t + 1e-9) - tol))
    print(
        f"Inverse volume: daily target = clip({k:,.0f} / volume, {lo}, {hi}), mean {level:.2f}; "
        f"cap = ceil(target)+{tol}, soft floor = max({lo}, floor(target)-{tol})."
    )
    return caps, soft, target, volumes, vol_audit


class Index:
    """TECH_ID-specific annual fairness, days, and desk/shift coverage buckets."""

    def __init__(self, allowed, all_groups=None):
        self.rows = list(allowed.itertuples(index=False))
        self.n = len(self.rows)
        self.groups = defaultdict(list)
        self.days = defaultdict(list)
        self.buckets = defaultdict(list)
        self.group_of, self.day_of, self.bucket_of = [], [], []
        for j, r in enumerate(self.rows):
            group = str(r.FAIR_GROUP)
            date = r.SHIFT_DATE
            bucket = (date, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
            self.group_of.append(group)
            self.day_of.append(date)
            self.bucket_of.append(bucket)
            self.groups[group].append(j)
            self.days[date].append(j)
            self.buckets[bucket].append(j)
        # Dispatchers with no eligible workday still count in the fairness
        # statistics (with 0 vacations), so they are never silently dropped.
        for g in all_groups or ():
            self.groups.setdefault(str(g), [])
        self.eg = {g: len(rows) for g, rows in self.groups.items()}
        self.group_list = sorted(self.groups)
        self.buckets_by_day = defaultdict(list)
        for bucket in self.buckets:
            self.buckets_by_day[bucket[0]].append(bucket)


def added_shortage(state, safe_caps):
    """Shortage created by vacations: vacations beyond the staff above demand."""
    return sum(max(0, n - safe_caps.get(b, 0)) for b, n in state.vb.items() if n)


def base_outside(total, base_range):
    if base_range is None:
        return 0
    return max(0, base_range[0] - total) + max(0, total - base_range[1])


class Allocation:
    """Maintain annual dispatcher and daily coverage counts after every LNS move."""

    def __init__(self, idx, chosen):
        self.chosen = list(chosen)
        self.v = 0
        self.vg = Counter()
        self.vd = Counter()
        self.vb = Counter()
        for j, flag in enumerate(chosen):
            if flag:
                self._counts_delta(idx, j, 1)

    def _counts_delta(self, idx, j, amount):
        self.v += amount
        self.vg[idx.group_of[j]] += amount
        self.vd[idx.day_of[j]] += amount
        self.vb[idx.bucket_of[j]] += amount

    def modify(self, idx, proposal):
        for j, flag in proposal.items():
            flag = int(flag)
            if self.chosen[j] != flag:
                self._counts_delta(idx, j, flag - self.chosen[j])
                self.chosen[j] = flag

    def clone(self):
        out = object.__new__(Allocation)
        out.chosen = self.chosen.copy()
        out.v = self.v
        out.vg = self.vg.copy()
        out.vd = self.vd.copy()
        out.vb = self.vb.copy()
        return out

    def statistics(self, idx, args):
        # PRIMARY fairness: annual vacation DAYS per individual TECH_ID.
        counts = [self.vg[g] for g in idx.group_list]
        mean_days = self.v / len(idx.group_list)
        signed = {g: self.vg[g] - mean_days for g in idx.group_list}
        average_days = sum(abs(v) for v in signed.values()) / len(signed)
        worst_days = max(map(abs, signed.values()), default=0.0)
        shortfall = sum(max(0, args.soft_targets[d] - self.vd[d]) for d in idx.days)
        outside = base_outside(self.v, args.base_range)
        added = added_shortage(self, args.safe_caps)
        D = len(idx.days)
        score = (
            args.annual_worst_weight * worst_days
            + args.annual_average_weight * average_days
            + args.soft_min_weight * shortfall / D
            + args.base_weight * outside / D
            + args.shortage_weight * added / D
        )
        rates = [self.vg[g] / idx.eg[g] if idx.eg[g] else 0.0 for g in idx.group_list]
        overall_rate = self.v / idx.n
        rate_devs = [r - overall_rate for r in rates]
        return dict(
            score_days=score,
            total_vacations=self.v,
            annual_average_deviation_days=average_days,
            annual_worst_deviation_days=worst_days,
            dispatcher_annual_vacation_day_mean=mean_days,
            dispatcher_annual_vacation_day_SD=float(np.std(counts, ddof=0)),
            dispatcher_annual_vacation_day_min=min(counts, default=0),
            dispatcher_annual_vacation_day_max=max(counts, default=0),
            dispatcher_annual_vacation_day_spread=max(counts, default=0) - min(counts, default=0),
            dispatcher_annual_rate_SD_pp=100 * float(np.std(rates, ddof=0)),
            annual_average_rate_deviation_pp=100 * sum(map(abs, rate_devs)) / len(rates),
            annual_worst_rate_deviation_pp=100 * max(map(abs, rate_devs), default=0),
            annual_signed=signed,
            soft_floor_deficit=shortfall,
            days_below_target=sum(self.vd[d] < args.soft_targets[d] for d in idx.days),
            days_below_daily_min=sum(self.vd[d] < args.day_lo for d in idx.days),
            base_total_range=list(args.base_range) if args.base_range else None,
            vacations_outside_base_range=outside,
            added_shortage=added,
        )


def flow_seed(idx, safe_cap, bucket_cap, day_cap, args):
    """Coverage-safe initial solution from an exact min-cost flow.

    Network: source -> dispatcher -> (one arc per eligible workday) -> bucket
    (capacity = staffing above demand) -> date (capacity = daily cap) -> total
    -> sink, plus a bypass so vacations are optional.  A bucket has a free
    arc (staff above demand) and, if added shortage is allowed, an arc priced
    at shortage.weight per vacation for the rest.  Prices, highest first:
      * band (if set): each dispatcher gets L - k .. L + 1 + k days, the first
        L - k forced;
      * daily floor and base: reward per vacation that fills them;
      * added shortage: cost per vacation beyond the staff above demand;
      * fairness: a dispatcher's u-th day costs 2(u - C - 1), i.e. a squared
        penalty around C + 0.5 days, so counts stay close together.
    The centre C is scanned (with the band anchored at L = C, or as near as
    the band allows) and the solution with the best real objective
    (Allocation.statistics) is kept.
    """
    from ortools.graph.python import min_cost_flow

    G, D = len(idx.group_list), len(idx.days)
    k = args.slack_days
    reward_floor = max(1, round(args.soft_min_weight * SEED_SCALE / D))
    reward_base = max(1, round(args.base_weight * SEED_SCALE / D)) if args.base_range else 0
    fair_unit = max(1, round(max(args.annual_average_weight, 1e-3) * SEED_SCALE / G / 2))
    shortage_price = max(1, round(args.shortage_weight * SEED_SCALE / D))
    forced_price = 10**13

    # Candidate rows: only buckets that have staff above demand.
    gpos = {g: i for i, g in enumerate(idx.group_list)}
    rows = [j for j in range(idx.n) if bucket_cap[idx.bucket_of[j]] > 0]
    cap_g = Counter(idx.group_of[j] for j in rows)
    buckets = sorted({idx.bucket_of[j] for j in rows})
    bpos = {b: i for i, b in enumerate(buckets)}
    days = sorted(idx.days)
    dpos = {d: i for i, d in enumerate(days)}
    SRC, SNK, TOT = 0, 1, 2
    t0 = 3
    b0 = t0 + G
    d0 = b0 + len(buckets)
    row_tail = np.array([t0 + gpos[idx.group_of[j]] for j in rows], dtype=np.int64)
    row_head = np.array([b0 + bpos[idx.bucket_of[j]] for j in rows], dtype=np.int64)
    bd_tail = np.array([b0 + i for i in range(len(buckets))], dtype=np.int64)
    bd_head = np.array([d0 + dpos[b[0]] for b in buckets], dtype=np.int64)
    bd_cap = np.array([min(safe_cap[b], bucket_cap[b]) for b in buckets], dtype=np.int64)
    bx_cap = np.array([max(0, bucket_cap[b] - safe_cap[b]) for b in buckets], dtype=np.int64)
    max_total = sum(min(day_cap[d], sum(bucket_cap[b] for b in idx.buckets_by_day[d])) for d in days)
    big = max_total + 1

    def solve(C, L):
        """C = fairness centre; L = band anchor (None: no band)."""
        lo = {g: max(0, L - k) if L is not None else 0 for g in idx.group_list}
        hi = {g: min(cap_g[g], L + 1 + k) if L is not None else cap_g[g] for g in idx.group_list}
        if any(lo[g] > cap_g[g] for g in idx.group_list):
            return None
        tails, heads, caps, costs = [], [], [], []

        def arc(t, h, c, p):
            if c > 0:
                tails.append(t)
                heads.append(h)
                caps.append(c)
                costs.append(p)

        for g in idx.group_list:
            node = t0 + gpos[g]
            arc(SRC, node, lo[g], -forced_price)
            for u in range(lo[g] + 1, hi[g] + 1):
                arc(SRC, node, 1, fair_unit * 2 * (u - C - 1))
        for d in days:
            floor = min(args.soft_targets[d], day_cap[d])
            arc(d0 + dpos[d], TOT, floor, -reward_floor)
            arc(d0 + dpos[d], TOT, day_cap[d] - floor, 0)
        if args.base_range:
            b_lo, b_hi = args.base_range
            arc(TOT, SNK, b_lo, -reward_base)
            arc(TOT, SNK, b_hi - b_lo, 0)
            arc(TOT, SNK, big, reward_base)
        else:
            arc(TOT, SNK, big, 0)
        total_hi = sum(hi.values())
        arc(SRC, SNK, total_hi, 0)
        f = min_cost_flow.SimpleMinCostFlow()
        f.add_arcs_with_capacity_and_unit_cost(
            np.array(tails, dtype=np.int64), np.array(heads, dtype=np.int64),
            np.array(caps, dtype=np.int64), np.array(costs, dtype=np.int64),
        )  # fmt: skip
        first_row = f.add_arcs_with_capacity_and_unit_cost(
            row_tail, row_head, np.ones(len(rows), dtype=np.int64), np.zeros(len(rows), dtype=np.int64)
        )[0] if rows else 0  # fmt: skip
        f.add_arcs_with_capacity_and_unit_cost(bd_tail, bd_head, bd_cap, np.zeros(len(buckets), dtype=np.int64))
        if bx_cap.any():
            f.add_arcs_with_capacity_and_unit_cost(
                bd_tail, bd_head, bx_cap, np.full(len(buckets), shortage_price, dtype=np.int64)
            )
        supplies = np.zeros(d0 + len(days), dtype=np.int64)
        supplies[SRC], supplies[SNK] = total_hi, -total_hi
        f.set_nodes_supplies(np.arange(len(supplies), dtype=np.int64), supplies)
        if f.solve() != f.OPTIMAL:
            raise RuntimeError("seed min-cost flow failed")
        chosen = [0] * idx.n
        flows = f.flows(np.arange(first_row, first_row + len(rows), dtype=np.int64)) if rows else []
        for j, fl in zip(rows, flows):
            if fl:
                chosen[j] = 1
        state = Allocation(idx, chosen)
        if any(state.vg[g] < lo[g] for g in idx.group_list):
            return None  # band infeasible at this level
        return state

    def spread(state):
        counts = [state.vg[g] for g in idx.group_list]
        return max(counts) - min(counts)

    anchor_max = None
    if k is not None:
        # Largest band anchor that is feasible (feasibility falls as L grows).
        lo_l, hi_l = 0, min(cap_g[g] for g in idx.group_list) + k
        while lo_l < hi_l:
            mid = (lo_l + hi_l + 1) // 2
            if solve(mid, mid) is not None:
                lo_l = mid
            else:
                hi_l = mid - 1
        anchor_max = lo_l

    best, best_key, scan = None, None, []
    flat, last_v = 0, None
    for C in range(0, max(cap_g.values(), default=0) + 1):
        free = solve(C, None)
        state, anchor = free, None
        if k is not None and spread(free) > 2 * k + 1:
            anchor = min(C, anchor_max)
            state = solve(C, anchor)
            if state is None:
                continue
        # Stop once the total has stopped growing for 3 centres (and, with a
        # band, the anchor has reached its highest feasible level).
        if last_v is not None and state.v <= last_v and (k is None or C > anchor_max):
            flat += 1
            if flat >= 3:
                break
        else:
            flat = 0
        last_v = state.v
        st = state.statistics(idx, args)
        key = (round(st["score_days"], 9), -state.v)
        scan.append(dict(FAIRNESS_CENTER=C, BAND_ANCHOR_L=anchor, VACATIONS=state.v, SCORE_DAYS=st["score_days"],
                         MIN_DAYS=st["dispatcher_annual_vacation_day_min"],
                         MAX_DAYS=st["dispatcher_annual_vacation_day_max"],
                         SOFT_DAILY_SHORTFALL=st["soft_floor_deficit"]))  # fmt: skip
        if best_key is None or key < best_key:
            best, best_key = state, key
    if best is None:
        raise RuntimeError("no feasible seed (this should not happen: L = 0 is always feasible)")
    return best, pd.DataFrame(scan)


def shortage(idx, state, req):
    available = {b: len(js) for b, js in idx.buckets.items()}
    all_b = set(req) | set(available)
    return sum(max(0, req.get(b, 0) - available.get(b, 0) + state.vb[b]) for b in all_b)


def neighborhood_buckets(idx, state, info, bucket_caps, rng, how_many, mode):
    """TECH_ID annual-vacation-day targeted neighborhoods.

    Swap and count-changing moves alternate. In count mode the number of
    vacations in the selected buckets is NOT fixed, so the annual total can
    change.
    """
    annual = info["annual_signed"]
    ranked = []
    for bucket, js in idx.buckets.items():
        chosen = [j for j in js if state.chosen[j]]
        free = [j for j in js if not state.chosen[j]]
        remaining_cap = bucket_caps[bucket] - len(chosen)
        if mode == "swap":
            if not (chosen and free and len({idx.group_of[j] for j in js}) > 1):
                continue
            potential = max(
                0.0, max(annual[idx.group_of[j]] for j in chosen) - min(annual[idx.group_of[j]] for j in free)
            )
        else:
            if not chosen and not (free and remaining_cap > 0):
                continue
            drop_gain = max([annual[idx.group_of[j]] for j in chosen] + [0.0])
            add_gain = -min([annual[idx.group_of[j]] for j in free] + [0.0]) if remaining_cap > 0 else 0.0
            potential = max(0.0, drop_gain, add_gain)
        ranked.append((potential + 0.002 * rng.random(), bucket))
    if not ranked:
        return []
    ranked.sort(reverse=True)
    shortlist = [b for _, b in ranked[: min(len(ranked), 200)]]
    first = shortlist[int(rng.integers(0, min(15, len(shortlist))))]
    picked = [first]
    if how_many > 1:
        other_dates = [b for b in shortlist if b[0] != first[0]]
        if other_dates:
            picked.append(other_dates[int(rng.integers(0, min(30, len(other_dates))))])
    while len(picked) < how_many:
        remain = [b for b in shortlist if b not in picked]
        if not remain:
            break
        picked.append(remain[int(rng.integers(0, min(60, len(remain))))])
    return picked


def solve_local(idx, state, buckets, bucket_caps, day_caps, args, mode, iteration, incumbent):
    """Solve a small CP-SAT annual-fairness neighborhood."""
    from ortools.sat.python import cp_model

    selected = sorted({j for bucket in buckets for j in idx.buckets[bucket]})
    if not selected:
        return "SKIPPED", {}, 0.0
    model = cp_model.CpModel()
    x = {j: model.NewBoolVar(f"v_{j}") for j in selected}
    for j in selected:
        model.AddHint(x[j], int(state.chosen[j]))
    for bucket in buckets:
        assigned = sum(x[j] for j in idx.buckets[bucket])
        if mode == "swap":
            model.Add(assigned == state.vb[bucket])
        else:
            model.Add(assigned <= bucket_caps[bucket])

    by_day, by_group = defaultdict(list), defaultdict(list)
    for j in selected:
        by_day[idx.day_of[j]].append(j)
        by_group[idx.group_of[j]].append(j)

    def updated(old, js):
        return old - sum(state.chosen[j] for j in js) + sum(x[j] for j in js)

    for day, js in by_day.items():
        model.Add(updated(state.vd[day], js) <= day_caps[day])

    total_expr = state.v if mode == "swap" else updated(state.v, selected)

    # delta_g = G * V_g - total, so z_g / G = |V_g - mean| in vacation days.
    G = len(idx.group_list)
    bound = G * len(idx.days)
    deviations, group_totals = [], []
    for g in idx.group_list:
        group_total = updated(state.vg[g], by_group[g]) if g in by_group else state.vg[g]
        group_totals.append(group_total)
        delta = G * group_total - total_expr
        z = model.NewIntVar(0, bound, f"dev_scaled_days_{len(deviations)}")
        model.Add(z >= delta)
        model.Add(z >= -delta)
        deviations.append(z)

    maximum = model.NewIntVar(0, bound, "worst_annual_deviation_scaled_days")
    model.AddMaxEquality(maximum, deviations)
    allowed_max = math.ceil(incumbent["annual_worst_deviation_days"] * G + 1e-7)
    model.Add(maximum <= allowed_max)

    # Hard fairness band in DAYS: max count - min count <= 2k + 1.
    if args.max_spread is not None:
        top = model.NewIntVar(0, len(idx.days), "max_dispatcher_days")
        low = model.NewIntVar(0, len(idx.days), "min_dispatcher_days")
        for t in group_totals:
            model.Add(top >= t)
            model.Add(low <= t)
        model.Add(top - low <= args.max_spread)

    k_worst = round(args.annual_worst_weight * 1000 * G)
    k_average = round(args.annual_average_weight * 1000)
    k_floor = round(args.soft_min_weight * 1000 * G * G / len(idx.days))
    k_base = round(args.base_weight * 1000 * G * G / len(idx.days))
    objective = k_worst * maximum + k_average * sum(deviations)
    for day, js in by_day.items():
        target = args.soft_targets[day]
        missing = model.NewIntVar(0, max(target, 0), f"daily_shortfall_{day:%Y%m%d}")
        model.Add(missing >= target - updated(state.vd[day], js))
        objective += k_floor * missing
    if mode != "swap" and args.allow_added_shortage:
        k_short = round(args.shortage_weight * 1000 * G * G / len(idx.days))
        for bucket in buckets:
            spare = args.safe_caps.get(bucket, 0)
            extra = model.NewIntVar(0, len(idx.buckets[bucket]), "")
            model.Add(extra >= sum(x[j] for j in idx.buckets[bucket]) - spare)
            objective += k_short * extra
    if args.base_range and mode != "swap":
        below = model.NewIntVar(0, args.base_range[0], "below_base")
        above = model.NewIntVar(0, idx.n, "above_base")
        model.Add(below >= args.base_range[0] - total_expr)
        model.Add(above >= total_expr - args.base_range[1])
        objective += k_base * (below + above)

    model.Minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.lns_seconds
    solver.parameters.num_search_workers = args.workers
    solver.parameters.random_seed = args.seed + iteration
    result = solver.Solve(model)
    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.StatusName(result), {}, solver.WallTime()
    return solver.StatusName(result), {j: int(solver.Value(var)) for j, var in x.items()}, solver.WallTime()


def spread_ok(idx, state, args):
    if args.max_spread is None:
        return True
    counts = [state.vg[g] for g in idx.group_list]
    return max(counts) - min(counts) <= args.max_spread


def optimize(idx, state, bucket_caps, day_caps, baseline_shortage, demand_req, args):
    rng = np.random.default_rng(args.seed)
    history = []
    current = state.statistics(idx, args)
    bucket_n = args.lns_buckets
    stagnant = 0
    for iteration in range(1, args.lns_iterations + 1):
        mode = "count" if iteration % 2 == 0 else "swap"
        k = min(5, bucket_n) if mode == "count" else bucket_n
        buckets = neighborhood_buckets(idx, state, current, bucket_caps, rng, k, mode)
        if not buckets:
            print(f"LNS {iteration}/{args.lns_iterations}: no eligible {mode} neighborhood")
            continue
        status, proposal, elapsed = solve_local(idx, state, buckets, bucket_caps, day_caps, args, mode, iteration, current)
        accepted = False
        if proposal:
            candidate = state.clone()
            candidate.modify(idx, proposal)
            if (
                all(candidate.vb[b] <= bucket_caps[b] for b in buckets)
                and all(candidate.vd[d] <= day_caps[d] for d in {b[0] for b in buckets})
                and (args.allow_added_shortage or shortage(idx, candidate, demand_req) == baseline_shortage)
                and spread_ok(idx, candidate, args)
            ):
                other = candidate.statistics(idx, args)
                if (
                    other["score_days"] < current["score_days"] - 1e-9
                    and other["annual_worst_deviation_days"] <= current["annual_worst_deviation_days"] + 1e-9
                ):
                    state, current = candidate, other
                    accepted, stagnant = True, 0
        if not accepted:
            stagnant += 1
        if stagnant >= 5:
            bucket_n = min(args.lns_max_buckets, bucket_n + 2)
            stagnant = 0
        elif accepted:
            bucket_n = max(2, bucket_n - 1)
        history.append(
            dict(
                ITERATION=iteration, MODE=mode, STATUS=status, ACCEPTED=accepted, BUCKETS=len(buckets),
                SECONDS=round(elapsed, 3), VACATIONS=state.v, SCORE_DAYS=round(current["score_days"], 6),
                ANNUAL_AVERAGE_DEVIATION_DAYS=round(current["annual_average_deviation_days"], 5),
                WORST_ANNUAL_DEVIATION_DAYS=round(current["annual_worst_deviation_days"], 5),
                SOFT_DAILY_SHORTFALL=current["soft_floor_deficit"],
            )
        )  # fmt: skip
        print(
            f"LNS {iteration:2d}/{args.lns_iterations} [{mode}, {len(buckets)} buckets]: "
            f"{status}, accepted={accepted}, annual_avg={current['annual_average_deviation_days']:.2f} days, "
            f"annual_worst={current['annual_worst_deviation_days']:.2f} days, "
            f"vacations={state.v}, shortfall={current['soft_floor_deficit']}",
            flush=True,
        )
    return state, history


STAT_KEYS = (
    "score_days", "total_vacations", "annual_average_deviation_days", "annual_worst_deviation_days",
    "dispatcher_annual_vacation_day_mean", "dispatcher_annual_vacation_day_SD",
    "dispatcher_annual_vacation_day_min", "dispatcher_annual_vacation_day_max",
    "dispatcher_annual_vacation_day_spread", "dispatcher_annual_rate_SD_pp",
    "annual_average_rate_deviation_pp", "annual_worst_rate_deviation_pp", "soft_floor_deficit",
    "days_below_target", "days_below_daily_min", "vacations_outside_base_range", "added_shortage",
)  # fmt: skip


def _label(series, labels):
    return series.map(lambda x: labels.get(x, x))


def export(ctx, state, outdir, before, args):
    """Write all outputs. Vacation and remaining-supply rows keep the input
    file's own spelling of every column (SHIFT_DATE written as YYYY-MM-DD), and
    the remaining supply includes the excluded dates untouched, so the flex
    module's consistency checks pass."""
    idx, req, raw = ctx["idx"], ctx["req"], ctx["raw"]
    bucket_caps, day_caps, baseline_shortage = ctx["limit_caps"], ctx["day_caps"], ctx["shortage_min"]
    safe_caps = ctx["safe_caps"]
    outdir.mkdir(parents=True, exist_ok=True)
    final = state.statistics(idx, args)
    assert shortage(idx, state, req) == baseline_shortage + final["added_shortage"]
    assert args.allow_added_shortage or final["added_shortage"] == 0
    assert all(state.vd[d] <= day_caps[d] for d in idx.days)
    assert all(state.vb[b] <= bucket_caps[b] for b in idx.buckets)
    assert spread_ok(idx, state, args), "fairness band violated"
    chosen_rows = [int(getattr(idx.rows[j], ROW)) for j, a in enumerate(state.chosen) if a]
    for j, a in enumerate(state.chosen):
        if a:
            assert (str(idx.rows[j].TECH_ID), idx.rows[j].SHIFT_DATE) not in ctx["train"]
    is_vac = raw[ROW].isin(chosen_rows)
    cols = [c for c in raw.columns if c not in (ROW, DATE)]
    vacations = raw.loc[is_vac, ["TECH_ID", DATE, "WORK_CENTER_ID", "SHIFT_TYPE"]].rename(columns={DATE: "SHIFT_DATE"})
    vacations = vacations.sort_values(["SHIFT_DATE", "TECH_ID"])
    vacations.to_csv(outdir / "vacation_initial.csv", index=False, date_format="%Y-%m-%d")
    remaining = raw.loc[~is_vac].copy()
    remaining["SHIFT_DATE"] = remaining[DATE]
    remaining = remaining[cols]
    assert len(remaining) + len(vacations) == len(raw)
    remaining.to_csv(outdir / "vacation_outputs_updatedFormat.csv", index=False, date_format="%Y-%m-%d")
    imputed = ctx["imputed"].copy()
    imputed["WORK_CENTER"] = _label(imputed["WORK_CENTER"], ctx["wc_labels"])
    imputed["SHIFT_TYPE"] = _label(imputed["SHIFT_TYPE"], ctx["shift_labels"])
    imputed.to_csv(outdir / "demand_imputation_audit.csv", index=False, date_format="%Y-%m-%d")
    ctx["volume_audit"].to_csv(outdir / "volume_imputation_audit.csv", index=False, date_format="%Y-%m-%d")
    ctx["seed_scan"].to_csv(outdir / "vacation_seed_level_scan.csv", index=False)

    home = ctx["home_wc"]
    staff = []
    overall_rate = state.v / idx.n
    for tech in idx.group_list:
        eligible = idx.eg[tech]
        allocated = state.vg[tech]
        day_delta = final["annual_signed"][tech]
        rate = allocated / eligible if eligible else 0.0
        staff.append(dict(
            TECH_ID=tech, WORK_CENTER_ID=ctx["wc_labels"].get(home.get(tech), home.get(tech)),
            ELIGIBLE_WORKDAYS=eligible, ANNUAL_VACATIONS=allocated,
            MEAN_ANNUAL_VACATIONS=final["dispatcher_annual_vacation_day_mean"],
            ANNUAL_DAY_DEVIATION=day_delta, ABS_ANNUAL_DAY_DEVIATION=abs(day_delta),
            ANNUAL_VACATION_RATE=rate, OVERALL_VACATION_RATE=overall_rate,
            ANNUAL_RATE_DEVIATION_PP=100 * (rate - overall_rate),
            ABS_ANNUAL_RATE_DEVIATION_PP=abs(100 * (rate - overall_rate)),
        ))  # fmt: skip
    dispatcher_df = pd.DataFrame(staff)
    dispatcher_df.sort_values("ABS_ANNUAL_DAY_DEVIATION", ascending=False).to_csv(
        outdir / "vacation_annual_dispatcher_fairness.csv", index=False
    )
    dispatcher_df.to_csv(outdir / "vacation_by_dispatcher_diagnostic.csv", index=False)
    dispatcher_df.rename(columns={"TECH_ID": "BIDLINE_GROUP"}).to_csv(
        outdir / "vacation_annual_bidline_fairness.csv", index=False
    )
    by_wc = (
        dispatcher_df.groupby("WORK_CENTER_ID")["ANNUAL_VACATIONS"]
        .agg(DISPATCHERS="count", MIN_DAYS="min", MEAN_DAYS="mean", MAX_DAYS="max", TOTAL_VACATIONS="sum")
        .reset_index()
    )
    by_wc.to_csv(outdir / "vacation_by_work_center.csv", index=False)

    daily = []
    for d in sorted(idx.days):
        t = ctx["targets"][d]
        daily.append(dict(
            SHIFT_DATE=d, OPERATIONAL_VOLUME=ctx["volumes"].get(d), PROPORTIONAL_TARGET=None if t is None else round(t, 3),
            DAILY_CAP=day_caps[d], DAILY_FLOOR=args.soft_targets[d], VACATIONS=state.vd[d],
            FLOOR_SHORTFALL=max(0, args.soft_targets[d] - state.vd[d]),
            SPARE_STAFF=sum(safe_caps[b] for b in idx.buckets_by_day[d]),
            SHORTAGE_ADDED=sum(max(0, state.vb[b] - safe_caps[b]) for b in idx.buckets_by_day[d]),
        ))  # fmt: skip
    daily_df = pd.DataFrame(daily)
    daily_df.to_csv(outdir / "vacation_daily_targets.csv", index=False, date_format="%Y-%m-%d")

    coverage = []
    for bucket, required in sorted(req.items()):
        eligible = len(idx.buckets.get(bucket, ()))
        coverage.append(dict(
            SHIFT_DATE=bucket[0], WORK_CENTER=ctx["wc_labels"].get(bucket[1], bucket[1]),
            SHIFT_TYPE=ctx["shift_labels"].get(bucket[2], bucket[2]), DEMAND=required,
            ELIGIBLE_SUPPLY_BEFORE_VACATIONS=eligible, UNAVOIDABLE_SHORTAGE=max(0, required - eligible),
            VACATIONS=state.vb[bucket], SUPPLY_AFTER_VACATIONS=eligible - state.vb[bucket],
            SHORTAGE_AFTER_VACATIONS=max(0, required - eligible + state.vb[bucket]),
        ))  # fmt: skip
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(outdir / "coverage_all_buckets.csv", index=False, date_format="%Y-%m-%d")
    coverage_deficits = coverage_df[coverage_df.UNAVOIDABLE_SHORTAGE > 0]
    coverage_deficits.to_csv(outdir / "coverage_shortage_audit.csv", index=False, date_format="%Y-%m-%d")
    assert int(coverage_df.UNAVOIDABLE_SHORTAGE.sum()) == baseline_shortage
    coverage_df[coverage_df.SHORTAGE_AFTER_VACATIONS > 0].to_csv(
        outdir / "coverage_shortage_after_vacation.csv", index=False, date_format="%Y-%m-%d"
    )
    by_desk = coverage_df.assign(
        SHORTAGE_ADDED=coverage_df.SHORTAGE_AFTER_VACATIONS - coverage_df.UNAVOIDABLE_SHORTAGE
    ).groupby("WORK_CENTER")[["DEMAND", "VACATIONS", "UNAVOIDABLE_SHORTAGE", "SHORTAGE_ADDED", "SHORTAGE_AFTER_VACATIONS"]].sum()
    by_desk.reset_index().to_csv(outdir / "vacation_shortage_by_work_center.csv", index=False)

    corr = None
    if ctx["volumes"]:
        corr = float(daily_df["VACATIONS"].corr(daily_df["OPERATIONAL_VOLUME"], method="spearman"))
    counts = [state.vg[g] for g in idx.group_list]
    report = dict(
        model="COVERAGE_OPTIMAL_PLUS_TECH_ID_ANNUAL_VACATION_DAY_FAIRNESS",
        allow_added_shortage=args.allow_added_shortage,
        shortage_before_vacation=baseline_shortage,
        shortage_added_by_vacation=final["added_shortage"],
        final_coverage_shortage=shortage(idx, state, req),
        coverage_shortage_proven_minimum=baseline_shortage,
        max_coverage_safe_vacations=ctx["max_safe_total"],
        fixed_annual_vacation_total=False,
        parameters=dict(
            rule1_enabled=args.daily_rule, rule1_min=args.daily_min, rule1_max=args.daily_max,
            rule1_tolerance_below=args.daily_tolerance_below, rule1_tolerance_above=args.daily_tolerance_above,
            daily_range=[args.day_lo, args.day_hi if args.daily_rule else None], min_weight=args.soft_min_weight,
            shortage_weight=args.shortage_weight,
            annual_worst_weight=args.annual_worst_weight, annual_average_weight=args.annual_average_weight,
            slack_days=args.slack_days, max_spread_days=args.max_spread,
            volume_policy=args.volume_policy, volume_tolerance=args.volume_tolerance,
            volume_cap_slack=args.volume_cap_slack, base_per_day=args.base_per_day,
            base_tolerance=args.base_tolerance, base_weight=args.base_weight,
            excluded_dates=[str(d.date()) for d in ctx["excluded"]],
            lns_iterations=args.lns_iterations, lns_seconds=args.lns_seconds, seed=args.seed,
        ),
        fairness_unit="ONE_TECH_ID_PER_DISPATCHER",
        fairness_primary_unit="ANNUAL_VACATION_DAYS_PER_TECH_ID",
        fairness_secondary_diagnostic="ANNUAL_VACATION_RATE",
        fairness_band_met=spread_ok(idx, state, args) if args.max_spread is not None else None,
        base_range_met=(final["vacations_outside_base_range"] == 0) if args.base_range else None,
        spearman_vacations_vs_volume=corr,
        dates=len(idx.days), eligible_rows=idx.n, group_count=len(idx.groups),
        training_calendar_supplied=ctx["training_supplied"],
        imputed_demand_buckets=len(imputed), imputed_volume_dates=len(ctx["volume_audit"]),
        before={k: before[k] for k in STAT_KEYS if k in before},
        after={k: final[k] for k in STAT_KEYS if k in final},
        unavoidable_shortage_buckets=len(coverage_deficits),
        vacation_rows=len(vacations), remaining_supply_rows=len(remaining), original_supply_rows=len(raw),
        dispatcher_vacation_min=min(counts, default=0), dispatcher_vacation_max=max(counts, default=0),
        zero_vacation_dispatchers=sum(c == 0 for c in counts),
    )  # fmt: skip
    if args.base_range:
        report["base_total_range"] = list(args.base_range)
    (outdir / "vacation_summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def build_parser():
    p = argparse.ArgumentParser(description="Shortage first + TECH_ID annual vacation-DAY fairness; daily max HARD, min SOFT")
    p.add_argument("--supply", default=DEFAULT_SUPPLY)
    p.add_argument("--demand", default=DEFAULT_DEMAND)
    p.add_argument("--volume", default=DEFAULT_VOLUME)
    p.add_argument("--training", default=DEFAULT_TRAINING)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume-vacations", default=None,
                   help="Recover a previously exported vacation_initial.csv without rerunning the search")  # fmt: skip
    p.add_argument("--exclude-date", default=DEFAULT_EXCLUDED_DATE, help="comma-separated dates, or none")
    p.add_argument("--demand-missing", choices=("impute", "error"), default="impute")
    # Rule 1: daily total
    p.add_argument("--daily-min", type=int, default=25)
    p.add_argument("--daily-max", type=int, default=29)
    p.add_argument("--daily-tolerance-below", type=int, default=0, help="allowed range starts at min - this")
    p.add_argument("--daily-tolerance-above", type=int, default=0, help="allowed range ends at max + this")
    p.add_argument("--no-daily-rule", dest="daily_rule", action="store_false",
                   help="switch rule 1 off: no daily range (eligible staff is the only cap)")  # fmt: skip
    p.add_argument("--min-weight", dest="soft_min_weight", type=float, default=10000.0,
                   help="penalty per missing vacation below the daily minimum, per date; 10000 = top priority, "
                        "2 = the original soft 25")  # fmt: skip
    # Shortage
    p.add_argument("--no-added-shortage", dest="allow_added_shortage", action="store_false",
                   help="original model: vacations only where a desk/shift has staff above demand")  # fmt: skip
    p.add_argument("--shortage-weight", type=float, default=1000.0,
                   help="penalty per unit of shortage a vacation adds, per date (below the daily minimum "
                        "and the base, above fairness)")  # fmt: skip
    # Rule 2: fairness in DAYS
    p.add_argument("--annual-worst-weight", type=float, default=1.0,
                   help="Weight on worst TECH_ID annual vacation-day deviation")  # fmt: skip
    p.add_argument("--annual-average-weight", type=float, default=1.0,
                   help="Weight on mean absolute TECH_ID annual vacation-day deviation")  # fmt: skip
    p.add_argument("--slack-days", type=int, default=None,
                   help="hard band: counts within [L-k, L+1+k] (max-min <= 2k+1); omit for no band")  # fmt: skip
    # Rule 3: inverse volume
    p.add_argument("--volume-policy", choices=("proportional", "rank-cap", "none"), default="proportional")
    p.add_argument("--volume-tolerance", type=int, default=1,
                   help="proportional: vacations per day allowed either side of the volume target")  # fmt: skip
    p.add_argument("--volume-cap-slack", type=int, default=1, help="rank-cap only")
    # Base per day
    p.add_argument("--base-per-day", type=float, default=None, help="e.g. 27; omit for no base")
    p.add_argument("--base-tolerance", type=float, default=0.0, help="+- vacations per day around the base")
    p.add_argument("--base-weight", type=float, default=10000.0,
                   help="Penalty per mean daily vacation outside the base range")  # fmt: skip
    # Search
    p.add_argument("--lns-iterations", type=int, default=80)
    p.add_argument("--lns-seconds", type=float, default=6.0)
    p.add_argument("--lns-buckets", type=int, default=6)
    p.add_argument("--lns-max-buckets", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p


RULE_PRIORITY = 10  # with added shortage allowed, rule weights >= 10 x shortage weight
NO_CAP = 10**6  # daily cap when rule 1 is off (eligible staff still caps each date)


def apply_daily_rule(args):
    """Effective daily range [day_lo, day_hi] from rule 1."""
    if args.daily_rule:
        args.day_lo = max(0, args.daily_min - args.daily_tolerance_below)
        args.day_hi = args.daily_max + args.daily_tolerance_above
    else:
        args.day_lo, args.day_hi = 0, NO_CAP


def check_args(args):
    errors = []
    if not (0 <= args.daily_min <= args.daily_max and args.daily_max > 0):
        errors.append("Require 0 <= daily min <= daily max")
    if args.daily_tolerance_below < 0 or args.daily_tolerance_above < 0:
        errors.append("rule 1 tolerances must be >= 0")
    apply_daily_rule(args)
    if not args.daily_rule and args.volume_policy == "rank-cap":
        errors.append("rank_cap needs rule 1 enabled (its caps count down from the daily max)")
    if not args.daily_rule and args.volume_policy == "none" and args.base_per_day is None:
        errors.append("with rule 1, rule 3 and the base all off nothing asks for vacations (the answer would be 0)")
    if any(x < 0 for x in (args.annual_worst_weight, args.annual_average_weight, args.soft_min_weight,
                           args.base_weight, args.shortage_weight)):
        errors.append("Objective weights must be nonnegative")
    if args.annual_worst_weight + args.annual_average_weight <= 0:
        errors.append("At least one annual fairness weight must be positive")
    if args.lns_iterations < 0 or args.lns_seconds <= 0 or args.workers < 1:
        errors.append("Invalid LNS iteration/time/worker settings")
    if args.lns_buckets < 2 or args.lns_max_buckets < args.lns_buckets:
        errors.append("Invalid LNS neighborhood sizes")
    if args.volume_cap_slack < 0 or args.volume_tolerance < 0:
        errors.append("Volume cap slack / tolerance must be >= 0")
    if args.slack_days is not None and args.slack_days < 0:
        errors.append("slack-days must be >= 0")
    if args.base_per_day is not None and (args.base_tolerance < 0 or not 0 <= args.base_per_day <= args.day_hi):
        errors.append("base value must be in [0, daily max + tolerance_above] and base tolerance >= 0")
    if errors:
        raise ValueError("; ".join(errors))


def run(args):
    """Run the whole vacation step for a parsed/constructed argument namespace."""
    check_args(args)
    args.max_spread = None if args.slack_days is None else 2 * args.slack_days + 1
    if args.allow_added_shortage:
        # The rules come before shortage: the daily minimum / rule-3 floors and
        # the base are priced well above a unit of added shortage, whatever
        # their configured weights say.
        rule_weight = RULE_PRIORITY * args.shortage_weight
        for name, label in (("soft_min_weight", "rule1 min_weight"), ("base_weight", "base_per_day weight")):
            if getattr(args, name) < rule_weight:
                if name == "soft_min_weight" or args.base_per_day is not None:
                    print(f"NOTE: {label} raised from {getattr(args, name):g} to {rule_weight:g} "
                          f"so the rule outranks shortage.")
                setattr(args, name, rule_weight)

    raw = csv_in(args.supply)
    # Keep the input rows exactly as read, so outputs use the input's own spelling.
    raw[ROW] = np.arange(len(raw))
    raw[DATE] = pd.to_datetime(raw["SHIFT_DATE"], format="mixed", errors="coerce").dt.normalize()
    full = normalize_supply(raw.drop(columns=[DATE]))
    wc_labels = {str(x).strip().upper(): str(x).strip() for x in raw["WORK_CENTER_ID"].dropna().unique()}
    shift_labels = {str(x).strip().upper(): str(x).strip() for x in raw["SHIFT_TYPE"].dropna().unique()}
    home_wc = full.groupby("TECH_ID")["WORK_CENTER_ID"].agg(lambda s: s.mode().iat[0]).to_dict()

    excluded = []
    if str(args.exclude_date).lower() not in ("none", "", "null"):
        excluded = [pd.Timestamp(x.strip()).normalize() for x in str(args.exclude_date).split(",") if x.strip()]
    supply = full.loc[~full.SHIFT_DATE.isin(excluded)].copy()
    for d in excluded:
        print(f"Excluded {d.date()} ({int((full.SHIFT_DATE == d).sum())} supply rows; kept unchanged in the output).")
    if supply.empty:
        raise ValueError("Supply empty after exclusions")
    supply, group_kind = attach_dispatcher_groups(supply)
    train, training_supplied = training_keys(args.training)
    mask = [(str(r.TECH_ID), r.SHIFT_DATE) not in train for r in supply.itertuples(index=False)]
    allowed = supply.loc[mask].reset_index(drop=True)
    if allowed.empty:
        raise ValueError("No vacation-eligible work shifts after training exclusions")
    idx = Index(allowed, all_groups=supply["FAIR_GROUP"].unique())
    dates = sorted(idx.days)

    base_range = None
    if args.base_per_day is not None:
        D = len(dates)
        base_range = (
            max(0, math.ceil((args.base_per_day - args.base_tolerance) * D - 1e-9)),
            math.floor((args.base_per_day + args.base_tolerance) * D + 1e-9),
        )
    args.base_range = base_range

    day_caps, soft, targets, volumes, vol_audit = daily_limits(dates, args.volume, args)
    day_caps = {d: min(day_caps[d], len(idx.days[d])) for d in dates}
    args.soft_targets = soft
    demand = normalize_demand(csv_in(args.demand))
    req_all, imputed = demand_map_with_audit(demand, full, set(full.SHIFT_DATE), args.demand_missing)
    req = {b: n for b, n in req_all.items() if b[0] in set(dates)}
    # PROOF of minimum shortage given fixed eligible supply: vacations only
    # come out of staff above demand in each (date, work center, shift).
    safe_caps = {b: min(len(js), max(0, len(js) - req.get(b, 0))) for b, js in idx.buckets.items()}
    shortage_min = sum(max(0, need - len(idx.buckets.get(b, ()))) for b, need in req.items())
    max_safe_total = sum(min(day_caps[d], sum(safe_caps[b] for b in idx.buckets_by_day[d])) for d in dates)
    print(f"Input: {idx.n:,} eligible shifts, {len(dates)} dates, {len(idx.groups)} TECH_ID fairness units.")
    # Vacations may use every eligible workday when added shortage is allowed.
    limit_caps = {b: len(js) for b, js in idx.buckets.items()} if args.allow_added_shortage else safe_caps
    args.safe_caps = safe_caps
    print(f"SHORTAGE BEFORE VACATIONS = {shortage_min} (no vacation allocation can reduce this).")
    print(f"Most vacations possible without adding shortage (within daily caps): {max_safe_total:,}.")
    if args.allow_added_shortage:
        print(f"Vacations may add shortage when the rules need them (price {args.shortage_weight} per unit).")
    else:
        print("Vacations never add shortage (shortage.allow_added: false).")
    if base_range:
        print(f"Base: annual total target {base_range[0]:,}..{base_range[1]:,} (weight {args.base_weight:g}).")
    if args.max_spread is not None:
        print(f"Fairness band: every dispatcher within [L-{args.slack_days}, L+1+{args.slack_days}] days (hard).")

    allocation, seed_scan = flow_seed(idx, safe_caps, limit_caps, day_caps, args)
    assert shortage(idx, allocation, req) == shortage_min + added_shortage(allocation, safe_caps)
    before = allocation.statistics(idx, args)
    print(
        f"Seed: {allocation.v} vacations; days per dispatcher {before['dispatcher_annual_vacation_day_min']}-"
        f"{before['dispatcher_annual_vacation_day_max']}; annual mean dev={before['annual_average_deviation_days']:.3f} "
        f"days, worst={before['annual_worst_deviation_days']:.3f} days, daily floor shortfall={before['soft_floor_deficit']}, "
        f"shortage added={before['added_shortage']}"
    )
    if args.resume_vacations:
        saved = normalize_supply(csv_in(args.resume_vacations))
        valid_rows = {
            (str(r.TECH_ID), r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE)): j for j, r in enumerate(idx.rows)
        }
        chosen = [0] * idx.n
        for r in saved.itertuples(index=False):
            key = (str(r.TECH_ID), r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
            if key not in valid_rows:
                raise ValueError(f"--resume-vacations includes ineligible or unknown assignment: {key}")
            chosen[valid_rows[key]] = 1
        allocation = Allocation(idx, chosen)
        if any(allocation.vd[d] > day_caps[d] for d in idx.days) or any(
            allocation.vb[b] > limit_caps[b] for b in idx.buckets
        ):
            raise ValueError("Recovered vacations exceed daily caps or the bucket limits")
        print(f"Recovered {allocation.v:,} vacation assignments from {args.resume_vacations}.")
        history = []
    elif args.lns_iterations:
        allocation, history = optimize(idx, allocation, limit_caps, day_caps, shortage_min, req, args)
    else:
        history = []

    ctx = dict(
        idx=idx, req=req, raw=raw, train=train, safe_caps=safe_caps, limit_caps=limit_caps, day_caps=day_caps,
        shortage_min=shortage_min, imputed=imputed, volume_audit=vol_audit, seed_scan=seed_scan,
        wc_labels=wc_labels, shift_labels=shift_labels, home_wc=home_wc, targets=targets,
        volumes=volumes, excluded=excluded, training_supplied=training_supplied, max_safe_total=max_safe_total,
    )  # fmt: skip
    output = Path(args.output_dir)
    summary = export(ctx, allocation, output, before, args)
    if history:
        pd.DataFrame(history).to_csv(output / "vacation_lns_history.csv", index=False)
    a = summary["after"]
    print(
        f"DONE: shortage={summary['final_coverage_shortage']}; vacations={a['total_vacations']}; "
        f"days per dispatcher {a['dispatcher_annual_vacation_day_min']}-{a['dispatcher_annual_vacation_day_max']}; "
        f"annual avg dev={a['annual_average_deviation_days']:.3f} days; worst={a['annual_worst_deviation_days']:.3f} days; "
        f"days below {args.day_lo}={a['days_below_daily_min']}"
    )
    print(f"Outputs and shortage audit: {output.resolve()}")
    return summary


# ---- YAML settings -> arguments ----------------------------------------------

YAML_KEYS = {
    "vacation": {"exclude_date", "demand_missing", "shortage", "rule1_daily_total", "rule2_fairness",
                 "rule3_inverse_volume", "base_per_day", "search"},
    "exclude_date": {"enabled", "date"},
    "shortage": {"allow_added", "weight"},
    "rule1_daily_total": {"enabled", "min", "max", "tolerance_below", "tolerance_above", "min_weight"},
    "rule2_fairness": {"worst_weight", "average_weight", "slack_days"},
    "rule3_inverse_volume": {"enabled", "mode", "tolerance", "cap_slack"},
    "base_per_day": {"enabled", "value", "tolerance", "weight"},
    "search": {"lns_iterations", "lns_seconds", "lns_buckets", "lns_max_buckets", "workers", "seed"},
}  # fmt: skip


def exclusion_from_settings(ex):
    """exclude_date: {enabled: true, date: "2026-01-11"} -> "2026-01-11"; enabled false -> "none".
    A missing block means the default (2026-01-11 excluded)."""
    if ex is None:
        return DEFAULT_EXCLUDED_DATE
    if not isinstance(ex, dict):
        raise ValueError('exclude_date must look like {enabled: true, date: "2026-01-11"}')
    unknown = set(ex) - YAML_KEYS["exclude_date"]
    if unknown:
        raise ValueError(f"unknown key(s) in 'exclude_date': {sorted(unknown)}; allowed: ['date', 'enabled']")
    if not ex.get("enabled", True):
        return "none"
    date = ex.get("date", DEFAULT_EXCLUDED_DATE)
    dates = date if isinstance(date, (list, tuple)) else [date]
    return ",".join(str(pd.Timestamp(str(x)).date()) for x in dates)


def args_from_settings(vacation: dict, **paths):
    """Build the argument namespace from the YAML 'vacation' section plus
    file paths (supply, demand, volume, training, output_dir)."""
    vacation = vacation or {}

    def section(name, d):
        d = d or {}
        unknown = set(d) - YAML_KEYS[name]
        if unknown:
            raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown)}; allowed: {sorted(YAML_KEYS[name])}")
        return d

    vacation = section("vacation", vacation)
    r1 = section("rule1_daily_total", vacation.get("rule1_daily_total"))
    r2 = section("rule2_fairness", vacation.get("rule2_fairness"))
    r3 = section("rule3_inverse_volume", vacation.get("rule3_inverse_volume"))
    base = section("base_per_day", vacation.get("base_per_day"))
    search = section("search", vacation.get("search"))
    a = build_parser().parse_args([])
    for k, v in paths.items():
        if v is not None:
            setattr(a, k, str(v))
    a.exclude_date = exclusion_from_settings(vacation.get("exclude_date"))
    a.demand_missing = vacation.get("demand_missing", a.demand_missing)
    sh = section("shortage", vacation.get("shortage"))
    a.allow_added_shortage = bool(sh.get("allow_added", True))
    a.shortage_weight = float(sh.get("weight", a.shortage_weight))
    a.daily_rule = bool(r1.get("enabled", True))
    a.daily_min = int(r1.get("min", a.daily_min))
    a.daily_max = int(r1.get("max", a.daily_max))
    a.daily_tolerance_below = int(r1.get("tolerance_below", 0))
    a.daily_tolerance_above = int(r1.get("tolerance_above", 0))
    a.soft_min_weight = float(r1.get("min_weight", a.soft_min_weight))
    a.annual_worst_weight = float(r2.get("worst_weight", a.annual_worst_weight))
    a.annual_average_weight = float(r2.get("average_weight", a.annual_average_weight))
    a.slack_days = None if r2.get("slack_days") is None else int(r2["slack_days"])
    if not r3.get("enabled", True):
        a.volume_policy = "none"
    else:
        mode = str(r3.get("mode", "proportional")).replace("_", "-")
        if mode not in ("proportional", "rank-cap"):
            raise ValueError("rule3_inverse_volume.mode must be proportional or rank_cap")
        a.volume_policy = mode
    a.volume_tolerance = int(r3.get("tolerance", a.volume_tolerance))
    a.volume_cap_slack = int(r3.get("cap_slack", a.volume_cap_slack))
    a.base_per_day = float(base.get("value", 27)) if base.get("enabled", False) else None
    a.base_tolerance = float(base.get("tolerance", a.base_tolerance))
    a.base_weight = float(base.get("weight", a.base_weight))
    for k in YAML_KEYS["search"]:
        if k in search:
            setattr(a, k, type(getattr(a, k))(search[k]))
    return a


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    try:
        check_args(args)
    except ValueError as exc:
        p.error(str(exc))
    return run(args)


if __name__ == "__main__":
    main()
