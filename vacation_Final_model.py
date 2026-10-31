"""American Airlines dispatcher vacation optimizer: shortage-first + TECH_ID annual fairness.

Priorities:
  1. Minimize staffing shortage. The mathematical minimum is computed from
     staffing after training, before vacations. Vacations are ONLY allowed
     in desk/day/shift buckets with staffing above the required demand.
  2. Balance ANNUAL vacation RATES across the 506 dispatcher-specific bid lines,
     identified by TECH_ID (one dispatcher holds one line for the whole year).
     Minimize both the worst and average absolute rate deviation.

Daily 29 is a hard maximum. Daily 25 is a soft target with a shortfall
penalty, NOT a lower bound. The objective also guards against the trivial
zero-vacation solution. LNS can alter WHICH people get vacation and HOW
MANY vacations are assigned. The total remains a decision, not fixed.

FAIRNESS UNIT: each TECH_ID, because each dispatcher retains one assigned
bid line for the whole year. No grouping of different dispatchers into inferred
schedule-pattern clusters. TECH_ID is NOT an official bid-line identifier and
multiple dispatchers may have equivalent scheduling patterns.

Standard deviations in the summary explicitly distinguish vacation-day SD
across dispatchers from annual vacation-rate SD in percentage points.

Outputs include coverage_shortage_audit.csv so you can determine which
staffing buckets require new capacity/flex days to reach zero shortage.

Run: python aa_vacation_TECH_ID_annual_fairness.py
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

DEFAULT_SUPPLY = 'data/Supply_Regular_qualified.csv'
DEFAULT_DEMAND = 'data/annual_desk_needs_reference.csv'
DEFAULT_VOLUME = 'data/demand_volume_per_day_reference.csv'
DEFAULT_TRAINING = 'data/all_training_data.csv'
DEFAULT_OUTPUT_DIR = 'outputs/tech_id_annual_fairness'
DEFAULT_EXCLUDED_DATE = '2026-01-11'  # As requested in earlier runs; use --exclude-date none to disable.
BP = 10_000  # 10,000 basis points = 100 percentage points


def csv_in(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Input file not found: {p.resolve()}')
    df = pd.read_csv(p, low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df


def require(df, names, label):
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f'{label}: missing {missing}. Available columns: {list(df.columns)}')


def dates_in(df, name):
    df = df.copy()
    df[name] = pd.to_datetime(df[name], errors='coerce').dt.normalize()
    if df[name].isna().any():
        raise ValueError(f'Invalid {name} values')
    return df


def normalize_supply(df):
    require(df, ['TECH_ID', 'SHIFT_DATE', 'WORK_CENTER_ID', 'SHIFT_TYPE'], 'supply')
    df = dates_in(df, 'SHIFT_DATE')
    df['TECH_ID'] = df['TECH_ID'].astype('string').str.strip()
    for col in ('WORK_CENTER_ID', 'SHIFT_TYPE'):
        df[col] = df[col].astype('string').str.strip().str.upper()
    if df[['TECH_ID', 'WORK_CENTER_ID', 'SHIFT_TYPE']].isna().any().any():
        raise ValueError('Supply contains missing dispatcher, work-center, or shift IDs')
    for col in ('STATUS', 'ASSIGNMENT_TYPE', 'DAY_TYPE'):
        if col in df:
            excluded = {'OFF', 'REST', 'DAY OFF', 'DAY_OFF', 'VACATION', 'FLEX', 'TRAINING'}
            df = df.loc[~df[col].astype(str).str.strip().str.upper().isin(excluded)].copy()
    if df.duplicated(['TECH_ID', 'SHIFT_DATE']).any():
        raise ValueError('Multiple scheduled supply rows for a TECH_ID and SHIFT_DATE')
    return df.sort_values(['SHIFT_DATE', 'TECH_ID']).reset_index(drop=True)


def attach_dispatcher_groups(supply):
    """One TECH_ID = one dispatcher-specific annual fairness unit.

    This does not claim TECH_ID is an official AA bid-line ID; it simply keeps
    each dispatcher's annual vacation allocation separate, as requested.
    """
    supply = supply.copy()
    supply['FAIR_GROUP'] = supply['TECH_ID'].astype(str)
    count = int(supply['TECH_ID'].nunique())
    print(f'Annual fairness groups: {count} TECH_ID dispatchers (one assigned bid line per dispatcher).')
    print('NOTE: TECH_ID is a dispatcher ID, not a verified official AA bid-line identifier.')
    return supply, 'TECH_ID_DISPATCHER_ANNUAL_LINE'


def normalize_demand(df):
    require(df, ['SHIFT_DATE', 'WORK_CENTER', 'SHIFT_TYPE', 'DEMAND'], 'demand')
    df = dates_in(df, 'SHIFT_DATE')
    df['WORK_CENTER'] = df.WORK_CENTER.astype('string').str.strip().str.upper()
    df['SHIFT_TYPE'] = df.SHIFT_TYPE.astype('string').str.strip().str.upper()
    df['DEMAND'] = pd.to_numeric(df.DEMAND, errors='raise')
    if df.DEMAND.isna().any() or (df.DEMAND < 0).any() or not (df.DEMAND % 1 == 0).all():
        raise ValueError('Demand must be nonnegative integers')
    if df.duplicated(['SHIFT_DATE', 'WORK_CENTER', 'SHIFT_TYPE']).any():
        raise ValueError('Duplicate desk demand bucket')
    return df


def demand_map_with_audit(df, supply, considered_dates, missing_mode):
    mapping = {(r.SHIFT_DATE, str(r.WORK_CENTER), str(r.SHIFT_TYPE)): int(r.DEMAND)
               for r in df.itertuples(index=False) if r.SHIFT_DATE in considered_dates}
    # Historical reference can supply the donor for an excluded or missing day.
    donors = {(r.SHIFT_DATE, str(r.WORK_CENTER), str(r.SHIFT_TYPE)): int(r.DEMAND)
              for r in df.itertuples(index=False)}
    missing = sorted(set(zip(supply.SHIFT_DATE, supply.WORK_CENTER_ID, supply.SHIFT_TYPE)) - set(mapping))
    if missing and missing_mode == 'error':
        raise ValueError(f'{len(missing)} missing demand buckets; examples {missing[:4]}')
    records = []
    for date, center, shift in missing:
        donor_date = next((z for z in (date - pd.Timedelta(days=7), date + pd.Timedelta(days=7))
                           if (z, center, shift) in donors), None)
        if donor_date is None:
            possible = [k[0] for k in donors if k[1:] == (center, shift)]
            if not possible:
                raise ValueError(f'No demand donor for {date}, {center}, {shift}')
            donor_date = min(possible, key=lambda x: abs((x - date).days))
        demand = donors[(donor_date, center, shift)]
        mapping[(date, center, shift)] = demand
        records.append(dict(SHIFT_DATE=date, WORK_CENTER=center, SHIFT_TYPE=shift,
                            IMPUTED_DEMAND=demand, DONOR_DATE=donor_date))
    if records:
        print(f'WARNING: {len(records)} demand buckets imputed. Coverage numbers are PROVISIONAL.')
    return mapping, pd.DataFrame(records, columns=['SHIFT_DATE', 'WORK_CENTER', 'SHIFT_TYPE',
                                                   'IMPUTED_DEMAND', 'DONOR_DATE'])


def training_keys(path):
    if not path or str(path).lower() == 'none':
        print('WARNING: No training calendar; vacation eligibility is provisional.')
        return set(), False
    train = csv_in(path)
    require(train, ['TECH_ID', 'SHIFT_DATE'], 'training')
    train = dates_in(train, 'SHIFT_DATE')
    return set(zip(train.TECH_ID.astype('string').str.strip(), train.SHIFT_DATE)), True


def daily_caps(dates, volume_path, policy, cap_slack, hard_max):
    caps = {d: hard_max for d in dates}
    volumes = {}
    if policy == 'none':
        return caps, volumes
    table = csv_in(volume_path)
    dcol = next((c for c in ('DATE', 'SHIFT_DATE', 'CALENDAR_DATE', 'DAY') if c in table), None)
    vcol = next((c for c in ('DEMAND_VOLUME_PER_DAY', 'DEMAND_VOLUME', 'DAILY_VOLUME',
                             'OPERATIONAL_VOLUME', 'VOLUME', 'DEMAND') if c in table), None)
    if dcol is None or vcol is None:
        raise ValueError(f'Cannot detect volume date/value columns: {list(table.columns)}')
    table = dates_in(table, dcol)
    if table[dcol].duplicated().any():
        raise ValueError('Duplicate dates in volume CSV')
    table[vcol] = pd.to_numeric(table[vcol], errors='raise')
    if table[vcol].isna().any():
        raise ValueError('Null/non-numeric operational volume')
    vol = dict(zip(table[dcol], table[vcol]))
    missing = [d for d in dates if d not in vol]
    if missing:
        raise ValueError(f'Volume file missing {len(missing)} dates, e.g. {missing[:3]}')
    volumes = {d: float(vol[d]) for d in dates}
    # Same historical approximate rank-based policy: *upper* bounds only.
    ordered = sorted(dates, key=lambda d: (volumes[d], d))
    n_levels = min(5, hard_max + 1)
    for rank, date in enumerate(ordered):
        band = min(n_levels - 1, rank * n_levels // len(ordered))
        suggested_max = hard_max - band
        caps[date] = min(hard_max, max(0, suggested_max + cap_slack))
    print('NOTE: Volume rank caps are approximate policy choices, NOT AA-specified cutoffs.')
    return caps, volumes


class Index:
    """TECH_ID-specific annual fairness, days, and desk/shift coverage buckets."""
    def __init__(self, allowed):
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
        self.eg = {g: len(rows) for g,rows in self.groups.items()}
        self.group_list = sorted(self.groups)
        self.buckets_by_day = defaultdict(list)
        for bucket in self.buckets:
            self.buckets_by_day[bucket[0]].append(bucket)


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
                self._counts_delta(idx,j,1)

    def _counts_delta(self, idx, j, amount):
        self.v += amount
        self.vg[idx.group_of[j]] += amount
        self.vd[idx.day_of[j]] += amount
        self.vb[idx.bucket_of[j]] += amount

    def modify(self, idx, proposal):
        for j, flag in proposal.items():
            flag=int(flag)
            if self.chosen[j] != flag:
                self._counts_delta(idx,j,flag-self.chosen[j])
                self.chosen[j]=flag

    def clone(self):
        out=object.__new__(Allocation)
        out.chosen=self.chosen.copy()
        out.v=self.v
        out.vg=self.vg.copy()
        out.vd=self.vd.copy()
        out.vb=self.vb.copy()
        return out

    def statistics(self, idx, soft_min, worst_weight, average_weight, floor_weight_pp):
        # Annual opportunity-normalized vacation rates for each TECH_ID.
        # Every group represents exactly one dispatcher; unlike inferred
        # work-pattern clusters, allocations cannot cancel across employees.
        overall=self.v/idx.n
        signed={g:self.vg[g]/idx.eg[g]-overall for g in idx.groups}
        avg_pp=100*sum(abs(v) for v in signed.values())/len(signed)
        worst_pp=100*max(map(abs,signed.values()),default=0)
        shortfall=sum(max(0,soft_min-self.vd[d]) for d in idx.days)
        score=worst_weight*worst_pp + average_weight*avg_pp + floor_weight_pp*shortfall/len(idx.days)
        rates=[self.vg[g]/idx.eg[g] for g in idx.group_list]
        vacations_per_dispatcher=[self.vg[g] for g in idx.group_list]
        rate_sd = float(np.std(rates, ddof=0))
        days_sd = float(np.std(vacations_per_dispatcher, ddof=0))
        return dict(score_pp=score,total_vacations=self.v,
                    annual_average_pp=avg_pp,annual_worst_pp=worst_pp,
                    annual_group_rate_SD=rate_sd,  # backward-compatible alias
                    annual_group_rate_SD_pp=100*rate_sd,  # now one group per TECH_ID
                    dispatcher_annual_rate_SD_pp=100*rate_sd,
                    dispatcher_annual_vacation_day_SD=days_sd,
                    dispatcher_annual_vacation_day_min=min(vacations_per_dispatcher, default=0),
                    dispatcher_annual_vacation_day_max=max(vacations_per_dispatcher, default=0),
                    annual_signed=signed,soft_floor_deficit=shortfall,
                    days_below_target=sum(self.vd[d]<soft_min for d in idx.days))



def seed_schedule(idx, bucket_cap, day_cap, soft_min, rng):
    """Coverage-safe initial solution around the SOFT daily target (not fixed)."""
    planned = {}
    for day in idx.days:
        safe_capacity = sum(bucket_cap[b] for b in idx.buckets_by_day[day])
        planned[day] = min(soft_min, day_cap[day], safe_capacity)
    target_total = sum(planned.values())
    target_year = {g: target_total * idx.eg[g] / idx.n for g in idx.groups}
    chosen = [0] * idx.n
    group_counts, bucket_counts = Counter(), Counter()
    ordering = list(idx.days)
    rng.shuffle(ordering)
    for day in ordering:
        for _ in range(planned[day]):
            best_j, best_priority = None, -float('inf')
            for j in idx.days[day]:
                if chosen[j] or bucket_counts[idx.bucket_of[j]] >= bucket_cap[idx.bucket_of[j]]:
                    continue
                g = idx.group_of[j]
                gap = (target_year[g] - group_counts[g]) / max(1.0, target_year[g])
                priority = gap + 0.03 * rng.random()
                if priority > best_priority:
                    best_j, best_priority = j, priority
            if best_j is None:
                raise AssertionError('Could not reach coverage-safe soft target for day ' + str(day))
            chosen[best_j] = 1
            group_counts[idx.group_of[best_j]] += 1
            bucket_counts[idx.bucket_of[best_j]] += 1
    return Allocation(idx, chosen), planned


def shortage(idx, state, req):
    available = {b: len(js) for b, js in idx.buckets.items()}
    all_b = set(req) | set(available)
    return sum(max(0, req.get(b, 0) - available.get(b, 0) + state.vb[b])
               for b in all_b)


def neighborhood_buckets(idx, state, info, bucket_caps, rng, how_many, mode):
    """TECH_ID annual-rate targeted neighborhoods; monthly fairness is never used.

    We deliberately alternate swap and count-changing moves. In count mode
    the sum of vacation assignments in selected buckets is NOT fixed, so the
    annual total can change without any artificial 8,800 or 9,912 equality.
    """
    annual = info['annual_signed']
    ranked = []
    for bucket, js in idx.buckets.items():
        chosen = [j for j in js if state.chosen[j]]
        free = [j for j in js if not state.chosen[j]]
        remaining_cap = bucket_caps[bucket] - len(chosen)
        if mode == 'swap':
            if not (chosen and free and len({idx.group_of[j] for j in js}) > 1):
                continue
            # Swap days from over-allocated to under-allocated TECH_IDs
            # in the same coverage bucket, preserving staffing exactly.
            potential = max(0., max(annual[idx.group_of[j]] for j in chosen) -
                            min(annual[idx.group_of[j]] for j in free))
        else:
            if not chosen and not (free and remaining_cap > 0):
                continue
            drop_gain = max([annual[idx.group_of[j]] for j in chosen] + [0.])
            add_gain = -min([annual[idx.group_of[j]] for j in free] + [0.]) if remaining_cap > 0 else 0.
            potential = max(0., drop_gain, add_gain)
        ranked.append((potential + 0.002 * rng.random(), bucket))
    if not ranked:
        return []
    ranked.sort(reverse=True)
    shortlist = [b for _, b in ranked[:min(len(ranked), 200)]]
    first = shortlist[int(rng.integers(0, min(15, len(shortlist))))]
    picked = [first]
    # Prefer more than one date, so count moves can add on one day and remove
    # on another if that is beneficial for the annual allocation balance.
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
    """Solve a small CP-SAT annual-fairness neighborhood (no monthly terms)."""
    from ortools.sat.python import cp_model
    selected = sorted({j for bucket in buckets for j in idx.buckets[bucket]})
    if not selected:
        return 'SKIPPED', {}, 0.
    model = cp_model.CpModel()
    x = {j: model.NewBoolVar(f'v_{j}') for j in selected}
    for j in selected:
        model.AddHint(x[j], int(state.chosen[j]))
    for bucket in buckets:
        assigned = sum(x[j] for j in idx.buckets[bucket])
        if mode == 'swap':
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
        vacations = updated(state.vd[day], js)
        model.Add(vacations <= day_caps[day])

    # The annual total is a DECISION in every count-changing neighborhood.
    # Only swap neighborhoods preserve it incidentally, as required by a swap.
    total_expr = state.v if mode == 'swap' else updated(state.v, selected)

    # Integer basis-point approximation to TECH_ID annual rate deviations.
    # z_g >= ceil(BP * abs(V_g/E_g - V/N)).
    deviations = []
    for g in idx.group_list:
        eg = idx.eg[g]
        group_total = updated(state.vg[g], by_group.get(g, [])) if g in by_group else state.vg[g]
        delta = idx.n * group_total - eg * total_expr
        z = model.NewIntVar(0, BP, f'dev_bp_{len(deviations)}')
        model.Add(z * (eg * idx.n) >= BP * delta)
        model.Add(z * (eg * idx.n) >= -BP * delta)
        deviations.append(z)

    maximum = model.NewIntVar(0, BP, 'worst_annual_deviation_bp')
    model.AddMaxEquality(maximum, deviations)
    # The incumbent is always feasible under this rounded rate guard.
    allowed_max_bp = min(BP, math.ceil(incumbent['annual_worst_pp'] * 100 + 1e-8))
    model.Add(maximum <= allowed_max_bp)

    # Match the evaluation weights as closely as practical in INTEGER units.
    # 100,000 * n_groups * score_pp corresponds to the following coefficients
    # because each z is expressed in basis points (z / 100 = percentage points).
    # Scaling by n_groups avoids rounding mean weight to zero with 506 TECH_IDs.
    n_groups = len(idx.groups)
    k_worst = round(args.annual_worst_weight * 1000 * n_groups)
    k_average = round(args.annual_average_weight * 1000)
    k_floor = round(args.soft_min_weight * 100000 * n_groups / len(idx.days))
    objective = k_worst * maximum + k_average * sum(deviations)
    for day, js in by_day.items():
        local_count = updated(state.vd[day], js)
        missing = model.NewIntVar(0, args.soft_min, f'daily_shortfall_{day:%Y%m%d}')
        model.Add(missing >= args.soft_min - local_count)
        objective += k_floor * missing

    model.Minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.lns_seconds
    solver.parameters.num_search_workers = args.workers
    solver.parameters.random_seed = args.seed + iteration
    result = solver.Solve(model)
    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.StatusName(result), {}, solver.WallTime()
    return solver.StatusName(result), {j: int(solver.Value(var)) for j, var in x.items()}, solver.WallTime()


def optimize(idx, state, bucket_caps, day_caps, baseline_shortage, demand_req, args):
    rng = np.random.default_rng(args.seed)
    history = []
    current = state.statistics(idx, args.soft_min, args.annual_worst_weight,
                               args.annual_average_weight, args.soft_min_weight)
    bucket_n = args.lns_buckets
    stagnant = 0
    for iteration in range(1, args.lns_iterations + 1):
        # Half the neighborhoods change counts; the annual total is NOT fixed.
        mode = 'count' if iteration % 2 == 0 else 'swap'
        k = min(5, bucket_n) if mode == 'count' else bucket_n
        buckets = neighborhood_buckets(idx, state, current, bucket_caps, rng, k, mode)
        if not buckets:
            print(f'LNS {iteration}/{args.lns_iterations}: no eligible {mode} neighborhood')
            continue
        status, proposal, elapsed = solve_local(idx, state, buckets, bucket_caps,
                                                day_caps, args, mode, iteration, current)
        accepted = False
        if proposal:
            candidate = state.clone()
            candidate.modify(idx, proposal)
            if (all(candidate.vb[b] <= bucket_caps[b] for b in buckets) and
                all(candidate.vd[d] <= day_caps[d] for d in {b[0] for b in buckets}) and
                shortage(idx, candidate, demand_req) == baseline_shortage):
                other = candidate.statistics(idx, args.soft_min, args.annual_worst_weight,
                                             args.annual_average_weight, args.soft_min_weight)
                # Both lower total score AND protect the worst annual bid line.
                if (other['score_pp'] < current['score_pp'] - 1e-9 and
                    other['annual_worst_pp'] <= current['annual_worst_pp'] + 1e-9):
                    state, current = candidate, other
                    accepted, stagnant = True, 0
        if not accepted:
            stagnant += 1
        if stagnant >= 5:
            bucket_n = min(args.lns_max_buckets, bucket_n + 2)
            stagnant = 0
        elif accepted:
            bucket_n = max(2, bucket_n - 1)
        history.append(dict(ITERATION=iteration, MODE=mode, STATUS=status,
                            ACCEPTED=accepted, BUCKETS=len(buckets), SECONDS=round(elapsed, 3),
                            VACATIONS=state.v, SCORE_PP=round(current['score_pp'], 6),
                            ANNUAL_AVERAGE_DEVIATION_PP=round(current['annual_average_pp'], 5),
                            WORST_ANNUAL_DEVIATION_PP=round(current['annual_worst_pp'], 5),
                            SOFT_DAILY_SHORTFALL=current['soft_floor_deficit']))
        print(f'LNS {iteration:2d}/{args.lns_iterations} [{mode}, {len(buckets)} buckets]: '
              f'{status}, accepted={accepted}, annual_avg={current["annual_average_pp"]:.2f}pp, '
              f'annual_worst={current["annual_worst_pp"]:.2f}pp, '
              f'vacations={state.v}, shortfall={current["soft_floor_deficit"]}', flush=True)
    return state, history


def export(idx, state, supply, source_columns, train, outdir,
           req, bucket_caps, day_caps, group_kind, volume, imputed,
           baseline_shortage, before, args, planned, training_supplied):
    outdir.mkdir(parents=True, exist_ok=True)
    final = state.statistics(idx, args.soft_min, args.annual_worst_weight,
                             args.annual_average_weight, args.soft_min_weight)
    assert shortage(idx, state, req) == baseline_shortage
    assert all(state.vd[d] <= day_caps[d] for d in idx.days)
    assert all(state.vb[b] <= bucket_caps[b] for b in idx.buckets)
    keys, vacation_rows = [], []
    for j, assigned in enumerate(state.chosen):
        if not assigned:
            continue
        r = idx.rows[j]
        key = (str(r.TECH_ID), r.SHIFT_DATE)
        assert key not in train
        keys.append(key)
        vacation_rows.append(dict(TECH_ID=r.TECH_ID, SHIFT_DATE=r.SHIFT_DATE,
                                  WORK_CENTER_ID=r.WORK_CENTER_ID, SHIFT_TYPE=r.SHIFT_TYPE))
    vacations = pd.DataFrame(vacation_rows,
                             columns=['TECH_ID','SHIFT_DATE','WORK_CENTER_ID','SHIFT_TYPE'])
    vacations.to_csv(outdir/'vacation_initial.csv', index=False, date_format='%Y-%m-%d')
    chosen_keys = pd.MultiIndex.from_tuples(keys, names=['TECH_ID','SHIFT_DATE'])
    input_keys = pd.MultiIndex.from_frame(supply[['TECH_ID','SHIFT_DATE']])
    assert chosen_keys.is_unique and chosen_keys.isin(input_keys).all()
    remaining = supply.loc[~input_keys.isin(chosen_keys), source_columns].copy()
    assert len(remaining) + len(vacations) == len(supply)
    remaining.to_csv(outdir/'vacation_outputs_updatedFormat.csv',
                     index=False, date_format='%Y-%m-%d')
    imputed.to_csv(outdir/'demand_imputation_audit.csv', index=False, date_format='%Y-%m-%d')

    # Every annual FAIR_GROUP is one TECH_ID. Report real dispatcher-level
    # rates and vacation-day SD, not inferred-cluster aggregates.
    staff = []
    for tech in idx.group_list:
        eligible = idx.eg[tech]
        allocated = state.vg[tech]
        delta = 100 * final['annual_signed'][tech]
        staff.append(dict(TECH_ID=tech,
                          ELIGIBLE_WORKDAYS=eligible,
                          ANNUAL_VACATIONS=allocated,
                          ANNUAL_VACATION_RATE=allocated / eligible,
                          OVERALL_VACATION_RATE=state.v / idx.n,
                          ANNUAL_RATE_DEVIATION_PP=delta,
                          ABS_ANNUAL_RATE_DEVIATION_PP=abs(delta)))
    dispatcher_df = pd.DataFrame(staff)
    dispatcher_df.sort_values('ABS_ANNUAL_RATE_DEVIATION_PP',ascending=False).to_csv(
        outdir/'vacation_annual_dispatcher_fairness.csv',index=False)
    dispatcher_df.to_csv(outdir/'vacation_by_dispatcher_diagnostic.csv',index=False)
    # Retain the previous fairness report name for downstream compatibility.
    dispatcher_df.rename(columns={'TECH_ID':'BIDLINE_GROUP'}).to_csv(
        outdir/'vacation_annual_bidline_fairness.csv', index=False)
    tech_counts = {row['TECH_ID']:int(row['ANNUAL_VACATIONS']) for row in staff}
    tech_group = {tech:tech for tech in idx.group_list}

    daily = []
    for d in sorted(idx.days):
        daily.append(dict(SHIFT_DATE=d, HARD_MAX=day_caps[d], SOFT_TARGET=args.soft_min,
                          INITIAL_SEED_TARGET=planned[d], VACATIONS=state.vd[d],
                          SOFT_TARGET_SHORTFALL=max(0, args.soft_min - state.vd[d]),
                          OPERATIONAL_VOLUME=volume.get(d)))
    pd.DataFrame(daily).to_csv(outdir/'vacation_daily_targets.csv',
                               index=False, date_format='%Y-%m-%d')

    # Vacations cannot fill staff vacancies: show precisely the buckets whose
    # demand exceeds eligible baseline supply BEFORE vacations are assigned.
    coverage = []
    for bucket, required in sorted(req.items()):
        eligible = len(idx.buckets.get(bucket, ()))
        short = max(0, required - eligible)
        coverage.append(dict(SHIFT_DATE=bucket[0], WORK_CENTER=bucket[1],
                             SHIFT_TYPE=bucket[2], DEMAND=required,
                             ELIGIBLE_SUPPLY_BEFORE_VACATIONS=eligible,
                             UNAVOIDABLE_SHORTAGE=short,
                             VACATIONS=state.vb[bucket],
                             SUPPLY_AFTER_VACATIONS=eligible-state.vb[bucket]))
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(outdir/'coverage_all_buckets.csv',index=False,date_format='%Y-%m-%d')
    coverage_deficits = coverage_df[coverage_df.UNAVOIDABLE_SHORTAGE > 0]
    coverage_deficits.to_csv(outdir/'coverage_shortage_audit.csv',
                             index=False, date_format='%Y-%m-%d')
    assert int(coverage_df.UNAVOIDABLE_SHORTAGE.sum()) == baseline_shortage

    report = dict(
        model='COVERAGE_OPTIMAL_PLUS_TECH_ID_ANNUAL_RATE_FAIRNESS',
        coverage_shortage_proven_minimum=baseline_shortage,
        final_coverage_shortage=shortage(idx,state,req),
        annual_vacation_maximized=False, fixed_annual_vacation_total=False,
        monthly_fairness_objective=False,
        hard_daily_max=args.hard_max, soft_daily_target=args.soft_min,
        group_definition=group_kind, group_count=len(idx.groups),
        fairness_unit='ONE_TECH_ID_PER_DISPATCHER',
        official_bid_line_id_verified=False,
        dates=len(idx.days), eligible_rows=idx.n,
        training_calendar_supplied=training_supplied,
        imputed_demand_buckets=len(imputed),
        volume_policy=args.volume_policy,
        before={k:before[k] for k in ('score_pp','total_vacations','annual_average_pp',
               'annual_worst_pp','annual_group_rate_SD_pp','dispatcher_annual_rate_SD_pp',
               'dispatcher_annual_vacation_day_SD','dispatcher_annual_vacation_day_min',
               'dispatcher_annual_vacation_day_max','soft_floor_deficit','days_below_target')},
        after={k:final[k] for k in ('score_pp','total_vacations','annual_average_pp',
               'annual_worst_pp','annual_group_rate_SD_pp','dispatcher_annual_rate_SD_pp',
               'dispatcher_annual_vacation_day_SD','dispatcher_annual_vacation_day_min',
               'dispatcher_annual_vacation_day_max','soft_floor_deficit','days_below_target')},
        unavoidable_shortage_buckets=len(coverage_deficits),
        vacation_rows=len(vacations), remaining_supply_rows=len(remaining),
        original_supply_rows=len(supply),
        dispatcher_vacation_min=min(tech_counts.values(), default=0) if not tech_group else
            min(tech_counts[t] for t in tech_group),
        dispatcher_vacation_max=max(tech_counts.values(), default=0),
        zero_vacation_dispatchers=sum(tech_counts[t]==0 for t in tech_group),
    )
    (outdir/'vacation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description='Shortage first + TECH_ID annual vacation-rate fairness only; daily 29 HARD, 25 SOFT')
    p.add_argument('--supply',default=DEFAULT_SUPPLY)
    p.add_argument('--demand',default=DEFAULT_DEMAND)
    p.add_argument('--volume',default=DEFAULT_VOLUME)
    p.add_argument('--training',default=DEFAULT_TRAINING)
    p.add_argument('--output-dir',default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--exclude-date',default=DEFAULT_EXCLUDED_DATE)
    p.add_argument('--demand-missing',choices=('impute','error'),default='impute')
    p.add_argument('--volume-policy',choices=('rank-cap','none'),default='rank-cap')
    p.add_argument('--volume-cap-slack',type=int,default=1)
    p.add_argument('--hard-max',type=int,default=29)
    p.add_argument('--soft-min',type=int,default=25)
    p.add_argument('--annual-worst-weight',type=float,default=1.0,
                   help='Weight on worst TECH_ID annual rate deviation in percentage points')
    p.add_argument('--annual-average-weight',type=float,default=1.0,
                   help='Weight on mean absolute TECH_ID annual rate deviation in percentage points')
    p.add_argument('--soft-min-weight',type=float,default=2.0,
                   help='PP penalty per mean daily vacation below the target (25 is SOFT)')
    p.add_argument('--lns-iterations',type=int,default=80)
    p.add_argument('--lns-seconds',type=float,default=6.0)
    p.add_argument('--lns-buckets',type=int,default=6)
    p.add_argument('--lns-max-buckets',type=int,default=12)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args(argv)
    if not (0<=args.soft_min<=args.hard_max and args.hard_max>0):
        p.error('Require 0 <= soft-min <= hard-max')
    if any(x<0 for x in (args.annual_worst_weight,args.annual_average_weight,args.soft_min_weight)):
        p.error('Fairness/target objective weights must be nonnegative')
    if args.annual_worst_weight+args.annual_average_weight<=0:
        p.error('At least one annual fairness weight must be positive')
    if args.lns_iterations<0 or args.lns_seconds<=0 or args.workers<1:
        p.error('Invalid LNS iteration/time/worker settings')
    if args.lns_buckets<2 or args.lns_max_buckets<args.lns_buckets:
        p.error('Invalid LNS neighborhood sizes')
    if args.volume_cap_slack<0:
        p.error('Volume cap slack must be >=0')

    supply=normalize_supply(csv_in(args.supply))
    if str(args.exclude_date).lower()!='none':
        excluded=pd.Timestamp(args.exclude_date).normalize()
        dropped=int((supply.SHIFT_DATE==excluded).sum())
        supply=supply.loc[supply.SHIFT_DATE!=excluded].copy()
        print(f'Excluded {excluded.date()} ({dropped} supply rows).')
    if supply.empty:
        raise ValueError('Supply empty after exclusions')
    source_cols=list(supply.columns)
    supply,group_kind=attach_dispatcher_groups(supply)
    train,training_supplied=training_keys(args.training)
    mask=[(str(r.TECH_ID),r.SHIFT_DATE) not in train
          for r in supply.itertuples(index=False)]
    allowed=supply.loc[mask].reset_index(drop=True)
    if allowed.empty:
        raise ValueError('No vacation-eligible work shifts after training exclusions')
    idx=Index(allowed)
    dates=sorted(idx.days)
    day_caps,volume=daily_caps(dates,args.volume,args.volume_policy,
                               args.volume_cap_slack,args.hard_max)
    day_caps={d:min(args.hard_max,day_caps[d],len(idx.days[d])) for d in dates}
    demand=normalize_demand(csv_in(args.demand))
    req,imputed=demand_map_with_audit(demand,supply,set(dates),args.demand_missing)
    # This is a PROOF of minimum shortage given fixed eligible supply and
    # coverage of each (date, work center, shift) independently.
    safe_caps={b:min(len(js),max(0,len(js)-req.get(b,0)))
               for b,js in idx.buckets.items()}
    shortage_min=sum(max(0,need-len(idx.buckets.get(b,()))) for b,need in req.items())
    print(f'Input: {idx.n:,} eligible shifts, {len(dates)} dates, {len(idx.groups)} TECH_ID fairness units.')
    print(f'PROVEN MINIMUM SHORTAGE = {shortage_min}; no vacation allocation can reduce this.')
    if shortage_min:
        print('To reduce the remaining shortage, investigate missing staffing/qualifications or run flex afterwards.')
    print(f'HARD <= {args.hard_max}/day; SOFT target {args.soft_min}/day. Annual vacations are VARIABLE.')
    print('FAIRNESS: individual TECH_ID annual vacation rates; no monthly fairness objective.')
    allocation,planned=seed_schedule(idx,safe_caps,day_caps,args.soft_min,
                                     np.random.default_rng(args.seed))
    assert shortage(idx,allocation,req)==shortage_min
    before=allocation.statistics(idx,args.soft_min,args.annual_worst_weight,
                                 args.annual_average_weight,args.soft_min_weight)
    print(f'Seed: {allocation.v} vacations; annual mean={before["annual_average_pp"]:.3f} pp, '
          f'worst={before["annual_worst_pp"]:.3f} pp, soft shortfall={before["soft_floor_deficit"]}')
    if args.lns_iterations:
        allocation,history=optimize(idx,allocation,safe_caps,day_caps,shortage_min,req,args)
    else:
        history=[]
    output=Path(args.output_dir)
    summary=export(idx,allocation,supply,source_cols,train,output,req,safe_caps,
                   day_caps,group_kind,volume,imputed,shortage_min,
                   before,args,planned,training_supplied)
    if history:
        pd.DataFrame(history).to_csv(output/'vacation_lns_history.csv',index=False)
    print(f'DONE: shortage={summary["final_coverage_shortage"]}; '
          f'vacations={summary["after"]["total_vacations"]}; '
          f'annual avg={summary["after"]["annual_average_pp"]:.3f}pp; '
          f'annual worst={summary["after"]["annual_worst_pp"]:.3f}pp; '
          f'individual vacation SD={summary["after"]["dispatcher_annual_vacation_day_SD"]:.3f} days')
    print(f'Outputs and shortage audit: {output.resolve()}')
    return summary


if __name__ == '__main__':
    main()
