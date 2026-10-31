"""Tests for the vacation allocation module (run:  pytest -q  from the folder that
contains ``vacation_allocation/``)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vacation_allocation import AllocationConfig, data_io, run_allocation, run_from_files  # noqa: E402
from vacation_allocation.allocator import InfeasibleAllocation, allocate, capacity_ok  # noqa: E402
from vacation_allocation.flex_capacity import FlexCapacity  # noqa: E402
from vacation_allocation.problem import build_problem  # noqa: E402
from vacation_allocation.volume_targets import (  # noqa: E402
    admissible_scales,
    daily_bounds_for_scale,
    fit_error,
    impute_volume,
)


# --------------------------------------------------------------------------
# synthetic data
# --------------------------------------------------------------------------
def make_instance(n_days=6, techs_per_group=3, seed=0, demand_slack=1):
    """Two work centres x two shifts; every dispatcher works ~70% of the days."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-03-03", periods=n_days)
    rows, tid = [], 1000
    groups = [("Domestic", "MORNING"), ("Domestic", "EVENING"), ("OC", "MORNING"), ("OC", "EVENING")]
    for wc, sh in groups:
        for _ in range(techs_per_group):
            tid += 1
            for d in dates:
                if rng.random() < 0.7:
                    rows.append(
                        {
                            "INDEX": np.nan, "START_TIME": f"{d:%m/%d/%Y} 05:00:00",
                            "WORK_CENTER_ID": wc, "TECH_ID": tid, "SHIFT_DATE": d,
                            "SHIFT_TYPE": sh, "DOMESTIC": 1,
                            "OC": int(wc == "OC" or tid % 2 == 0), "SF_LOAD_TMS": np.nan,
                        }
                    )  # fmt: skip
    supply = pd.DataFrame(rows)
    sched = supply.groupby(["SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"]).size()
    demand = sched.sub(demand_slack).clip(lower=0).rename("DEMAND").reset_index()
    demand = demand.rename(columns={"WORK_CENTER_ID": "WORK_CENTER"})
    volume = pd.DataFrame({"DATE": dates, "DEMAND": rng.integers(900, 1100, n_days)})
    return supply, demand, volume


SMALL = dict(daily_min=1, daily_max=3, flex_pricing_iterations=0, scale_grid_points=9)


# --------------------------------------------------------------------------
# volume targets
# --------------------------------------------------------------------------
def test_admissible_scales_and_daily_bounds():
    vol = np.array([1583.0, 2800, 3000, 3200, 3300, 3400, 3500, 3700])
    grid, k_star, e_star = admissible_scales(vol, 25, 29, 0.05, 11)
    dense = np.linspace(25 * vol.min(), 29 * vol.max(), 200_001)
    assert e_star <= min(fit_error(k, vol, 25, 29) for k in dense[::50]) + 1e-6  # global best
    assert all(fit_error(k, vol, 25, 29) <= e_star * 1.05 + 1e-9 for k in grid)
    assert k_star in grid
    only, k1, _ = admissible_scales(vol, 25, 29, 0.0, 11)
    assert len(only) == 1 and only[0] == k1
    ideal, lo, hi = daily_bounds_for_scale(k_star, vol, 25, 29)
    assert (lo >= 25).all() and (hi <= 29).all() and ((hi - lo) <= 1).all() and (lo <= ideal + 1e-9).all()
    order = np.argsort(vol)
    assert (np.diff(ideal[order]) <= 1e-12).all()  # busier day never gets a higher target
    free = (ideal > 25) & (ideal < 29)
    assert np.allclose((ideal * vol)[free], k_star)  # exactly K / volume where not clipped


def test_impute_volume_prefers_previous_week_then_next_then_errors():
    dates = list(pd.date_range("2025-01-01", periods=15))
    vol = pd.DataFrame({"DATE": dates[:14], "DEMAND": np.arange(100.0, 114.0)})
    series, audit = impute_volume(vol, dates)  # day 15 missing -> day 8 (7 days earlier)
    assert series.iloc[-1] == vol["DEMAND"].iloc[7]
    assert audit["DONOR_DATE"].iloc[0] == dates[7]
    # first day missing and there is a following week
    series, audit = impute_volume(vol.iloc[1:], dates[:14])
    assert series.iloc[0] == vol["DEMAND"].iloc[7]
    with pytest.raises(ValueError):
        impute_volume(vol.iloc[:3], dates[:14])


# --------------------------------------------------------------------------
# demand imputation (the flex module consumes the audit)
# --------------------------------------------------------------------------
def test_fill_missing_demand_audit_and_donor_rule():
    supply, demand, _ = make_instance(n_days=12, seed=0)
    supply = data_io.clean_supply(supply)
    demand = data_io.clean_demand(demand)
    dates = [pd.Timestamp(d) for d in sorted(supply["SHIFT_DATE"].unique())]
    drop = demand.index[(demand["SHIFT_DATE"] == dates[0]) & (demand["WORK_CENTER"] == "OC")]
    assert len(drop) > 0
    filled, audit = data_io.fill_missing_demand(demand.drop(drop), supply, dates)
    assert len(filled) == len(demand) and len(audit) == len(drop)
    assert list(audit.columns) == data_io.AUDIT_COLUMNS
    known = {(r.SHIFT_DATE, r.WORK_CENTER, r.SHIFT_TYPE): r.DEMAND for r in demand.itertuples()}
    for r in audit.itertuples():
        week_later = (dates[0] + pd.Timedelta(days=7), r.WORK_CENTER, r.SHIFT_TYPE)
        if week_later in known:  # no earlier week exists, so the following week must be used
            assert r.DONOR_DATE == week_later[0] and r.METHOD == "same_weekday"
            assert r.IMPUTED_DEMAND == known[week_later]
        else:
            assert r.METHOD == "nearest_date"


# --------------------------------------------------------------------------
# allocator
# --------------------------------------------------------------------------
def _problem(supply, demand, volume, training=frozenset()):
    sup = data_io.clean_supply(supply)
    dem = data_io.clean_demand(demand)
    dates = [pd.Timestamp(d) for d in sorted(sup["SHIFT_DATE"].unique())]
    dem, _ = data_io.fill_missing_demand(dem, sup, dates)
    return build_problem(sup, dem, dates, set(training)), dates


def test_allocation_meets_hard_requirements():
    supply, demand, volume = make_instance(n_days=8, techs_per_group=4, seed=3)
    cfg = AllocationConfig(**SMALL)
    res = run_allocation(supply, demand, volume, set(), cfg)
    daily = res.assignments.groupby("SHIFT_DATE").size()
    assert daily.between(1, 3).all() and len(daily) == 8
    counts = res.assignments.groupby("TECH_ID").size().reindex(res.fairness["TECH_ID"], fill_value=0)
    assert counts.max() - counts.min() <= 1
    # each vacation is an existing workday, used once; supply is partitioned
    keys = ["TECH_ID", "SHIFT_DATE"]
    assert not res.assignments.duplicated(keys).any()
    assert len(res.assignments) + len(res.remaining) == len(supply)
    merged = res.assignments.merge(res.remaining, on=keys)
    assert merged.empty


def test_flow_is_optimal_against_brute_force():
    """Shortage first, then most vacations: the flow must hit the true optimum."""
    supply, demand, volume = make_instance(n_days=3, techs_per_group=2, seed=11, demand_slack=1)
    cfg = AllocationConfig(daily_min=1, daily_max=3, workday_tie_break=False, flex_pricing_iterations=0)
    problem, _ = _problem(supply, demand, volume)
    n_rows, n = len(problem.supply), problem.n_tech
    assert n_rows <= 22, "keep the brute force small"
    day_lo = np.array([1, 2, 1])
    day_hi = np.array([2, 3, 2])
    level = 0  # every dispatcher 0 or 1 day
    price = np.full(problem.n_bucket, 1000)
    sol = allocate(problem, day_lo, day_hi, level, price, cfg)
    got = (int(problem.shortage(sol.per_bucket).sum()), -int(sol.selected.sum()))

    best = None
    for total in range(int(day_lo.sum()), int(day_hi.sum()) + 1):
        for combo in itertools.combinations(range(n_rows), total):
            sel = np.zeros(n_rows, bool)
            sel[list(combo)] = True
            per_day = np.bincount(problem.row_day[sel], minlength=problem.n_day)
            per_tech = np.bincount(problem.row_tech[sel], minlength=n)
            if (per_day < day_lo).any() or (per_day > day_hi).any():
                continue
            if (per_tech < level).any() or (per_tech > level + 1).any():
                continue
            s = int(problem.shortage(np.bincount(problem.row_bucket[sel], minlength=problem.n_bucket)).sum())
            best = (s, -total) if best is None else min(best, (s, -total))
    assert best is not None
    assert got == best


def test_training_days_never_become_vacation_and_reduce_coverage():
    supply, demand, volume = make_instance(n_days=8, techs_per_group=4, seed=5)
    sup = data_io.clean_supply(supply)
    pick = sup.sample(15, random_state=1)
    training = set(zip(pick["TECH_ID"], pick["SHIFT_DATE"]))
    res = run_allocation(supply, demand, volume, training, AllocationConfig(**SMALL))
    taken = set(zip(res.assignments["TECH_ID"], res.assignments["SHIFT_DATE"]))
    assert not (taken & training)
    assert res.stats["inputs"]["training_validated"] is True
    # training rows lower the productive supply of their bucket
    problem, _ = _problem(supply, demand, volume, training)
    assert problem.bucket_training.sum() == 15


def test_infeasible_requirements_are_reported():
    supply, demand, volume = make_instance(n_days=4, techs_per_group=1, seed=2)
    problem, _ = _problem(supply, demand, volume)
    cfg = AllocationConfig(daily_min=40, daily_max=50)
    assert "fewer eligible workdays" in capacity_ok(problem, cfg, np.full(problem.n_day, 40), 0)
    with pytest.raises(InfeasibleAllocation):
        run_allocation(supply, demand, volume, set(), AllocationConfig(daily_min=40, daily_max=50,
                                                                       flex_pricing_iterations=0))


def test_total_is_chosen_by_shortage_then_maximised():
    # no slack anywhere: every vacation creates shortage -> fewest vacations allowed
    supply, demand, volume = make_instance(n_days=10, techs_per_group=4, seed=8, demand_slack=0)
    tight = run_allocation(supply, demand, volume, set(), AllocationConfig(**SMALL))
    log = tight.search_log[tight.search_log["status"] == "solved"]
    # objective: least unmet demand after vacations (flex ignored), then most vacations
    assert tight.stats["coverage"]["shortage_after_vacation"] == log["shortage_after_vacation"].min()
    best = log[log["shortage_after_vacation"] == log["shortage_after_vacation"].min()]
    assert len(tight.assignments) == best["vacations"].max()
    # plenty of slack: no shortage possible -> as many vacations as R1-R3 allow
    supply, demand, volume = make_instance(n_days=10, techs_per_group=4, seed=8, demand_slack=6)
    loose = run_allocation(supply, demand, volume, set(), AllocationConfig(**SMALL))
    log = loose.search_log[loose.search_log["status"] == "solved"]
    assert loose.stats["coverage"]["shortage_after_vacation"] == 0
    assert len(loose.assignments) == log["vacations"].max()
    assert len(loose.assignments) > len(tight.assignments)


def test_relaxed_fairness_trades_equal_split_for_less_shortage():
    supply, demand, volume = make_instance(n_days=10, techs_per_group=5, seed=9, demand_slack=0)
    strict = run_allocation(supply, demand, volume, set(), AllocationConfig(**SMALL))
    loose = run_allocation(
        supply, demand, volume, set(),
        AllocationConfig(fairness_slack_days=3, fairness_weight=1, **SMALL),
    )  # fmt: skip
    s, l = strict.stats["coverage"], loose.stats["coverage"]
    assert strict.stats["requirements"]["per_dispatcher_spread"] <= 1
    assert l["shortage_after_vacation"] <= s["shortage_after_vacation"]


# --------------------------------------------------------------------------
# flex headroom: exact max-flow agrees with an independent implementation
# --------------------------------------------------------------------------
def test_flex_capacity_matches_independent_maxflow():
    nx = pytest.importorskip("networkx")
    supply, demand, volume = make_instance(n_days=6, techs_per_group=4, seed=21, demand_slack=0)
    problem, _ = _problem(supply, demand, volume)
    cfg = AllocationConfig(max_flex_days=2, **{k: v for k, v in SMALL.items()})
    flex = FlexCapacity(problem, cfg)
    rng = np.random.default_rng(0)
    for _ in range(5):
        vac = rng.integers(0, 3, problem.n_bucket)
        vac = np.minimum(vac, problem.bucket_eligible)
        short = problem.shortage(vac)
        got = flex.evaluate(short)

        g = nx.DiGraph()
        for b in np.flatnonzero(short > 0):
            wc, sh, d = problem.bucket_wc[b], problem.bucket_shift[b], problem.bucket_day[b]
            for t in np.flatnonzero(problem.tech_quals[wc] & (np.array(problem.tech_home_wc) == wc)):
                if sh in problem.tech_shifts[t] and not problem.busy[t, d]:
                    g.add_edge(("T", t), ("TD", t, d), capacity=1)
                    g.add_edge(("TD", t, d), ("B", b), capacity=1)
            g.add_edge(("B", b), "snk", capacity=int(short[b]))
        for t in range(problem.n_tech):
            if ("T", t) in g:
                g.add_edge("src", ("T", t), capacity=cfg.max_flex_days)
        expected = nx.maximum_flow_value(g, "src", "snk") if "snk" in g and "src" in g else 0
        assert got.covered == expected
        assert got.residual == got.raw_shortage - expected


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def test_run_from_files_writes_expected_outputs(tmp_path):
    supply, demand, volume = make_instance(n_days=8, techs_per_group=4, seed=4)
    supply.to_csv(tmp_path / "supply.csv", index=False)
    demand.to_csv(tmp_path / "demand.csv", index=False)
    volume.to_csv(tmp_path / "volume.csv", index=False)
    res, paths = run_from_files(
        tmp_path / "supply.csv", tmp_path / "demand.csv", tmp_path / "volume.csv",
        tmp_path / "out", None, AllocationConfig(**SMALL),
    )  # fmt: skip
    vac = pd.read_csv(paths["vacation"])
    rem = pd.read_csv(paths["remaining"])
    original = pd.read_csv(tmp_path / "supply.csv")
    assert list(vac.columns) == list(original.columns) == list(rem.columns)
    assert len(vac) + len(rem) == len(original)
    assert vac["SHIFT_DATE"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()  # YYYY-MM-DD like the reference
    assert vac["INDEX"].isna().all()
    for name in ("audit", "summary", "targets", "fairness", "shortage"):
        assert paths[name].exists()


def test_validator_rejects_broken_allocations():
    """The independent checker must actually fail when a requirement is violated."""
    from vacation_allocation.validation import validate_allocation

    supply, demand, volume = make_instance(n_days=8, techs_per_group=4, seed=3)
    cfg = AllocationConfig(**SMALL)
    res = run_allocation(supply, demand, volume, set(), cfg)
    sup = data_io.clean_supply(supply)
    total = len(res.assignments)
    ok = validate_allocation(sup, res.assignments, res.remaining, cfg, set(), total)
    assert ok["daily_min"] >= 1

    # (a) a whole day emptied of vacations violates R1
    day = res.assignments["SHIFT_DATE"].iloc[0]
    moved = res.assignments[res.assignments["SHIFT_DATE"] == day]
    bad_a = res.assignments.drop(moved.index)
    bad_r = pd.concat([res.remaining, moved]).sort_values(["SHIFT_DATE", "TECH_ID"])
    with pytest.raises(AssertionError):
        validate_allocation(sup, bad_a, bad_r, cfg, set(), total)

    # (b) the same workday used twice violates the partition
    with pytest.raises(AssertionError):
        validate_allocation(sup, pd.concat([res.assignments, res.assignments.iloc[:1]]), res.remaining, cfg, set(), total)

    # (c) giving one dispatcher many extra days violates R2
    tech = res.assignments["TECH_ID"].iloc[0]
    extra = res.remaining[res.remaining["TECH_ID"] == tech]
    assert len(extra) >= 2
    bad_a = pd.concat([res.assignments, extra.iloc[:2]])
    bad_r = res.remaining.drop(extra.index[:2])
    with pytest.raises(AssertionError):
        validate_allocation(sup, bad_a, bad_r, cfg, set(), len(bad_a))


def test_shortage_goal_relaxes_rules_in_the_agreed_order():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shortage_goal import ladder

    steps = list(ladder(False, 2, 23, AllocationConfig()))
    assert [s[1] for s in steps] == [
        "all rules", "R3 tolerance 0.5", "R3 off",
        "band +-1 days", "band +-2 days", "daily_min 24", "daily_min 23",
    ]  # fmt: skip
    assert [s[0] for s in steps] == ["none", "R3", "R3", "R2", "R2", "R1", "R1"]
    assert all(not c.enforce_proportionality for _, label, c in steps[2:])
    assert steps[-1][2].fairness_slack_days == 2 and steps[-1][2].daily_min == 23
    kept = list(ladder(True, 1, 24, AllocationConfig()))
    assert [s[0] for s in kept] == ["none", "R2", "R1"]
    assert all(c.enforce_proportionality for _, _, c in kept)


def test_base_level_fixes_or_bounds_the_total():
    supply, demand, volume = make_instance(n_days=10, techs_per_group=4, seed=8, demand_slack=0)
    fixed = run_allocation(supply, demand, volume, set(), AllocationConfig(base_daily=2, **SMALL))
    assert len(fixed.assignments) == 2 * 10
    band = run_allocation(
        supply, demand, volume, set(),
        AllocationConfig(base_daily=2, base_tolerance=0.5, enforce_proportionality=False, **SMALL),
    )  # fmt: skip
    assert 15 <= len(band.assignments) <= 25
    fixed_r3off = run_allocation(
        supply, demand, volume, set(),
        AllocationConfig(base_daily=2, enforce_proportionality=False, **SMALL),
    )  # fmt: skip
    assert len(fixed_r3off.assignments) == 20
    # the band contains the fixed total, so it can only do as well or better
    b, f = band.stats["coverage"], fixed_r3off.stats["coverage"]
    key = lambda c: (c["residual_shortage_after_best_case_flex"], c["shortage_after_vacation"])  # noqa: E731
    assert key(b) <= key(f)
    with pytest.raises(ValueError):
        AllocationConfig(base_daily=40, **SMALL).validate()


# --------------------------------------------------------------------------
# flex assignment
# --------------------------------------------------------------------------
def _flex_case(seed=4, max_flex=2):
    from vacation_allocation.problem import clean_preflex

    supply, demand, volume = make_instance(n_days=8, techs_per_group=4, seed=seed, demand_slack=0)
    sup = data_io.clean_supply(supply)
    techs = sorted(sup["TECH_ID"].unique())
    dates = sorted(sup["SHIFT_DATE"].unique())
    work = set(zip(sup["TECH_ID"], sup["SHIFT_DATE"]))
    off = [(t, d) for t in techs for d in dates if (t, d) not in work]
    training = set(off[::5])
    pre = [k for k in off if k not in training][1::7]
    home = sup.groupby("TECH_ID")[["WORK_CENTER_ID", "SHIFT_TYPE"]].first()
    pre_rows = [{"TECH_ID": t, "SHIFT_DATE": d, "WORK_CENTER_ID": home.loc[t, "WORK_CENTER_ID"],
                 "SHIFT_TYPE": home.loc[t, "SHIFT_TYPE"]} for t, d in pre]  # fmt: skip
    t0, d0 = sorted(work)[0]  # one conflicting row: already a workday
    pre_rows.append({"TECH_ID": t0, "SHIFT_DATE": d0, "WORK_CENTER_ID": "Domestic", "SHIFT_TYPE": "MORNING"})
    preflex = clean_preflex(pd.DataFrame(pre_rows))
    cfg = AllocationConfig(max_flex_days=max_flex, **SMALL)
    dem = data_io.clean_demand(demand)
    dts = [pd.Timestamp(d) for d in dates]
    dem, _ = data_io.fill_missing_demand(dem, sup, dts)
    return sup, demand, volume, training, preflex, cfg, build_problem(sup, dem, dts, training, preflex)


def test_preassigned_flex_budget_cover_and_conflicts():
    sup, demand, volume, training, preflex, cfg, problem = _flex_case()
    assert len(problem.preflex_conflicts) == 1
    assert problem.preflex_conflicts["REASON"].iloc[0] == "already a scheduled workday"
    assert problem.bucket_extra.sum() == len(preflex) - 1  # valid ones add a person
    assert problem.tech_preflex.sum() == len(preflex)  # all of them use budget
    flex = FlexCapacity(problem, cfg)
    assert (flex.budget == np.maximum(0, 2 - problem.tech_preflex)).all()


def test_flex_assignment_rules_and_optimality():
    nx = pytest.importorskip("networkx")
    from vacation_allocation.flex_assignment import FlexConfig, assign_flex

    sup, demand, volume, training, preflex, cfg, problem = _flex_case()
    flex = FlexCapacity(problem, cfg)
    rng = np.random.default_rng(1)
    vac = np.minimum(rng.integers(0, 3, problem.n_bucket), problem.bucket_eligible)
    short = problem.shortage(vac)
    fa = assign_flex(problem, flex, short, FlexConfig())
    # rules
    pk = set(zip(preflex["TECH_ID"], preflex["SHIFT_DATE"]))
    work = set(zip(sup["TECH_ID"], sup["SHIFT_DATE"]))
    keys = [(problem.tech_ids[t], problem.dates[d]) for t, d in zip(fa.tech, fa.day)]
    assert len(set(keys)) == len(keys)
    assert not (set(keys) & (work | training | pk))
    assert ((fa.per_tech + problem.tech_preflex)[fa.per_tech > 0] <= 2).all()
    for t, b in zip(fa.tech, fa.bucket):
        assert problem.bucket_wc[b] == problem.tech_home_wc[t]  # home desk only
        assert problem.bucket_shift[b] in problem.tech_shifts[t]
    assert (fa.per_bucket <= short).all()
    # optimal: equals an independent max-flow
    g = nx.DiGraph()
    for b in np.flatnonzero(short > 0):
        for t in range(problem.n_tech):
            d = problem.bucket_day[b]
            if (problem.tech_home_wc[t] == problem.bucket_wc[b] and problem.bucket_shift[b] in problem.tech_shifts[t]
                    and not problem.busy[t, d]):  # fmt: skip
                g.add_edge(("T", t), ("TD", t, d), capacity=1)
                g.add_edge(("TD", t, d), ("B", b), capacity=1)
        g.add_edge(("B", b), "snk", capacity=int(short[b]))
    for t in range(problem.n_tech):
        if ("T", t) in g:
            g.add_edge("src", ("T", t), capacity=int(max(0, 2 - problem.tech_preflex[t])))
    best = nx.maximum_flow_value(g, "src", "snk") if "src" in g else 0
    assert fa.covered == best == flex.evaluate(short).covered


def test_flex_pipeline_end_to_end(tmp_path):
    from vacation_allocation.flex_pipeline import run_flex, write_flex
    from vacation_allocation.flex_assignment import FlexConfig

    sup, demand, volume, training, preflex, cfg, problem = _flex_case()
    vac = run_allocation(sup, demand, volume, training, cfg, preflex)
    res = run_flex(sup, vac.assignments, demand, training, preflex, cfg, FlexConfig())
    s = res["summary"]
    assert s["covered_by_new_flex"] == vac.stats["coverage"]["best_case_flex_coverable"]
    assert s["shortage_after_flex"] == vac.stats["coverage"]["residual_shortage_after_best_case_flex"]
    assert len(res["combined"]) == len(preflex) + s["new_flex_days"]
    assert len(res["final"]) == len(sup) - len(vac.assignments) + (len(preflex) - 1) + s["new_flex_days"]
    paths = write_flex(res, tmp_path)
    assert all(p.exists() for p in paths.values())


# --------------------------------------------------------------------------
# YAML settings
# --------------------------------------------------------------------------
def test_yaml_settings(tmp_path):
    from vacation_allocation.settings import load_settings

    p = tmp_path / "c.yaml"
    p.write_text("""
vacation:
  rule1_daily_total: {enabled: true, min: 25, max: 29, tolerance_below: 2, tolerance_above: 1}
  rule2_fairness: {slack_days: 3, weight: 50}
  rule3_inverse_volume: {enabled: false, tolerance: 0.2}
  base_per_day: {enabled: true, value: 27, tolerance: 0.5}
flex: {max_days_per_dispatcher: 3, same_shift_only: false, home_work_center_only: false, balance_cost: 5}
""")
    s = load_settings(p)
    v = s.vacation
    assert (v.daily_min, v.daily_max, v.fairness_slack_days, v.fairness_weight) == (23, 30, 3, 50)
    assert not v.enforce_proportionality and v.base_daily == 27 and v.base_tolerance == 0.5
    assert v.max_flex_days == 3 and not v.flex_same_shift_only and not v.flex_home_work_center_only
    assert s.flex.balance_cost == 5
    p.write_text("files: {supply: s.csv}")  # file paths are no longer configured in YAML
    with pytest.raises(ValueError, match="unknown top-level"):
        load_settings(p)
    p.write_text("vacation: {rule2_fairness: {slak_days: 3}}")
    with pytest.raises(ValueError, match="unknown key"):
        load_settings(p)
    shipped = Path(__file__).resolve().parents[1] / "vacation_flex_config.yaml"
    if shipped.exists():
        load_settings(shipped)


def test_rule1_can_be_switched_off():
    supply, demand, volume = make_instance(n_days=6, techs_per_group=3, seed=2, demand_slack=0)
    r = run_allocation(supply, demand, volume, set(),
                       AllocationConfig(enforce_daily_bounds=False, enforce_proportionality=False,
                                        base_daily=1, base_tolerance=0, flex_pricing_iterations=0))  # fmt: skip
    assert len(r.assignments) == 6  # base 1/day x 6 days; daily totals unconstrained


def test_secondary_quals_used_only_when_allowed():
    from vacation_allocation.flex_assignment import FlexConfig, assign_flex

    sup, demand, volume, training, preflex, cfg, problem = _flex_case()
    short = problem.shortage(np.minimum(3, problem.bucket_eligible))
    home_only = assign_flex(problem, FlexCapacity(problem, cfg), short, FlexConfig())
    assert all(problem.bucket_wc[b] == problem.tech_home_wc[t] for t, b in zip(home_only.tech, home_only.bucket))
    import dataclasses
    wide_cfg = dataclasses.replace(cfg, flex_home_work_center_only=False)
    wide = assign_flex(problem, FlexCapacity(problem, wide_cfg), short, FlexConfig())
    assert wide.covered >= home_only.covered
    assert any(problem.bucket_wc[b] != problem.tech_home_wc[t] for t, b in zip(wide.tech, wide.bucket))


def test_base_unreachable_within_r3_tolerance_widens_r3():
    vol = np.array([1583.0, 2800, 3000, 3200, 3300, 3400, 3500, 3700] * 10)
    nd = vol.size
    info = {}
    target = 28 * nd  # far from the best fit for this data: needs a looser R3
    grid, k, e = admissible_scales(vol, 24, 29, 0.05, 11, (target, target), info)
    assert grid.size > 0 and info["effective_tolerance"] > 0.05
    for kk in grid:  # every scale returned can actually reach the base exactly
        _, lo, hi = daily_bounds_for_scale(kk, vol, 24, 29)
        assert lo.sum() <= target <= hi.sum()
        assert fit_error(kk, vol, 24, 29) <= e * (1 + info["effective_tolerance"]) + 1e-9
    info2 = {}
    admissible_scales(vol, 24, 29, 0.05, 11, None, info2)
    assert info2["effective_tolerance"] == 0.05  # untouched without a base


def test_merged_schedule_days_never_get_flex():
    from vacation_allocation.flex_assignment import FlexConfig
    from vacation_allocation.flex_pipeline import run_flex

    sup, demand, volume, _, preflex, cfg, _ = _flex_case()
    techs = sorted(sup["TECH_ID"].unique())
    dates = sorted(sup["SHIFT_DATE"].unique())
    work = set(zip(sup["TECH_ID"], sup["SHIFT_DATE"]))
    pk = set(zip(preflex["TECH_ID"], preflex["SHIFT_DATE"]))
    off = [(t, d) for t in techs for d in dates if (t, d) not in work and (t, d) not in pk]
    blocked = work | set(off[::2])  # merged schedule = workdays + "training" on half the off-days
    vac = run_allocation(sup, demand, volume, set(), cfg, preflex, blocked)
    res = run_flex(sup, vac.assignments, demand, set(), preflex, cfg, FlexConfig(), blocked)
    keys = set(zip(res["new"]["TECH_ID"], res["new"]["SHIFT_DATE"]))
    assert len(keys) > 0 and not (keys & blocked)
    assert res["summary"]["covered_by_new_flex"] == vac.stats["coverage"]["best_case_flex_coverable"]
    # regular supply is untouched by the merged schedule
    assert len(vac.assignments) + len(vac.remaining) == len(sup)
