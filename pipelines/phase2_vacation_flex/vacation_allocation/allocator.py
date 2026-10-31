"""Exact vacation allocation as a minimum-cost network flow.

Network (one unit of flow = one vacation day)::

    SRC ──▶ dispatcher ──▶ bucket(date, work centre, shift) ──▶ date ──▶ SNK
     │       V_t units       one arc per scheduled workday        N_d units  ▲
     └──────────────────────────── unused capacity ───────────────────────────┘

* dispatcher -> bucket arcs have capacity 1: at most one vacation per workday,
  and only on a day the dispatcher actually works.
* **R2** Each dispatcher must send ``L`` units (forced) and may send one more
  (optional), so every count is L or L+1.  (With ``fairness_slack_days = k``
  the range widens to [L-k, L+1+k] with a squared penalty.)
* **R1/R3** Each date must receive ``day_lo[d]`` units and may receive up to
  ``day_hi[d]`` -- the floor and ceiling of the proportional target K/v_d,
  which already lie inside 25..29.
* **Shortage** Each bucket has two parallel arcs to its date: the first
  ``free`` units cost nothing (more people than desk demand), further units
  cost the bucket's shortage price.
* **Maximise vacations** Every optional unit earns a small reward, smaller than
  the cheapest shortage could ever be offset by, so a vacation that creates a
  shortage is only taken when R1-R3 force it.
* The SRC -> SNK bypass absorbs optional capacity that is not used, which is
  what lets the total float.

All costs are convex and capacities integral, so the optimum is integral and
exact (well under a second at full size).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ortools.graph.python import min_cost_flow

from .config import AllocationConfig
from .problem import Problem

log = logging.getLogger(__name__)

PRICE_SCALE = 1_000  # internal cost of one config "price unit"
VACATION_REWARD = 100  # internal reward per optional vacation taken
TIE_BREAK_MAX = 50  # cap on the workday tie-break, below the reward


@dataclass
class FlowSolution:
    selected: np.ndarray  # bool per supply row: turned into vacation
    per_tech: np.ndarray
    per_day: np.ndarray
    per_bucket: np.ndarray
    objective: int


class InfeasibleAllocation(RuntimeError):
    pass


def level_range(problem: Problem, cfg: AllocationConfig, day_lo, day_hi) -> range:
    """Fairness levels L for which counts L..L+1 can meet the day totals.

    With fairness_slack_days > 0, L is only the centre of the wider band
    [L-k, L+1+k], so the same centres are enough.
    """
    n = problem.n_tech
    a, b = int(np.sum(day_lo)), int(np.sum(day_hi))
    first = max(0, -(-(a - n) // n))  # n*(L+1) >= a
    last = b // n  # n*L <= b
    return range(first, last + 1)


def capacity_ok(problem: Problem, cfg: AllocationConfig, day_lo, level: int) -> str | None:
    """Reason the candidate is impossible, or None."""
    elig_day = np.bincount(problem.row_day[problem.row_eligible], minlength=problem.n_day)
    bad = np.flatnonzero(elig_day < day_lo)
    if bad.size:
        d = bad[0]
        return (
            f"{bad.size} date(s) have fewer eligible workdays than their required "
            f"vacations, e.g. {problem.dates[d].date()}: {elig_day[d]} < {day_lo[d]}"
        )
    floor = max(0, level - cfg.fairness_slack_days)
    poor = np.flatnonzero(problem.tech_eligible_days < floor)
    if poor.size:
        t = poor[0]
        return (
            f"{poor.size} dispatcher(s) have fewer eligible workdays than {floor}, "
            f"e.g. {problem.tech_ids[t]} has {problem.tech_eligible_days[t]}"
        )
    return None


def allocate(
    problem: Problem,
    day_lo: np.ndarray,
    day_hi: np.ndarray,
    level: int,
    bucket_price: np.ndarray,
    cfg: AllocationConfig,
    total_range: tuple[int, int] | None = None,
) -> FlowSolution:
    """Solve one min-cost-flow instance for a given scale (day_lo/day_hi) and level L.

    ``total_range=(lo, hi)`` additionally bounds the annual total: optional day
    units then pass through one TOT node that must carry between
    lo - sum(day_lo) and hi - sum(day_lo) units.
    """
    nt, nb, nd = problem.n_tech, problem.n_bucket, problem.n_day
    k = cfg.fairness_slack_days
    TECH0, BUCK0 = 0, nt
    DAY0 = BUCK0 + nb
    SRC = DAY0 + nd
    SNK = SRC + 1
    TOT = SNK + 1  # used only with total_range
    forced_lo = int(np.sum(day_lo))
    tot_min = tot_max = None
    if total_range is not None:
        tot_min = max(0, int(total_range[0]) - forced_lo)
        tot_max = int(total_range[1]) - forced_lo
        if tot_max < tot_min:
            raise InfeasibleAllocation("total range is below the forced daily minimum")

    tails, heads, caps, costs = [], [], [], []

    def add(t, h, c, w):
        tails.append(np.asarray(t, dtype=np.int64))
        heads.append(np.asarray(h, dtype=np.int64))
        caps.append(np.asarray(c, dtype=np.int64))
        costs.append(np.asarray(w, dtype=np.int64))

    # 1. dispatcher -> bucket: one arc per eligible scheduled workday
    rows = np.flatnonzero(problem.row_eligible)
    add(
        TECH0 + problem.row_tech[rows],
        BUCK0 + problem.row_bucket[rows],
        np.ones(rows.size),
        np.zeros(rows.size),
    )
    n_row_arcs = rows.size

    # 2. bucket -> date: free units, then shortage-priced units
    free = problem.bucket_free
    excess = problem.bucket_eligible - free
    b_free = np.flatnonzero(free > 0)
    add(BUCK0 + b_free, DAY0 + problem.bucket_day[b_free], free[b_free], np.zeros(b_free.size))
    b_exc = np.flatnonzero(excess > 0)
    add(
        BUCK0 + b_exc,
        DAY0 + problem.bucket_day[b_exc],
        excess[b_exc],
        PRICE_SCALE * np.asarray(bucket_price, dtype=np.int64)[b_exc],
    )

    # 3. date -> sink: optional units between floor and ceiling of the target
    opt = np.asarray(day_hi) - np.asarray(day_lo)
    d_opt = np.flatnonzero(opt > 0)
    day_sink = SNK if total_range is None else TOT
    add(DAY0 + d_opt, np.full(d_opt.size, day_sink), opt[d_opt], np.zeros(d_opt.size))
    if total_range is not None:  # always added (even at capacity 0) so TOT exists
        add([TOT], [SNK], [tot_max - tot_min], [0])

    # 4. source -> dispatcher: optional units above the guaranteed minimum
    base_cnt = max(0, level - k)
    n_inc = level + 1 + k - base_cnt
    v_old = base_cnt + np.arange(n_inc)
    centre = level + 0.5
    fair = cfg.fairness_weight * PRICE_SCALE * ((v_old + 1 - centre) ** 2 - (v_old - centre) ** 2)
    marg = np.rint(fair).astype(np.int64)[None, :].repeat(nt, axis=0) - VACATION_REWARD
    if cfg.workday_tie_break:
        extra = np.minimum(problem.tech_workdays - problem.tech_workdays.min(), TIE_BREAK_MAX)
        marg = marg - extra[:, None]
    t_ix = np.repeat(np.arange(nt), n_inc)
    add(np.full(t_ix.size, SRC), TECH0 + t_ix, np.ones(t_ix.size), marg.ravel())

    # 5. bypass for optional capacity that is not used
    spare = nt * n_inc
    add([SRC], [SNK], [spare], [0])

    tails, heads = np.concatenate(tails), np.concatenate(heads)
    caps, costs = np.concatenate(caps), np.concatenate(costs)

    smcf = min_cost_flow.SimpleMinCostFlow()
    arc_ids = smcf.add_arcs_with_capacity_and_unit_cost(tails, heads, caps, costs)

    n_nodes = SNK + 1 if total_range is None else TOT + 1
    supply = np.zeros(n_nodes, dtype=np.int64)
    supply[TECH0 : TECH0 + nt] = base_cnt
    supply[SRC] = spare
    supply[DAY0 : DAY0 + nd] = -np.asarray(day_lo, dtype=np.int64)
    supply[SNK] = -(spare + nt * base_cnt - forced_lo)
    if total_range is not None:  # TOT keeps tot_min units: total >= range lower bound
        supply[TOT] = -tot_min
        supply[SNK] += tot_min
    if supply[SNK] > 0:
        raise InfeasibleAllocation("fairness level too low for the required daily totals")
    smcf.set_nodes_supplies(np.arange(n_nodes), supply)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise InfeasibleAllocation(
            f"min-cost-flow status {status.name}: the daily totals, fairness level and "
            "dispatcher availability cannot all be satisfied together"
        )

    flow = smcf.flows(arc_ids[:n_row_arcs])
    selected = np.zeros(len(problem.row_tech), dtype=bool)
    selected[rows[flow > 0]] = True
    return FlowSolution(
        selected=selected,
        per_tech=np.bincount(problem.row_tech[selected], minlength=nt),
        per_day=np.bincount(problem.row_day[selected], minlength=nd),
        per_bucket=np.bincount(problem.row_bucket[selected], minlength=nb),
        objective=int(smcf.optimal_cost()),
    )
