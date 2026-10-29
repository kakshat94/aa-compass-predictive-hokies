"""Optimal flex-day assignment after vacations.

A flex day turns a dispatcher's *off-day* into a workday on a desk that would
otherwise be short.  Rules:

* only on a true off-day: not a scheduled workday (vacation days are scheduled
  workdays), not a training day, not a day that already has pre-assigned flex;
* by default only on the dispatcher's own WORK_CENTER_ID (not on desks held
  only as a secondary qualification) and only on the shift they normally work;
* at most one flex day per dispatcher per date;
* each dispatcher's remaining budget is ``max_flex_days`` minus their
  pre-assigned flex days.

Model: a minimum-cost flow on the same candidate arcs the vacation step uses
to estimate flex headroom (``FlexCapacity``)::

    SRC ─(budget, k-th day costs balance·(2k-1))─▶ dispatcher ─(1)─▶ [dispatcher, date]
        ─(1, cost if other desk / other shift)─▶ bucket ─(shortage, reward per unit)─▶ SNK

The reward per covered unit exceeds every other cost, so the solution first
covers as much shortage as possible (the max-flow value the vacation step
reported), then prefers each dispatcher's own desk and shift, then spreads flex
days evenly.  Capacities are integral and costs convex, so it is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.graph.python import min_cost_flow

from .flex_capacity import FlexCapacity
from .problem import Problem


@dataclass(frozen=True)
class FlexConfig:
    cross_work_center_cost: int = (
        1_000  # per flex day on a desk other than the home desk
    )
    cross_shift_cost: int = 1_000  # per flex day on another shift (only if allowed)
    balance_cost: int = 10  # k-th flex day of one dispatcher costs balance_cost*(2k-1)

    def validate(self) -> None:
        if (
            min(self.cross_work_center_cost, self.cross_shift_cost, self.balance_cost)
            < 0
        ):
            raise ValueError("flex costs must be >= 0")


@dataclass
class FlexAssignment:
    tech: np.ndarray  # dispatcher code per new flex day
    day: np.ndarray
    bucket: np.ndarray
    per_bucket: np.ndarray
    per_tech: np.ndarray
    covered: int


def assign_flex(
    problem: Problem, flex: FlexCapacity, shortage: np.ndarray, fcfg: FlexConfig
) -> FlexAssignment:
    fcfg.validate()
    sh = np.asarray(shortage, dtype=np.int64)
    nt, nb, nd = problem.n_tech, problem.n_bucket, problem.n_day

    # keep only candidate arcs into buckets that are actually short
    keep = sh[flex.cand_b] > 0
    ct, cb, cd = flex.cand_t[keep], flex.cand_b[keep], flex.cand_d[keep]
    if ct.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return FlexAssignment(
            z, z, z, np.zeros(nb, np.int64), np.zeros(nt, np.int64), 0
        )

    home = np.array(problem.tech_home_wc, dtype=object)
    bwc = np.array(problem.bucket_wc, dtype=object)
    bsh = np.array(problem.bucket_shift, dtype=object)
    cross_wc = home[ct] != bwc[cb]
    cross_sh = np.array(
        [bsh[b] not in problem.tech_shifts[t] for t, b in zip(ct, cb)], dtype=bool
    )
    arc_cost = cross_wc * fcfg.cross_work_center_cost + cross_sh * fcfg.cross_shift_cost

    budget = flex.budget
    reward = 1 + int(fcfg.cross_work_center_cost + fcfg.cross_shift_cost
                     + fcfg.balance_cost * (2 * max(1, int(budget.max())) - 1))  # fmt: skip
    reward *= 1_000  # covering one more unit always beats any mix of secondary costs

    key = ct * nd + cd
    uniq, inv = np.unique(key, return_inverse=True)
    SRC, SNK = 0, 1
    T0 = 2
    TD0 = T0 + nt
    B0 = TD0 + uniq.size
    n_nodes = B0 + nb

    tails, heads, caps, costs = [], [], [], []

    def add(t, h, c, w):
        tails.append(np.asarray(t, dtype=np.int64))
        heads.append(np.asarray(h, dtype=np.int64))
        caps.append(np.asarray(c, dtype=np.int64))
        costs.append(np.asarray(w, dtype=np.int64))

    techs = np.unique(ct)
    # source -> dispatcher: one arc per budget unit, convex cost spreads flex days
    for k in range(1, int(budget.max()) + 1):
        t_k = techs[budget[techs] >= k]
        add(
            np.full(t_k.size, SRC),
            T0 + t_k,
            np.ones(t_k.size),
            np.full(t_k.size, fcfg.balance_cost * (2 * k - 1)),
        )
    # dispatcher -> [dispatcher, date] (at most one flex per date)
    td_tech = uniq // nd
    add(
        T0 + td_tech,
        TD0 + np.arange(uniq.size),
        np.ones(uniq.size),
        np.zeros(uniq.size),
    )
    n_before = sum(a.size for a in tails)
    # [dispatcher, date] -> bucket
    add(TD0 + inv, B0 + cb, np.ones(ct.size), arc_cost)
    cand_slice = slice(n_before, n_before + ct.size)
    # bucket -> sink: each covered unit earns the reward
    short_b = np.flatnonzero(sh > 0)
    add(
        B0 + short_b,
        np.full(short_b.size, SNK),
        sh[short_b],
        np.full(short_b.size, -reward),
    )
    # unused budget
    total_budget = int(budget[techs].sum())
    add([SRC], [SNK], [total_budget], [0])

    smcf = min_cost_flow.SimpleMinCostFlow()
    arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        np.concatenate(tails),
        np.concatenate(heads),
        np.concatenate(caps),
        np.concatenate(costs),
    )
    supply = np.zeros(n_nodes, dtype=np.int64)
    supply[SRC], supply[SNK] = total_budget, -total_budget
    smcf.set_nodes_supplies(np.arange(n_nodes), supply)
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"flex assignment failed: {status.name}")

    used = smcf.flows(arcs[cand_slice]) > 0
    t_sel, b_sel, d_sel = ct[used], cb[used], cd[used]
    order = np.lexsort((t_sel, d_sel))
    t_sel, b_sel, d_sel = t_sel[order], b_sel[order], d_sel[order]
    per_bucket = np.bincount(b_sel, minlength=nb).astype(np.int64)
    return FlexAssignment(
        tech=t_sel, day=d_sel, bucket=b_sel, per_bucket=per_bucket,
        per_tech=np.bincount(t_sel, minlength=nt).astype(np.int64),
        covered=int(np.minimum(per_bucket, sh).sum()),
    )  # fmt: skip


# --------------------------------------------------------------------------
# turning assignments into supply-format rows
# --------------------------------------------------------------------------
def _fmt_ts(ts: pd.Timestamp) -> str:
    """Same style as the input files: 9/10/2025 05:00:00 (no zero padding)."""
    return f"{ts.month}/{ts.day}/{ts.year} {ts:%H:%M:%S}"


def shift_times(supply: pd.DataFrame) -> dict:
    """(start offset from SHIFT_DATE, duration) per shift, taken from the supply."""
    if not {"START_TIME", "END_TIME"} <= set(supply.columns):
        return {}
    st = pd.to_datetime(supply["START_TIME"], errors="coerce")
    en = pd.to_datetime(supply["END_TIME"], errors="coerce")
    df = pd.DataFrame(
        {"sh": supply["SHIFT_TYPE"], "o": st - supply["SHIFT_DATE"], "d": en - st}
    ).dropna()
    return {
        sh: (g["o"].mode().iloc[0], g["d"].mode().iloc[0]) for sh, g in df.groupby("sh")
    }


def flex_rows(problem: Problem, fa: FlexAssignment, columns: list[str]) -> pd.DataFrame:
    """New flex days in supply layout: the dispatcher's own row with date,
    desk, shift and times replaced."""
    sup = problem.supply
    first_row = pd.Series(np.arange(len(sup))).groupby(problem.row_tech).first()
    times = shift_times(sup)
    rows = []
    for t, d, b in zip(fa.tech, fa.day, fa.bucket):
        r = sup.iloc[int(first_row[t])].to_dict()
        date = problem.dates[d]
        wc_label, sh = problem.bucket_keys[b][1], problem.bucket_shift[b]
        r["SHIFT_DATE"] = date
        r["WORK_CENTER_ID"] = wc_label
        r["SHIFT_TYPE"] = sh
        if "_WC" in r:
            r["_WC"] = wc_label
        if sh in times:
            off, dur = times[sh]
            r["START_TIME"] = _fmt_ts(date + off)
            r["END_TIME"] = _fmt_ts(date + off + dur)
        if "INDEX" in r:
            r["INDEX"] = np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    if "SECONDARY_QUALS" in columns and len(out):
        if "FINAL_QUALS" in out.columns and "PRIMARY_QUAL" in out.columns:
            out["SECONDARY_QUALS"] = [
                _secondary(fq, pq)
                for fq, pq in zip(out["FINAL_QUALS"], out["PRIMARY_QUAL"])
            ]
        else:
            out["SECONDARY_QUALS"] = "[]"
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    return out[columns] if len(out) else pd.DataFrame(columns=columns)


def _secondary(final_quals: str, primary: str) -> str:
    items = [
        x
        for x in str(final_quals).strip("[]").replace("'", " ").split()
        if x and x != primary
    ]
    return "[" + " ".join(f"'{x}'" for x in items) + "]"
