"""Best-case flex headroom, estimated -- never assigned.

Flex days themselves are assigned by a separate module.  What this module needs
from flex is only an answer to: *if vacations leave desk X short on date D, can
flex still cover it?*  Whether it can depends on who is off that day, who is
qualified for the desk, and the 4-flex-days-per-dispatcher budget that every
other shortage competes for.  That is a bipartite max-flow::

    SRC ─(max_flex)─▶ dispatcher ─(1)─▶ [dispatcher,date] ─(1)─▶ bucket ─(shortage)─▶ SNK

A dispatcher is a flex candidate for bucket (date, work centre, shift) when
that date is a true off-day for them (not scheduled, not training, no
pre-assigned flex), they are qualified for the work centre (qualification
columns), and -- by default -- they normally work that shift.  Each
dispatcher's budget is max_flex_days minus their pre-assigned flex days.

``evaluate`` returns how much shortage the best possible flex assignment would
absorb and, from the residual graph, for each bucket whether one *more* short
unit there could still be absorbed.  The allocator uses that to price shortages
(coverable ones are cheaper); the exact max-flow value is what is reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ortools.graph.python import max_flow

from .config import AllocationConfig
from .problem import Problem

log = logging.getLogger(__name__)


@dataclass
class FlexResult:
    raw_shortage: int
    covered: int
    residual: int
    covered_by_bucket: np.ndarray
    coverable_at_margin: np.ndarray  # bool per bucket


class FlexCapacity:
    def __init__(self, problem: Problem, cfg: AllocationConfig):
        self.problem = problem
        self.cfg = cfg
        nt, nb, nd = problem.n_tech, problem.n_bucket, problem.n_day

        # Buckets that can ever be short: those where vacations may exceed slack,
        # plus any bucket that is already below demand before vacations.
        self.buckets = np.flatnonzero(
            (problem.bucket_eligible > problem.bucket_free) | (problem.bucket_slack < 0)
        )

        pools: dict[tuple, np.ndarray] = {}
        shift_of = problem.tech_shifts
        all_shifts = sorted({s for s in problem.bucket_shift})
        home = np.array(problem.tech_home_wc, dtype=object)
        for wc, flag in problem.tech_quals.items():
            if cfg.flex_home_work_center_only:
                flag = flag & (home == wc)
            for sh in all_shifts:
                if cfg.flex_same_shift_only:
                    ok = flag & np.array([sh in s for s in shift_of])
                else:
                    ok = flag
                pools[(wc, sh)] = np.flatnonzero(ok)

        self.pools = pools
        cand_t, cand_b = [], []
        for b in self.buckets:
            pool = pools.get((problem.bucket_wc[b], problem.bucket_shift[b]))
            if pool is None or pool.size == 0:
                continue
            avail = pool[~problem.busy[pool, problem.bucket_day[b]]]
            if avail.size:
                cand_t.append(avail)
                cand_b.append(np.full(avail.size, b, dtype=np.int64))
        if cand_t:
            cand_t = np.concatenate(cand_t)
            cand_b = np.concatenate(cand_b)
        else:
            cand_t = np.zeros(0, dtype=np.int64)
            cand_b = np.zeros(0, dtype=np.int64)
        cand_d = problem.bucket_day[cand_b]
        self.n_candidate_arcs = int(cand_t.size)
        self.cand_t, self.cand_b, self.cand_d = cand_t, cand_b, cand_d
        self.budget = np.maximum(0, cfg.max_flex_days - problem.tech_preflex).astype(
            np.int64
        )

        # Node ids:  0 = SRC, 1 = SNK, then dispatchers, buckets, [dispatcher,date] nodes
        self.SRC, self.SNK = 0, 1
        self.T0 = 2
        self.B0 = self.T0 + nt
        self.TD0 = self.B0 + nb

        key = cand_t * nd + cand_d
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        multi = cnt[inv] > 1  # dispatcher could serve 2+ desks that day -> share 1 unit
        single = ~multi

        st, en, cp = [], [], []
        techs_used = np.unique(cand_t)
        st.append(np.full(techs_used.size, self.SRC))
        en.append(self.T0 + techs_used)
        cp.append(self.budget[techs_used])
        # dispatchers with a single reachable desk that day connect directly
        st.append(self.T0 + cand_t[single])
        en.append(self.B0 + cand_b[single])
        cp.append(np.ones(single.sum()))
        # others go through a [dispatcher, date] node of capacity 1
        if multi.any():
            m_keys, m_first = np.unique(inv[multi], return_index=True)
            td_node = self.TD0 + np.arange(m_keys.size)
            lookup = np.full(uniq.size, -1, dtype=np.int64)
            lookup[m_keys] = td_node
            st.append(self.T0 + cand_t[multi][m_first])
            en.append(td_node)
            cp.append(np.ones(m_keys.size))
            st.append(lookup[inv[multi]])
            en.append(self.B0 + cand_b[multi])
            cp.append(np.ones(int(multi.sum())))
            self.n_nodes = self.TD0 + m_keys.size
        else:
            self.n_nodes = self.TD0
        self._st = np.concatenate(st).astype(np.int64)
        self._en = np.concatenate(en).astype(np.int64)
        self._cp = np.concatenate(cp).astype(np.int64)

    # ------------------------------------------------------------------
    def evaluate(self, shortage: np.ndarray) -> FlexResult:
        """Exact best-case flex coverage for a per-bucket shortage vector."""
        sh = np.asarray(shortage, dtype=np.int64)
        b = self.buckets
        mf = max_flow.SimpleMaxFlow()
        mf.add_arcs_with_capacity(self._st, self._en, self._cp)
        sink_ids = mf.add_arcs_with_capacity(
            self.B0 + b, np.full(b.size, self.SNK, dtype=np.int64), sh[b]
        )
        status = mf.solve(self.SRC, self.SNK)
        if status != mf.OPTIMAL:
            raise RuntimeError(f"flex max-flow failed: {status.name}")

        covered_by_bucket = np.zeros(self.problem.n_bucket, dtype=np.int64)
        covered_by_bucket[b] = mf.flows(sink_ids)

        # Buckets reachable from SRC in the residual graph can take one more
        # unit of shortage and still have it covered.
        reachable = np.zeros(self.n_nodes, dtype=bool)
        reachable[list(mf.get_source_side_min_cut())] = True
        coverable = np.zeros(self.problem.n_bucket, dtype=bool)
        coverable[b] = reachable[self.B0 + b]

        raw = int(sh[b].sum())
        cov = int(mf.optimal_flow())
        return FlexResult(
            raw_shortage=raw,
            covered=cov,
            residual=raw - cov,
            covered_by_bucket=covered_by_bucket,
            coverable_at_margin=coverable,
        )
