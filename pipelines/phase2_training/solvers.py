"""Solver-agnostic backend for the Phase II training MIP (members/daniel).

Purpose
-------
Provide a single, minimal interface over Gurobi and FICO Xpress so the exact
training model in :mod:`members.daniel.training_exact` is written once and can be
built against either engine. This isolates every engine-specific API call
(variable creation, constraints, objective, solve, status, solution queries) to
one small class per solver.

Inputs / outputs
----------------
* Input: a solver name (``"gurobi"`` or ``"xpress"``) and modelling primitives
  (names, linear ``(coefficient, variable)`` term lists, senses, right-hand
  sides).
* Output: a :class:`BaseBackend` instance exposing variable handles and, after
  :meth:`optimize`, a normalised status string plus per-variable and objective
  values.

Assumptions
-----------
* Models are linear mixed-integer programs with a single aggregate objective and
  a mix of binary and non-negative continuous variables (exactly the T1 training
  model).
* The engine is installed: ``pip install gurobipy`` / ``pip install xpress`` and
  a valid licence are required to *solve* (import + model build work without one).

References
----------
* Gurobi Optimizer Reference Manual, Python API (Model / Var).
  https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html
* FICO Xpress Python API reference.
  https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/index.html
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

Term = Tuple[float, object]  # (coefficient, variable-handle)

# Normalised status vocabulary shared by both engines.
OPTIMAL = "optimal"
FEASIBLE = "feasible"
INFEASIBLE = "infeasible"
TIME_LIMIT = "time_limit"
ERROR = "error"


class BaseBackend:
    """Common interface implemented by every solver backend.

    A backend owns a single model. Variables are returned as opaque handles; the
    caller stores them and asks the backend for their values after solving.
    """

    name: str = "base"

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.solve_time: float = 0.0

    # -- model construction ------------------------------------------------- #
    def add_binary(self, name: str) -> object:
        raise NotImplementedError

    def add_continuous(self, name: str, lb: float = 0.0) -> object:
        raise NotImplementedError

    def add_constraint(
        self, terms: Sequence[Term], sense: str, rhs: float, name: str = ""
    ) -> None:
        """Add ``sum(coef * var) (sense) rhs`` where ``sense`` is ``<=``/``>=``/``==``."""
        raise NotImplementedError

    def set_min_objective(self, terms: Sequence[Term]) -> None:
        raise NotImplementedError

    # -- solving / querying ------------------------------------------------- #
    def optimize(
        self, time_limit: Optional[float] = None, mip_gap: Optional[float] = None
    ) -> str:
        raise NotImplementedError

    def status(self) -> str:
        raise NotImplementedError

    def value(self, var: object) -> float:
        raise NotImplementedError

    def obj_value(self) -> float:
        raise NotImplementedError


def _pick_python_sense(sense: str) -> str:
    if sense not in ("<=", ">=", "=="):
        raise ValueError(f"unknown constraint sense {sense!r}")
    return sense


# --------------------------------------------------------------------------- #
# Gurobi
# --------------------------------------------------------------------------- #
class GurobiBackend(BaseBackend):
    """Gurobi implementation of :class:`BaseBackend` (``gurobipy``)."""

    name = "gurobi"

    def __init__(self, log_path: Optional[str] = None, mip_focus: bool = False):
        super().__init__(log_path=log_path)
        try:
            import gurobipy as gp
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "gurobipy is not installed. Run `pip install gurobipy` and ensure a "
                "valid Gurobi licence is available (an academic licence works)."
            ) from exc
        self._gp = gp
        self.model = gp.Model("training_exact")
        self.model.Params.OutputFlag = 0
        if log_path:
            self.model.Params.OutputFlag = 1
            self.model.Params.LogFile = log_path
        if mip_focus:
            self.model.Params.MIPFocus = 1

    def add_binary(self, name: str) -> object:
        return self.model.addVar(vtype=self._gp.GRB.BINARY, name=name)

    def add_continuous(self, name: str, lb: float = 0.0) -> object:
        return self.model.addVar(vtype=self._gp.GRB.CONTINUOUS, lb=lb, name=name)

    def _expr(self, terms: Sequence[Term]):
        gp = self._gp
        return gp.quicksum(coef * var for coef, var in terms)

    def add_constraint(
        self, terms: Sequence[Term], sense: str, rhs: float, name: str = ""
    ) -> None:
        _pick_python_sense(sense)
        gp = self._gp
        sense_map = {
            "<=": gp.GRB.LESS_EQUAL,
            ">=": gp.GRB.GREATER_EQUAL,
            "==": gp.GRB.EQUAL,
        }
        self.model.addLConstr(self._expr(terms), sense_map[sense], rhs, name=name)

    def set_min_objective(self, terms: Sequence[Term]) -> None:
        self.model.setObjective(self._expr(terms), self._gp.GRB.MINIMIZE)

    def optimize(
        self, time_limit: Optional[float] = None, mip_gap: Optional[float] = None
    ) -> str:
        if time_limit is not None:
            self.model.Params.TimeLimit = float(time_limit)
        if mip_gap is not None:
            self.model.Params.MIPGap = float(mip_gap)
        t0 = time.time()
        self.model.optimize()
        self.solve_time = time.time() - t0
        return self.status()

    def status(self) -> str:
        gp = self._gp
        st = self.model.Status
        if st == gp.GRB.OPTIMAL:
            return OPTIMAL
        if st in (gp.GRB.SUBOPTIMAL,) or self.model.SolCount > 0:
            return FEASIBLE if self.model.SolCount > 0 else FEASIBLE
        if st == gp.GRB.INFEASIBLE:
            return INFEASIBLE
        if st == gp.GRB.TIME_LIMIT or st == gp.GRB.INTERRUPTED:
            return TIME_LIMIT
        return ERROR

    def value(self, var: object) -> float:
        return float(var.X)

    def obj_value(self) -> float:
        return float(self.model.ObjVal)


# --------------------------------------------------------------------------- #
# FICO Xpress
# --------------------------------------------------------------------------- #
class XpressBackend(BaseBackend):
    """FICO Xpress implementation of :class:`BaseBackend` (``xpress``)."""

    name = "xpress"

    def __init__(self, log_path: Optional[str] = None, mip_focus: bool = False):
        super().__init__(log_path=log_path)
        try:
            import xpress as xp
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "xpress is not installed. Run `pip install xpress` and ensure a valid "
                "FICO Xpress licence is available (a community/student licence works)."
            ) from exc
        self._xp = xp
        self.model = xp.problem()
        self.model.controls.outputlog = 0
        if log_path:
            try:  # Xpress >= 9.8
                self.model.setLogFile(log_path)
            except Exception:  # pragma: no cover - older bindings
                self.model.setlogfile(log_path)

    def add_binary(self, name: str) -> object:
        return self.model.addVariable(vartype=self._xp.binary, name=name)

    def add_continuous(self, name: str, lb: float = 0.0) -> object:
        return self.model.addVariable(vartype=self._xp.continuous, lb=lb, name=name)

    def _expr(self, terms: Sequence[Term]):
        xp = self._xp
        return xp.Sum([coef * var for coef, var in terms]) if terms else 0.0

    def add_constraint(
        self, terms: Sequence[Term], sense: str, rhs: float, name: str = ""
    ) -> None:
        _pick_python_sense(sense)
        expr = self._expr(terms)
        if sense == "<=":
            con = expr <= rhs
        elif sense == ">=":
            con = expr >= rhs
        else:
            con = expr == rhs
        self.model.addConstraint(con)

    def set_min_objective(self, terms: Sequence[Term]) -> None:
        self.model.setObjective(self._expr(terms), sense=self._xp.minimize)

    def optimize(
        self, time_limit: Optional[float] = None, mip_gap: Optional[float] = None
    ) -> str:
        if time_limit is not None:
            self.model.controls.maxtime = int(round(float(time_limit)))
        if mip_gap is not None:
            self.model.controls.miprelstop = float(mip_gap)
        t0 = time.time()
        self.model.solve()
        self.solve_time = time.time() - t0
        return self.status()

    def status(self) -> str:
        xp = self._xp
        sv = self.model.attributes.solvestatus
        ss = self.model.attributes.solstatus
        try:
            xp_solved = xp.solvestatus.completed
            xp_optimal = xp.solstatus.optimal
            xp_feasible = xp.solstatus.feasible
            xp_infeas = xp.solvestatus.infeasible
            xp_stopped = xp.solvestatus.stopped
        except Exception:  # pragma: no cover - constant names vary by version
            xp_solved, xp_optimal, xp_feasible, xp_infeas, xp_stopped = 3, 1, 2, 4, 5
        if sv == xp_solved and ss in (xp_optimal, xp_feasible):
            return OPTIMAL if ss == xp_optimal else FEASIBLE
        if sv in (xp_solved, xp_stopped) and ss == xp_feasible:
            return FEASIBLE
        if sv == xp_infeas:
            return INFEASIBLE
        if sv == xp_stopped:
            return TIME_LIMIT
        return ERROR

    def value(self, var: object) -> float:
        return float(self.model.getSolution(var))

    def obj_value(self) -> float:
        return float(self.model.attributes.objval)


_BACKENDS = {"gurobi": GurobiBackend, "xpress": XpressBackend}


def make_backend(
    solver: str, log_path: Optional[str] = None, mip_focus: bool = False
) -> BaseBackend:
    """Factory returning a backend for ``solver`` in ``{"gurobi", "xpress"}``."""
    key = (solver or "").strip().lower()
    if key not in _BACKENDS:
        raise ValueError(f"unknown solver {solver!r}; choose from {sorted(_BACKENDS)}")
    return _BACKENDS[key](log_path=log_path, mip_focus=mip_focus)
