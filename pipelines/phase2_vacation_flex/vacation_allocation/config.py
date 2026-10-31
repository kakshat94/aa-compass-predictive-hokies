"""Tunable parameters for the vacation allocation module.

Every modelling choice that is a *policy* rather than a fact about the data
lives here, so that nothing is hidden inside the solver code.

Priority order (highest first).  The annual total is NOT fixed; it is the
outcome of these rules:

    R1  daily total vacations in [daily_min, daily_max]                    (hard)
    R2  every dispatcher gets L or L+1 days; the model chooses L            (hard at slack 0)
    R3  daily total = K / volume (to the nearest integer, clipped to R1);
        the model chooses the scale K among those whose proportional fit
        is within ``proportionality_tolerance`` of the best possible fit    (hard)
    --  minimise desk shortage left after best-case flex                    (objective 1)
    --  maximise the number of vacations                                    (objective 2)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AllocationConfig:
    # ---- R1: daily bounds ------------------------------------------------
    daily_min: int = 25
    daily_max: int = 29
    # False switches R1 off: each day may take 0 .. (eligible workdays) vacations.
    enforce_daily_bounds: bool = True

    # ---- base level (optional) --------------------------------------------
    # None -> the annual total is decided by the model.
    # base_daily=27, base_tolerance=t -> the annual total must lie within
    # (27 - t) x dates .. (27 + t) x dates, i.e. an average of 27 +- t per day.
    # t = 0 fixes the total at exactly 27 x dates.
    base_daily: int | None = None
    base_tolerance: float = 0.0

    # ---- R2: fairness across dispatchers ---------------------------------
    # 0  -> every dispatcher gets L or L+1 vacation days (spread <= 1), with
    #       L chosen by the model.
    # k>0 -> counts may range over [L-k, L+1+k]; every day a dispatcher moves
    #       away from L+0.5 is charged fairness_weight * (2*dev) in the same
    #       price units as a shortage (see cost_* below).  At the default
    #       weight that price exceeds any shortage it could save, so k has no
    #       effect until the weight is lowered too (roughly 300 -> 30).
    fairness_slack_days: int = 0
    fairness_weight: int = 1_000

    # ---- R3: inverse proportionality to volume ----------------------------
    # Fit error of scale K:  sum_d |clip(K/v_d) - K/v_d| / (K/v_d)
    # (how far the 25..29 bounds force each day away from exact proportion).
    # Scales whose error is at most (1 + tolerance) x the best error are
    # admissible; shortage and vacation count then choose among them.
    # 0 -> only the best-fitting scale (the total is then decided by volume).
    proportionality_tolerance: float = 0.05
    # False switches R3 off: each day may then take any total in [daily_min,
    # daily_max] (used to measure what R3 costs in shortage).
    enforce_proportionality: bool = True
    scale_grid_points: int = 41  # scales tried across the admissible band

    # ---- flex headroom (estimated, never assigned) ------------------------
    max_flex_days: int = 4  # per dispatcher over the whole horizon
    flex_same_shift_only: bool = True  # candidate must normally work that shift
    # True: a dispatcher may only flex onto their own WORK_CENTER_ID, never onto
    # a desk they hold only as a secondary qualification.
    flex_home_work_center_only: bool = True
    # Vacation objective.  False (default): minimise the desk demand left
    # uncovered by the vacations themselves, ignoring any flex days still to
    # be assigned.  True: minimise what would remain after best-case flex.
    flex_aware_vacations: bool = False
    flex_pricing_iterations: int = 4  # flex-aware re-pricing (only if flex_aware_vacations)
    pricing_candidates: int = 3  # best (scale, level) candidates that get re-priced

    # Price of one vacation that creates a shortage in a bucket, in "price
    # units".  A shortage flex can still absorb is cheaper than one it cannot.
    cost_uncoverable: int = 1_000
    cost_coverable: int = 300
    pricing_damping: float = 0.5  # weight on the previous price when updating

    # Tiny preference for giving the "extra" vacation day to dispatchers who
    # have more scheduled workdays.
    workday_tie_break: bool = True

    def validate(self) -> None:
        if not 0 <= self.daily_min <= self.daily_max:
            raise ValueError("require 0 <= daily_min <= daily_max")
        if self.base_daily is not None:
            if self.base_tolerance < 0:
                raise ValueError("base_tolerance must be >= 0")
            if self.enforce_daily_bounds and not self.daily_min <= self.base_daily <= self.daily_max:
                raise ValueError("base_daily must lie within [daily_min, daily_max]")
        if self.fairness_slack_days < 0 or self.fairness_weight < 0:
            raise ValueError("fairness_slack_days and fairness_weight must be >= 0")
        if self.proportionality_tolerance < 0:
            raise ValueError("proportionality_tolerance must be >= 0")
        if self.scale_grid_points < 1:
            raise ValueError("scale_grid_points must be >= 1")
        if self.max_flex_days < 0:
            raise ValueError("max_flex_days must be >= 0")
        if not 0.0 <= self.pricing_damping < 1.0:
            raise ValueError("pricing_damping must be in [0, 1)")
        if self.cost_coverable <= 0 or self.cost_uncoverable < self.cost_coverable:
            raise ValueError("require 0 < cost_coverable <= cost_uncoverable")

    def to_dict(self) -> dict:
        return asdict(self)
