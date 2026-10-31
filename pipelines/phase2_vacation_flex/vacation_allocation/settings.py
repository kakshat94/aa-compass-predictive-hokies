"""Load the YAML configuration (see vacation_flex_config.yaml).

The YAML holds only policy settings (rules, fairness, base, flex, solver).
Input files and output folders are passed to the run_* functions directly.
Unknown keys raise an error, so a misspelt setting is never silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import AllocationConfig
from .flex_assignment import FlexConfig

SCHEMA = {
    "vacation": {"rule1_daily_total", "rule2_fairness", "rule3_inverse_volume", "base_per_day"},
    "rule1_daily_total": {"enabled", "min", "max", "tolerance_below", "tolerance_above"},
    "rule2_fairness": {"slack_days", "weight"},
    "rule3_inverse_volume": {"enabled", "tolerance"},
    "base_per_day": {"enabled", "value", "tolerance"},
    "flex": {"max_days_per_dispatcher", "same_shift_only", "home_work_center_only", "cross_work_center_cost",
             "cross_shift_cost", "balance_cost"},
    "solver": {"cost_uncoverable", "cost_coverable", "scale_grid_points",
               "flex_pricing_iterations", "pricing_candidates", "flex_aware_vacations"},
}  # fmt: skip


@dataclass
class Settings:
    vacation: AllocationConfig
    flex: FlexConfig
    raw: dict


def _check(section: str, d: dict) -> None:
    unknown = set(d) - SCHEMA[section]
    if unknown:
        raise ValueError(f"unknown key(s) in '{section}': {sorted(unknown)}; allowed: {sorted(SCHEMA[section])}")


def load_settings(path) -> Settings:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    vac, flx, sol = (raw.get(k) or {} for k in ("vacation", "flex", "solver"))
    for name, d in (("vacation", vac), ("flex", flx), ("solver", sol)):
        _check(name, d)
    r1, r2, r3, base = (vac.get(k) or {} for k in
                        ("rule1_daily_total", "rule2_fairness", "rule3_inverse_volume", "base_per_day"))  # fmt: skip
    for name, d in (("rule1_daily_total", r1), ("rule2_fairness", r2),
                    ("rule3_inverse_volume", r3), ("base_per_day", base)):  # fmt: skip
        _check(name, d)

    lo = int(r1.get("min", 25)) - int(r1.get("tolerance_below", 0))
    hi = int(r1.get("max", 29)) + int(r1.get("tolerance_above", 0))
    kw = dict(
        daily_min=max(0, lo),
        daily_max=hi,
        enforce_daily_bounds=bool(r1.get("enabled", True)),
        fairness_slack_days=int(r2.get("slack_days", 0)),
        fairness_weight=int(r2.get("weight", 1000)),
        enforce_proportionality=bool(r3.get("enabled", True)),
        proportionality_tolerance=float(r3.get("tolerance", 0.05)),
        base_daily=int(base["value"]) if base.get("enabled", False) else None,
        base_tolerance=float(base.get("tolerance", 0.0)),
        max_flex_days=int(flx.get("max_days_per_dispatcher", 4)),
        flex_same_shift_only=bool(flx.get("same_shift_only", True)),
        flex_home_work_center_only=bool(flx.get("home_work_center_only", True)),
    )
    for k in SCHEMA["solver"]:
        if k in sol:
            kw[k] = sol[k]
    vcfg = AllocationConfig(**kw)
    vcfg.validate()
    fcfg = FlexConfig(
        cross_work_center_cost=int(flx.get("cross_work_center_cost", 1000)),
        cross_shift_cost=int(flx.get("cross_shift_cost", 1000)),
        balance_cost=int(flx.get("balance_cost", 10)),
    )
    fcfg.validate()

    return Settings(vcfg, fcfg, raw)
