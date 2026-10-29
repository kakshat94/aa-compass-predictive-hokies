"""Writing result files.

The two files the rest of the pipeline consumes keep their established format:

* ``vacation_initial.csv`` -- the supply rows turned into vacation (all supply
  columns, SHIFT_DATE as YYYY-MM-DD).
* ``vacation_outputs_updatedFormat.csv`` -- every other supply row, same columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

VACATION_FILE = "vacation_initial.csv"
REMAINING_FILE = "vacation_outputs_updatedFormat.csv"
AUDIT_FILE = "demand_imputation_audit.csv"  # read by the flex module
SUMMARY_FILE = "vacation_initial_summary.json"
TARGETS_FILE = "vacation_daily_targets.csv"
FAIRNESS_FILE = "vacation_fairness_by_dispatcher.csv"
SHORTAGE_FILE = "vacation_shortage_by_bucket.csv"
SEARCH_FILE = "vacation_scale_search.csv"  # every (scale, level) candidate tried

# Column order of the supplied supply view; extra upstream columns follow it.
REFERENCE_SUPPLY_COLUMNS = [
    "INDEX", "MAJOR_WORK_TYPE", "START_TIME", "EMPLOYEE_TITLE", "WORK_CENTER_ID",
    "CONTRACTOR_FLAG", "CLUSTER", "EMPLOYEE_WORK_CENTER_ID", "TECH_ID", "END_TIME",
    "SHIFT_DATE", "SHIFT_LENGTH", "SHIFT_TYPE", "FINAL_QUALS", "PRIMARY_QUAL",
    "DOMESTIC", "ATC", "ROC", "CHF", "OC", "LATIN", "EUROPE", "PACIFIC",
    "QUAL_COUNT", "SF_LOAD_TMS",
]  # fmt: skip


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    first = [c for c in REFERENCE_SUPPLY_COLUMNS if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_all(result, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "vacation": out / VACATION_FILE,
        "remaining": out / REMAINING_FILE,
        "audit": out / AUDIT_FILE,
        "summary": out / SUMMARY_FILE,
        "targets": out / TARGETS_FILE,
        "fairness": out / FAIRNESS_FILE,
        "shortage": out / SHORTAGE_FILE,
        "search": out / SEARCH_FILE,
    }
    fmt = "%Y-%m-%d"
    _ordered(result.assignments).to_csv(paths["vacation"], index=False, date_format=fmt)
    _ordered(result.remaining).to_csv(paths["remaining"], index=False, date_format=fmt)
    result.demand_audit.to_csv(paths["audit"], index=False, date_format=fmt)
    result.daily.to_csv(paths["targets"], index=False, date_format=fmt)
    result.fairness.to_csv(paths["fairness"], index=False)
    result.shortage_by_bucket.to_csv(paths["shortage"], index=False, date_format=fmt)
    if result.search_log is not None:
        result.search_log.to_csv(paths["search"], index=False)
    paths["summary"].write_text(
        json.dumps(result.stats, indent=2, default=_json_default)
    )
    return paths
