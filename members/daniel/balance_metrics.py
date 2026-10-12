"""Balance metrics for a vacation assignment (COMPASS Phase II, section 4.3).

Two families of metrics, using Gini, coefficient of variation (CV) and
max-to-mean ratio throughout:

  * Across bid lines -- proportional distribution of vacation days.
    Computed overall, by primary qualification, and by every held qualification
    (overlapping), plus the between-group dispersion for each grouping.
  * Across the year -- temporal spread, measured on daily / weekly / monthly
    buckets (raw and normalised by the number of working days), plus the
    per-bid-line month concentration (Herfindahl index).

All metrics are dispersion measures where 0 is best (perfectly even).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from members.daniel.qualification_mapper import (
    DEFAULT_PRIORITY_LIST,
    DISPLAY_TO_COLUMN,
    parse_final_quals,
)


def _gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) - (n + 1) * total) / (n * total))


def _cv(values) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or x.mean() == 0:
        return 0.0
    return float(x.std(ddof=0) / x.mean())


def _max_to_mean(values) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or x.mean() == 0:
        return 0.0
    return float(x.max() / x.mean())


def _metrics(values) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "gini": 0.0, "cv": 0.0, "max_to_mean": 0.0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        "gini": _gini(x),
        "cv": _cv(x),
        "max_to_mean": _max_to_mean(x),
    }


def _tech_qualification_lookup(roster: pd.DataFrame):
    """One row per bid line with its primary qualification and held-qual flags."""
    tech = roster.drop_duplicates("TECH_ID").set_index("TECH_ID")
    primary = tech["PRIMARY_QUAL"].to_dict() if "PRIMARY_QUAL" in tech else {}
    held = {
        name: set(tech.index[tech[col] == 1])
        for name, col in DISPLAY_TO_COLUMN.items()
        if col in tech.columns
    }
    return primary, held


def bidline_balance(vacation_data: pd.DataFrame, roster: pd.DataFrame,
                    priority_list: Optional[List[str]] = None) -> Dict:
    priority_list = priority_list or DEFAULT_PRIORITY_LIST
    all_techs = roster["TECH_ID"].unique()
    counts = (
        vacation_data.groupby("TECH_ID").size()
        .reindex(all_techs, fill_value=0)
    )

    primary_lookup, held_sets = _tech_qualification_lookup(roster)

    # By primary qualification.
    by_primary: Dict[str, Dict] = {}
    primary_means = []
    for qual in priority_list:
        techs = [t for t in all_techs if primary_lookup.get(t) == qual]
        if not techs:
            continue
        m = _metrics(counts.loc[techs].values)
        by_primary[qual] = m
        primary_means.append(m["mean"])

    # By every held qualification (overlapping groups).
    by_held: Dict[str, Dict] = {}
    held_means = []
    for qual in priority_list:
        techs = list(held_sets.get(qual, set()))
        if not techs:
            continue
        m = _metrics(counts.reindex(techs, fill_value=0).values)
        by_held[qual] = m
        held_means.append(m["mean"])

    return {
        "overall": _metrics(counts.values),
        "by_primary": by_primary,
        "primary_between": _metrics(primary_means),
        "by_held": by_held,
        "held_between": _metrics(held_means),
    }


def _bucket_counts(dates: pd.Series, freq: str) -> pd.Series:
    period = dates.dt.to_period(freq)
    return period.value_counts().sort_index()


def _working_days_per_bucket(roster: pd.DataFrame, freq: str) -> pd.Series:
    days = pd.to_datetime(roster["SHIFT_DATE"]).dt.normalize().drop_duplicates()
    return days.dt.to_period(freq).value_counts().sort_index()


def temporal_balance(vacation_data: pd.DataFrame, roster: pd.DataFrame) -> Dict:
    dates = pd.to_datetime(vacation_data["SHIFT_DATE"])

    daily = _bucket_counts(dates, "D")
    weekly = _bucket_counts(dates, "W")
    monthly = _bucket_counts(dates, "M")

    wd_week = _working_days_per_bucket(roster, "W").reindex(weekly.index, fill_value=1)
    wd_month = _working_days_per_bucket(roster, "M").reindex(monthly.index, fill_value=1)
    weekly_norm = weekly / wd_week
    monthly_norm = monthly / wd_month

    # Per-bid-line month concentration (Herfindahl across the months it takes vacation).
    months = dates.dt.month
    frame = pd.DataFrame({"TECH_ID": vacation_data["TECH_ID"].values, "M": months.values})
    herf = (
        frame.groupby(["TECH_ID", "M"]).size()
        .groupby(level=0)
        .apply(lambda s: float(np.sum((s / s.sum()) ** 2)))
    )

    return {
        "daily": _metrics(daily.values),
        "weekly": _metrics(weekly.values),
        "monthly": _metrics(monthly.values),
        "weekly_norm": _metrics(weekly_norm.values),
        "monthly_norm": _metrics(monthly_norm.values),
        "bidline_month_herfindahl": _metrics(herf.values),
    }


def compute_balance_metrics(vacation_data: pd.DataFrame, roster: pd.DataFrame,
                            priority_list: Optional[List[str]] = None) -> Dict:
    return {
        "bidlines": bidline_balance(vacation_data, roster, priority_list),
        "temporal": temporal_balance(vacation_data, roster),
    }


def _fmt(m: Dict) -> str:
    return (f"mean={m['mean']:.3f} range=[{m['min']:.3f},{m['max']:.3f}] "
            f"gini={m['gini']:.3f} cv={m['cv']:.3f} max/mean={m['max_to_mean']:.3f}")


def print_balance_report(metrics: Dict) -> None:
    b = metrics["bidlines"]
    t = metrics["temporal"]
    print("\n================ BALANCE METRICS ================")
    print("ACROSS BID LINES (0 = perfectly proportional)")
    print(f"  overall                      : {_fmt(b['overall'])}")
    print("  by primary qualification:")
    for qual, m in b["by_primary"].items():
        print(f"    {qual:8s} : {_fmt(m)}")
    print(f"  between primary groups       : {_fmt(b['primary_between'])}")
    print("  by held qualification (overlapping):")
    for qual, m in b["by_held"].items():
        print(f"    {qual:8s} : {_fmt(m)}")
    print(f"  between held groups          : {_fmt(b['held_between'])}")

    print("ACROSS THE YEAR (0 = perfectly spread)")
    print(f"  daily                        : {_fmt(t['daily'])}")
    print(f"  weekly                       : {_fmt(t['weekly'])}")
    print(f"  monthly                      : {_fmt(t['monthly'])}")
    print(f"  weekly per working day       : {_fmt(t['weekly_norm'])}")
    print(f"  monthly per working day      : {_fmt(t['monthly_norm'])}")
    print(f"  bid-line month Herfindahl    : {_fmt(t['bidline_month_herfindahl'])}")
    print("=================================================\n")
