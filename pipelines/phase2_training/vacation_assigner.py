"""Vacation scaling and assignment (COMPASS Phase II, section 4).

Implements the rules in COMPASS_Phase_II_Problem_Statement.md and the helpers in
"Vacation information summary.docx":

  * Scale ``inputs/demand_volume_per_day_reference.csv`` to [-1, 1]
    (``_scale_to_range_neg_1_to_1``) and derive the daily vacation quota:
    base 27 dispatchers +/- up to 2 -> 25..29 per calendar day. The adjustment is
    inversely related to the day's operational volume (quieter days carry more
    dispatchers on vacation), per Problem Statement section 4.1.
  * Assign which bid lines take vacation so that:
      - vacation falls only on a day the bid line is scheduled to work;
      - the daily count is met;
      - no bid line absorbs a disproportionate share (sampling is biased away
        from bid lines that already have many vacation days).

Writes:
    outputs/vacation_data.csv                    (25-column bid line schema)
    outputs/supply_data_after_vacation_shrinkage.csv
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pipelines.phase2_training.demand_scaling import _scale_to_range_neg_1_to_1

BASE_VACATION_DEFAULT = 27
MAX_VARIABLE_DEFAULT = 2
VACATION_SEED_DEFAULT = 1


def scale_demand_volume(
    volume_path: str = "inputs/demand_volume_per_day_reference.csv",
    base_vacation: int = BASE_VACATION_DEFAULT,
    max_variable: int = MAX_VARIABLE_DEFAULT,
) -> pd.DataFrame:
    """Scale the daily operational volume to [-1, 1] and derive the vacation quota.

    Returns one row per day with the scaled/inverse volume, the integer variable
    adjustment in ``{-max_variable..+max_variable}`` and ``TOTAL_VACATION`` in
    ``{base-max_variable .. base+max_variable}``.
    """
    df = pd.read_csv(volume_path)
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Rank days by operational volume and split into 2*max_variable+1 bands.
    ranked = df.sort_values("DEMAND", kind="stable").reset_index(drop=True)
    n = len(ranked)
    total_buckets = 2 * max_variable + 1
    base_size, extra = divmod(n, total_buckets)

    adjustments = []
    for bucket in range(total_buckets):
        size = base_size + (1 if bucket < extra else 0)
        # bucket 0 = quietest -> +max_variable (most vacation); last -> -max_variable.
        adjustments.extend([max_variable - bucket] * size)
    ranked["VARIABLE_VACATION"] = adjustments

    ranked["SCALED_DEMAND"] = _scale_to_range_neg_1_to_1(ranked["DEMAND"])
    ranked["INVERSE_DEMAND"] = -ranked["SCALED_DEMAND"]
    ranked["TOTAL_VACATION"] = base_vacation + ranked["VARIABLE_VACATION"]

    ranked = ranked.sort_values("DATE").reset_index(drop=True)
    return ranked[
        [
            "DATE",
            "DEMAND",
            "SCALED_DEMAND",
            "INVERSE_DEMAND",
            "VARIABLE_VACATION",
            "TOTAL_VACATION",
        ]
    ]


def quota_by_date(
    scaled_volume: pd.DataFrame, base_vacation: int = BASE_VACATION_DEFAULT
) -> Dict:
    """Map ``datetime.date -> total_vacation`` from a scaled-volume frame."""
    return {
        row.DATE.date(): int(row.TOTAL_VACATION) for row in scaled_volume.itertuples()
    }


def assign_vacation(
    regular_df: pd.DataFrame,
    totals_by_date: Dict,
    base_vacation: int = BASE_VACATION_DEFAULT,
    seed: int = VACATION_SEED_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pick which bid lines take vacation each day.

    ``regular_df`` is the working-day roster (training and days off already
    removed), one row per bid line-day. Returns ``(effective_supply, vacation_data)``.
    """
    df = regular_df.copy().reset_index(drop=True)
    df["_DATE"] = pd.to_datetime(df["SHIFT_DATE"]).dt.date

    rng = np.random.RandomState(seed)
    freq: Dict[object, int] = {}
    chosen_idx: list = []

    for date, grp in df.groupby("_DATE", sort=True):
        slots = int(totals_by_date.get(date, base_vacation))
        if len(grp) == 0:
            continue
        if len(grp) <= slots:
            picked = grp.index.tolist()
        else:
            techs = grp["TECH_ID"].to_numpy()
            if not freq:
                weights = np.ones(len(grp), dtype=float)
            else:
                f = np.array([freq.get(t, 0) for t in techs], dtype=float)
                # 1 / freq**10 for already-used bid lines, else 1 (anti-concentration).
                weights = np.ones(len(grp), dtype=float)
                used = f > 0
                weights[used] = 1.0 / np.power(f[used], 10.0)
            probs = weights / weights.sum()
            picked = rng.choice(
                grp.index.to_numpy(), size=slots, replace=False, p=probs
            )

        for idx in picked:
            chosen_idx.append(idx)
            tech = df.at[idx, "TECH_ID"]
            freq[tech] = freq.get(tech, 0) + 1

    mask = df.index.isin(chosen_idx)
    vacation = df[mask].drop(columns=["_DATE"]).reset_index(drop=True)
    effective = df[~mask].drop(columns=["_DATE"]).reset_index(drop=True)
    return effective, vacation


def write_vacation_outputs(
    effective_supply: pd.DataFrame,
    vacation_data: pd.DataFrame,
    output_dir: str = "outputs",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "vacation": os.path.join(output_dir, "vacation_data.csv"),
        "effective_supply": os.path.join(
            output_dir, "supply_data_after_vacation_shrinkage.csv"
        ),
    }
    vacation_data.to_csv(paths["vacation"], index=False)
    effective_supply.to_csv(paths["effective_supply"], index=False)
    return paths


def run_vacation(
    regular_df: pd.DataFrame,
    volume_path: str = "inputs/demand_volume_per_day_reference.csv",
    base_vacation: int = BASE_VACATION_DEFAULT,
    max_variable: int = MAX_VARIABLE_DEFAULT,
    seed: int = VACATION_SEED_DEFAULT,
    output_dir: str = "outputs",
    write: bool = True,
):
    """Scale the volume, assign vacation, optionally write CSVs. Returns a bundle dict."""
    scaled_volume = scale_demand_volume(volume_path, base_vacation, max_variable)
    totals = quota_by_date(scaled_volume, base_vacation)
    effective, vacation = assign_vacation(
        regular_df, totals, base_vacation=base_vacation, seed=seed
    )
    paths = write_vacation_outputs(effective, vacation, output_dir) if write else {}
    return {
        "scaled_volume": scaled_volume,
        "effective_supply": effective,
        "vacation_data": vacation,
        "paths": paths,
    }
