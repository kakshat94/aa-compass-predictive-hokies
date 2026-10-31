"""Independent post-hoc checks of a finished allocation.

These re-derive every requirement from the two *output tables* with plain
pandas, sharing no code with the solver, so a modelling or indexing bug cannot
pass its own test.  ``validate_allocation`` raises ``AssertionError`` on the
first violated hard requirement and returns a dict of facts for the summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AllocationConfig


def validate_allocation(
    supply: pd.DataFrame,
    assignments: pd.DataFrame,
    remaining: pd.DataFrame,
    cfg: AllocationConfig,
    training: set,
    total: int,
    day_lo=None,
    day_hi=None,
) -> dict:
    keys = ["TECH_ID", "SHIFT_DATE"]

    # every vacation is an existing workday, used at most once
    assert not assignments.duplicated(keys).any(), "a workday was used twice as vacation"
    sk = pd.MultiIndex.from_frame(supply[keys])
    vk = pd.MultiIndex.from_frame(assignments[keys])
    assert vk.isin(sk).all(), "vacation assigned on a day the dispatcher does not work"

    # supply is partitioned exactly into vacation + remaining
    assert len(assignments) + len(remaining) == len(supply), "row count mismatch"
    rk = pd.MultiIndex.from_frame(remaining[keys])
    assert not rk.isin(vk).any(), "a vacation row is still in the remaining supply"
    assert rk.isin(sk).all(), "remaining supply contains a row not in the input"

    # no vacation on a training day
    if training:
        hit = [(t, d) in training for t, d in zip(assignments["TECH_ID"], assignments["SHIFT_DATE"])]
        assert not any(hit), "vacation assigned on a training day"

    # R1: daily totals within bounds (every schedule date counts, including zeros)
    all_dates = pd.Index(sorted(supply["SHIFT_DATE"].unique()))
    daily = assignments.groupby("SHIFT_DATE").size().reindex(all_dates, fill_value=0)
    assert (daily >= cfg.daily_min).all() and (daily <= cfg.daily_max).all(), (
        f"daily total outside [{cfg.daily_min},{cfg.daily_max}]: "
        f"min={daily.min()} max={daily.max()}"
    )
    assert int(daily.sum()) == total, f"total vacations {int(daily.sum())} != {total}"

    # R3: each day within the floor/ceiling of its proportional target
    if day_lo is not None and day_hi is not None:
        n = daily.to_numpy()
        assert (n >= np.asarray(day_lo)).all() and (n <= np.asarray(day_hi)).all(), (
            "a daily total is outside the floor/ceiling of its proportional target"
        )

    # R2: spread of vacation counts across all dispatchers
    all_tech = pd.Index(sorted(supply["TECH_ID"].unique()))
    counts = assignments.groupby("TECH_ID").size().reindex(all_tech, fill_value=0)
    spread = int(counts.max() - counts.min())
    if cfg.fairness_slack_days == 0:
        assert spread <= 1, f"vacation counts differ by {spread} days between dispatchers"
    else:
        assert spread <= 1 + 2 * cfg.fairness_slack_days

    return {
        "daily_min": int(daily.min()),
        "daily_max": int(daily.max()),
        "per_dispatcher_min": int(counts.min()),
        "per_dispatcher_max": int(counts.max()),
        "per_dispatcher_spread": spread,
        "per_dispatcher_std": float(counts.std(ddof=0)),
        "dispatchers_with_zero_vacation": int((counts == 0).sum()),
        "total_vacations": int(daily.sum()),
        "daily_totals": daily,
        "dispatcher_counts": counts,
    }


def volume_relationship(daily: pd.Series, volume: pd.Series) -> dict:
    """How closely the allocated daily totals follow the inverse-volume rule."""
    v = volume.reindex(daily.index).to_numpy(dtype=float)
    n = daily.to_numpy(dtype=float)
    out = {}
    if len(n) > 2 and n.std() > 0 and v.std() > 0:
        out["pearson_vacations_vs_volume"] = float(np.corrcoef(n, v)[0, 1])
        out["pearson_vacations_vs_inverse_volume"] = float(np.corrcoef(n, 1.0 / v)[0, 1])
        rank = lambda x: pd.Series(x).rank().to_numpy()  # noqa: E731
        out["spearman_vacations_vs_volume"] = float(np.corrcoef(rank(n), rank(v))[0, 1])
    return out
