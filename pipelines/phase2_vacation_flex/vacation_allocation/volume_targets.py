"""Daily vacation totals driven by operational volume (requirement R3).

Replaces ``make_quota``.  Kept from it: the rule for imputing the volume of a
date with no observation (same weekday previous week, else following week,
else error) and the logging of every imputation.  Dropped: the quintile
bucketing and the fixed quota level.

Rule
----
Vacations on a day are *inversely proportional* to that day's volume:

    N_d  ~  K / volume_d

Proportionality fixes the shape, not the level: the scale ``K`` is free, and
it is exactly what decides the annual total.  ``K`` is not arbitrary, though.
The busiest and quietest days differ by a factor of ~2.4 in volume while the
daily bounds only allow 25..29 (a factor of 1.16), so for any ``K`` some days
are pushed onto a bound.  The fit error of a scale measures that distortion::

    E(K) = sum_d |clip(K/v_d, 25, 29) - K/v_d| / (K/v_d)

A scale is *admissible* if ``E(K) <= (1 + tolerance) * min E``.  For each
admissible scale the daily total must equal ``clip(K/v_d)`` rounded down or up;
the allocator chooses which, and the pipeline chooses the scale, by shortage
and then by number of vacations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def impute_volume(
    volume: pd.DataFrame, dates: list[pd.Timestamp]
) -> tuple[pd.Series, pd.DataFrame]:
    """Volume per schedule date, imputing absent dates from the same weekday.

    Returns (series indexed by date, audit table of imputed dates).
    """
    vol = volume.set_index("DATE")["DEMAND"].astype(float).copy()
    observed = set(vol.index)
    audit = []
    for d in sorted(set(dates) - observed):
        prior = d - pd.Timedelta(days=7)
        following = d + pd.Timedelta(days=7)
        if prior in observed:
            donor = prior
        elif following in observed:
            donor = following
        else:
            raise ValueError(
                f"No same-weekday observed volume for {d.date()} (+/- 7 days). "
                "Provide the missing data or explicitly choose another policy."
            )
        vol.loc[d] = vol.loc[donor]
        audit.append(
            {"DATE": d, "DONOR_DATE": donor, "IMPUTED_VOLUME": float(vol.loc[donor])}
        )
        log.warning(
            "Imputed operational volume on %s from %s (%g); unobserved estimate.",
            d.date(),
            donor.date(),
            vol.loc[donor],
        )
    return vol.loc[list(dates)], pd.DataFrame(
        audit, columns=["DATE", "DONOR_DATE", "IMPUTED_VOLUME"]
    )


def fit_error(k: float, volume: np.ndarray, lo: float, hi: float) -> float:
    """Relative distortion the daily bounds impose on the proportional curve K/v."""
    ideal = k / np.asarray(volume, dtype=float)
    return float(np.sum(np.abs(np.clip(ideal, lo, hi) - ideal) / ideal))


def _error_curve(v: np.ndarray, lo: float, hi: float, n: int = 20_001):
    """Fit error on a dense grid of scales.  E(K) mixes convex and concave
    pieces, so a scan is used rather than a line search."""
    ks = np.linspace(max(lo, 0.5) * v.min(), hi * v.max(), n)
    ideal = ks[:, None] / v[None, :]
    err = (np.abs(np.clip(ideal, lo, hi) - ideal) / ideal).sum(axis=1)
    return ks, err


def admissible_scales(
    volume: np.ndarray, lo: float, hi: float, tolerance: float, points: int,
    total_range: tuple[int, int] | None = None,
    info: dict | None = None,
) -> tuple[np.ndarray, float, float]:
    """Scales K whose fit error is within (1+tolerance) of the best.

    With ``total_range`` only scales whose rounded daily totals can sum to a
    value inside the range are kept (the base-per-day constraint).  If no scale
    within the tolerance can reach that range, the tolerance is widened to the
    smallest value that can (R3 gives way before the base); the tolerance
    actually used is reported in ``info["effective_tolerance"]``.
    Returns (up to ``points`` scales spread over the admissible set, best K, best error).
    """
    v = np.asarray(volume, dtype=float)
    if v.size == 0:
        raise ValueError("No schedule dates provided")
    if (v <= 0).any():
        raise ValueError("Volume must be strictly positive")
    if lo == hi:
        return np.array([lo * v.mean()]), lo * v.mean(), 0.0
    ks, err = _error_curve(v, lo, hi)
    i = int(np.argmin(err))
    k_star, e_star = float(ks[i]), float(err[i])
    mask = err <= e_star * (1 + tolerance) + 1e-12
    if total_range is not None:
        ideal = np.clip(ks[:, None] / v[None, :], lo, hi)
        fl = np.floor(ideal + 1e-9).sum(axis=1)
        ce = np.ceil(ideal - 1e-9).sum(axis=1)
        reach = (fl <= total_range[1]) & (ce >= total_range[0])
        if not reach.any():  # the daily bounds themselves cannot produce this total
            return np.zeros(0), k_star, e_star
        if not (mask & reach).any():  # widen R3 just enough to reach the base
            needed = float(err[reach].min()) / e_star - 1 if e_star > 0 else 0.0
            tolerance = max(tolerance, needed)
            mask = err <= e_star * (1 + tolerance) + 1e-12
        mask &= reach
        if not mask[i]:  # best fit cannot reach the base range: use the best that can
            j = np.flatnonzero(mask)[np.argmin(err[mask])]
            k_star_ok = float(ks[j])
        else:
            k_star_ok = k_star
    else:
        k_star_ok = k_star
    if info is not None:
        info["effective_tolerance"] = float(tolerance)
    ok = ks[mask]
    if points <= 1 or ok.size <= 1:
        if ok.size == 1:
            return ok, k_star, e_star
        return np.array([k_star]), k_star, e_star
    pick = ok[np.unique(np.linspace(0, ok.size - 1, points).round().astype(int))]
    return np.unique(np.concatenate([pick, [k_star_ok]])), k_star, e_star


def daily_bounds_for_scale(
    k: float, volume: np.ndarray, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ideal clipped target, lowest allowed integer, highest allowed integer)."""
    ideal = np.clip(k / np.asarray(volume, dtype=float), lo, hi)
    floor = np.floor(ideal + 1e-9).astype(np.int64)
    ceil = np.ceil(ideal - 1e-9).astype(np.int64)
    return ideal, np.clip(floor, lo, hi), np.clip(ceil, lo, hi)
