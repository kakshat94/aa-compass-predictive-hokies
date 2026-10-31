"""Reading and cleaning of the module's input files.

Behaviour is intentionally the same as the previous implementation wherever the
downstream flex module depends on it (column handling, date parsing, the
desk-demand imputation and its audit table).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

AUDIT_COLUMNS = [
    "SHIFT_DATE",
    "WORK_CENTER",
    "SHIFT_TYPE",
    "IMPUTED_DEMAND",
    "DONOR_DATE",
    "METHOD",
]


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p.resolve()}")
    df = pd.read_csv(p, low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df


def require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(
            f"{name}: missing required columns {missing}; available: {list(df.columns)}"
        )


def parse_dates(df: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    if df[column].isna().any():
        raise ValueError(
            f"{name}: invalid/missing {column} values in {int(df[column].isna().sum())} rows"
        )
    return df


def _clean_ids(series: pd.Series) -> pd.Series:
    """Dispatcher ids as plain text, without a spurious '.0' from float parsing."""
    if pd.api.types.is_float_dtype(series):
        if series.notna().all() and (series % 1 == 0).all():
            series = series.astype("int64")
    return series.astype(str).str.strip()


def norm_wc(value: str) -> str:
    """Work-centre key used for *matching* (output files keep the original text)."""
    return str(value).strip().upper()


# --------------------------------------------------------------------------
# cleaners
# --------------------------------------------------------------------------
def clean_supply(df: pd.DataFrame) -> pd.DataFrame:
    require(df, ["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"], "supply")
    df = parse_dates(df, "SHIFT_DATE", "supply")
    if df[["TECH_ID", "WORK_CENTER_ID", "SHIFT_TYPE"]].isna().any().any():
        raise ValueError("Supply has missing identifier, work center or shift")
    df["TECH_ID"] = _clean_ids(df["TECH_ID"])
    df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype(str).str.strip().str.upper()
    dup = df.duplicated(["TECH_ID", "SHIFT_DATE"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"]]
        raise ValueError(
            "Duplicate employee-date rows in supply; cannot model one vacation/day:\n"
            + str(bad.head(8))
        )
    return df.sort_values(["SHIFT_DATE", "TECH_ID"]).reset_index(drop=True)


def clean_demand(df: pd.DataFrame) -> pd.DataFrame:
    require(df, ["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "DEMAND"], "desk demand")
    df = parse_dates(df, "SHIFT_DATE", "desk demand")
    df["WORK_CENTER"] = df["WORK_CENTER"].astype(str).str.strip()
    df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype(str).str.strip().str.upper()
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="raise")
    if (df["DEMAND"] < 0).any() or not (df["DEMAND"] % 1 == 0).all():
        raise ValueError("Desk demand values must be nonnegative integers")
    key = ["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE"]
    if df.assign(_k=df["WORK_CENTER"].map(norm_wc)).duplicated(
        ["SHIFT_DATE", "_k", "SHIFT_TYPE"]
    ).any():
        raise ValueError(f"Desk demand contains duplicate {key} keys")
    return df


def clean_volume(df: pd.DataFrame) -> pd.DataFrame:
    require(df, ["DATE", "DEMAND"], "daily operational volume")
    df = parse_dates(df, "DATE", "daily operational volume")
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="raise")
    if df["DEMAND"].isna().any() or df.duplicated(["DATE"]).any():
        raise ValueError("Daily volume has missing values or duplicated dates")
    if (df["DEMAND"] <= 0).any():
        raise ValueError("Daily volume must be strictly positive")
    return df


def load_training(path: str | Path | None) -> set[tuple[str, pd.Timestamp]]:
    """Training calendar as {(TECH_ID, date)}; empty if no file is given."""
    if path is None:
        return set()
    df = read_csv(path)
    require(df, ["TECH_ID", "SHIFT_DATE"], "training calendar")
    df = parse_dates(df, "SHIFT_DATE", "training calendar")
    return set(zip(_clean_ids(df["TECH_ID"]), df["SHIFT_DATE"]))


def load_blocked_days(path: str | Path | None) -> set[tuple[str, pd.Timestamp]]:
    """Dates a dispatcher is already assigned a shift (work or training), e.g.
    Supply_Regular_Merged.csv; no flex day may be placed on them."""
    if path is None:
        return set()
    df = read_csv(path)
    require(df, ["TECH_ID", "SHIFT_DATE"], "merged schedule")
    df = parse_dates(df, "SHIFT_DATE", "merged schedule")
    return set(zip(_clean_ids(df["TECH_ID"]), df["SHIFT_DATE"]))


# --------------------------------------------------------------------------
# desk-demand imputation  (same rules as before; the flex module reads the audit)
# --------------------------------------------------------------------------
def fill_missing_demand(
    demand: pd.DataFrame, supply: pd.DataFrame, dates: list[pd.Timestamp]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate desk demand for scheduled buckets that have no demand record.

    Donor preference: same work-centre/shift one week earlier, then one week
    later, then the closest observed date. Never fills with zero. Returns the
    completed demand table and an audit table of every imputed bucket.
    """
    date_set = set(dates)
    known = {
        (r.SHIFT_DATE, norm_wc(r.WORK_CENTER), r.SHIFT_TYPE): int(r.DEMAND)
        for r in demand.itertuples(index=False)
    }
    by_bucket: dict[tuple[str, str], list[pd.Timestamp]] = {}
    for d, c, sh in known:
        by_bucket.setdefault((c, sh), []).append(d)

    label = {}  # first-seen supply label per normalised work centre
    for wc in supply["WORK_CENTER_ID"].unique():
        label.setdefault(norm_wc(wc), wc)
    scheduled = set(
        zip(
            supply["SHIFT_DATE"],
            supply["WORK_CENTER_ID"].map(norm_wc),
            supply["SHIFT_TYPE"],
        )
    )
    missing = sorted(b for b in scheduled if b[0] in date_set and b not in known)

    audit, new_rows = [], []
    for d, c, sh in missing:
        donor = next(
            (
                cand
                for cand in (d - pd.Timedelta(days=7), d + pd.Timedelta(days=7))
                if (cand, c, sh) in known
            ),
            None,
        )
        if donor is None:
            cands = by_bucket.get((c, sh), [])
            if not cands:
                raise ValueError(
                    f"Cannot estimate desk demand for {d.date()}, {label[c]}, {sh}: "
                    "no donor exists for this bucket"
                )
            donor = min(cands, key=lambda dd: (abs((dd - d).days), dd))
        value = known[(donor, c, sh)]
        audit.append(
            {
                "SHIFT_DATE": d,
                "WORK_CENTER": label[c],
                "SHIFT_TYPE": sh,
                "IMPUTED_DEMAND": value,
                "DONOR_DATE": donor,
                "METHOD": (
                    "same_weekday" if abs((donor - d).days) == 7 else "nearest_date"
                ),
            }
        )
        new_rows.append(
            {
                "SHIFT_DATE": d,
                "WORK_CENTER": label[c],
                "SHIFT_TYPE": sh,
                "DEMAND": value,
            }
        )
    if new_rows:
        # Donors are looked up in ``known`` (observed values only), so imputed
        # values never chain onto each other.
        demand = pd.concat([demand, pd.DataFrame(new_rows)], ignore_index=True)
        log.warning(
            "Estimated %d missing desk-demand buckets; see demand_imputation_audit.csv "
            "(results are provisional for those buckets).",
            len(new_rows),
        )
    return demand, pd.DataFrame(audit, columns=AUDIT_COLUMNS)


def demand_lookup(demand: pd.DataFrame) -> dict[tuple, int]:
    """{(date, normalised work centre, shift): demand}"""
    return {
        (r.SHIFT_DATE, norm_wc(r.WORK_CENTER), r.SHIFT_TYPE): int(r.DEMAND)
        for r in demand.itertuples(index=False)
    }


__all__ = [
    "read_csv",
    "require",
    "parse_dates",
    "clean_supply",
    "clean_demand",
    "clean_volume",
    "load_training",
    "fill_missing_demand",
    "demand_lookup",
    "norm_wc",
]
