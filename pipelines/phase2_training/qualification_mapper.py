"""Qualification mapping algorithms (COMPASS Phase II).

Contains the three algorithms in ``Current_Process_Pseudo_code.pdf``:

  * Algorithm 1 (Phase 1) -> ``map_qualifications_to_supply``
  * Algorithm 2 (Phase 2) -> ``build_prop_table``
  * Algorithm 3 (Phase 3) -> ``sprinkle_qualifications``

They populate the qualification columns of a supply / bid-line frame:
``FINAL_QUALS``, ``PRIMARY_QUAL``, the eight binary flags
(``DOMESTIC, ATC, ROC, CHF, OC, LATIN, EUROPE, PACIFIC``) and ``QUAL_COUNT``.

The pipeline entry point is :func:`apply_qualification_pipeline`, which is meant
to be called from ``members/akshat/supply_generator.py`` right after the three
supply files are written.
"""

from __future__ import annotations

import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Display name (as used in the pseudocode / Final_Quals strings) -> column name.
DISPLAY_TO_COLUMN: Dict[str, str] = {
    "Domestic": "DOMESTIC",
    "ATC": "ATC",
    "ROC": "ROC",
    "CHF": "CHF",
    "OC": "OC",
    "Latin": "LATIN",
    "Europe": "EUROPE",
    "Pacific": "PACIFIC",
}

# Binary flag order required by the data specification (section 4.1).
FLAG_COLUMNS: List[str] = list(DISPLAY_TO_COLUMN.values())

# Qualification priority order required by the data specification (section 4.1).
DEFAULT_PRIORITY_LIST: List[str] = list(DISPLAY_TO_COLUMN.keys())

RANDOM_SEED = 42

# Column names of a standard 23/25-column bid line record.
_SCHEMA_COLUMNS = [
    "INDEX",
    "MAJOR_WORK_TYPE",
    "START_TIME",
    "EMPLOYEE_TITLE",
    "WORK_CENTER_ID",
    "CONTRACTOR_FLAG",
    "CLUSTER",
    "EMPLOYEE_WORK_CENTER_ID",
    "TECH_ID",
    "END_TIME",
    "SHIFT_DATE",
    "SHIFT_LENGTH",
    "SHIFT_TYPE",
    "FINAL_QUALS",
    "PRIMARY_QUAL",
    "DOMESTIC",
    "ATC",
    "ROC",
    "CHF",
    "OC",
    "LATIN",
    "EUROPE",
    "PACIFIC",
    "QUAL_COUNT",
    "SF_LOAD_TMS",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def parse_final_quals(cell) -> List[str]:
    """Parse the ``['Latin' 'Pacific' 'Domestic']`` encoding into a list of names."""
    if isinstance(cell, (list, tuple)):
        return [str(x) for x in cell]
    if cell is None:
        return []
    if isinstance(cell, float) and math.isnan(cell):
        return []
    return re.findall(r"'([^']+)'", str(cell))


def format_final_quals(quals: List[str]) -> str:
    """Render a list of names back into the bracketed, space-separated encoding."""
    return "[" + " ".join(f"'{q}'" for q in quals) + "]"


def _display_to_column(name: str) -> str:
    return DISPLAY_TO_COLUMN.get(name, name.upper())


def _default_out_path(path: str) -> str:
    """``Supply_Regular.csv`` -> ``Supply_Regular_qualified.csv``."""
    root, ext = os.path.splitext(path)
    return f"{root}_qualified{ext}"


def build_qual_data(qual_data_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize historical qualification records.

    Implements Algorithm 1 step 3: ``QualData[Qual Count] <- |QualData[Final Quals]|``.
    """
    df = qual_data_df.copy()
    for col in FLAG_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "FINAL_QUALS" not in df.columns:
        df["FINAL_QUALS"] = "[]"
    df["QUAL_COUNT"] = df["FINAL_QUALS"].map(lambda c: len(parse_final_quals(c)))
    if "PRIMARY_QUAL" not in df.columns:
        # Fall back to whichever flag is set for the home work center.
        df["PRIMARY_QUAL"] = df["WORK_CENTER_ID"]
    return df


def _order_final_quals(
    primary: str, flag_row: Dict[str, int], priority_list: List[str]
) -> List[str]:
    """Primary qualification first, then the remaining held quals in priority order."""
    quals = [primary]
    for name in priority_list:
        if name != primary and flag_row.get(name, 0) == 1:
            quals.append(name)
    return quals


# --------------------------------------------------------------------------- #
# Algorithm 1 - Phase 1
# --------------------------------------------------------------------------- #
def map_qualifications_to_supply(
    supply_df: pd.DataFrame,
    qual_data_df: pd.DataFrame,
    priority_list: Optional[List[str]] = None,
    dedupe: bool = False,
) -> pd.DataFrame:
    """Algorithm 1: map supply employees to historical qualification records.

    For each work center, rank historical employees by qualification breadth and
    map them one-to-one onto supply bid-line IDs. The resulting profiles are
    merged into the supply data; unmatched rows default to Domestic-only.

    ``dedupe=False`` (the current default) uses the raw per-shift rows as the
    candidate pool, matching the literal pseudocode.
    """
    priority_list = priority_list or DEFAULT_PRIORITY_LIST
    qual = build_qual_data(qual_data_df)
    supply = supply_df.copy()

    cand_source = qual.drop_duplicates(subset=["TECH_ID"]) if dedupe else qual

    # SupplyAgg: unique tech ids per work center.
    supply_tech = supply[["TECH_ID", "WORK_CENTER_ID"]].drop_duplicates()
    requirements = supply.groupby("WORK_CENTER_ID")["TECH_ID"].nunique().to_dict()

    emp_qual_map: Dict[object, object] = {}
    for qual_name, requirement in requirements.items():
        cands = cand_source[cand_source["PRIMARY_QUAL"] == qual_name].sort_values(
            "QUAL_COUNT", ascending=False
        )
        used_qual_emps = cands["TECH_ID"].head(int(requirement)).tolist()
        supply_emps = supply_tech.loc[
            supply_tech["WORK_CENTER_ID"] == qual_name, "TECH_ID"
        ].tolist()
        for qual_id, supply_id in zip(used_qual_emps, supply_emps):
            emp_qual_map[supply_id] = qual_id

    supply["_MAPPED_QUAL_ID"] = supply["TECH_ID"].map(emp_qual_map)

    hist_cols = ["TECH_ID", "FINAL_QUALS", "PRIMARY_QUAL", "QUAL_COUNT"] + FLAG_COLUMNS
    hist = (
        qual[hist_cols]
        .rename(columns={c: f"_HIST_{c}" for c in hist_cols if c != "TECH_ID"})
        .rename(columns={"TECH_ID": "_MAPPED_QUAL_ID"})
    )
    hist = hist.drop_duplicates(subset=["_MAPPED_QUAL_ID"])

    merged = supply.merge(hist, on="_MAPPED_QUAL_ID", how="left")

    # Merge the historical profiles into the standard supply columns.
    merged["FINAL_QUALS"] = merged["_HIST_FINAL_QUALS"].fillna("['Domestic']")
    merged["PRIMARY_QUAL"] = merged["_HIST_PRIMARY_QUAL"].fillna(
        merged["EMPLOYEE_WORK_CENTER_ID"]
    )
    for col in FLAG_COLUMNS:
        merged[col] = merged[f"_HIST_{col}"].fillna(0).astype(int)
    merged["QUAL_COUNT"] = merged["_HIST_QUAL_COUNT"].fillna(0).astype(int)

    # The pseudocode keeps a Mapped Qual ID column; expose it but let the pipeline drop it.
    merged = merged.rename(columns={"_MAPPED_QUAL_ID": "MAPPED_QUAL_ID"})
    return merged


# --------------------------------------------------------------------------- #
# Algorithm 2 - Phase 2
# --------------------------------------------------------------------------- #
def build_prop_table(
    qual_data_df: pd.DataFrame,
    priority_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Algorithm 2: historical ``P(holds y | primary = x)`` proportion table."""
    priority_list = priority_list or DEFAULT_PRIORITY_LIST
    qual = build_qual_data(qual_data_df)

    rows = []
    for x in priority_list:
        primary_df = qual[qual["PRIMARY_QUAL"] == x]
        row: Dict[str, float] = {
            name: float(primary_df[_display_to_column(name)].sum())
            for name in priority_list
        }
        row["_Primary_Qual"] = x
        row["_total"] = row[x]
        rows.append(row)

    prop = pd.DataFrame(rows).set_index("_Primary_Qual")
    for x in priority_list:
        total = prop["_total"].to_numpy()
        values = prop[x].to_numpy()
        prop[x] = [v / t if t else 0.0 for v, t in zip(values, total)]
    prop = prop.drop(columns=["_total"])
    return prop


# --------------------------------------------------------------------------- #
# Algorithm 3 - Phase 3
# --------------------------------------------------------------------------- #
def sprinkle_qualifications(
    merged_df: pd.DataFrame,
    prop_table: pd.DataFrame,
    priority_list: Optional[List[str]] = None,
    seed: int = RANDOM_SEED,
    prop_scales: Optional[Dict[Tuple[str, str], float]] = None,
) -> pd.DataFrame:
    """Algorithm 3: sprinkle secondary qualifications onto the supply data.

    ``prop_scales`` optionally scales ``prop_table`` per ``(primary, secondary)``
    pair. A missing pair or ``None`` keeps the historical proportion. A scale of
    zero assigns nobody, which in turn removes that training requirement.
    """
    priority_list = priority_list or DEFAULT_PRIORITY_LIST
    rng = random.Random(seed)
    df = merged_df.copy().reset_index(drop=True)

    # Step 1: initialize binary skill columns from each dispatcher's primary work center.
    for name in priority_list:
        col = _display_to_column(name)
        if name == "Domestic":
            df[col] = 1
        else:
            df[col] = (df["EMPLOYEE_WORK_CENTER_ID"] == name).astype(int)

    # Step 2: randomly assign secondary skills in proportion to historical rates.
    for s1, skill1 in enumerate(priority_list):
        if skill1 == "Domestic":
            continue
        w1 = df.loc[df["EMPLOYEE_WORK_CENTER_ID"] == skill1, "TECH_ID"].unique()
        n = len(w1)
        for s2, skill2 in enumerate(priority_list):
            if skill2 == "Domestic" or s1 == s2:
                continue
            scale = (
                1.0
                if prop_scales is None
                else float(prop_scales.get((skill1, skill2), 1.0))
            )
            prop = float(prop_table.loc[skill1, skill2]) * scale
            if prop <= 0:
                continue
            k = int(math.floor(n * prop)) + 1
            col2 = _display_to_column(skill2)
            eligible = df.loc[
                (df["EMPLOYEE_WORK_CENTER_ID"] == skill1) & (df[col2] == 0), "TECH_ID"
            ].unique()
            if len(eligible) == 0:
                continue
            k = min(k, len(eligible))
            sampled = rng.sample(list(eligible), k)
            df.loc[df["TECH_ID"].isin(sampled), col2] = 1

    # Step 3: derive Qual_Count, Primary_Qual and Final_Quals.
    df["QUAL_COUNT"] = df[FLAG_COLUMNS].sum(axis=1).astype(int)
    df["PRIMARY_QUAL"] = df["EMPLOYEE_WORK_CENTER_ID"]

    flag_records = (
        df[FLAG_COLUMNS]
        .rename(columns={v: k for k, v in DISPLAY_TO_COLUMN.items()})
        .to_dict(orient="records")
    )
    df["FINAL_QUALS"] = [
        format_final_quals(_order_final_quals(primary, flags, priority_list))
        for primary, flags in zip(df["PRIMARY_QUAL"], flag_records)
    ]
    return df


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def apply_qualification_pipeline(
    regular_path: str,
    standard_training_path: str,
    special_training_path: str,
    qual_data_path: str = "inputs/Supply_sample.csv",
    priority_list: Optional[List[str]] = None,
    seed: int = RANDOM_SEED,
    dedupe: bool = False,
    prop_scales: Optional[Dict[Tuple[str, str], float]] = None,
    output_paths: Optional[Dict[str, str]] = None,
    write: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run Algorithms 1-3 over the three supply files as a single cohort.

    The three files are concatenated so every ``TECH_ID`` receives one consistent
    qualification profile, the algorithms run once, and the result is split back
    into the original files. Output paths default to ``<input>_qualified.csv``.
    """
    priority_list = priority_list or DEFAULT_PRIORITY_LIST

    inputs = {
        "regular": regular_path,
        "standard": standard_training_path,
        "special": special_training_path,
    }

    frames = []
    base_columns: Optional[List[str]] = None
    for source, path in inputs.items():
        frame = pd.read_csv(path)
        if base_columns is None:
            base_columns = list(frame.columns)
        frame["_SOURCE"] = source
        frames.append(frame)

    cohort = pd.concat(frames, ignore_index=True)
    qual_data = pd.read_csv(qual_data_path)

    merged = map_qualifications_to_supply(
        cohort, qual_data, priority_list, dedupe=dedupe
    )
    prop_table = build_prop_table(qual_data, priority_list)
    merged = sprinkle_qualifications(
        merged, prop_table, priority_list, seed=seed, prop_scales=prop_scales
    )

    if output_paths is None:
        output_paths = {
            source: _default_out_path(path) for source, path in inputs.items()
        }

    results: Dict[str, pd.DataFrame] = {}
    for source in inputs:
        sub = merged.loc[merged["_SOURCE"] == source].drop(columns=["_SOURCE"])
        # Preserve the input schema exactly (drops MAPPED_QUAL_ID and any helper columns).
        ordered = [c for c in (base_columns or _SCHEMA_COLUMNS) if c in sub.columns]
        sub = sub[ordered]
        results[source] = sub
        if write:
            sub.to_csv(output_paths[source], index=False)

    return results


def validate_quals(
    df: pd.DataFrame, priority_list: Optional[List[str]] = None
) -> List[str]:
    """Return a list of consistency problems (empty == valid)."""
    priority_list = priority_list or DEFAULT_PRIORITY_LIST
    problems: List[str] = []

    if (df["DOMESTIC"] != 1).any():
        problems.append("some rows do not hold the Domestic qualification")

    flag_sums = df[FLAG_COLUMNS].sum(axis=1)
    if not (flag_sums == df["QUAL_COUNT"]).all():
        problems.append("QUAL_COUNT != sum of the eight binary flags on some rows")

    parsed = df["FINAL_QUALS"].map(lambda c: set(parse_final_quals(c)))
    if not (parsed.map(len) == df["QUAL_COUNT"]).all():
        problems.append("QUAL_COUNT != number of entries in FINAL_QUALS on some rows")

    flagged = df[FLAG_COLUMNS].rename(
        columns={v: k for k, v in DISPLAY_TO_COLUMN.items()}
    )
    flagged_sets = flagged.apply(
        lambda r: {name for name in priority_list if r[name] == 1}, axis=1
    )
    if not (flagged_sets == parsed).all():
        problems.append("FINAL_QUALS set != flagged columns set on some rows")

    prim_flag = df.apply(
        lambda r: r[_display_to_column(r["PRIMARY_QUAL"])] == 1, axis=1
    )
    if not prim_flag.all():
        problems.append("PRIMARY_QUAL flag is not 1 on some rows")

    if not (
        (df["PRIMARY_QUAL"] == df["EMPLOYEE_WORK_CENTER_ID"]).all()
        and (df["PRIMARY_QUAL"] == df["WORK_CENTER_ID"]).all()
    ):
        problems.append("PRIMARY_QUAL != work center id on some rows")

    return problems
