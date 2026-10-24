"""Load and qualify pre-assigned flex days from the Phase 1 MIP.

Phase 1 writes ``Flex_assignment.csv`` with the same 25-column schema as a
supply file. Its qualification columns are empty because qualifications are
assigned later, in Phase 2. This module reads those rows, exposes their
``(TECH_ID, SHIFT_DATE)`` keys for training exclusion, and writes a qualified
copy once the secondary qualifications are known.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

from pipelines.phase2_training.qualification_mapper import (
    FLAG_COLUMNS,
    format_final_quals,
    parse_final_quals,
)
from pipelines.phase2_training.training_assigner import _format_date

QUAL_COLUMNS = ["FINAL_QUALS", "PRIMARY_QUAL", *FLAG_COLUMNS, "QUAL_COUNT"]
FlexKey = Tuple[str, str]


def _normalized_dates(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, format="mixed", errors="coerce")
    if parsed.isna().any():
        examples = values[parsed.isna()].head(5).tolist()
        raise ValueError(f"{name} has invalid SHIFT_DATE values: {examples}")
    return parsed.map(_format_date)


def load_flex_assignments(
    path: Optional[str],
    *,
    optional: bool = True,
    name: str = "pre-assigned flex",
) -> Tuple[pd.DataFrame, Set[FlexKey]]:
    """Return the flex rows and their normalized ``(TECH_ID, date)`` keys.

    ``path`` may be ``None`` or a missing optional file, in which case the
    result is an empty frame and empty key set.
    """
    columns = ["TECH_ID", "SHIFT_DATE", *QUAL_COLUMNS]
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return pd.DataFrame(columns=columns), set()

    source = Path(path)
    if not source.is_file():
        if optional:
            print(f"WARNING: {name} file not found at {source}; continuing without pre-assigned flex.")
            return pd.DataFrame(columns=columns), set()
        raise FileNotFoundError(f"Missing {name}: {source.resolve()}")

    frame = pd.read_csv(source, low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip().str.upper()
    missing = {"TECH_ID", "SHIFT_DATE"} - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns {sorted(missing)}: {source}")

    tech = frame["TECH_ID"].astype("string").str.strip()
    if tech.isna().any() or (tech == "").any():
        raise ValueError(f"{name} has missing TECH_ID values: {source}")
    dates = _normalized_dates(frame["SHIFT_DATE"], name)

    keys = set(zip(tech.astype(str), dates.astype(str)))
    if len(keys) != len(frame):
        duplicates = frame.loc[
            pd.DataFrame({"TECH_ID": tech.astype(str), "SHIFT_DATE": dates}).duplicated(keep=False)
        ]
        examples = duplicates[["TECH_ID", "SHIFT_DATE"]].head(5).to_dict("records")
        raise ValueError(f"{name} has duplicate (TECH_ID, SHIFT_DATE) rows: {examples}")

    return frame, keys


def enrich_flex_assignments(flex_df: pd.DataFrame, qualified_roster: pd.DataFrame) -> pd.DataFrame:
    """Fill qualification columns and add ``SECONDARY_QUALS`` to flex rows."""
    if flex_df.empty:
        return flex_df.copy()

    roster = qualified_roster.copy()
    if "TECH_ID" not in roster:
        raise ValueError("qualified roster is missing TECH_ID")
    roster["_TECH_KEY"] = roster["TECH_ID"].astype("string").str.strip()
    profiles = roster.drop_duplicates("_TECH_KEY").set_index("_TECH_KEY")

    missing_columns = [col for col in QUAL_COLUMNS if col not in profiles]
    if missing_columns:
        raise ValueError(f"qualified roster is missing qualification columns {missing_columns}")

    result = flex_df.copy()
    result["_TECH_KEY"] = result["TECH_ID"].astype("string").str.strip()
    unknown = sorted(set(result["_TECH_KEY"]) - set(profiles.index))
    if unknown:
        raise ValueError(
            f"{len(unknown)} flex TECH_IDs are absent from the qualified roster; "
            f"examples: {unknown[:5]}"
        )

    for column in QUAL_COLUMNS:
        result[column] = result["_TECH_KEY"].map(profiles[column])

    final_quals = result["FINAL_QUALS"].map(parse_final_quals)
    primary = result["PRIMARY_QUAL"].astype(str)
    secondary = [
        [qual for qual in quals if qual != home]
        for quals, home in zip(final_quals, primary)
    ]
    if "SECONDARY_QUALS" in result:
        result = result.drop(columns=["SECONDARY_QUALS"])
    result["SECONDARY_QUALS"] = [format_final_quals(quals) for quals in secondary]
    return result.drop(columns=["_TECH_KEY"])


def write_flex_assignments(flex_df: pd.DataFrame, path: str) -> Path:
    """Write the qualified flex copy, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flex_df.to_csv(destination, index=False)
    return destination
