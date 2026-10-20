"""Random training assignment for bid lines (COMPASS Phase II).

Builds a random training solution over the full working-day schedule, honouring
the rules from "Training assignments information.docx" and the eligible month
windows in ``members/akshat/params.yaml`` (``training_months``):

  * one training day per held qualification per month in that qualification's
    ``training_months`` (holds for primary AND secondary qualifications);
  * training falls only on a day the bid line is scheduled to work;
  * no more than one training per bid line per day;
  * across all work centers/skills, at most ``CLASS_SIZE`` trainees per
    ``(SHIFT_DATE, SHIFT_TYPE)`` (the docx's shared per-shift class limit);
  * flex-day and vacation rules are accepted as optional sets but are not yet
    populated by the pipeline (ignored for now).

Outputs match the competition reference files:
  * ``standard_training_data`` / ``intl_special_training_data``  (shift-like rows)
  * ``standard_training_scheduled_df`` / ``intl_special_training_scheduled_df``
"""

from __future__ import annotations

import calendar
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

from members.daniel.qualification_mapper import parse_final_quals

CLASS_SIZE_DEFAULT = 12
RANDOM_SEED = 42

# 23-column bid line schema (Data Specification section 4.1, without INDEX/SF_LOAD_TMS).
SCHEMA_23 = [
    "MAJOR_WORK_TYPE", "START_TIME", "EMPLOYEE_TITLE", "WORK_CENTER_ID",
    "CONTRACTOR_FLAG", "CLUSTER", "EMPLOYEE_WORK_CENTER_ID", "TECH_ID", "END_TIME",
    "SHIFT_DATE", "SHIFT_LENGTH", "SHIFT_TYPE", "FINAL_QUALS", "PRIMARY_QUAL",
    "DOMESTIC", "ATC", "ROC", "CHF", "OC", "LATIN", "EUROPE", "PACIFIC",
    "QUAL_COUNT",
]

SCHEDULED_COLUMNS = [
    "EMP_ID", "SKILL", "TRAINING_SCHEDULE_DATE", "PERIOD",
    "FROM_SHIFT_TYPE", "FROM_WORK_CENTER_ID",
]

_MONTH_NAME_TO_INT = {
    calendar.month_name[m].lower(): m for m in range(1, 13)
} | {
    calendar.month_abbr[m].lower(): m for m in range(1, 13)
}


@dataclass
class TrainingResult:
    regular: pd.DataFrame
    standard_data: pd.DataFrame
    intl_data: pd.DataFrame
    standard_scheduled: pd.DataFrame
    intl_scheduled: pd.DataFrame
    failures: List[Tuple[object, str, int]]
    report: dict = field(default_factory=dict)


def _format_date(dt) -> str:
    """Cross-platform non-zero-padded M/D/YYYY."""
    return f"{dt.month}/{dt.day}/{dt.year}"


def load_frozen_assignments(
    standard_path: str,
    special_path: str,
    training_months: Dict[str, List[int]],
) -> Dict[Tuple[object, str], str]:
    """Read the Phase I training files as ``(tech, date) -> skill`` assignments.

    Phase I splits training by month, not by skill: every row in the standard
    file is the Domestic qualification, while a special-file row trains the
    dispatcher's primary (home work center) qualification. Dates are kept in the
    non-padded ``M/D/YYYY`` form used everywhere else in the pipeline.
    """
    frames = [
        (pd.read_csv(standard_path), "Domestic"),
        (pd.read_csv(special_path), None),
    ]

    frozen: Dict[Tuple[object, str], str] = {}
    for frame, fixed_skill in frames:
        for _, row in frame.iterrows():
            tech = row["TECH_ID"]
            date = _format_date(pd.to_datetime(row["SHIFT_DATE"]))
            skill = fixed_skill or str(row["WORK_CENTER_ID"])
            key = (tech, date)
            if key in frozen:
                raise ValueError(
                    f"Phase I training has two rows for TECH_ID {tech} on {date}"
                )
            month = pd.to_datetime(row["SHIFT_DATE"]).month
            if month not in training_months.get(skill, []):
                raise ValueError(
                    f"Phase I training for TECH_ID {tech} uses skill {skill!r} "
                    f"in month {month}, which is not an eligible training month"
                )
            frozen[key] = skill
    return frozen


def load_training_months(params_path: str = "members/akshat/params.yaml") -> Dict[str, List[int]]:
    """Read ``training_months`` per work center from params.yaml as month ints."""
    with open(params_path, "r") as fh:
        params = yaml.safe_load(fh)
    work_centers = params.get("work_centers", {})
    result: Dict[str, List[int]] = {}
    for wc, rules in work_centers.items():
        months = []
        for name in (rules or {}).get("training_months", []) or []:
            key = str(name).strip().lower()
            if key in _MONTH_NAME_TO_INT:
                months.append(_MONTH_NAME_TO_INT[key])
            else:
                raise ValueError(f"Unknown training month {name!r} for work center {wc!r}")
        result[wc] = sorted(set(months))
    return result


def _period_standard(month: int) -> str:
    """Domestic scheduled-view PERIOD: ``(m, m)`` for the row's training month."""
    return f"({month}, {month})"


def _period_intl(months: List[int]) -> str:
    """Non-Domestic scheduled-view PERIOD: the skill's full eligible window, e.g. ``[3, 11]``."""
    return "[" + ", ".join(str(m) for m in sorted(months)) + "]"


def scheduled_view(sub: pd.DataFrame, training_months: Dict[str, List[int]]) -> pd.DataFrame:
    """Build the ``EMP_ID,SKILL,...,PERIOD`` scheduling view for training rows.

    Expects the helper columns ``_TRAIN_SKILL``, ``_MONTH``, ``_DATE`` produced
    while a solution is assembled (see :func:`build_result_from_marks`).
    """
    rows = []
    for _, r in sub.iterrows():
        skill = r["_TRAIN_SKILL"]
        if skill == "Domestic":
            period = _period_standard(int(r["_MONTH"]))
        else:
            period = _period_intl(training_months.get(skill, []))
        rows.append({
            "EMP_ID": r["TECH_ID"],
            "SKILL": skill,
            "TRAINING_SCHEDULE_DATE": r["_DATE"],
            "PERIOD": period,
            "FROM_SHIFT_TYPE": r["SHIFT_TYPE"],
            "FROM_WORK_CENTER_ID": r["EMPLOYEE_WORK_CENTER_ID"],
        })
    return pd.DataFrame(rows, columns=SCHEDULED_COLUMNS)


def build_result_from_marks(
    df: pd.DataFrame,
    training_months: Dict[str, List[int]],
    failures: List[Tuple[object, str, int]],
) -> TrainingResult:
    """Assemble a :class:`TrainingResult` from a roster with ``_TRAIN_SKILL`` marks.

    ``df`` carries every scheduled working-day row plus the helper columns
    ``_DATE`` (non-padded ``M/D/YYYY``), ``_MONTH`` (int) and ``_TRAIN_SKILL``
    (the skill name for a training row, else ``NaN``/``None``). This is the single
    place the competition output schema is produced, so the random and exact
    solvers stay byte-for-byte compatible downstream.
    """
    df = df.copy()
    is_training = df["_TRAIN_SKILL"].notna()

    training = df[is_training].copy()
    regular = df[~is_training].drop(
        columns=["_DT", "_DATE", "_MONTH", "_TRAIN_SKILL"], errors="ignore"
    )

    standard = training[training["_TRAIN_SKILL"] == "Domestic"].copy()
    intl = training[training["_TRAIN_SKILL"] != "Domestic"].copy()

    standard_scheduled = scheduled_view(standard, training_months)
    intl_scheduled = scheduled_view(intl, training_months)

    standard_data = standard[SCHEMA_23].reset_index(drop=True)
    intl_cols = SCHEMA_23[:] + ["MONTH"]
    intl_data = intl[SCHEMA_23].copy()
    intl_data["MONTH"] = intl["_MONTH"].values
    intl_data = intl_data[intl_cols].reset_index(drop=True)

    return TrainingResult(
        regular=regular,
        standard_data=standard_data,
        intl_data=intl_data,
        standard_scheduled=standard_scheduled,
        intl_scheduled=intl_scheduled,
        failures=failures,
    )


def assign_training(
    schedule: pd.DataFrame,
    training_months: Dict[str, List[int]],
    class_size: int = CLASS_SIZE_DEFAULT,
    seed: int = RANDOM_SEED,
    flex_days: Optional[Set[Tuple[object, str]]] = None,
    vacation_days: Optional[Set[Tuple[object, str]]] = None,
    frozen: Optional[Dict[Tuple[object, str], str]] = None,
) -> TrainingResult:
    """Assign one random training day per (bid line, held qualification, eligible month).

    ``schedule`` is the full working-day roster (regular + training concatenated),
    one row per dispatcher-day, with qualification columns populated. ``frozen``
    optionally pins ``(tech, date) -> skill`` assignments from Phase I; those
    rows stay put, their requirements are not reassigned, and only the remaining
    secondary-qualification requirements are placed.
    """
    flex_days = flex_days or set()
    vacation_days = vacation_days or set()
    frozen = dict(frozen or {})
    rng = random.Random(seed)

    df = schedule.copy().reset_index(drop=True)
    df["_DT"] = pd.to_datetime(df["SHIFT_DATE"])
    df["_DATE"] = df["_DT"].map(_format_date)
    df["_MONTH"] = df["_DT"].dt.month

    # Per-tech working days and shift type (a tech has a single shift type).
    tech_days: Dict[object, List[Tuple[str, int]]] = {}
    tech_shift: Dict[object, str] = {}
    for tech, grp in df.groupby("TECH_ID", sort=False):
        tech_days[tech] = list(zip(grp["_DATE"], grp["_MONTH"]))
        tech_shift[tech] = str(grp["SHIFT_TYPE"].iloc[0])

    # Held qualifications per tech.
    tech_quals: Dict[object, List[str]] = (
        df.drop_duplicates("TECH_ID")
        .set_index("TECH_ID")["FINAL_QUALS"]
        .map(parse_final_quals)
        .to_dict()
    )

    def _cap_key(date_str: str, shift: str) -> Tuple[str, str]:
        return (date_str, shift)

    capacity: Dict[Tuple[str, str], int] = {}
    tech_used: Dict[object, Set[str]] = {}
    chosen: Dict[Tuple[object, str], str] = {}  # (tech, date) -> skill
    failures: List[Tuple[object, str, int]] = []

    # Pin the Phase I schedule first so it consumes days and class seats.
    schedule_rows = set(zip(df["TECH_ID"], df["_DATE"]))
    date_month = dict(zip(df["_DATE"], df["_MONTH"]))
    frozen_requirements: Set[Tuple[object, str, int]] = set()
    for (tech, date_str), skill in frozen.items():
        if (tech, date_str) not in schedule_rows:
            raise ValueError(
                f"frozen training for TECH_ID {tech} on {date_str} is not a working day"
            )
        if skill not in tech_quals.get(tech, []):
            raise ValueError(
                f"frozen training for TECH_ID {tech} uses unheld skill {skill!r}"
            )
        month = date_month[date_str]
        if month not in training_months.get(skill, []):
            raise ValueError(
                f"frozen training for TECH_ID {tech} uses skill {skill!r} "
                f"in month {month}, which is not an eligible training month"
            )
        frozen_requirements.add((tech, skill, month))
        tech_used.setdefault(tech, set()).add(date_str)
        cap_key = _cap_key(date_str, tech_shift[tech])
        capacity[cap_key] = capacity.get(cap_key, 0) + 1
        chosen[(tech, date_str)] = skill

    # Requirements: (tech, skill, month) for every held qual x eligible month,
    # minus whatever the frozen Phase I schedule already covers.
    requirements: List[Tuple[object, str, int]] = []
    for tech, quals in tech_quals.items():
        for skill in quals:
            for month in training_months.get(skill, []):
                req = (tech, skill, month)
                if req not in frozen_requirements:
                    requirements.append(req)
    rng.shuffle(requirements)

    for tech, skill, month in requirements:
        shift = tech_shift.get(tech)
        used = tech_used.setdefault(tech, set())
        candidates = [
            date_str
            for date_str, m in tech_days.get(tech, [])
            if m == month
            and date_str not in used
            and (tech, date_str) not in flex_days
            and (tech, date_str) not in vacation_days
            and capacity.get(_cap_key(date_str, shift), 0) < class_size
        ]
        if not candidates:
            failures.append((tech, skill, month))
            continue
        date_str = rng.choice(candidates)
        used.add(date_str)
        key = _cap_key(date_str, shift)
        capacity[key] = capacity.get(key, 0) + 1
        chosen[(tech, date_str)] = skill

    # Mark training rows and assemble the shared output bundle.
    df["_TRAIN_SKILL"] = [
        chosen.get((tech, date_str)) for tech, date_str in zip(df["TECH_ID"], df["_DATE"])
    ]
    return build_result_from_marks(df, training_months, failures)


def write_training_outputs(result: TrainingResult, output_dir: str = "outputs") -> Dict[str, str]:
    """Write the regular roster plus the four reference-style training files."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "regular": os.path.join(output_dir, "Supply_Regular_qualified.csv"),
        "standard_data": os.path.join(output_dir, "standard_training_data.csv"),
        "intl_data": os.path.join(output_dir, "intl_special_training_data.csv"),
        "standard_scheduled": os.path.join(output_dir, "standard_training_scheduled_df.csv"),
        "intl_scheduled": os.path.join(output_dir, "intl_special_training_scheduled_df.csv"),
    }
    result.regular.to_csv(paths["regular"], index=False)
    result.standard_data.to_csv(paths["standard_data"], index=False)
    result.intl_data.to_csv(paths["intl_data"], index=False)
    result.standard_scheduled.to_csv(paths["standard_scheduled"], index=False)
    result.intl_scheduled.to_csv(paths["intl_scheduled"], index=False)
    return paths
