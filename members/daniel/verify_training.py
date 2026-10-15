"""Verify a training solution against the params.yaml ``training_months`` windows.

Run from the repo root:

    python3 members/daniel/verify_training.py

It reads the regular roster and the four training files produced by
``members/daniel/main.py`` and checks:

  * every held qualification is trained exactly once per eligible month
    (per-tech x month count identity);
  * each scheduled training's month is an eligible month for its ``SKILL``,
    and ``PERIOD`` matches the reference encoding;
  * the data files and scheduled views correspond one-to-one;
  * ``MONTH`` (intl data) equals the shift-date month;
  * no dispatcher trains twice on the same day, and no training day is also a
    regular working day;
  * at most ``class_size`` trainees per ``(SHIFT_DATE, SHIFT_TYPE)``.
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd  # noqa: E402

from members.daniel.qualification_mapper import parse_final_quals  # noqa: E402
from members.daniel.training_assigner import (  # noqa: E402
    CLASS_SIZE_DEFAULT,
    _period_intl,
    _period_standard,
    load_training_months,
)


def _month(series) -> pd.Series:
    return pd.to_datetime(series).dt.month


def verify_training(
    regular_path: str,
    standard_data_path: str,
    intl_data_path: str,
    standard_scheduled_path: str,
    intl_scheduled_path: str,
    training_months: dict,
    class_size: int = CLASS_SIZE_DEFAULT,
) -> list:
    problems = []

    regular = pd.read_csv(regular_path)
    std = pd.read_csv(standard_data_path)
    intl = pd.read_csv(intl_data_path)
    sched_std = pd.read_csv(standard_scheduled_path)
    sched_intl = pd.read_csv(intl_scheduled_path)

    # --- data <-> scheduled one-to-one correspondence ------------------------ #
    std_pairs = set(zip(std["TECH_ID"], std["SHIFT_DATE"]))
    intl_pairs = set(zip(intl["TECH_ID"], intl["SHIFT_DATE"]))
    if std_pairs != set(zip(sched_std["EMP_ID"], sched_std["TRAINING_SCHEDULE_DATE"])):
        problems.append("standard data and scheduled view are not 1:1 on (TECH_ID, date)")
    if intl_pairs != set(zip(sched_intl["EMP_ID"], sched_intl["TRAINING_SCHEDULE_DATE"])):
        problems.append("intl data and scheduled view are not 1:1 on (TECH_ID, date)")

    # --- skill placement ----------------------------------------------------- #
    if set(sched_std["SKILL"].dropna().unique()) - {"Domestic"}:
        problems.append("standard scheduled view contains non-Domestic skills")
    intl_skills = set(sched_intl["SKILL"].dropna().unique())
    if "Domestic" in intl_skills:
        problems.append("intl scheduled view contains Domestic rows")
    unknown = (intl_skills | {"Domestic"}) - set(training_months)
    if unknown:
        problems.append(f"scheduled views use skills absent from params.yaml: {sorted(unknown)}")

    # --- schedule PERIOD / eligible-month correctness ------------------------ #
    for name, sched, is_std in [("standard", sched_std, True), ("intl", sched_intl, False)]:
        if sched.empty:
            continue
        months = _month(sched["TRAINING_SCHEDULE_DATE"])
        exp_period = [
            _period_standard(m) if is_std else _period_intl(training_months.get(s, []))
            for m, s in zip(months, sched["SKILL"])
        ]
        for m, s in zip(months, sched["SKILL"]):
            if m not in training_months.get(s, []):
                problems.append(f"{name}: {s} trained in month {m}, not in training_months")
                break
        if list(sched["PERIOD"]) != exp_period:
            problems.append(f"{name}: PERIOD encoding does not match the reference format")

    # --- intl MONTH column --------------------------------------------------- #
    intl_months = _month(intl["SHIFT_DATE"])
    if not (intl["MONTH"].astype(int) == intl_months).all():
        problems.append("intl data MONTH != shift-date month on some rows")

    # --- per-tech x month identity ------------------------------------------- #
    all_rows = pd.concat([regular, std, intl], ignore_index=True)
    tech_quals = (
        all_rows.drop_duplicates("TECH_ID")
        .set_index("TECH_ID")["FINAL_QUALS"]
        .map(parse_final_quals)
        .to_dict()
    )
    training = pd.concat(
        [
            std.assign(_m=_month(std["SHIFT_DATE"])),
            intl.assign(_m=_month(intl["SHIFT_DATE"])),
        ],
        ignore_index=True,
    )
    actual = training.groupby(["TECH_ID", "_m"]).size()
    for tech, quals in tech_quals.items():
        for m in range(1, 13):
            expected = sum(1 for q in quals if m in training_months.get(q, []))
            got = int(actual.get((tech, m), 0))
            if got != expected:
                problems.append(
                    f"TECH_ID {tech} month {m}: {got} trainings but expected {expected}"
                )
                break  # one example per tech is enough

    # --- one training per tech per day, and not on a regular day ------------- #
    if training.duplicated(subset=["TECH_ID", "SHIFT_DATE"]).any():
        problems.append("a dispatcher has more than one training on the same day")
    reg_pairs = set(zip(regular["TECH_ID"], regular["SHIFT_DATE"]))
    overlap = (std_pairs | intl_pairs) & reg_pairs
    if overlap:
        problems.append(f"{len(overlap)} training rows also appear in the regular roster")

    # --- class size per (date, shift) ---------------------------------------- #
    caps = training.groupby(["SHIFT_DATE", "SHIFT_TYPE"]).size()
    if (caps > class_size).any():
        worst = int(caps.max())
        problems.append(f"class size exceeded: max {worst} > {class_size} on some (date, shift)")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the training solution.")
    parser.add_argument("--regular", default="outputs/Supply_Regular_qualified.csv")
    parser.add_argument("--standard-data", default="outputs/standard_training_data.csv")
    parser.add_argument("--intl-data", default="outputs/intl_special_training_data.csv")
    parser.add_argument("--standard-scheduled", default="outputs/standard_training_scheduled_df.csv")
    parser.add_argument("--intl-scheduled", default="outputs/intl_special_training_scheduled_df.csv")
    parser.add_argument("--params", default="members/akshat/params.yaml")
    parser.add_argument("--class-size", type=int, default=CLASS_SIZE_DEFAULT)
    args = parser.parse_args()

    training_months = load_training_months(args.params)
    problems = verify_training(
        args.regular, args.standard_data, args.intl_data,
        args.standard_scheduled, args.intl_scheduled, training_months, args.class_size,
    )

    print("\n================ TRAINING VERIFICATION ================")
    print(f"eligible months per skill: {training_months}")
    if problems:
        print(f"FAIL ({len(problems)} problems):")
        for p in problems:
            print("  -", p)
    else:
        print("PASS: training respects params.yaml training_months and all rules")
    print("=======================================================\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
