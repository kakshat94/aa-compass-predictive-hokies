"""Effective-supply coverage scoring + shrinkage reconciliation (members/daniel).

Reuses ``common.coverage_report.CoverageReporter`` for the scoring metrics (it is
the single source of truth -- this file does not fork it). On top of that it adds:

  * a convenience for scoring the *effective* supply (the roster minus vacation and
    training), which is the file that should be measured against demand;
  * reconciliation checks proving the shrinkage files are consistent, so the
    effective supply is trustworthy:
      - regular == effective + vacation (disjoint partition);
      - vacation is a subset of the roster;
      - training days are disjoint from the roster and from vacation;
      - the two training files do not overlap;
      - no duplicate (tech, date) within vacation or within a training file.

Run from the repo root:

    python3 members/daniel/coverage_report.py
    python3 members/daniel/coverage_report.py --score-only
    python3 members/daniel/coverage_report.py --reconcile-only
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd  # noqa: E402

from common.coverage_report import CoverageReporter  # noqa: E402

DEMAND_PATH = "data/Demand.csv"
REGULAR_PATH = "outputs/Supply_Regular_qualified.csv"
STANDARD_TRAINING_PATH = "outputs/standard_training_data.csv"
INTL_TRAINING_PATH = "outputs/intl_special_training_data.csv"
VACATION_PATH = "outputs/vacation_data.csv"
EFFECTIVE_PATH = "outputs/supply_data_after_vacation_shrinkage.csv"


def _key_set(df: pd.DataFrame) -> set:
    """Set of (TECH_ID as str, YYYY-MM-DD) identifying a bid line-day."""
    dates = pd.to_datetime(df["SHIFT_DATE"]).dt.strftime("%Y-%m-%d")
    return set(zip(df["TECH_ID"].astype(str), dates))


def evaluate_effective(demand_path: str = DEMAND_PATH, effective_path: str = EFFECTIVE_PATH) -> pd.DataFrame:
    """Score the effective supply against demand using the shared scorer."""
    return CoverageReporter.evaluate(demand_path, effective_path)


def score(demand_path: str = DEMAND_PATH, effective_path: str = EFFECTIVE_PATH,
          name: str = "Effective supply", output: str = None) -> pd.DataFrame:
    report = evaluate_effective(demand_path, effective_path)
    CoverageReporter.print_summary(report, stage=name)
    if output:
        report.to_csv(output, index=False)
        print(f"Detailed coverage report saved to {output}")
    return report


def reconciliation_checks(
    regular_path: str = REGULAR_PATH,
    standard_training_path: str = STANDARD_TRAINING_PATH,
    intl_training_path: str = INTL_TRAINING_PATH,
    vacation_path: str = VACATION_PATH,
    effective_path: str = EFFECTIVE_PATH,
) -> list:
    """Return a list of consistency problems (empty == the shrinkage reconciles)."""
    regular = pd.read_csv(regular_path)
    standard = pd.read_csv(standard_training_path)
    intl = pd.read_csv(intl_training_path)
    vacation = pd.read_csv(vacation_path)
    effective = pd.read_csv(effective_path)

    reg_k = _key_set(regular)
    eff_k = _key_set(effective)
    vac_k = _key_set(vacation)
    std_k = _key_set(standard)
    intl_k = _key_set(intl)
    train_k = std_k | intl_k

    problems = []

    # 1. regular == effective + vacation (disjoint partition).
    if eff_k & vac_k:
        problems.append(f"effective ∩ vacation != empty ({len(eff_k & vac_k)} shared bid line-days)")
    if (eff_k | vac_k) != reg_k:
        missing = (reg_k - eff_k - vac_k)
        extra = (eff_k | vac_k) - reg_k
        problems.append(
            f"effective ∪ vacation != regular ({len(missing)} regular day(s) unaccounted, "
            f"{len(extra)} non-regular day(s) present)"
        )
    if len(regular) != len(effective) + len(vacation):
        problems.append(
            f"row count mismatch: regular {len(regular)} != effective {len(effective)} + vacation {len(vacation)}"
        )

    # 2. vacation is a subset of the roster.
    if not vac_k <= reg_k:
        problems.append(f"vacation not a subset of regular ({len(vac_k - reg_k)} days outside the roster)")

    # 3. training disjoint from the roster and from vacation.
    if reg_k & train_k:
        problems.append(f"training overlaps the regular roster ({len(reg_k & train_k)} shared days)")
    if vac_k & train_k:
        problems.append(f"vacation overlaps training ({len(vac_k & train_k)} shared days)")

    # 4. the two training files do not overlap.
    if std_k & intl_k:
        problems.append(f"standard and intl training overlap ({len(std_k & intl_k)} shared days)")

    # 5. no duplicate (tech, date) within vacation or a training file.
    if len(vac_k) != len(vacation):
        problems.append(f"vacation has {len(vacation) - len(vac_k)} duplicate (tech, date) row(s)")
    if len(std_k) != len(standard):
        problems.append(f"standard training has {len(standard) - len(std_k)} duplicate (tech, date) row(s)")
    if len(intl_k) != len(intl):
        problems.append(f"intl training has {len(intl) - len(intl_k)} duplicate (tech, date) row(s)")

    return problems


def run(
    demand_path: str = DEMAND_PATH,
    regular_path: str = REGULAR_PATH,
    standard_training_path: str = STANDARD_TRAINING_PATH,
    intl_training_path: str = INTL_TRAINING_PATH,
    vacation_path: str = VACATION_PATH,
    effective_path: str = EFFECTIVE_PATH,
    name: str = "Effective supply",
    output: str = None,
    do_score: bool = True,
    do_reconcile: bool = True,
):
    """Score the effective supply and/or reconcile the shrinkage files."""
    report = None
    problems = []
    if do_score:
        report = score(demand_path, effective_path, name=name, output=output)
    if do_reconcile:
        problems = reconciliation_checks(
            regular_path, standard_training_path, intl_training_path, vacation_path, effective_path
        )
        if problems:
            print(f"RECONCILIATION FAIL ({len(problems)} problem(s)):")
            for p in problems:
                print("  -", p)
        else:
            print("RECONCILIATION OK: regular == effective + vacation; vacation ⊆ roster; "
                  "training disjoint from roster and vacation\n")
    return report, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Score effective supply and reconcile shrinkage files.")
    parser.add_argument("--demand", default=DEMAND_PATH)
    parser.add_argument("--effective", "--supply", dest="effective", default=EFFECTIVE_PATH)
    parser.add_argument("--regular", default=REGULAR_PATH)
    parser.add_argument("--standard", default=STANDARD_TRAINING_PATH)
    parser.add_argument("--intl", default=INTL_TRAINING_PATH)
    parser.add_argument("--vacation", default=VACATION_PATH)
    parser.add_argument("--name", default="Effective supply")
    parser.add_argument("--output", default=None, help="Optional path to save the detail report CSV")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--reconcile-only", action="store_true")
    args = parser.parse_args()

    _, problems = run(
        demand_path=args.demand,
        regular_path=args.regular,
        standard_training_path=args.standard,
        intl_training_path=args.intl,
        vacation_path=args.vacation,
        effective_path=args.effective,
        name=args.name,
        output=args.output,
        do_score=not args.reconcile_only,
        do_reconcile=not args.score_only,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
