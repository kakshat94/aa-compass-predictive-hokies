"""End-to-end verification for the qualification pipeline (Algorithms 1-3).

Run from the repo root:

    python3 members/daniel/verify_pipeline.py

Checks performed:
  1. Reference-data identity: the ``EMP_ID`` values in the training
     ``*_scheduled_df_reference`` files must equal, as a set, the ``TECH_ID``
     values in the reference supply (``initial_supply_data_reference.csv``).
     This is the "emp_id matches tech_id" assumption that Algorithm 1's
     one-to-one mapping relies on.
  2. The generated ``*_qualified.csv`` files pass ``validate_quals`` (Domestic
     held everywhere, Qual_Count == flags == len(Final_Quals), primary flagged,
     primary == work center).
  3. Every ``TECH_ID`` carries exactly one qualification profile across all
     three output files (cross-file consistency).

Exit code is 0 when every check passes, 1 otherwise.
"""

import argparse
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd  # noqa: E402

from pipelines.phase2_training.qualification_mapper import (  # noqa: E402
    parse_final_quals,
    validate_quals,
)

# Competition reference files live outside the repo on this machine.
DEFAULT_REFERENCE_DIR = os.path.expanduser(
    "~/Desktop/University work/AA Competition/data"
)

_SUPPLY_REFERENCE = ["initial_supply_data_reference.csv", "Supply_sample.csv"]
_SCHEDULED_REFERENCES = [
    "standard_training_scheduled_df_reference.csv",
    "intl_special_training_scheduled_df_reference.csv",
]


def _find(reference_dir, candidates):
    for name in candidates:
        path = os.path.join(reference_dir, name)
        if os.path.exists(path):
            return path
    return None


def verify_emp_id_tech_id(reference_dir):
    """Verify EMP_ID (training scheduled views) == TECH_ID (reference supply) as sets.

    Returns a report dict with ``ok`` and details, or ``None`` if the reference
    files are not available.
    """
    supply_path = _find(reference_dir, _SUPPLY_REFERENCE)
    scheduled_paths = [
        os.path.join(reference_dir, name) for name in _SCHEDULED_REFERENCES
    ]
    if supply_path is None or not all(os.path.exists(p) for p in scheduled_paths):
        return None

    tech_ids = set(pd.read_csv(supply_path, usecols=["TECH_ID"])["TECH_ID"].unique())
    emp_ids = set()
    per_file = {}
    for path in scheduled_paths:
        ids = set(pd.read_csv(path, usecols=["EMP_ID"])["EMP_ID"].unique())
        per_file[os.path.basename(path)] = ids
        emp_ids |= ids

    return {
        "ok": emp_ids == tech_ids,
        "supply_path": supply_path,
        "n_tech_id": len(tech_ids),
        "n_emp_id": len(emp_ids),
        "emp_only": sorted(emp_ids - tech_ids),
        "tech_only": sorted(tech_ids - emp_ids),
        "per_file": per_file,
    }


def verify_outputs(paths):
    """Validate each output file and check cross-file profile consistency."""
    frames = {name: pd.read_csv(path) for name, path in paths.items()}

    per_file_problems = {name: validate_quals(df) for name, df in frames.items()}

    combined = pd.concat(frames.values(), ignore_index=True)
    profiles = combined.drop_duplicates(subset=["TECH_ID", "FINAL_QUALS", "QUAL_COUNT"])
    dupes = profiles.groupby("TECH_ID").size()
    inconsistent = sorted(dupes[dupes > 1].index.tolist())

    n_tech = combined["TECH_ID"].nunique()
    return {
        "ok": all(not p for p in per_file_problems.values()) and not inconsistent,
        "per_file_problems": per_file_problems,
        "n_tech": n_tech,
        "inconsistent_tech_ids": inconsistent,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify the qualification pipeline.")
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--regular", default="outputs/Supply_Regular_qualified.csv")
    parser.add_argument("--standard", default="outputs/standard_training_data.csv")
    parser.add_argument("--special", default="outputs/intl_special_training_data.csv")
    args = parser.parse_args()

    failures = 0
    logging.info("\n================ VERIFICATION: members/daniel ================")

    emp_report = verify_emp_id_tech_id(args.reference_dir)
    logging.info("\n[1] EMP_ID <-> TECH_ID (reference data)")
    if emp_report is None:
        logging.info(
            f"    SKIPPED: reference files not found under {args.reference_dir!r}"
        )
    else:
        status = "PASS" if emp_report["ok"] else "FAIL"
        logging.info(f"    {status}: EMP_ID set == TECH_ID set")
        logging.info(
            f"      supply source    : {os.path.basename(emp_report['supply_path'])}"
        )
        logging.info(
            f"      |TECH_ID| = {emp_report['n_tech_id']}   |EMP_ID| = {emp_report['n_emp_id']}"
        )
        for name, ids in emp_report["per_file"].items():
            logging.info(f"      {name}: {len(ids)} ids")
        if not emp_report["ok"]:
            logging.info(f"      EMP_ID-only : {emp_report['emp_only'][:10]}")
            logging.info(f"      TECH_ID-only: {emp_report['tech_only'][:10]}")
            failures += 1

    out_report = verify_outputs(
        {"regular": args.regular, "standard": args.standard, "special": args.special}
    )
    logging.info("\n[2] Generated *_qualified.csv files")
    for name, problems in out_report["per_file_problems"].items():
        logging.info(f"    {name:9s}: {'PASS' if not problems else problems}")
        if problems:
            failures += 1

    logging.info("\n[3] Cross-file profile consistency")
    if out_report["inconsistent_tech_ids"]:
        logging.info(
            f"    FAIL: {len(out_report['inconsistent_tech_ids'])} TECH_IDs have >1 profile"
        )
        failures += 1
    else:
        logging.info(
            f"    PASS: all {out_report['n_tech']} TECH_IDs have one profile across the 3 files"
        )

    logging.info("\n=============================================================")
    logging.info(
        "RESULT:",
        "ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED",
    )
    logging.info("=============================================================\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
