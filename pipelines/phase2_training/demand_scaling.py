"""Scaled demand features from ``inputs/annual_desk_needs_reference.csv`` (COMPASS Phase II).

Reproduces the ``_scale_to_range_neg_1_to_1`` helper from the "Vacation
information summary.docx" and applies it to two accumulations of desk demand:

  1. per ``(SHIFT_TYPE, SHIFT_DATE)`` -- summed across work centers;
  2. per ``(SHIFT_TYPE, SHIFT_DATE, WORK_CENTER)``.

Each output's ``DEMAND_TOTAL`` is linearly rescaled so its minimum maps to -1 and
its maximum to +1 (flat series -> 0).

Run from the repo root:

    python3 members/daniel/demand_scaling.py
"""

import argparse
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd  # noqa: E402


def _scale_to_range_neg_1_to_1(series: pd.Series) -> pd.Series:
    """Linearly rescale to [-1, 1] (min -> -1, max -> +1); zeros if flat."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    scaled = (series - min_val) / (max_val - min_val)  # [0, 1]
    return scaled * 2 - 1


def build_scaled_demand(
    demand_path: str = "inputs/annual_desk_needs_reference.csv",
    out_dir: str = "pipelines/phase2_training/outputs",
):
    df = pd.read_csv(demand_path)
    df["_DT"] = pd.to_datetime(df["SHIFT_DATE"])
    df["SHIFT_DATE"] = df["_DT"].map(lambda d: f"{d.month}/{d.day}/{d.year}")

    def _aggregate(keys):
        agg = (
            df.groupby(keys + ["_DT"], as_index=False)["DEMAND"]
            .sum()
            .rename(columns={"DEMAND": "DEMAND_TOTAL"})
            .sort_values(keys + ["_DT"])
        )
        agg["SCALED_DEMAND"] = _scale_to_range_neg_1_to_1(agg["DEMAND_TOTAL"])
        return agg[keys + ["DEMAND_TOTAL", "SCALED_DEMAND"]].reset_index(drop=True)

    per_shift_date = _aggregate(["SHIFT_TYPE", "SHIFT_DATE"])
    per_shift_date_wc = _aggregate(["SHIFT_TYPE", "SHIFT_DATE", "WORK_CENTER"])

    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "demand_shift_date_scaled.csv")
    p2 = os.path.join(out_dir, "demand_shift_date_work_center_scaled.csv")
    per_shift_date.to_csv(p1, index=False)
    per_shift_date_wc.to_csv(p2, index=False)
    return per_shift_date, per_shift_date_wc, p1, p2


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scaled demand features.")
    parser.add_argument("--demand", default="inputs/annual_desk_needs_reference.csv")
    parser.add_argument("--out-dir", default="pipelines/daniel/outputs")
    args = parser.parse_args()

    per_sd, per_sd_wc, p1, p2 = build_scaled_demand(args.demand, args.out_dir)
    logging.info("\n================ DEMAND SCALING ================")
    logging.info(f"(SHIFT_TYPE, SHIFT_DATE)           rows={len(per_sd):,}  -> {p1}")
    logging.info(
        f"   SCALED_DEMAND range [{per_sd['SCALED_DEMAND'].min():.3f}, {per_sd['SCALED_DEMAND'].max():.3f}]"
    )
    logging.info(
        f"(SHIFT_TYPE, SHIFT_DATE, WORK_CENTER) rows={len(per_sd_wc):,}  -> {p2}"
    )
    logging.info(
        f"   SCALED_DEMAND range [{per_sd_wc['SCALED_DEMAND'].min():.3f}, {per_sd_wc['SCALED_DEMAND'].max():.3f}]"
    )
    logging.info("================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
