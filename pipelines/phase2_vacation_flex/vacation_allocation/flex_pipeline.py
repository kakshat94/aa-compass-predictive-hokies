"""Flex step: read the vacation result, assign flex days optimally, write outputs.

Writes (same names as the previous flex step, in ``out_dir``):
    flex_days_new.csv                         one line per new flex day
    flex_added_supply_rows.csv                the same days as supply rows
    Flex_assignment_combined.csv              pre-assigned + new flex, pre-assigned layout
    supply_after_vacation.csv                 regular supply minus vacations
    supply_after_vacation_and_flex.csv        + valid pre-assigned flex + new flex
    staffing_coverage_before_after_flex.csv   per date / desk / shift
    flex_by_dispatcher.csv                    pre-assigned, new and remaining flex days
    preassigned_flex_conflicts.csv            pre-assigned rows that are not on an off-day
    flex_summary.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_io
from .config import AllocationConfig
from .flex_assignment import FlexConfig, assign_flex, flex_rows
from .flex_capacity import FlexCapacity
from .outputs import _json_default, _ordered
from .problem import build_problem, clean_preflex

log = logging.getLogger(__name__)
FMT = "%Y-%m-%d"


def run_flex(
    supply_df: pd.DataFrame,
    vacation_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    training: set,
    preflex_df: pd.DataFrame | None,
    cfg: AllocationConfig,
    fcfg: FlexConfig,
) -> dict:
    t0 = time.time()
    supply = data_io.clean_supply(supply_df)
    demand = data_io.clean_demand(demand_df)
    dates = [pd.Timestamp(d) for d in sorted(supply["SHIFT_DATE"].unique())]
    demand, _ = data_io.fill_missing_demand(demand, supply, dates)
    preflex = clean_preflex(preflex_df)
    problem = build_problem(supply, demand, dates, training, preflex)

    # vacations -> rows of the regular supply
    vac = vacation_df.copy()
    vac = data_io.parse_dates(vac, "SHIFT_DATE", "vacations")
    vac["TECH_ID"] = data_io._clean_ids(vac["TECH_ID"])
    vkeys = set(zip(vac["TECH_ID"], vac["SHIFT_DATE"]))
    sel = np.array(
        [
            (t, d) in vkeys
            for t, d in zip(problem.supply["TECH_ID"], problem.supply["SHIFT_DATE"])
        ]
    )
    if sel.sum() != len(vkeys):
        raise ValueError(
            f"{len(vkeys) - int(sel.sum())} vacation rows are not workdays in the supply file"
        )
    vac_b = np.bincount(problem.row_bucket[sel], minlength=problem.n_bucket)

    flex = FlexCapacity(problem, cfg)
    short = problem.shortage(vac_b)
    est = flex.evaluate(short)
    fa = assign_flex(problem, flex, short, fcfg)
    if fa.covered != est.covered:  # same arcs, same budgets: must agree exactly
        raise AssertionError(
            f"flex assignment covers {fa.covered}, max-flow says {est.covered}"
        )
    short_after = problem.shortage(vac_b, fa.per_bucket)

    # ---- tables
    home = np.array(problem.tech_home_wc, dtype=object)
    tid = np.array(problem.tech_ids, dtype=object)
    new_days = pd.DataFrame(
        {
            "TECH_ID": tid[fa.tech],
            "SHIFT_DATE": [problem.dates[d] for d in fa.day],
            "WORK_CENTER_ID": [problem.bucket_keys[b][1] for b in fa.bucket],
            "SHIFT_TYPE": [problem.bucket_shift[b] for b in fa.bucket],
            "HOME_WORK_CENTER": [problem.wc_label.get(w, w) for w in home[fa.tech]],
            "HOME_SHIFT": ["/".join(sorted(problem.tech_shifts[t])) for t in fa.tech],
        }
    )
    new_days["OTHER_WORK_CENTER"] = [
        problem.bucket_wc[b] != home[t] for t, b in zip(fa.tech, fa.bucket)
    ]
    new_days["OTHER_SHIFT"] = [problem.bucket_shift[b] not in problem.tech_shifts[t]
                               for t, b in zip(fa.tech, fa.bucket)]  # fmt: skip

    supply_cols = list(problem.supply.columns)
    added = flex_rows(problem, fa, supply_cols)
    pf_cols = (
        list(preflex.columns)
        if preflex is not None
        else supply_cols + ["SECONDARY_QUALS"]
    )
    combined = pd.concat(
        [
            preflex if preflex is not None else pd.DataFrame(columns=pf_cols),
            flex_rows(problem, fa, pf_cols),
        ],
        ignore_index=True,
    )
    after_vac = problem.supply.loc[~sel]
    pf_valid_rows = pd.DataFrame(columns=supply_cols)
    if preflex is not None and len(problem.preflex_valid):
        k = set(
            zip(problem.preflex_valid["TECH_ID"], problem.preflex_valid["SHIFT_DATE"])
        )
        pf_valid_rows = preflex[
            [(t, d) in k for t, d in zip(preflex["TECH_ID"], preflex["SHIFT_DATE"])]
        ].copy()
        if "_WC" in supply_cols and "_WC" not in pf_valid_rows.columns:
            pf_valid_rows["_WC"] = pf_valid_rows["WORK_CENTER_ID"]
        pf_valid_rows = pf_valid_rows.reindex(columns=supply_cols)
    final = pd.concat([after_vac, pf_valid_rows, added], ignore_index=True)
    final = final.sort_values(["SHIFT_DATE", "TECH_ID"], kind="stable").reset_index(
        drop=True
    )

    cov = pd.DataFrame(
        {
            "SHIFT_DATE": [k[0] for k in problem.bucket_keys],
            "WORK_CENTER": [k[1] for k in problem.bucket_keys],
            "SHIFT_TYPE": problem.bucket_shift,
            "DEMAND": problem.bucket_demand,
            "REGULAR_SUPPLY": problem.bucket_total,
            "TRAINING": problem.bucket_training,
            "VACATIONS": vac_b,
            "PREASSIGNED_FLEX": problem.bucket_extra,
            "SUPPLY_AFTER_VACATION": problem.supply_after(vac_b),
            "SHORTAGE_AFTER_VACATION": short,
            "NEW_FLEX": fa.per_bucket,
            "SUPPLY_AFTER_FLEX": problem.supply_after(vac_b, fa.per_bucket),
            "SHORTAGE_AFTER_FLEX": short_after,
        }
    )
    by_person = pd.DataFrame(
        {
            "TECH_ID": tid,
            "HOME_WORK_CENTER": [problem.wc_label.get(w, w) for w in home],
            "PREASSIGNED_FLEX": problem.tech_preflex,
            "NEW_FLEX": fa.per_tech,
            "TOTAL_FLEX": problem.tech_preflex + fa.per_tech,
            "REMAINING_FLEX": np.maximum(
                0, cfg.max_flex_days - problem.tech_preflex - fa.per_tech
            ),
        }
    )

    checks = validate_flex(problem, new_days, training, preflex, cfg)
    summary = {
        "new_flex_days": int(len(new_days)),
        "preassigned_flex_rows": int(problem.tech_preflex.sum()),
        "preassigned_flex_adding_cover": int(problem.bucket_extra.sum()),
        "preassigned_flex_conflicts": int(len(problem.preflex_conflicts)),
        "shortage_after_vacation": int(short.sum()),
        "shortage_after_flex": int(short_after.sum()),
        "covered_by_new_flex": fa.covered,
        "flex_budget_remaining": int(by_person["REMAINING_FLEX"].sum()),
        "dispatchers_using_all_flex": int((by_person["REMAINING_FLEX"] == 0).sum()),
        "new_flex_on_other_work_center": int(new_days["OTHER_WORK_CENTER"].sum()),
        "new_flex_on_other_shift": int(new_days["OTHER_SHIFT"].sum()),
        "shortage_after_flex_by_work_center": cov.groupby("WORK_CENTER")[
            "SHORTAGE_AFTER_FLEX"
        ]
        .sum()
        .to_dict(),
        "flex_config": {**vars(fcfg), "max_flex_days": cfg.max_flex_days,
                        "same_shift_only": cfg.flex_same_shift_only,
                        "home_work_center_only": cfg.flex_home_work_center_only},  # fmt: skip
        "checks": checks,
        "seconds": round(time.time() - t0, 1),
    }
    return {
        "new": new_days, "added": added, "combined": combined, "after": after_vac, "final": final,
        "cov": cov, "by_person": by_person, "conflicts": problem.preflex_conflicts, "summary": summary,
    }  # fmt: skip


def validate_flex(problem, new_days: pd.DataFrame, training: set, preflex, cfg) -> dict:
    """Independent checks of the flex rules on the output table."""
    work = set(zip(problem.supply["TECH_ID"], problem.supply["SHIFT_DATE"]))
    pf = (
        set(zip(preflex["TECH_ID"], preflex["SHIFT_DATE"]))
        if preflex is not None
        else set()
    )
    keys = list(zip(new_days["TECH_ID"], new_days["SHIFT_DATE"]))
    assert len(keys) == len(set(keys)), "two flex days for one dispatcher on one date"
    assert not any(k in work for k in keys), "flex day on a scheduled workday"
    assert not any(k in training for k in keys), "flex day on a training day"
    assert not any(k in pf for k in keys), "flex day on a pre-assigned flex day"
    pre = pd.Series(problem.tech_preflex, index=problem.tech_ids)
    total = (
        new_days.groupby("TECH_ID").size().reindex(problem.tech_ids, fill_value=0) + pre
    )
    assert (
        total[new_days["TECH_ID"].unique()] <= cfg.max_flex_days
    ).all(), "flex budget exceeded"
    tcode = {t: i for i, t in enumerate(problem.tech_ids)}
    for t, wc in zip(new_days["TECH_ID"], new_days["WORK_CENTER_ID"]):
        assert problem.tech_quals[data_io.norm_wc(wc)][
            tcode[t]
        ], f"{t} not qualified for {wc}"
    if cfg.flex_home_work_center_only:
        assert not new_days[
            "OTHER_WORK_CENTER"
        ].any(), "flex on a desk other than the dispatcher's WORK_CENTER_ID"
    if cfg.flex_same_shift_only:
        assert not new_days["OTHER_SHIFT"].any(), "flex on another shift"
    return {"off_day_only": True, "budget_respected": True, "qualified": True,
            "one_per_day": True, "max_flex_days_used_by_anyone": int(total.max()) if len(total) else 0}  # fmt: skip


def write_flex(res: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "new": "flex_days_new.csv",
        "added": "flex_added_supply_rows.csv",
        "combined": "Flex_assignment_combined.csv",
        "after": "supply_after_vacation.csv",
        "final": "supply_after_vacation_and_flex.csv",
        "cov": "staffing_coverage_before_after_flex.csv",
        "by_person": "flex_by_dispatcher.csv",
        "conflicts": "preassigned_flex_conflicts.csv",
    }
    paths = {}
    for k, name in files.items():
        df = res[k]
        if k in ("added", "after", "final"):
            df = _ordered(df)
        paths[k] = out / name
        df.to_csv(paths[k], index=False, date_format=FMT)
    paths["summary"] = out / "flex_summary.json"
    paths["summary"].write_text(
        json.dumps(res["summary"], indent=2, default=_json_default)
    )
    return paths


def run_flex_from_files(
    supply_path, vacation_path, demand_path, out_dir, training_path=None, preflex_path=None,
    cfg: AllocationConfig | None = None, fcfg: FlexConfig | None = None,
):  # fmt: skip
    cfg, fcfg = cfg or AllocationConfig(), fcfg or FlexConfig()
    training = data_io.load_training(training_path) if training_path else set()
    preflex = data_io.read_csv(preflex_path) if preflex_path else None
    res = run_flex(
        data_io.read_csv(supply_path), data_io.read_csv(vacation_path), data_io.read_csv(demand_path),
        training, preflex, cfg, fcfg,
    )  # fmt: skip
    return res, write_flex(res, out_dir)
