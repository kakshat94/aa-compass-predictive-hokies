"""AA dispatcher flex placement after vacations with preassigned flex days.

Inputs:
  --original-supply: employee-day schedule BEFORE vacation shrinkage (full schema)
  --after-vacation: supply AFTER removing vacation work rows (full schema)
  --vacation: vacation assignments (TECH_ID, SHIFT_DATE)
  --preassigned-flex: existing Flex_assignment.csv (TECH_ID, SHIFT_DATE; optional
                     WORK_CENTER_ID, SHIFT_TYPE and/or complete supply columns)
  --training: merged training calendar (TECH_ID, SHIFT_DATE)
  --blocked-schedule: optional schedule of every assigned work/training day
                     (e.g. Supply_Regular_Merged.csv); no flex day goes on these
  --exclude-date: date removed from every input (no flex, no demand on it);
                  "none" disables. Its after-vacation rows are passed through
                  unchanged into the final supply.

Pre-assigned flex rows that clash with a workday, vacation, training day or
blocked day, fall outside the horizon, or name an unknown TECH_ID are not
added as supply; they are listed in preassigned_flex_conflicts.csv. Rows for
known dispatchers still count against that dispatcher's flex budget.

Outputs include flex assignments, added supply-shaped rows and final supply.
Exact max-flow optimality is conditional on fixed inputs, fixed preassignments and
modeled eligibility; this is not FAA/CBA or qualifications certification.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

SCHEMA = ["INDEX", "MAJOR_WORK_TYPE", "START_TIME", "EMPLOYEE_TITLE",
          "WORK_CENTER_ID", "CONTRACTOR_FLAG", "CLUSTER", "EMPLOYEE_WORK_CENTER_ID",
          "TECH_ID", "END_TIME", "SHIFT_DATE", "SHIFT_LENGTH", "SHIFT_TYPE",
          "FINAL_QUALS", "PRIMARY_QUAL", "DOMESTIC", "ATC", "ROC", "CHF",
          "OC", "LATIN", "EUROPE", "PACIFIC", "QUAL_COUNT", "SF_LOAD_TMS"]
ASSIGNMENT_COLS = ["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE", "ASSIGNMENT_TYPE"]


def load_csv(path, *, optional=False, name="input"):
    if str(path).lower() in ("none", "", "null"):
        if not optional:
            raise ValueError(f"Required {name} cannot be omitted")
        print(f"WARNING: No {name} specified; assuming no records.", flush=True)
        return pd.DataFrame(columns=["TECH_ID", "SHIFT_DATE"])
    p = Path(path)
    if not p.is_file():
        if optional:
            print(f"WARNING: {name} file not found at {p}; assuming no records.", flush=True)
            return pd.DataFrame(columns=["TECH_ID", "SHIFT_DATE"])
        raise FileNotFoundError(f"Missing required {name}: {p.resolve()}")
    df = pd.read_csv(p, dtype={"TECH_ID": str}, low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.upper()
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate column headers in {p}")
    return df


def normalize(df, name, center=False):
    df = df.copy()
    for col in ("TECH_ID", "SHIFT_DATE"):
        if col not in df:
            raise ValueError(f"{name} missing {col}; columns: {df.columns.tolist()}")
    df["TECH_ID"] = df["TECH_ID"].astype("string").str.strip()
    if df["TECH_ID"].isna().any() or (df["TECH_ID"] == "").any():
        raise ValueError(f"{name}: missing TECH_ID")
    raw = df["SHIFT_DATE"].copy()
    df["SHIFT_DATE"] = pd.to_datetime(df["SHIFT_DATE"], format="mixed", errors="coerce").dt.normalize()
    if df["SHIFT_DATE"].isna().any():
        raise ValueError(f"{name}: bad SHIFT_DATE: {raw[df['SHIFT_DATE'].isna()].head().tolist()}")
    if center:
        for c in ("WORK_CENTER_ID", "SHIFT_TYPE"):
            if c not in df:
                raise ValueError(f"{name} missing {c}")
            if df[c].isna().any() or (df[c] == "").any():
                raise ValueError(f"{name} invalid {c}")
        df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype("string").str.strip().str.upper()
    if "SHIFT_TYPE" in df:
        df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype("string").str.strip().str.upper()
    return df


def pairset(df):
    return set(zip(df["TECH_ID"].astype(str), df["SHIFT_DATE"]))


def check_unique(df, name):
    if df.duplicated(["TECH_ID", "SHIFT_DATE"]).any():
        examples = df.loc[df.duplicated(["TECH_ID", "SHIFT_DATE"], keep=False), ["TECH_ID", "SHIFT_DATE"]].head()
        raise ValueError(f"{name} duplicate employee-date rows:\n{examples}")


def load_demand(path, audit_path):
    df = load_csv(path)
    if not {"SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "DEMAND"}.issubset(df):
        raise ValueError("Demand must have SHIFT_DATE, WORK_CENTER, SHIFT_TYPE, DEMAND")
    df["SHIFT_DATE"] = pd.to_datetime(df["SHIFT_DATE"], format="mixed", errors="raise").dt.normalize()
    df["SHIFT_TYPE"] = df["SHIFT_TYPE"].astype(str).str.strip().str.upper()
    if str(audit_path).lower() != "none" and Path(audit_path).exists():
        audit = load_csv(audit_path)
        required = {"SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "IMPUTED_DEMAND"}
        if not required.issubset(audit):
            raise ValueError(f"Demand audit requires {sorted(required)}")
        audit["SHIFT_DATE"] = pd.to_datetime(audit["SHIFT_DATE"], format="mixed").dt.normalize()
        audit["SHIFT_TYPE"] = audit["SHIFT_TYPE"].astype(str).str.strip().str.upper()
        present = set(zip(df.SHIFT_DATE, df.WORK_CENTER, df.SHIFT_TYPE))
        extra = audit.loc[[key not in present for key in zip(audit.SHIFT_DATE, audit.WORK_CENTER, audit.SHIFT_TYPE)], ["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE", "IMPUTED_DEMAND"]].rename(columns={"IMPUTED_DEMAND": "DEMAND"})
        if len(extra):
            print(f"WARNING: Using {len(extra)} estimated demand records from {audit_path}", flush=True)
            df = pd.concat([df, extra], ignore_index=True)
    df["DEMAND"] = pd.to_numeric(df["DEMAND"], errors="raise")
    if df["DEMAND"].isna().any() or (df["DEMAND"] < 0).any() or (df["DEMAND"] % 1 != 0).any():
        raise ValueError("Demand must be nonnegative integers")
    if df.duplicated(["SHIFT_DATE", "WORK_CENTER", "SHIFT_TYPE"]).any():
        raise ValueError("Duplicate demand buckets")
    return df


def get_people_assignments(original):
    centers = defaultdict(set)
    for r in original[["TECH_ID", "WORK_CENTER_ID", "SHIFT_TYPE"]].itertuples(index=False, name=None):
        centers[str(r[0])].add((str(r[1]), str(r[2])))
    ambiguous = {p: sorted(v) for p, v in centers.items() if len(v) != 1}
    if ambiguous:
        raise ValueError(f"Cannot infer one fixed work center/shift for {len(ambiguous)} employees: {list(ambiguous.items())[:3]}. An employee-level flex qualification mapping is needed.")
    return {p: next(iter(v)) for p, v in centers.items()}


def prepare_preassigned(raw, assignment, original_pairs, after_pairs, vac_pairs, training_pairs, begin, finish, max_flex,
                        blocked_pairs=frozenset()):
    pre = normalize(raw, "preassigned flex", center=False)
    check_unique(pre, "preassigned flex")
    for col in ("WORK_CENTER_ID", "SHIFT_TYPE"):
        if col not in pre:
            pre[col] = pd.NA
    people = set(assignment)
    # Clashing rows are audited instead of stopping the run.
    reasons = []
    for r in pre.itertuples(index=False):
        p, d = str(r.TECH_ID), r.SHIFT_DATE
        if p not in people:
            reasons.append("dispatcher not in supply")
        elif not begin <= d <= finish:
            reasons.append("date outside planning horizon")
        elif (p, d) in original_pairs or (p, d) in after_pairs:
            reasons.append("already a scheduled workday")
        elif (p, d) in vac_pairs:
            reasons.append("vacation day")
        elif (p, d) in training_pairs:
            reasons.append("training day")
        elif (p, d) in blocked_pairs:
            reasons.append("day already assigned in blocked schedule")
        elif any(pd.notna(g) and str(g).strip() not in ("", e)
                 for g, e in zip((r.WORK_CENTER_ID, r.SHIFT_TYPE), assignment[p])):
            reasons.append("work center/shift differs from the dispatcher's own")
        else:
            reasons.append("")
    pre["CONFLICT_REASON"] = reasons
    conflicts = pre.loc[pre.CONFLICT_REASON != ""].copy()
    if len(conflicts):
        print(f"WARNING: {len(conflicts)} preassigned flex rows clash and are not added as supply: "
              f"{conflicts.CONFLICT_REASON.value_counts().to_dict()}", flush=True)
    # Every row of a known dispatcher uses that dispatcher's flex budget.
    pre_counts = Counter(t for t in pre.TECH_ID.astype(str) if t in people)
    pre = pre.loc[pre.CONFLICT_REASON == ""].drop(columns="CONFLICT_REASON").copy()
    for i, r in pre.iterrows():
        p, d = str(r.TECH_ID), r.SHIFT_DATE
        c, s = assignment[p]
        for col, expected in (("WORK_CENTER_ID", c), ("SHIFT_TYPE", s)):
            given = r[col]
            if pd.notna(given) and str(given).strip() not in ("", expected):
                raise ValueError(f"Preassigned flex {p} {d.date()} has {col}={given}, expected {expected}")
            pre.at[i, col] = expected
    excess = {p: n for p, n in pre_counts.items() if n > max_flex}
    if excess:
        raise ValueError(f"Preassigned flex exceeds {max_flex} days per dispatcher: {list(excess.items())[:5]}")
    pre["ASSIGNMENT_TYPE"] = "FLEX_PREASSIGNED"
    return pre, pre_counts, conflicts


def template_row(original, person, date, center, shift):
    # Employee-specific historical metadata is preferred over a different employee.
    personal = original.loc[(original.TECH_ID == person) & (original.WORK_CENTER_ID == center) & (original.SHIFT_TYPE == shift)]
    if personal.empty:
        raise ValueError(f"No employee-specific work template for {person}, {center}, {shift}")
    same_weekday = personal.loc[personal.SHIFT_DATE.dt.dayofweek == date.dayofweek]
    pool = same_weekday if len(same_weekday) else personal
    source_idx = (pool.SHIFT_DATE - date).abs().idxmin()
    row = original.loc[source_idx].copy()
    old_date = row["SHIFT_DATE"]
    row["SHIFT_DATE"] = date
    # START_TIME and END_TIME may be clock strings or full timestamps. Preserve
    # clock strings; move timestamp dates by the same day offset as SHIFT_DATE.
    for key in ("START_TIME", "END_TIME"):
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value and ("-" in value or "/" in value or "T" in value):
                parsed = pd.to_datetime(value, errors="coerce")
                if pd.notna(parsed):
                    row[key] = (parsed + (date - old_date)).isoformat(sep=" ")
    return row


def full_supply_rows(original, assignments, *, index_start):
    rows = []
    original_columns = original.columns.tolist()
    for r in assignments.itertuples(index=False):
        p, d, c, s = str(r.TECH_ID), r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE)
        row = template_row(original, p, d, c, s)
        for col in original_columns:
            if col in assignments and col not in ("TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE", "ASSIGNMENT_TYPE"):
                value = getattr(r, col)
                if pd.notna(value):
                    row[col] = value
        if "INDEX" in original_columns:
            row["INDEX"] = index_start + len(rows)
        if "SF_LOAD_TMS" in original_columns:
            row["SF_LOAD_TMS"] = pd.NA  # new record; do not invent upstream load timestamp
        rows.append(row)
    if not rows:
        return original.iloc[0:0].copy()
    return pd.DataFrame(rows).reindex(columns=original_columns)


def run(original, after, vacation, training, pre_raw, demand, *, max_flex, solver, start, end, exclude_date="2026-01-11",
        blocked=None):
    original = normalize(original, "original supply", center=True)
    after = normalize(after, "after-vacation supply", center=True)
    vacation = normalize(vacation, "vacation assignments")
    training = normalize(training, "training")
    # Match the vacation optimizer's planning horizon. Remove the excluded
    # date from ALL inputs before checking that only vacation rows were removed.
    # This also prevents excluded dates from contributing demand or flex rows.
    excluded = None if exclude_date is None or str(exclude_date).strip().lower() in ("", "none", "null") else pd.Timestamp(exclude_date).normalize()
    passthrough = after.iloc[0:0].copy()
    if excluded is not None:
        # The excluded date's after-vacation rows go back into the final supply unchanged.
        passthrough = after.loc[after["SHIFT_DATE"] == excluded].copy()
        datasets = {
            "original supply": original,
            "after-vacation supply": after,
            "vacation assignments": vacation,
            "training": training,
            "preassigned flex": normalize(pre_raw, "preassigned flex"),
            "demand": demand,
        }
        filtered = {}
        for name, df in datasets.items():
            removed = int((df["SHIFT_DATE"] == excluded).sum())
            filtered[name] = df.loc[df["SHIFT_DATE"] != excluded].copy()
            if removed:
                print(f"Excluded {removed} {name} rows for {excluded.date()}", flush=True)
        original = filtered["original supply"]
        after = filtered["after-vacation supply"]
        vacation = filtered["vacation assignments"]
        training = filtered["training"]
        pre_raw = filtered["preassigned flex"]
        demand = filtered["demand"]
    if original.empty:
        raise ValueError("No original supply rows remain after date exclusion")
    check_unique(original, "original supply")
    check_unique(after, "after-vacation supply")
    check_unique(vacation, "vacation")
    assignment = get_people_assignments(original)
    original_pairs, after_pairs, vac_pairs, tr_pairs = map(pairset, (original, after, vacation, training))
    # Every work/training day in the blocked schedule: never a flex day.
    blocked_pairs = pairset(normalize(blocked, "blocked schedule")) if blocked is not None and len(blocked) else set()
    if not vac_pairs <= original_pairs:
        raise ValueError("Vacation file contains employee-date not present in original supply")
    if tr_pairs & vac_pairs:
        raise ValueError("Training and vacation overlap")
    if set(after.columns) != set(original.columns):
        raise ValueError(f"After-vacation columns mismatch original supply: missing={sorted(set(original)-set(after))}, extra={sorted(set(after)-set(original))}")
    if not after_pairs <= original_pairs:
        raise ValueError("After-vacation supply contains new employee-date rows not in original supply")
    if after_pairs != original_pairs - vac_pairs:
        missing = (original_pairs - vac_pairs) - after_pairs
        unwanted = after_pairs & vac_pairs
        raise ValueError(f"After-vacation supply inconsistent with vacation assignments: missing non-vacation work rows={len(missing)}, vacation rows still present={len(unwanted)}. This program requires vacation-only shrinkage; training is separately excluded for coverage.")
    begin = pd.Timestamp(start).normalize() if start else original.SHIFT_DATE.min()
    finish = pd.Timestamp(end).normalize() if end else original.SHIFT_DATE.max()
    if finish < begin:
        raise ValueError("End before start")
    pre, pre_counts, pre_conflicts = prepare_preassigned(pre_raw, assignment, original_pairs, after_pairs,
                                                        vac_pairs, tr_pairs, begin, finish, max_flex, blocked_pairs)
    pre_pairs = pairset(pre)
    # Only actual productive rows count; exclude training from after-vacation supply.
    coverage = Counter()
    for r in after[["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"]].itertuples(index=False):
        if (str(r.TECH_ID), r.SHIFT_DATE) not in tr_pairs:
            coverage[(r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))] += 1
    coverage_without_preassigned = coverage.copy()
    for r in pre[["SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"]].itertuples(index=False):
        coverage[(r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))] += 1
    required = {(r.SHIFT_DATE, str(r.WORK_CENTER), str(r.SHIFT_TYPE)): int(r.DEMAND)
                for r in demand.itertuples(index=False) if begin <= r.SHIFT_DATE <= finish}
    represented = {(r.SHIFT_DATE, str(r.WORK_CENTER_ID), str(r.SHIFT_TYPE))
                   for r in original.itertuples(index=False) if begin <= r.SHIFT_DATE <= finish}
    missing_demand = represented - required.keys()
    if missing_demand:
        raise ValueError(f"Missing demand in {len(missing_demand)} represented buckets. Example: {next(iter(missing_demand))}")
    shortage = {b: max(0, n - coverage[b]) for b, n in required.items()}
    before = sum(shortage.values())
    by_bucket = defaultdict(list)
    original_days = defaultdict(set)
    for p, d in original_pairs:
        original_days[p].add(d)
    for p, bucket in assignment.items():
        by_bucket[bucket].append(p)
    for bucket in by_bucket:
        by_bucket[bucket].sort()
    remaining = {p: max(0, max_flex - pre_counts[p]) for p in assignment}
    allocated = []
    greedy_used = Counter()
    if solver == "greedy":
        for b in sorted(shortage, key=lambda b: (-shortage[b], b)):
            d, c, s = b
            for _ in range(shortage[b]):
                eligible = [p for p in by_bucket.get((c, s), []) if greedy_used[p] < remaining[p]
                            and d not in original_days[p] and (p, d) not in pre_pairs
                            and (p, d) not in tr_pairs and (p, d) not in vac_pairs
                            and (p, d) not in blocked_pairs]
                if not eligible:
                    break
                p = min(eligible, key=lambda p: (greedy_used[p], p))
                greedy_used[p] += 1
                allocated.append((p, d, c, s))
    else:
        try:
            from ortools.graph.python import max_flow
        except ImportError as exc:
            raise ImportError("Install ortools for exact maximum-flow scheduling") from exc
        flow = max_flow.SimpleMaxFlow()
        src, dst, next_node = 0, 1, 2
        person_node = {}
        for p in sorted(assignment):
            person_node[p] = next_node
            next_node += 1
            flow.add_arc_with_capacity(src, person_node[p], remaining[p])
        bucket_node = {}
        for b, need in sorted(shortage.items()):
            if need:
                bucket_node[b] = next_node
                next_node += 1
                flow.add_arc_with_capacity(bucket_node[b], dst, need)
        assignment_arcs = []
        for b, node in bucket_node.items():
            d, c, s = b
            for p in by_bucket.get((c, s), []):
                if remaining[p] == 0 or d in original_days[p] or (p, d) in pre_pairs or (p, d) in tr_pairs or (p, d) in vac_pairs \
                        or (p, d) in blocked_pairs:
                    continue
                arc = flow.add_arc_with_capacity(person_node[p], node, 1)
                assignment_arcs.append((arc, p, b))
        status = flow.solve(src, dst)
        if status != flow.OPTIMAL:
            raise RuntimeError(f"Maximum flow failed, status={status}")
        for arc, p, b in assignment_arcs:
            if flow.flow(arc):
                allocated.append((p, *b))
        assert len(allocated) == flow.optimal_flow()
    new = pd.DataFrame(allocated, columns=["TECH_ID", "SHIFT_DATE", "WORK_CENTER_ID", "SHIFT_TYPE"])
    new["ASSIGNMENT_TYPE"] = "FLEX_OPTIMIZED"
    pre_assign = pre[ASSIGNMENT_COLS]
    all_assign = pd.concat([pre_assign, new[ASSIGNMENT_COLS]], ignore_index=True)
    check_unique(all_assign, "combined flex")
    counts = Counter(all_assign.TECH_ID.astype(str)) + Counter(
        t for t in pre_conflicts.TECH_ID.astype(str) if t in assignment)
    if any(v > max_flex for v in counts.values()):
        raise AssertionError("Annual flex cap violated")
    # Create complete-format new supply records; use employee-specific templates.
    # INDEX is not guaranteed to be numeric in the teammate/AA data. Preserve
    # every existing INDEX unchanged, and allocate collision-free flex IDs.
    # The schema is unchanged, but INDEX for new rows is a string identifier.
    existing_ids = set()
    if "INDEX" in original:
        existing_ids.update(original["INDEX"].dropna().astype(str).str.strip())
    if "INDEX" in after:
        existing_ids.update(after["INDEX"].dropna().astype(str).str.strip())
    generated_ids = []
    suffix = 1
    while len(generated_ids) < len(pre) + len(new):
        candidate = f"FLEX_NEW_{suffix:06d}"
        if candidate not in existing_ids:
            generated_ids.append(candidate)
            existing_ids.add(candidate)
        suffix += 1
    pre_full = full_supply_rows(original, pre, index_start=0)
    new_full = full_supply_rows(original, new, index_start=0)
    if "INDEX" in original:
        pre_full["INDEX"] = generated_ids[:len(pre_full)]
        new_full["INDEX"] = generated_ids[len(pre_full):]
    # For provided full-schema preassignments use their fields where available,
    # otherwise replicate employee metadata and adjust the shift date.
    added = pd.concat([pre_full, new_full], ignore_index=True).reindex(columns=original.columns)
    final = pd.concat([after.reindex(columns=original.columns), added,
                       passthrough.reindex(columns=original.columns)], ignore_index=True)
    check_unique(final, "final supply")
    if "INDEX" in final and final["INDEX"].dropna().astype(str).str.strip().duplicated().any():
        raise ValueError("INDEX collision in final supply")
    # Verify numeric coverage accounting for both preassigned and newly placed days.
    before_new = coverage.copy()
    new_bucket = Counter((d, c, s) for p, d, c, s in allocated)
    for b, n in new_bucket.items():
        coverage[b] += n
    rows = []
    for b, dem in sorted(required.items()):
        d, c, s = b
        rows.append({"SHIFT_DATE": d, "WORK_CENTER": c, "SHIFT_TYPE": s, "DEMAND": dem,
                     "AVAILABLE_BEFORE_FLEX": before_new[b], "AVAILABLE_AFTER_FLEX": coverage[b],
                     "SHORTAGE_BEFORE_FLEX": max(0, dem-before_new[b]),
                     "SHORTAGE_AFTER_FLEX": max(0, dem-coverage[b]),
                     "FLEX_ASSIGNED": new_bucket[b],
                     "OVERCOVERAGE_AFTER_FLEX": max(0, coverage[b]-dem)})
    cov = pd.DataFrame(rows)
    by_person = pd.DataFrame({"TECH_ID": sorted(assignment)})
    by_person["PREASSIGNED_FLEX_DAYS"] = by_person.TECH_ID.map(pre_counts).fillna(0).astype(int)
    new_counts = Counter(new.TECH_ID.astype(str))
    by_person["NEW_FLEX_DAYS"] = by_person.TECH_ID.map(new_counts).fillna(0).astype(int)
    by_person["TOTAL_FLEX_DAYS"] = by_person.PREASSIGNED_FLEX_DAYS + by_person.NEW_FLEX_DAYS
    # PREASSIGNED_FLEX_DAYS includes clashing rows (they use budget but add no cover).
    by_person["PREASSIGNED_CONFLICT_DAYS"] = by_person.PREASSIGNED_FLEX_DAYS - by_person.TECH_ID.map(
        Counter(pre.TECH_ID.astype(str))).fillna(0).astype(int)
    by_person["REMAINING_FLEX_CAPACITY"] = max_flex - by_person.TOTAL_FLEX_DAYS
    summary = {"solver": "exact_bipartite_max_flow" if solver == "maxflow" else "greedy",
               "max_flex_per_dispatcher": max_flex, "preassigned_flex_days": len(pre),
               "preassigned_flex_conflicts": len(pre_conflicts), "blocked_schedule_days": len(blocked_pairs),
               "excluded_date_rows_passed_through": len(passthrough),
               "new_flex_days": len(new), "total_flex_days": len(all_assign),
               "dispatchers_with_flex": int((by_person.TOTAL_FLEX_DAYS > 0).sum()),
               "shortage_after_vacation_before_preassigned": sum(max(0, n - coverage_without_preassigned[b]) for b, n in required.items()),
               "shortage_after_preassigned_before_new_flex": before,
               "shortage_after_all_flex": int(cov.SHORTAGE_AFTER_FLEX.sum()),
               "shortage_reduction_from_new_flex": before - int(cov.SHORTAGE_AFTER_FLEX.sum()),
               "supply_rows_after_vacation": len(after), "supply_rows_after_all_flex": len(final),
               "full_supply_columns": original.columns.tolist(),
               "schema_columns_missing_from_SQL_screenshot": [c for c in SCHEMA if c not in original],
               "qualifications_verified": False,
               "new_flex_row_metadata": "inherited from same employee existing work shift; review timestamp and shift metadata",
               "optimal_under_modeled_constraints": solver == "maxflow",
                "excluded_date": str(excluded.date()) if excluded is not None else None}
    assert len(final) == len(after) + len(pre) + len(new) + len(passthrough)
    assert int(cov.FLEX_ASSIGNED.sum()) == len(new)
    assert summary["shortage_reduction_from_new_flex"] == len(new)
    return all_assign, new, added, final, cov, by_person, summary, pre_conflicts


def run_from_files(original_supply, after_vacation, vacation, training, preassigned_flex, demand, demand_audit,
                   output_dir, *, max_flex=4, solver="maxflow", start=None, end=None, exclude_date="2026-01-11",
                   blocked_schedule=None):
    """Load the inputs, run, and write every output to output_dir."""
    blocked = load_csv(blocked_schedule, optional=True, name="blocked schedule") if blocked_schedule is not None else None
    result = run(load_csv(original_supply), load_csv(after_vacation), load_csv(vacation),
                 load_csv(training if training is not None else "none", optional=True, name="training"),
                 load_csv(preassigned_flex if preassigned_flex is not None else "none", optional=True,
                          name="preassigned flex"),
                 load_demand(demand, demand_audit if demand_audit is not None else "none"),
                 max_flex=max_flex, solver=solver, start=start, end=end, exclude_date=exclude_date, blocked=blocked)
    combined, new, added, final, cov, by_person, summary, conflicts = result
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "Flex_assignment_combined.csv", index=False, date_format="%Y-%m-%d")
    new.to_csv(out / "flex_days_new.csv", index=False, date_format="%Y-%m-%d")
    added.to_csv(out / "flex_added_supply_rows.csv", index=False, date_format="%Y-%m-%d")
    final.to_csv(out / "supply_after_vacation_and_flex.csv", index=False, date_format="%Y-%m-%d")
    cov.to_csv(out / "staffing_coverage_before_after_flex.csv", index=False, date_format="%Y-%m-%d")
    by_person.to_csv(out / "flex_by_dispatcher.csv", index=False)
    conflicts.to_csv(out / "preassigned_flex_conflicts.csv", index=False, date_format="%Y-%m-%d")
    (out / "flex_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved outputs to {out.resolve()}", flush=True)
    return summary, out


def main():
    p = argparse.ArgumentParser(description="Flex max-flow with preassignments and full supply-format output")
    p.add_argument("--original-supply", default="data/Supply_Regular_qualified.csv")
    p.add_argument("--after-vacation", default="outputs/vacation_outputs_updatedFormat.csv")
    p.add_argument("--vacation", default="outputs/vacation_initial.csv")
    p.add_argument("--preassigned-flex", default="data/Flex_assignment.csv",
                   help="Preexisting Flex_assignment.csv; missing file or none means no preassignments")
    p.add_argument("--training", default="data/all_training_data.csv",
                   help="Merged training calendar; missing file or none means no training records")
    p.add_argument("--blocked-schedule", default=None,
                   help="Optional schedule of all work/training days (e.g. Supply_Regular_Merged.csv); never flexed")
    p.add_argument("--demand", default="data/annual_desk_needs_reference.csv")
    p.add_argument("--demand-audit", default="outputs/demand_imputation_audit.csv")
    p.add_argument("--output-dir", default="outputs/flex_maxflow")
    p.add_argument("--max-flex", type=int, default=4)
    p.add_argument("--solver", choices=["maxflow", "greedy"], default="maxflow")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--exclude-date", default="2026-01-11",
                   help="Exclude this date consistently from supply, vacations, training, flex and demand; use none to disable")
    args = p.parse_args()
    if args.max_flex < 0:
        p.error("max-flex must be nonnegative")
    summary, _ = run_from_files(args.original_supply, args.after_vacation, args.vacation, args.training,
                                args.preassigned_flex, args.demand, args.demand_audit, args.output_dir,
                                max_flex=args.max_flex, solver=args.solver, start=args.start, end=args.end,
                                exclude_date=args.exclude_date, blocked_schedule=args.blocked_schedule)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
