import os
import logging
import yaml
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Any

_PARAMS_PATH = os.path.join("params.yaml")
with open(_PARAMS_PATH) as _f:
    _PARAMS = yaml.safe_load(_f)
_MORNING_EQUALS_EVENING: bool = _PARAMS.get("shifts", {}).get(
    "morning_equals_evening", False
)


class SupplyGenerator:
    """
    Iterates over all shift models in `orch.models`, processes their X (bidline)
    and Y (training) solution variables, and consolidates the output into a single
    Supply_Regular.csv and a single Supply_Training.csv file.
    Works seamlessly across Windows, macOS, and Linux.
    """

    SHIFT_SCHEDULES = {
        "MORNING": {"start": "05:00:00", "end": "13:00:00", "length": 8, "code": "M"},
        "EVENING": {"start": "13:00:00", "end": "21:00:00", "length": 8, "code": "E"},
        "MIDNIGHT": {"start": "21:00:00", "end": "05:00:00", "length": 8, "code": "MD"},
    }

    # tech_id block convention (Data Specification section 4.2)
    CORE = ["Domestic", "Latin", "Europe", "Pacific"]
    SPEC = ["ATC", "ROC", "CHF", "OC"]
    SHIFTS = ["MORNING", "EVENING", "MIDNIGHT"]

    SUPPLY_COLUMNS = [
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

    def __init__(self, coverage_vectors: Dict):
        self.coverage_vectors = coverage_vectors

    @staticmethod
    def assign_tech_ids(lines) -> Dict[int, Tuple[str, str]]:
        """lines: iterable of (primary_work_center, shift_type) for every bid line you create.
        Returns {tech_id: (work_center, shift)} following DS section 4.2."""
        counts = Counter(lines)
        ids, nxt = {}, 100000
        for wc in SupplyGenerator.CORE:
            for sh in SupplyGenerator.SHIFTS:
                for _ in range(counts[(wc, sh)]):
                    ids[nxt] = (wc, sh)
                    nxt += 1
        nxt = 9100000
        for wc in SupplyGenerator.SPEC:
            for sh in SupplyGenerator.SHIFTS:
                for _ in range(counts[(wc, sh)]):
                    ids[nxt] = (wc, sh)
                    nxt += 1
        return ids

    @staticmethod
    def _format_date(dt: datetime) -> str:
        """Cross-platform non-zero-padded date string (M/D/YYYY)."""
        return f"{dt.month}/{dt.day}/{dt.year}"

    def process_all_models(
        self,
        orch_models: Dict[str, Any],
        demand_df: pd.DataFrame,
        filter_by_demand: bool = False,
        regular_out_path: str = "Supply_Regular.csv",
        standard_training_out_path: str = "Supply_Standard_Training.csv",
        special_training_out_path: str = "Supply_Special_Training.csv",
        flex_out_path: str = "Flex_assignment.csv",
        all_out_path: str = "Supply_all.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        all_regular_rows: List[Dict] = []
        all_special_training_rows: List[Dict] = []
        all_standard_training_rows: List[Dict] = []
        all_flex_rows: List[Dict] = []

        # Build a set of (wc, shift_type, date_str) present in demand for fast lookup.
        # Only populated when filter_by_demand=True; otherwise left empty and never consulted.
        _demand_dates: set = set()
        if filter_by_demand:
            for _, row in demand_df.iterrows():
                try:
                    _dt = datetime.strptime(str(row["SHIFT_DATE"]).strip(), "%m/%d/%Y")
                except ValueError:
                    _dt = datetime.strptime(str(row["SHIFT_DATE"]).strip(), "%Y-%m-%d")
                _demand_dates.add(
                    (
                        str(row["WORK_CENTER"]).strip(),
                        str(row["SHIFT_TYPE"]).strip().upper(),
                        _dt.strftime("%Y-%m-%d"),
                    )
                )

        X_dict = {}
        Y_dict = {}
        F_dict = {}

        # Pass A: collect every selected bidline across all shift models.
        # When morning_equals_evening, also count EVENING lines for tech_id allocation.
        selected: List[Tuple[str, str, str, str]] = (
            []
        )  # (shift_type, wc, num_id, off_str)
        for shift_key, model in orch_models.items():
            X_dict[shift_key] = model.model.getSolution(model.X)
            Y_dict[shift_key] = model.model.getSolution(model.Y)
            F_dict[shift_key] = model.model.getSolution(model.F)
            st = shift_key.upper()
            if st not in self.SHIFT_SCHEDULES:
                continue
            for (wc, num_id, off_str), var in X_dict[shift_key].items():
                if var > 0.5:
                    selected.append((st, wc, num_id, off_str))
                    if _MORNING_EQUALS_EVENING and st == "MORNING":
                        selected.append(("EVENING", wc, num_id, off_str))

        # Assign DS section 4.2 integer tech ids once over the whole solution.
        ids = self.assign_tech_ids([(wc, st) for st, wc, *_ in selected])
        pool: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for tid, group in ids.items():
            pool[group].append(tid)
        for group in pool.values():
            group.sort()
        used: Dict[Tuple[str, str], int] = defaultdict(int)
        tech_id_of: Dict[Tuple[str, str, str, str], int] = {}
        for st, wc, num_id, off_str in selected:
            group = (wc, st)
            if group not in pool:
                raise ValueError(
                    f"No DS section 4.2 id block for work center/shift {group}"
                )
            tech_id_of[(st, wc, num_id, off_str)] = pool[group][used[group]]
            used[group] += 1

        for shift_key, model in orch_models.items():
            shift_type = shift_key.upper()
            if shift_type not in self.SHIFT_SCHEDULES:
                logging.warning(
                    f"Warning: Shift type '{shift_key}' not recognized. Skipping."
                )
                continue

            shift_info = self.SHIFT_SCHEDULES[shift_type]
            start_str = shift_info["start"]
            end_str = shift_info["end"]
            length = shift_info["length"]
            shift_code = shift_info["code"]

            # Extract solution dictionaries

            shift_regular_records: Dict[Tuple[str, str, str, str], Dict] = {}

            # Step 1: Populate regular records
            for (wc, num_id, off_str), val in X_dict[shift_key].items():
                if val > 0.5:
                    # Old string tech id — superseded by the DS section 4.2 integer ids below.
                    # wc_prefix = wc[0].upper()
                    # num_code = num_id.replace('Number', 'N')
                    # off_code = off_str.replace('Offset', 'O')
                    # tech_id = f"{wc_prefix}_{num_code}_{off_code}_{shift_code}"
                    tech_id = tech_id_of[(shift_type, wc, num_id, off_str)]

                    cov_vec = self.coverage_vectors.get(off_str, {})

                    for date_str, is_working in cov_vec.items():
                        if is_working == 1 and (
                            not filter_by_demand
                            or (wc, shift_type, date_str) in _demand_dates
                        ):
                            dt = datetime.strptime(date_str, "%Y-%m-%d")
                            fmt_date = self._format_date(dt)

                            if shift_type == "MIDNIGHT":
                                start_dt = dt - timedelta(days=1)
                                start_date_fmt = self._format_date(start_dt)
                                end_date_fmt = fmt_date
                            else:
                                start_date_fmt = fmt_date
                                end_date_fmt = fmt_date

                            start_ts = f"{start_date_fmt} {start_str}"
                            end_ts = f"{end_date_fmt} {end_str}"

                            row = {
                                "INDEX": np.nan,
                                "MAJOR_WORK_TYPE": "All",
                                "START_TIME": start_ts,
                                "EMPLOYEE_TITLE": wc,
                                "WORK_CENTER_ID": wc,
                                "CONTRACTOR_FLAG": "N",
                                "CLUSTER": 1,
                                "EMPLOYEE_WORK_CENTER_ID": wc,
                                "TECH_ID": tech_id,
                                "END_TIME": end_ts,
                                "SHIFT_DATE": fmt_date,
                                "SHIFT_LENGTH": length,
                                "SHIFT_TYPE": shift_type,
                                "FINAL_QUALS": np.nan,
                                "PRIMARY_QUAL": wc,
                                "DOMESTIC": np.nan,
                                "ATC": np.nan,
                                "ROC": np.nan,
                                "CHF": np.nan,
                                "OC": np.nan,
                                "LATIN": np.nan,
                                "EUROPE": np.nan,
                                "PACIFIC": np.nan,
                                "QUAL_COUNT": np.nan,
                                "SF_LOAD_TMS": np.nan,
                            }

                            key = (wc, num_id, off_str, date_str)
                            shift_regular_records[key] = row

            # Step 2: Extract training shifts
            for (w, n, o, d), v in Y_dict[shift_key].items():
                if v > 0.5:
                    key = (w, n, o, d)
                    if key in shift_regular_records:
                        training_row = shift_regular_records.pop(key)
                        if datetime.strptime(d, "%Y-%m-%d").month in (2, 5, 9):
                            all_standard_training_rows.append(training_row)
                        else:
                            all_special_training_rows.append(training_row)
                    else:
                        logging.error("Error")

            all_regular_rows.extend(shift_regular_records.values())

            # Step 2.5: Extract flex assignment rows from model.F
            for (w, n, o, d), v in F_dict[shift_key].items():
                if v > 0.5:
                    tech_id = tech_id_of.get((shift_type, w, n, o))
                    dt = datetime.strptime(d, "%Y-%m-%d")
                    fmt_date = self._format_date(dt)
                    if shift_type == "MIDNIGHT":
                        start_dt = dt - timedelta(days=1)
                        start_date_fmt = self._format_date(start_dt)
                        end_date_fmt = fmt_date
                    else:
                        start_date_fmt = fmt_date
                        end_date_fmt = fmt_date
                    all_flex_rows.append(
                        {
                            "INDEX": np.nan,
                            "MAJOR_WORK_TYPE": "All",
                            "START_TIME": f"{start_date_fmt} {start_str}",
                            "EMPLOYEE_TITLE": w,
                            "WORK_CENTER_ID": w,
                            "CONTRACTOR_FLAG": "N",
                            "CLUSTER": 1,
                            "EMPLOYEE_WORK_CENTER_ID": w,
                            "TECH_ID": tech_id,
                            "END_TIME": f"{end_date_fmt} {end_str}",
                            "SHIFT_DATE": fmt_date,
                            "SHIFT_LENGTH": length,
                            "SHIFT_TYPE": shift_type,
                            "FINAL_QUALS": np.nan,
                            "PRIMARY_QUAL": w,
                            "DOMESTIC": np.nan,
                            "ATC": np.nan,
                            "ROC": np.nan,
                            "CHF": np.nan,
                            "OC": np.nan,
                            "LATIN": np.nan,
                            "EUROPE": np.nan,
                            "PACIFIC": np.nan,
                            "QUAL_COUNT": np.nan,
                            "SF_LOAD_TMS": np.nan,
                        }
                    )

        # Step 2.6: If morning_equals_evening, duplicate MORNING rows as EVENING
        if _MORNING_EQUALS_EVENING and "MORNING" in orch_models:
            eve_info = self.SHIFT_SCHEDULES["EVENING"]
            eve_start, eve_end, eve_length = (
                eve_info["start"],
                eve_info["end"],
                eve_info["length"],
            )
            # Build a mapping from MORNING tech_id -> EVENING tech_id for the same (wc, num_id, off_str)
            morning_to_evening_tid: Dict[int, int] = {}
            for st, wc, num_id, off_str in selected:
                if st == "MORNING":
                    m_tid = tech_id_of.get(("MORNING", wc, num_id, off_str))
                    e_tid = tech_id_of.get(("EVENING", wc, num_id, off_str))
                    if m_tid is not None and e_tid is not None:
                        morning_to_evening_tid[m_tid] = e_tid

            def _to_evening(row):
                return {
                    **row,
                    "SHIFT_TYPE": "EVENING",
                    "START_TIME": f"{row['SHIFT_DATE']} {eve_start}",
                    "END_TIME": f"{row['SHIFT_DATE']} {eve_end}",
                    "SHIFT_LENGTH": eve_length,
                    "TECH_ID": morning_to_evening_tid.get(
                        row["TECH_ID"], row["TECH_ID"]
                    ),
                }

            for row in [r for r in all_regular_rows if r["SHIFT_TYPE"] == "MORNING"]:
                eve_row = _to_evening(row)
                date_str = datetime.strptime(row["SHIFT_DATE"], "%m/%d/%Y").strftime(
                    "%Y-%m-%d"
                )
                if (
                    not filter_by_demand
                    or (row["WORK_CENTER_ID"], "EVENING", date_str) in _demand_dates
                ):
                    all_regular_rows.append(eve_row)
            for row in [r for r in all_flex_rows if r["SHIFT_TYPE"] == "MORNING"]:
                all_flex_rows.append(_to_evening(row))
            for row in [
                r for r in all_standard_training_rows if r["SHIFT_TYPE"] == "MORNING"
            ]:
                all_standard_training_rows.append(_to_evening(row))
            for row in [
                r for r in all_special_training_rows if r["SHIFT_TYPE"] == "MORNING"
            ]:
                all_special_training_rows.append(_to_evening(row))

        # Step 3: Combine and export
        def _to_supply_df(rows):
            df = pd.DataFrame(rows)
            return (
                pd.DataFrame(columns=self.SUPPLY_COLUMNS)
                if df.empty
                else df[self.SUPPLY_COLUMNS]
            )

        master_regular_df = _to_supply_df(all_regular_rows)
        master_standard_training_df = _to_supply_df(all_standard_training_rows)
        master_special_training_df = _to_supply_df(all_special_training_rows)
        master_flex_df = _to_supply_df(all_flex_rows)
        master_all_df = pd.concat(
            [master_regular_df, master_flex_df], ignore_index=True
        )

        master_regular_df.to_csv(regular_out_path, index=False)
        master_standard_training_df.to_csv(standard_training_out_path, index=False)
        master_special_training_df.to_csv(special_training_out_path, index=False)
        master_flex_df.to_csv(flex_out_path, index=False)
        master_all_df.to_csv(all_out_path, index=False)

        """ Uncomment if want it to be applied in the pipeline
        # --- qualification + training + vacation pipeline (members/daniel) ---
        # Runs Phase 1-3 (primary/secondary qualifications), Then, the random
        # training assignment, THEN the vacation assignment. Training needs the
        # full working-day roster (regular + both training files) plus the
        # qualifications; vacation then runs on the roster left after training.
        # Invoked once, here, after the three files are written; it writes:
        #   outputs/Supply_Regular_qualified.csv
        #   outputs/standard_training_data.csv / outputs/standard_training_scheduled_df.csv
        #   outputs/intl_special_training_data.csv / outputs/intl_special_training_scheduled_df.csv
        #   outputs/vacation_data.csv / outputs/supply_data_after_vacation_shrinkage.csv
        # This supersedes the qualification-only call below.
        from members.daniel.main import run_pipeline
        run_pipeline(
            regular_path=regular_out_path,
            standard_training_path=standard_training_out_path,
            special_training_path=special_training_out_path,
        )

        # --- qualification pipeline only (Phase 1-3, members/daniel) ---
        # from members.daniel.qualification_mapper import apply_qualification_pipeline
        # apply_qualification_pipeline(
        #     regular_out_path, standard_training_out_path, special_training_out_path
        # )
        """

        logging.info(f"\n================ EXPORT COMPLETE ================")
        logging.info(
            f"Total Regular Shifts       : {len(master_regular_df):,} -> Saved to '{regular_out_path}'"
        )
        logging.info(
            f"Total Standard Training    : {len(master_standard_training_df):,} -> Saved to '{standard_training_out_path}'"
        )
        logging.info(
            f"Total Special Training     : {len(master_special_training_df):,} -> Saved to '{special_training_out_path}'"
        )
        logging.info(
            f"Total Flex Assignments     : {len(master_flex_df):,} -> Saved to '{flex_out_path}'"
        )
        logging.info(
            f"Total Supply (Regular+Flex): {len(master_all_df):,} -> Saved to '{all_out_path}'"
        )
        logging.info("=================================================\n")

        return (
            master_regular_df,
            master_standard_training_df,
            master_special_training_df,
            master_flex_df,
            master_all_df,
        )
