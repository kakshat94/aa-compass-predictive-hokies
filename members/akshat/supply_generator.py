import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Any


class SupplyGenerator:
    """
    Iterates over all shift models in `orch.models`, processes their X (bidline) 
    and Y (training) solution variables, and consolidates the output into a single
    Supply_Regular.csv and a single Supply_Training.csv file.
    Works seamlessly across Windows, macOS, and Linux.
    """

    SHIFT_SCHEDULES = {
        'MORNING':  {'start': '05:00:00', 'end': '13:00:00', 'length': 8, 'code': 'M'},
        'EVENING':  {'start': '13:00:00', 'end': '21:00:00', 'length': 8, 'code': 'E'},
        'MIDNIGHT': {'start': '21:00:00', 'end': '05:00:00', 'length': 8, 'code': 'MD'}
    }

    SUPPLY_COLUMNS = [
        'INDEX', 'MAJOR_WORK_TYPE', 'START_TIME', 'EMPLOYEE_TITLE', 'WORK_CENTER_ID',
        'CONTRACTOR_FLAG', 'CLUSTER', 'EMPLOYEE_WORK_CENTER_ID', 'TECH_ID', 'END_TIME',
        'SHIFT_DATE', 'SHIFT_LENGTH', 'SHIFT_TYPE', 'FINAL_QUALS', 'PRIMARY_QUAL',
        'DOMESTIC', 'ATC', 'ROC', 'CHF', 'OC', 'LATIN', 'EUROPE', 'PACIFIC',
        'QUAL_COUNT', 'SF_LOAD_TMS'
    ]

    def __init__(self, coverage_vectors: Dict):
        self.coverage_vectors = coverage_vectors

    @staticmethod
    def _format_date(dt: datetime) -> str:
        """Cross-platform non-zero-padded date string (M/D/YYYY)."""
        return f"{dt.month}/{dt.day}/{dt.year}"

    def process_all_models(
        self,
        orch_models: Dict[str, Any],
        regular_out_path: str = "Supply_Regular.csv",
        standard_training_out_path: str = "Supply_Standard_Training.csv",
        special_training_out_path: str = "Supply_Standard_Training.csv"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_regular_rows: List[Dict] = []
        all_special_training_rows: List[Dict] = []
        all_standard_training_rows: List[Dict] = []

        for shift_key, model in orch_models.items():
            shift_type = shift_key.upper()
            if shift_type not in self.SHIFT_SCHEDULES:
                print(f"Warning: Shift type '{shift_key}' not recognized. Skipping.")
                continue

            shift_info = self.SHIFT_SCHEDULES[shift_type]
            start_str = shift_info['start']
            end_str = shift_info['end']
            length = shift_info['length']
            shift_code = shift_info['code']

            # Extract solution dictionaries
            X_dict = {i: e.X for i, e in model.X.items()}
            Y_dict = {i: e.X for i, e in model.Y.items()}

            shift_regular_records: Dict[Tuple[str, str, str, str], Dict] = {}

            # Step 1: Populate regular records
            for (wc, num_id, off_str), val in X_dict.items():
                if val > 0.5:
                    wc_prefix = wc[0].upper()
                    num_code = num_id.replace('Number', 'N')
                    off_code = off_str.replace('Offset', 'O')
                    tech_id = f"{wc_prefix}_{num_code}_{off_code}_{shift_code}"

                    cov_vec = self.coverage_vectors.get(off_str, {})

                    for date_str, is_working in cov_vec.items():
                        if is_working == 1:
                            dt = datetime.strptime(date_str, "%Y-%m-%d")
                            fmt_date = self._format_date(dt)

                            if shift_type == 'MIDNIGHT':
                                end_dt = dt + timedelta(days=1)
                                end_date_fmt = self._format_date(end_dt)
                            else:
                                end_date_fmt = fmt_date

                            start_ts = f"{fmt_date} {start_str}"
                            end_ts = f"{end_date_fmt} {end_str}"

                            row = {
                                'INDEX': np.nan,
                                'MAJOR_WORK_TYPE': 'All',
                                'START_TIME': start_ts,
                                'EMPLOYEE_TITLE': wc,
                                'WORK_CENTER_ID': wc,
                                'CONTRACTOR_FLAG': 'N',
                                'CLUSTER': 1,
                                'EMPLOYEE_WORK_CENTER_ID': wc,
                                'TECH_ID': tech_id,
                                'END_TIME': end_ts,
                                'SHIFT_DATE': fmt_date,
                                'SHIFT_LENGTH': length,
                                'SHIFT_TYPE': shift_type,
                                'FINAL_QUALS': np.nan,
                                'PRIMARY_QUAL': wc,
                                'DOMESTIC': np.nan,
                                'ATC': np.nan,
                                'ROC': np.nan,
                                'CHF': np.nan,
                                'OC': np.nan,
                                'LATIN': np.nan,
                                'EUROPE': np.nan,
                                'PACIFIC': np.nan,
                                'QUAL_COUNT': np.nan,
                                'SF_LOAD_TMS': np.nan
                            }

                            key = (wc, num_id, off_str, date_str)
                            shift_regular_records[key] = row

            # Step 2: Extract training shifts
            for (w, n, o, d), v in Y_dict.items():
                if v > 0.5:
                    key = (w, n, o, d)
                    if key in shift_regular_records:
                        training_row = shift_regular_records.pop(key)
                        if datetime.strptime(d, "%Y-%m-%d").month in (2, 5, 9):
                            all_standard_training_rows.append(training_row)
                        else:
                            all_special_training_rows.append(training_row)
                    else:
                        print('Error')

            all_regular_rows.extend(shift_regular_records.values())

        # Step 3: Combine and export
        master_regular_df = pd.DataFrame(all_regular_rows)
        master_standard_training_df = pd.DataFrame(all_standard_training_rows)
        master_special_training_df = pd.DataFrame(all_special_training_rows)

        if master_regular_df.empty:
            master_regular_df = pd.DataFrame(columns=self.SUPPLY_COLUMNS)
        else:
            master_regular_df = master_regular_df[self.SUPPLY_COLUMNS]

        if master_standard_training_df.empty:
            master_standard_training_df = pd.DataFrame(columns=self.SUPPLY_COLUMNS)
        else:
            master_standard_training_df = master_standard_training_df[self.SUPPLY_COLUMNS]

        if master_special_training_df.empty:
            master_special_training_df = pd.DataFrame(columns=self.SUPPLY_COLUMNS)
        else:
            master_special_training_df = master_special_training_df[self.SUPPLY_COLUMNS]

        master_regular_df.to_csv(regular_out_path, index=False)
        master_standard_training_df.to_csv(standard_training_out_path, index=False)
        master_special_training_df.to_csv(special_training_out_path, index=False)

        print(f"\n================ EXPORT COMPLETE ================")
        print(f"Total Regular Shifts  : {len(master_regular_df):,} -> Saved to '{regular_out_path}'")
        print(f"Total Standard Training Shifts : {len(master_standard_training_df):,} -> Saved to '{standard_training_out_path}'")
        print(f"Total Special Training Shifts : {len(master_special_training_df):,} -> Saved to '{special_training_out_path}'")
        print("=================================================\n")

        return master_regular_df, master_standard_training_df, master_special_training_df
