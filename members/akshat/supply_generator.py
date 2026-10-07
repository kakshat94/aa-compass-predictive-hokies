import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from members.akshat.config import ProjectConfig
from members.akshat.bidline import BidlineGenerator

class SupplyGenerator:
    """
    Translates optimized bidline counts into individual technician shift schedules
    matching the Supply.csv structure.
    """
    def __init__(self, bidline_solutions_df: pd.DataFrame, start_date: str, end_date: str):
        self.solutions = bidline_solutions_df
        self.bidline_gen = BidlineGenerator(start_date, end_date)
        self.coverage_vectors = self.bidline_gen.get_all_offset_vectors()

    def generate() -> pd.DataFrame:
        supply_rows = []
        tech_id_counter = 100000

        for _, row in self.solutions.iterrows():
            wc = row['WORK_CENTER']
            shift = row['SHIFT_TYPE']
            offset = row['OFFSET']
            count = row['BIDLINE_COUNT']
            
            shift_info = ProjectConfig.SHIFT_SCHEDULES[shift]
            cov_vec = self.coverage_vectors[offset]

            for _ in range(count):
                tech_id_counter += 1
                
                for date_str, is_working in cov_vec.items():
                    if is_working == 1:
                        # Compute exact datetime strings
                        start_dt_str = f"{date_str} {shift_info['start']}"
                        start_dt = datetime.strptime(start_dt_str, "%Y-%m-%d %H:%M")
                        
                        if shift == 'MIDNIGHT':
                            # Midnight shifts end on the next calendar day
                            end_dt = start_dt + timedelta(hours=8)
                        else:
                            end_dt = start_dt + timedelta(hours=8)

                        fmt_date = start_dt.strftime("%-m/%-d/%Y")
                        fmt_start = start_dt.strftime("%-m/%-d/%Y %H:%M")
                        fmt_end = end_dt.strftime("%-m/%-d/%Y %H:%M")

                        supply_rows.append({
                            'INDEX': np.nan,
                            'MAJOR_WORK_TYPE': 'All',
                            'START_TIME': fmt_start,
                            'EMPLOYEE_TITLE': wc,
                            'WORK_CENTER_ID': wc,
                            'CONTRACTOR_FLAG': 'N',
                            'CLUSTER': 1,
                            'EMPLOYEE_WORK_CENTER_ID': wc,
                            'TECH_ID': tech_id_counter,
                            'END_TIME': fmt_end,
                            'SHIFT_DATE': fmt_date,
                            'SHIFT_LENGTH': shift_info['length'],
                            'SHIFT_TYPE': shift,
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
                        })

        return pd.DataFrame(supply_rows)