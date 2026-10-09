import os
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any


class ProjectConfig:
    # 4-3-4-4 Bidline Structure Parameters
    BIDLINE_PATTERN = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    CYCLE_LENGTH = len(BIDLINE_PATTERN)  # 15
    TOTAL_OFFSETS = 15

    # Shift Timings
    SHIFT_SCHEDULES = {
        'MORNING':  {'start': '05:00', 'end': '13:00', 'length': 8},
        'EVENING':  {'start': '13:00', 'end': '21:00', 'length': 8},
        'MIDNIGHT': {'start': '21:00', 'end': '05:00', 'length': 8}
    }

    # Penalties
    PENALTY_UNDERCOVERAGE = 100.0
    PENALTY_OVERCOVERAGE = 10.0

    @staticmethod
    def load_parameters(yaml_path: str = None) -> Dict[str, Any]:
        """Loads work center rules from params.yaml."""
        if yaml_path is None:
            yaml_path = os.path.join(os.path.dirname(__file__), "params.yaml")
        
        with open(yaml_path, "r") as file:
            params = yaml.safe_load(file)
        return params.get("work_centers", {})

    @classmethod
    def get_workcenter_day_sets(
        cls, demand_df: pd.DataFrame, yaml_path: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parses Demand.csv and params.yaml to generate date sets for each WorkCenter:
          - 'demand_start_window': 30-day window from start date [List[str]]
          - 'training_optimizer': List of lists where each inner list corresponds 
                                  to dates in a specific TrainingMonthOptimizer month [List[List[str]]]
          - 'training_month': List of lists corresponding to TrainingMonths [List[List[str]]]
        """
        params = cls.load_parameters(yaml_path)
        
        df = demand_df.copy()
        df['SHIFT_DATE_DT'] = pd.to_datetime(df['SHIFT_DATE'])
        df['DATE_STR'] = df['SHIFT_DATE_DT'].dt.strftime('%Y-%m-%d')
        df['MONTH_NAME'] = df['SHIFT_DATE_DT'].dt.strftime('%B')

        workcenter_sets = {}

        for wc, rules in params.items():
            wc_df = df[df['WORK_CENTER'] == wc]
            available_years = wc_df['SHIFT_DATE_DT'].dt.year.unique()
            primary_year = available_years[0] if len(available_years) > 0 else 2025

            # 1. Demand Start Date 30-Day Window Set (Flat List)
            demand_start_dates = []
            start_date_str = rules.get('demand_start_date')
            
            if start_date_str:
                full_start_str = f"{start_date_str}/{primary_year}"
                start_dt = datetime.strptime(full_start_str, "%m/%d/%Y")
                end_dt = start_dt + pd.Timedelta(days=29)  # 30 days total inclusive
                
                demand_start_dates = sorted(
                    wc_df[
                        (wc_df['SHIFT_DATE_DT'] >= start_dt) & 
                        (wc_df['SHIFT_DATE_DT'] <= end_dt)
                    ]['DATE_STR'].unique().tolist()
                )

            # 2. Training Month Optimizer Set (List of Lists per month)
            opt_months = rules.get('training_month_optimizer', [])
            training_opt_month_lists = []
            for month in opt_months:
                month_dates = sorted(
                    wc_df[wc_df['MONTH_NAME'] == month]['DATE_STR'].unique().tolist()
                )
                training_opt_month_lists.append(month_dates)

            # 3. Other Training Months Set (List of Lists per month)
            training_months = rules.get('training_months', [])
            training_month_lists = []
            for month in training_months:
                month_dates = sorted(
                    wc_df[wc_df['MONTH_NAME'] == month]['DATE_STR'].unique().tolist()
                )
                training_month_lists.append(month_dates)

            workcenter_sets[wc] = {
                'demand_days': demand_start_dates,
                'training_optimizer': training_opt_month_lists,
                'training_days': training_month_lists
            }

        return workcenter_sets