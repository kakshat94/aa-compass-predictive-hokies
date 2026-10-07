import pandas as pd
from datetime import datetime

class ProjectConfig:
    # Bidline Structure Parameters
    # 4 working, 3 off, 4 working, 4 off = 15-day cycle
    BIDLINE_PATTERN = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    CYCLE_LENGTH = len(BIDLINE_PATTERN)  # 15
    TOTAL_OFFSETS = 15

    # Shift Timings (Start Time, End Time, Length)
    SHIFT_SCHEDULES = {
        'MORNING':  {'start': '05:00', 'end': '13:00', 'length': 8},
        'EVENING':  {'start': '13:00', 'end': '21:00', 'length': 8},
        'MIDNIGHT': {'start': '21:00', 'end': '05:00', 'length': 8}
    }

    # Optimization Penalties
    PENALTY_UNDERCOVERAGE = 100.0  # High penalty for unmet demand
    PENALTY_OVERCOVERAGE = 10.0    # Lower penalty for surplus supply

    @staticmethod
    def get_workcenter_day_sets(demand_df: pd.DataFrame, custom_filters: dict = None) -> dict:
        """
        Creates set of valid workcenter-day combinations supplied through parameters.
        `custom_filters` can specify allowed date windows or day-of-week rules per workcenter.
        """
        df = demand_df.copy()
        df['SHIFT_DATE'] = pd.to_datetime(df['SHIFT_DATE'])
        
        workcenter_days = {}
        unique_workcenters = df['WORK_CENTER'].unique()

        for wc in unique_workcenters:
            wc_dates = df[df['WORK_CENTER'] == wc]['SHIFT_DATE'].dt.strftime('%Y-%m-%d').unique().tolist()
            
            # Apply custom a-priori filtering if provided
            if custom_filters and wc in custom_filters:
                allowed_dates = custom_filters[wc]
                wc_dates = [d for d in wc_dates if d in allowed_dates]
                
            workcenter_days[wc] = sorted(wc_dates)

        return workcenter_days