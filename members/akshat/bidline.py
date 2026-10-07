import pandas as pd
from typing import Dict
from members.akshat.config import ProjectConfig

class BidlineGenerator:
    """
    Generates 4-3-4-4 bidline coverage vectors across 15 offsets.
    """
    def __init__(self, start_date: str, end_date: str):
        self.date_range = pd.date_range(start=start_date, end=end_date)
        self.date_list = [dt.strftime('%Y-%m-%d') for dt in self.date_range]
        self.pattern = ProjectConfig.BIDLINE_PATTERN
        self.cycle_length = ProjectConfig.CYCLE_LENGTH

    def get_coverage_vector(self, offset: int) -> Dict[str, int]:
        """
        Generates a dictionary with date as key and 1 (working) / 0 (off) as value
        for a bidline starting with the specified offset.
        """
        if offset < 0 or offset >= self.cycle_length:
            raise ValueError(f"Offset must be between 0 and {self.cycle_length - 1}")

        coverage_vector = {}
        for day_idx, date_str in enumerate(self.date_list):
            pattern_pos = (day_idx - offset) % self.cycle_length
            coverage_vector[date_str] = self.pattern[pattern_pos]

        return coverage_vector

    def get_all_offset_vectors() -> Dict[int, Dict[str, int]]:
        """
        Returns coverage vectors for all 15 offsets.
        """
        return {offset: self.get_coverage_vector(offset) for offset in range(self.cycle_length)}