import pandas as pd
from datetime import datetime
from typing import Dict, List, Union
import logging


class BidlineGenerator:
    """
    Generates 4-3-4-4 bidline coverage vectors for 15 offsets across a specified date horizon.

    Pattern Structure (15-day cycle):
      Days 0..3   : 4 Working Days (1)
      Days 4..6   : 3 Off Days     (0)
      Days 7..10  : 4 Working Days (1)
      Days 11..14 : 4 Off Days     (0)
    """

    CYCLE_LENGTH = 15
    PATTERN = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

    def __init__(self, start_date: str = "2025-01-14", end_date: str = "2026-01-11"):
        """
        Initializes the date range horizon. Dates can be provided as YYYY-MM-DD or MM/DD/YYYY.
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        # Build list of date strings (YYYY-MM-DD)
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date)
        self.date_strings = [dt.strftime("%Y-%m-%d") for dt in self.date_range]

    def get_coverage_vector(self, offset: int) -> Dict[str, int]:
        """
        Returns a dictionary {date_str: 0 or 1} for a specific offset (0 to 14).

        Args:
            offset (int): Offset integer from 0 to 14 representing cycle shift.

        Returns:
            Dict[str, int]: Key is YYYY-MM-DD, Value is 1 (working) or 0 (off).
        """
        if not (0 <= offset < self.CYCLE_LENGTH):
            raise ValueError(
                f"Offset must be an integer between 0 and {self.CYCLE_LENGTH - 1}."
            )

        coverage = {}
        for day_idx, date_str in enumerate(self.date_strings):
            pattern_pos = (day_idx - offset) % self.CYCLE_LENGTH
            coverage[date_str] = self.PATTERN[pattern_pos]

        return coverage

    def get_all_coverage_vectors(self) -> Dict[int, Dict[str, int]]:
        """
        Generates coverage vectors for all 15 offsets.

        Returns:
            Dict[int, Dict[str, int]]: {offset_id: {date_str: 0_or_1}}
        """
        return {
            f"Offset{offset}": self.get_coverage_vector(offset)
            for offset in range(self.CYCLE_LENGTH)
        }

    def get_coverage_dataframe(self) -> pd.DataFrame:
        """
        Returns a full tabular coverage matrix where:
          - Rows: Dates in horizon (YYYY-MM-DD)
          - Columns: Offset_0 through Offset_14
        """
        all_vecs = self.get_all_coverage_vectors()
        df = pd.DataFrame(all_vecs, index=self.date_strings)
        df.columns = [f"Offset_{i}" for i in range(self.CYCLE_LENGTH)]
        return df


if __name__ == "__main__":
    # Sanity Test & Sample Verification
    gen = BidlineGenerator("2025-01-14", "2026-01-11")
    cov_df = gen.get_coverage_dataframe()

    logging.info(f"Total dates generated: {len(cov_df)}")
    logging.info("\nSample Coverage Matrix (First 10 days for Offsets 0, 1, 5, 6):")
    logging.info(cov_df[["Offset_0", "Offset_1", "Offset_5", "Offset_6"]].head(10))
