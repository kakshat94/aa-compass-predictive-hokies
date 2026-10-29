"""Vacation allocation for dispatcher supply.

Turns a fair, volume-aware set of scheduled workdays into vacation days and
returns (a) the vacation rows and (b) the remaining supply.

    from .vacation_allocation import run_from_files, AllocationConfig
"""

from .config import AllocationConfig
from .pipeline import AllocationResult, run_allocation, run_from_files

__all__ = ["AllocationConfig", "AllocationResult", "run_allocation", "run_from_files"]
