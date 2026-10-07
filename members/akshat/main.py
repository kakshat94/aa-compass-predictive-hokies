import sys
import os
import pandas as pd

# Allow importing common tools from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import ProjectConfig
from model import WorkforceOptimizer
from supply_generator import SupplyGenerator
from common.coverage_report import CoverageReporter

def run():
    demand_file = "../../data/Demand.csv"
    supply_out = "Supply.csv"

    demand_df = pd.read_csv(demand_file)
    
    # 1. Run your unique strategy
    wc_sets = ProjectConfig.get_workcenter_day_sets(demand_df)
    optimizer = WorkforceOptimizer(demand_df, wc_sets)
    sol_out = optimizer.build_and_solve()
    
    # 2. Output Supply file
    supply_gen = SupplyGenerator(sol_out['solution_df'], "2025-01-14", "2026-01-11")
    supply_df = supply_gen.generate()
    supply_df.to_csv(supply_out, index=False)
    
    # 3. Directly evaluate using common tool
    report_df = CoverageReporter.evaluate(demand_file, supply_out)
    CoverageReporter.print_summary(report_df, member_name="Member 1 (FICO Xpress)")

if __name__ == "__main__":
    run()