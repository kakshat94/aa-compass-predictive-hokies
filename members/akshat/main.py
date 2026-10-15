import sys
import os
import pandas as pd
from datetime import datetime
# Allow importing common tools from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import gurobipy as gp
from config import ProjectConfig
from model import WorkforceOptimizer
from supply_generator import SupplyGenerator
from orchestrator import Orchestrator
from common.coverage_report import CoverageReporter

def run():
    
    start = datetime.now()
    demand_file = "data/Demand.csv"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    demand_df = pd.read_csv(demand_file)
    
    # 1. Run your unique strategy
    wc_sets = ProjectConfig.get_workcenter_day_sets(demand_df)
    orch = Orchestrator(demand_df, wc_sets)
    orch.initiate_models()

    # Instantiate and process
    exporter = SupplyGenerator(orch.coverage_vectors)

    # Process all shift models in orch.models simultaneously
    regular_df, standard_training_df, special_training_df = exporter.process_all_models(
        orch_models=orch.models,
        regular_out_path=os.path.join(out_dir, "Supply_Regular.csv"),
        standard_training_out_path=os.path.join(out_dir, "Supply_Standard_Training.csv"),
        special_training_out_path=os.path.join(out_dir, "Supply_Special_Training.csv")
    )

    end = datetime.now() - start
    print(f"Total Time Elapsed: {end}")
    
    # 3. Directly evaluate using common tool
    report_df = CoverageReporter.evaluate(demand_file, os.path.join(out_dir, "Supply_Regular.csv"))
    CoverageReporter.print_summary(report_df, member_name="Member 1 (FICO Xpress)")

if __name__ == "__main__":
    run()