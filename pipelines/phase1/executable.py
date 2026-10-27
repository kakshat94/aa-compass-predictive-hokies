from pipelines.phase1.config import ProjectConfig
from pipelines.phase1.orchestrator import Orchestrator
from pipelines.phase1.supply_generator import SupplyGenerator
import pandas as pd


def main(PHASE1_DIR, ROOT):

    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    demand_df = pd.read_csv(ROOT / "inputs" / "annual_desk_needs_reference.csv")
    wc_sets = ProjectConfig.get_workcenter_day_sets(demand_df)
    orch = Orchestrator(demand_df, wc_sets)
    bidlines = orch.initiate_models()
    SupplyGenerator(orch.coverage_vectors).process_all_models(
        orch_models=orch.models,
        demand_df=demand_df,
        filter_by_demand=False,
        regular_out_path=str(PHASE1_DIR / "Supply_Regular.csv"),
        standard_training_out_path=str(PHASE1_DIR / "Supply_Standard_Training.csv"),
        special_training_out_path=str(PHASE1_DIR / "Supply_Special_Training.csv"),
        flex_out_path=str(PHASE1_DIR / "Flex_assignment.csv"),
        all_out_path=str(PHASE1_DIR / "Supply_all.csv"),
    )
    return bidlines
