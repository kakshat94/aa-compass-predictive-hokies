from pathlib import Path
import pandas as pd
import logging
from pipelines.phase2_training.main import run_pipeline
from pipelines.phase2_training.training_assigner import (
    CLASS_SIZE_DEFAULT,
    load_training_months,
)
from pipelines.phase2_training.verify_training import verify_training


def main(
    PHASE1_DIR: Path, PHASE2_DIR: Path, ROOT: Path, PARAMS_PATH: Path, SOLVER: str
):
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    flex_assignment = PHASE1_DIR / "Flex_assignment.csv"
    flex_assignment_output = PHASE2_DIR / "Flex_assignment_qualified.csv"
    run_pipeline(
        regular_path=str(PHASE1_DIR / "Supply_Regular.csv"),
        standard_training_path=str(PHASE1_DIR / "Supply_Standard_Training.csv"),
        special_training_path=str(PHASE1_DIR / "Supply_Special_Training.csv"),
        qual_data_path=str(ROOT / "inputs" / "Supply_sample.csv"),
        params_path=str(PARAMS_PATH),
        volume_path=str(ROOT / "inputs" / "demand_volume_per_day_reference.csv"),
        demand_path=str(ROOT / "inputs" / "annual_desk_needs_reference.csv"),
        method="exact-mip",
        solver=SOLVER,
        log=True,
        class_size=CLASS_SIZE_DEFAULT,
        output_dir=str(PHASE2_DIR),
        freeze_phase1_training=True,
        run_vacation_stage=False,
        write=True,
        flex_assignment_path=str(flex_assignment),
        flex_assignment_output_path=str(flex_assignment_output),
    )

    problems = verify_training(
        str(PHASE2_DIR / "Supply_Regular_qualified.csv"),
        str(PHASE2_DIR / "standard_training_data.csv"),
        str(PHASE2_DIR / "intl_special_training_data.csv"),
        str(PHASE2_DIR / "standard_training_scheduled_df.csv"),
        str(PHASE2_DIR / "intl_special_training_scheduled_df.csv"),
        load_training_months(str(PARAMS_PATH)),
        CLASS_SIZE_DEFAULT,
        flex_assignment_path=str(flex_assignment),
    )
    if problems:
        raise RuntimeError("Training verification failed: " + "; ".join(problems))
    logging.info("training check         : OK")

    # write all_training_data.csv
    std = pd.read_csv(PHASE2_DIR / "standard_training_data.csv")
    intl = pd.read_csv(PHASE2_DIR / "intl_special_training_data.csv")
    std = std.loc[:, ~std.columns.str.startswith("Unnamed:")]
    intl = intl.loc[:, ~intl.columns.str.startswith("Unnamed:")]
    cols = list(std.columns) + [c for c in intl.columns if c not in std.columns]
    combined = pd.concat(
        [std.reindex(columns=cols), intl.reindex(columns=cols)], ignore_index=True
    )
    if combined.duplicated(["TECH_ID", "SHIFT_DATE"]).any():
        raise RuntimeError("all_training_data.csv has duplicate (tech, date) rows")
    all_training_path = PHASE2_DIR / "all_training_data.csv"
    combined.to_csv(all_training_path, index=False)
    logging.info(
        f"all_training_data      : {len(combined):,} rows -> {all_training_path}"
    )
