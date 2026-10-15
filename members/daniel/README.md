# members/daniel: Phase II (qualifications, training, vacation)

Reads the Phase I bid-line roster written by `members/akshat/main.py`
(`outputs/Supply_Regular.csv` plus the two training files), adds the Phase II
shrinkage layers, and scores the *effective* supply (the roster less training and
vacation) against `data/Demand.csv` with the shared evaluator
`common/coverage_report.py`.

Paths: `data/` holds the competition inputs (`Demand.csv`, `Supply_sample.csv`,
`demand_volume_per_day_reference.csv`); `outputs/` holds everything generated,
including the Phase I roster and the Phase II results. The pipeline exposes
`--input-data-dir` (default `data`) and `--output-data-dir` (default `outputs`)
if you need to relocate them.

Environment: run every command from the repo root with the repo virtualenv
(`.venv/bin/python …`, or `source .venv/bin/activate` first). The exact training
layer needs a Gurobi licence; to use FICO Xpress, point `$XPRESS` at the folder
that holds `xpauth.xpr`.

## Commands (all from the repo root)

| Command | What you get |
|---|---|
| `.venv/bin/python members/daniel/main.py --random --score --reconcile` | Full pipeline with the random training baseline. Writes `outputs/Supply_Regular_qualified.csv`, the four `*training*` files, `vacation_data.csv` and `supply_data_after_vacation_shrinkage.csv`; prints the vacation balance metrics, the coverage summary and the reconciliation check. |
| `.venv/bin/python members/daniel/main.py --exact-mip --solver gurobi --output-data-dir outputs/exact_gurobi --score --reconcile --log` | Same, but training is placed by the exact BLT MIP (Gurobi); the files land in `outputs/exact_gurobi/`, plus `training_run_log.txt` with per-(shift, month) status, objective and seconds. |
| `XPRESS="…/xpauth.xpr-dir" .venv/bin/python members/daniel/main.py --exact-mip --solver xpress --output-data-dir outputs/exact_xpress --score --reconcile --log` | Same with FICO Xpress. |
| `.venv/bin/python members/daniel/training_exact.py --solver gurobi --output-data-dir outputs/exact_gurobi` | Training layer only (no vacation); writes the regular/training files for the exact solution. |
| `.venv/bin/python members/daniel/demand_scaling.py` | `members/daniel/outputs/demand_shift_date_scaled.csv` and `…_work_center_scaled.csv`: desk demand rescaled to `[-1,1]`. |
| `.venv/bin/python members/daniel/coverage_report.py` | Scores the effective supply and reconciles the shrinkage files (`--score-only`, `--reconcile-only`). |
| `.venv/bin/python members/daniel/verify_training.py` | Verifies the training solution against the `params.yaml` windows (exit code 0 = OK). |
| `.venv/bin/python members/daniel/verify_pipeline.py` | Verifies the qualification outputs against the reference data (exit code 0 = OK). |
| `.venv/bin/python common/coverage_report.py --supply outputs/Supply_Regular.csv --name "Akshat"` | Scores any single supply file with the shared evaluator. |
| `.venv/bin/python common/plot_demand.py` / `.venv/bin/python common/plot_coverage.py` | Interactive demand / coverage HTML plots under `outputs/`. |
