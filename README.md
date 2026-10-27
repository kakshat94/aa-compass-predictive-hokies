# aa-compass-predictive-hokies

## Quick Start

```bash
python main.py
```

No arguments are needed. Solver settings and time limits are configured as
constants at the top of `main.py`.

---

## Repository Layout

```
main.py                             <- single entry point, run from repo root

pipelines/
  phase1/
    bidline.py                      <- 4-3-4-4 coverage vector generation
    config.py                       <- workcenter day-set construction
    model.py                        <- Gurobi bidline MIP
    orchestrator.py                 <- builds and solves shift models
    supply_generator.py             <- converts model solution to supply CSVs
    executable.py                   <- Executable file called from root
  phase2_training/
    main.py / run_pipeline()        <- qualification + training pipeline
    training_exact.py               <- exact training MIP (xpress/gurobi)
    training_assigner.py            <- training assignment logic
    qualification_mapper.py         <- primary/secondary qual mapping
    solvers.py                      <- solver-agnostic backend (xpress/gurobi)
    verify_training.py              <- training integrity checks
    vacation_assigner.py            <- vacation assignment logic
    balance_metrics.py              <- balance metric utilities
    executable.py                   <- Executable file called from root
  phase2_vacation_flex/
    vacation_cp_sat.py              <- Phase 3a: vacation scheduling (CP-SAT)
    flex_day_maxflow.py             <- Phase 3b: flex-day scheduling (max-flow)
    executable.py                   <- Executable file called from root
    README_vacation.md
    README_flex_days.md

common/
  coverage_report.py                <- CoverageReporter utility
bidline_check.py                    <- BidlineChecker integrity checks
params.yaml                         <- all parameterization (shifts, flex days)

inputs/
  annual_desk_needs_reference.csv
  demand_volume_per_day_reference.csv
  Supply_sample.csv

outputs/
  phase1/                           <- Phase 1 outputs
  phase2_training/                  <- Phase 2 outputs
  phase2_vacation_flex/             <- Phase 3 outputs
```

---

## Pipeline Procedure (`main.py`)

`main.py` runs the full pipeline in-process — no subprocesses, no CLI flags.
Each phase is timed and a summary is printed at the end.

Solver settings are module-level constants in `main.py`:

```python
SOLVER       = "xpress"   # solver for Phase 2 training MIP
FAIR_SECONDS = 120.0      # vacation fairness stage time budget
```

### Phase 1 — Bidline Creation

1. Loads `inputs/annual_desk_needs_reference.csv` and builds workcenter day-sets via `ProjectConfig`.
2. Runs `Orchestrator.initiate_models()` — solves a Xpress Binary Program for MORNING
   and MIDNIGHT shifts (EVENING is derived from MORNING when
   `shifts.morning_equals_evening: true` in `params.yaml`).
3. Calls `SupplyGenerator.process_all_models()` to convert the solution into
   five CSV files.

Outputs written to `outputs/phase1/`:
- `Supply_Regular.csv` — regular working shifts
- `Supply_Standard_Training.csv` — standard training shifts (Feb, May, Sep)
- `Supply_Special_Training.csv` — special/international training shifts
- `Flex_assignment.csv` — flex day assignments from the MIP
- `Supply_all.csv` — concatenation of regular + flex supply

Parameterization (`params.yaml`):
- `shifts.morning_equals_evening: true` — mirror MORNING solution as EVENING
  supply with distinct TECH_IDs and shift times
- `flex_days.<WorkCenter>.max_flex_days` — which work centers get flex days
  and the per-bidline cap

### Check 1 — Phase 1 Integrity

```python
validator = BidlinePipelineValidator(reg, std, spc)
all_initial_tech_ids = set(reg['TECH_ID']).union(std['TECH_ID'], spc['TECH_ID'])
unmapped_tech_ids = all_initial_tech_ids - set(validator.tech_id_patterns.keys())

if unmapped_tech_ids:
    logging.info(f"Stage 1 Structural Error: The following TECH_IDs violate the 4-3-4-4 structure: {list(unmapped_tech_ids)}")

stage1_flex_results = validator.stage4_flex_checks(flx)
if stage1_flex_results['invalid_flex_days'] or stage1_flex_results['max_count_violations']:
    logging.info("Stage 1 (Baseline) Errors Found:")
    logging.info(f" - Initial Flex days assigned on original WORKING days: {stage1_flex_results['invalid_flex_days']}")
    logging.info(f" - Initial TECH_IDs exceeding 4 flex days: {stage1_flex_results['max_count_violations']}")        # 
CoverageReporter.print_summary(...)   # supply vs demand summary
```

### Phase 2 — Qualification + Training


1. Calls `pipelines/phase2_training/run_pipeline()` with Phase 1 outputs as inputs.
2. Applies primary/secondary qualifications to the regular roster.
3. Loads Phase 1 `Flex_assignment.csv`, blocks those dispatcher-days from every
   training assignment, and writes a qualified flex copy with `FINAL_QUALS`,
   the eight qualification flags, `QUAL_COUNT`, and `SECONDARY_QUALS`.
4. Solves the exact training MIP using the solver set in `SOLVER`
   (`"xpress"` by default), freezing the Phase 1 training day assignments.
5. Writes training outputs, then calls `verify_training()`, which also rejects
   any training that falls on a pre-assigned flex day.
6. Concatenates standard + international training into `all_training_data.csv`.

Inputs (from `outputs/phase1/`):
- `Supply_Regular.csv`, `Supply_Standard_Training.csv`, `Supply_Special_Training.csv`, `Flex_assignment.csv`

Outputs written to `outputs/phase2_training/`:
- `Supply_Regular_qualified.csv`
- `standard_training_data.csv` / `standard_training_scheduled_df.csv`
- `intl_special_training_data.csv` / `intl_special_training_scheduled_df.csv`
- `all_training_data.csv`
- `Flex_assignment_qualified.csv`
- `training_run_log.txt`, `training_xpress.log`

### Check 2 — Phase 2 Integrity

reusing the `BidlinePipelineValidator` from Check 1:

```python
stage2_results = validator.stage2_training_checks(std2, spc2, flx)

if stage2_results['invalid_training_days'] or stage2_results['duplicate_trainings']:
    logging.info("Stage 2 Errors Found:")
    logging.info(f" - Invalid training days (assigned on OFF day): {stage2_results['invalid_training_days']}")
    logging.info(f" - Duplicate trainings on same day: {stage2_results['duplicate_trainings']}")

CoverageReporter.print_summary(...)
```

### Phase 3a — Vacation Scheduling

1. Builds daily vacation quotas from operational volume quintiles (25–29/day).
2. Solves a CP-SAT coverage model, then a fairness model (minimize max
   absolute deviation across dispatchers).
3. Runs adaptive LNS and zero-vacation targeted swap repair.

Inputs:
- `outputs/phase2_training/Supply_Regular_qualified.csv`
- `outputs/phase2_training/all_training_data.csv`
- `inputs/demand_volume_per_day_reference.csv`
- `inputs/annual_desk_needs_reference.csv`

Outputs written to `outputs/phase2_vacation_flex/`:
- `vacation_initial.csv`, `vacation_daily_targets.csv`
- `vacation_fairness_by_dispatcher.csv`, `vacation_initial_summary.json`
- `demand_imputation_audit.csv`

### Check 3a — Phase 3a Integrity

Placeholder — final pipeline check to be implemented.

### Phase 3b — Flex Day Scheduling


1. Builds a bipartite max-flow network: dispatchers → shortage buckets.
2. Solves for the maximum number of flex days that reduce staffing shortages,
   subject to the per-dispatcher cap from `params.yaml`.

Inputs:
- `outputs/phase2_training/Supply_Regular_qualified.csv`
- `outputs/phase2_training/all_training_data.csv`
- `outputs/phase2_training/Flex_assignment_qualified.csv`
- `outputs/phase2_vacation_flex/vacation_initial.csv`
- `outputs/phase2_vacation_flex/demand_imputation_audit.csv`
- `inputs/annual_desk_needs_reference.csv`

Outputs written to `outputs/phase2_vacation_flex/`:
- `flex_days.csv` (new flex only), `Flex_assignment_final.csv` (pre-assigned + new)
- `flex_by_dispatcher.csv`
- `staffing_coverage_before_after_flex.csv`, `flex_summary.json`

### Check 3b — Phase 3b Integrity

Reads `Flex_assignment_final.csv` and verifies every flex day is on an original
off day and no dispatch exceeds four flex days across the year. The pipeline
stops if either check fails.

---

## Evaluate Sample Solution

```bash
python common/coverage_report.py --supply inputs/Supply_sample.csv --name "Sample"

# Save detailed report
python common/coverage_report.py --supply inputs/Supply_sample.csv \
    --output outputs/coverage_report.csv --name "Sample"
```