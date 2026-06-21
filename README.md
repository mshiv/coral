# CORAL Emulator Project

*COastal Response emulator & Adaptation Learning* | *Coastal Options for Resilience & Adaptation Learning* (name tbd)

This project develops a machine-learning emulator for coupled coastal flooding simulations over Savannah, Georgia in the Southeastern USA. The goal is to approximate expensive physics-based GeoClaw-LISFLOOD-FP outputs with a fast surrogate model that can explore many storm, sea-level-rise, and (bathymetric) intervention scenarios.

## Project Goal

Predict flood depth fields and/or summary inundation metrics from:
- tropical cyclone parameters
- sea-level-rise scenarios
- topographic intervention parameters
- static landscape features

The emulator will be trained on a dataset generated from physics-based simulations and evaluated on held-out scenarios and historical events with observational constraints.

## Repository Layout

- `src/coral/` # the installable package (`pip install -e .`)
  - `preprocess/` # clip DEM, Manning grid, coastline gauges/bci 
  - `couple/` # GeoClaw -> LISFLOOD-FP to create the storm surge boundary conditions, `.bdy`
  - `geoclaw/` # run-dir assembly
  - `lisflood/` # run-dir assembly
  - `postprocess/` # COG, Zarr, animation, maps, hazard
  - `validate/` # HWM + gauge/surge calibration
  - `viz/` # AMR/domain/QC figures
  - `emulator/` # the ML model
- `configs/` # scenario YAMLs (`base.yaml` + `scenarios/*.yaml`) and `hpc/` profiles; one file per scenario
- `templates/` # jinja templates (`savannah.par.j2`, setrun) rendered from a scenario config
- `workflows/` # `Snakefile` pipeline + `slurm/` submit scripts
- `patches/` + `docs/LISFLOOD_build.md` # LISFLOOD-FP source fixes (stock 8.0.3 segfaults) as patch + fork/submodule
- `notebooks/` — exploration (Zarr, emulator dev)
- `data/` , `runs/` # gitignored; input data files and generated HPC run dirs/outputs
- `docs/` # methods, HPC setup, data dictionary, migration status

## Workflow (config-driven)

Each run is one scenario YAML. Example:
```bash
pip install -e .
coral show       configs/scenarios/savannah_matthew_LR.yaml      # resolved config
coral render-par configs/scenarios/savannah_matthew_LR.yaml runs/sav/savannah.par
coral build-bdy  configs/scenarios/savannah_matthew_LR.yaml <geoclaw_output> <bci_in> runs/sav/inputs
# full pipeline / sweeps:
snakemake -c1 all                # every scenario in configs/scenarios/
```
Each new scenario (different storm, Manning, SLR, drag, resolution) will require that you copy one YAML and
change a field. The scenario set doubles as the ML emulator's training index. See`docs/LISFLOOD_build.md` to build the solver (it includes bug fixes and patches for building on the GT PACE cluster).

## Milestones

1. Define the scenario table and emulator inputs/outputs.
2. Build one working GeoClaw-LISFLOOD-FP baseline run.
3. Convert physics outputs into a training dataset.
4. Train a baseline emulator.
5. Evaluate on held-out scenarios and measure speedup.

## Notes

Older SSLS/ERA5 surge/flood and coursework notebooks have been moved to `archive/` for reference.

## Contact

Shiva Muruganandham - smurugan9@gatech.edu
