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

## Repo Layout

```Shell
/coral/
├── data/
    ├── raw/
    ├── external/
    ├── interim/
    ├── processed/
    ├── tmp/
├── notebooks/ # Data exploration notebooks
├── configs/ # YAML files for experiment scenarios (`base.yaml` + `scenarios/*.yaml`) and `hpc/` profiles - defined for GT PACE, added to default LISFLOOD-FP directory
├── templates/ # for GeoClaw (setrun) and LISFLOOD-FP (.par) run scripts, rendered from scenario config files
├── workflows/ # `Snakefile` pipeline and `slurm/` job submit scripts (and some older shell scripts, to be migrated)
├── runs/ # GeoClaw and LISFLOOD-FP model run directories, with full inputs and outputs, one for each scenario
├── docs/ # methods, HPC setup, data dictionary, status, bug fixes and patches etc
├── src/coral/ # installable package 
    ├── preprocess/ # clip DEM, Manning grid, coastline gauges/bci etc.
    ├── couple/ # boundary conditions coupling script for the storm surge file (`.bdy`) GeoClaw -> LISFLOOD-FP
    ├── geoclaw/ # run dir assembly scripts
    ├── lisflood-fp/ # run-dir assembly scripts
    ├── postprocess/ ## COG, Zarr, animation, maps, csv/json generation for CHORUS front-end tooling
    ├── validate/ # HWM + gauge/surge calibration scripts
    ├── viz/ # plotting scripts
    ├── emulator/ # ML model code

```


## Setup and workflow in general

Each run is based on a scenario YAML. 

Example:
```bash
pip install -e .
coral show       configs/scenarios/savannah_matthew_LR.yaml     
coral render-par configs/scenarios/savannah_matthew_LR.yaml runs/sav/savannah.par
coral build-bdy  configs/scenarios/savannah_matthew_LR.yaml <geoclaw_output> <bci_in> runs/sav/inputs
# full pipeline / sweeps:
snakemake -c1 all                # every scenario to be listed in configs/scenarios/
```

Each new scenario (different storm, Manning, SLR, drag, resolution) will require that you copy one YAML and change a field. 
The scenario set doubles as the ML emulator's training index. See`docs/LISFLOOD_build.md` to build the solver (it includes bug fixes and patches for compiling on the GT PACE cluster). 
I made use of a an existing `spack` environment on my PACE account with compatible dependencies.

Details on setting such an environment up can be found on this GT Github (you'll need an account), detailed for an ice sheet model: 
https://github.gatech.edu/smurugan9/software-build-doc/blob/main/MALI/build-notes.md

Will migrate this to its own env and docs once the workflow is setup, and will be stored here:
https://github.gatech.edu/smurugan9/build-docs

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
