# CORAL Emulator Project

*COastal Response emulator & Adaptation Learning* | *Coastal Options for Resilience & Adaptation Learning* (name tbd)

This project develops a machine-learning emulator for coupled coastal flooding simulations over coastal Georgia. The goal is to approximate expensive physics-based GeoClaw-LISFLOOD-FP outputs with a fast surrogate model that can explore many storm, sea-level-rise, and (bathymetric) intervention scenarios.

## Project Goal

Predict flood depth fields and/or summary inundation metrics from:
- tropical cyclone parameters
- sea-level-rise scenarios
- topographic intervention parameters
- static landscape features

The emulator will be trained on a synthetic dataset generated from physics-based simulations and evaluated on held-out scenarios and historical events with observational constraints.

## Repository Layout

- `docs/` - project brief, experiment plan, presentation notes, and figures
- `notebooks/` - workflow notebooks for data audit, physics dataset building, emulator baselines, and evaluation
- `src/coral_emulator/` - code for data handling, physics runs, model training, evaluation, and plotting
- `scripts/` - command-line entry points for batch runs and reproducible workflows
- `data/` - raw inputs, intermediate scenario tables, processed training data, and split definitions
- `results/` - metrics, figures, model artifacts, and logs
- `tests/` - smoke tests for scenario specs, dataset building, and evaluation code

## Milestones

1. Define the scenario table and emulator inputs/outputs.
2. Build one working GeoClaw-LISFLOOD-FP baseline run.
3. Convert physics outputs into a training dataset.
4. Train a baseline emulator.
5. Evaluate on held-out scenarios and measure speedup.

## Notes

Older SSLS/ERA5 surge/flood and coursework notebooks have been moved to `archive/` for provenance; the emulator project is the current focus. Those notebooks are useful references.

## Contact

Shiva Muruganandham - smurugan9@gatech.edu
