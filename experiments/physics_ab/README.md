# Experiment (a): does NWI marsh-roughness improve the compound run?

A controlled A/B on the existing 30 m Matthew compound run: **baseline** (current NLCD
Manning) vs **nwi** (NWI-refined Manning), scored against USGS high-water marks. Tests the
headline calibration finding — NLCD undersells marsh roughness (median n 0.061→0.110 over
54.7% of Pin Point cells), concentrated in the coastal fringe the model actually covers.

**Manning-only by design.** Do NOT also swap POLARIS→SSURGO infiltration here: the
full-domain comparison showed Ksat reverses sign off Pin Point and storage agreement
doesn't generalize (`reports/physics/full_domain/README.md`). Infiltration is a separate,
region-specific experiment.

Harness: `coral.analysis.physics_ab` (`assemble` stages the two run dirs; `compare` scores
both `.mxe` against HWMs). Runs happen on HPC; keep run dirs on scratch (see
`docs/CONSOLIDATION_TODO.md` for the project-vs-scratch layout).

## Prerequisites
- A prepared 30 m compound **base run** dir (the current infilcap compound setup — DEM,
  `Manning_<name>.asc`, `.bci`/`.bdy`, rain, infil, `.par`). Reuse the existing compound
  run inputs (sibling `coastalFlood/savannah_matthew_workflow/…compound_infilcap…`).
- `nlcd_savannah.tif` (current NLCD source) and the domain-wide NWI wetlands
  (`data/raw/sagis_savannah/sagis_wetlands_nwi.geojson`, already fetched).
- The LISFLOOD-FP binary (HPC).

## 1. Stage baseline/ + nwi/
```bash
python - <<'PY'
from coral.analysis.physics_ab import assemble
assemble(
    base_run="runs/compound_base", out_root="runs/physics_ab",
    name="savannah_matthew_compound",
    nlcd="/path/to/nlcd_savannah.tif",
    wetlands="data/raw/sagis_savannah/sagis_wetlands_nwi.geojson",
    dem="runs/compound_base/SUB_DEM_savannah_matthew_compound.asc",
)
PY
```
`baseline/` is a verbatim copy; `nwi/` gets `Manning_*.asc` regenerated with the NWI
overlay. (Omit nlcd/wetlands/dem to stage only and print the make_manning command.)

## 2. Run both (HPC)
```bash
cd runs/physics_ab/baseline && <lisflood> savannah_matthew_compound.par
cd runs/physics_ab/nwi      && <lisflood> savannah_matthew_compound.par
```

## 3. Score against HWMs
```bash
python -m coral.analysis.physics_ab compare \
    runs/physics_ab/baseline/results_*/res_*.mxe \
    runs/physics_ab/nwi/results_*/res_*.mxe \
    --dem runs/physics_ab/baseline/SUB_DEM_savannah_matthew_compound.asc \
    --out-fig reports/physics/physics_ab_hwm.png
```
Reports bias + RMSE for each and the winner (current baseline: filtered RMSE ~0.18 m,
coastal bias −0.09 m — the number to beat). Also writes an obs-vs-modelled scatter.

## Reading the result
- **NWI wins (lower RMSE / smaller |bias|):** the marsh-roughness correction is real and
  worth adopting as the default Manning field. Fold `--nwi` into the standard pipeline.
- **No improvement / worse:** the HWMs don't constrain marsh roughness strongly (they tend
  to sit on structures, not in marsh), or the 30 m grid smears it. Revisit at 2 m, where
  marsh cells are resolved and the effect should be sharper.
