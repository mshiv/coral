# Drainage sensitivity test: Tier 0 vs Tier 1, whether drainage justifies SWMM

Run the Pin Point compound scenario twice, once with no storm drainage (Tier 0) and once
with the native drainage representation (Tier 1), and measure how much the flood changes.
The change is the decision: if it's large, build the coupled SWMM model (Tier 2); if it's
small, SWMM is unjustified scope. Background and the three-tier plan are in the wiki,
`syntheses/Drainage backup and SWMM coupling`.

- Tier 0: baseline. SGC channels off, no drain point sources. (The stock `.par` already
  is Tier 0: `sgc_enable` is on but with no `SGCwidth` raster it does nothing.)
- Tier 1: open channels (ditches/canals) as sub-grid channels, plus tide-conditioned inlet
  sinks. Native LISFLOOD-FP only, no second model.

Harness: `coral.analysis.drainage_sensitivity` (`assemble` stages both run dirs; `compare`
scores them + issues the verdict). The comparison is standalone-runnable on any two `.max`
rasters; `assemble` wires `preprocess/burn_drainage.py` + `preprocess/make_drainage_proxy.py`.

## Prerequisites (the same inputs the 2 m Pin Point run needs)

1. CoNED 2 m DEM at `data/raw/coned_pinpoint_2m.asc`. See `docs/highres_pinpoint.md`
   step 1 (still the open blocker for all Pin Point building-scale work).
2. SAGIS vectors, already fetched at `data/raw/sagis_savannah/` (or re-clip to the Pin
   Point bbox with `python -m coral.preprocess.fetch_sagis --group stormwater --bbox
   -81.1557 -81.1181 31.9278 31.9601 --out data/raw/sagis_pinpoint`).
3. A staged base run dir, the normal pipeline output, containing
   `SUB_DEM_pinpoint_highres.asc`, `Manning_pinpoint_highres.asc`,
   `pinpoint_highres_coastline.bci`, `pinpoint_highres.bdy`, `pinpoint_highres.par`, and the
   rain/infil files. Build it the usual way:
   ```bash
   S=configs/scenarios/pinpoint_highres.yaml
   coral render-par $S runs/pp/pinpoint_highres.par
   python -m coral.couple.nest_bdy --coarse <coarse_compound_run> \
       --bbox -81.1557 -81.1181 31.9278 31.9601 \
       --out-bci runs/pp/pinpoint_highres_coastline.bci --out-bdy runs/pp/pinpoint_highres.bdy
   python -m coral.preprocess.make_manning --dem data/raw/coned_pinpoint_2m.asc --nlcd nlcd.tif \
       --out runs/pp/Manning_pinpoint_highres.asc
   python -m coral.preprocess.make_infil --config $S --dem data/raw/coned_pinpoint_2m.asc
   # (stage DEM as SUB_DEM_pinpoint_highres.asc + rain into runs/pp/ per the .par names)
   ```
4. Stage series for the proxy, the surge+tide WSE at the outfall, i.e. a
   `seconds level_m` file. The coupling `.bdy` already holds this; extract one representative
   boundary block near the Pin Point outfalls, or reuse the `make_tide` series.
5. LISFLOOD-FP binary (on HPC).

## 1. Assemble the two variants

```bash
python - <<'PY'
from coral.analysis.drainage_sensitivity import assemble
assemble(
    base_run="runs/pp", out_root="runs/pp_sens", name="pinpoint_highres",
    channels="data/raw/sagis_pinpoint/sagis_ditches_chatham.geojson,"
             "data/raw/sagis_pinpoint/sagis_ditches_maint_sav.geojson,"
             "data/raw/sagis_pinpoint/sagis_canals_chatham.geojson",
    inlets="data/raw/sagis_pinpoint/sagis_inlets_sav.geojson",
    stage_series="runs/pp/outfall_stage.txt",
    capacity=0.05,          # per-inlet drain rate (m^3/s) - see calibration note
    outfall_invert=0.5,     # stage (m, boundary datum) above which the outfall drowns
    backup_frac=0.2,        # 0 = just stop draining; >0 = surcharge back out
)
PY
```

This writes `runs/pp_sens/tier0/` and `runs/pp_sens/tier1/` and prints the two run commands.

> Widths/inverts: the public SAGIS ditches carry no width (a per-type default is used) but do
> carry `INV_ELV_1/2`. `burn_drainage` uses their difference as a relative bed slope
> (the values are feet, `FT_DATUM_UNKNOWN`, so not usable as absolute elevations). Canals
> want `--kind canal` (wider default) if burned separately.

## 2. Run both (HPC)

```bash
cd runs/pp_sens/tier0 && <lisflood> pinpoint_highres.par
cd runs/pp_sens/tier1 && <lisflood> pinpoint_highres.par
```

## 3. Compare + verdict

```bash
python -m coral.analysis.drainage_sensitivity \
    runs/pp_sens/tier0/results_pinpoint_highres/res_pinpoint_highres.max \
    runs/pp_sens/tier1/results_pinpoint_highres/res_pinpoint_highres.max \
    --decision-pct 10 --out-fig reports/sagis/drainage_sensitivity.png
```

Prints wet-area and flood-volume percent change and a 3-panel figure (Tier 0, Tier 1, signed
diff), then the verdict:

- `SWMM NOT justified` (|Δ| < threshold): drainage is second-order here, ship Tier 1, close
  the SWMM question.
- `SWMM JUSTIFIED` (|Δ| >= threshold): build Tier 2. The SAGIS `conduit_pipes` layer already
  carries `DIAMETER`/`INVERT_IN/OUT`/`MATERIAL` (and inlets `MEASURED_DEPTH`), so the SWMM
  network can be built from data in hand, with no further data acquisition needed.

## Robustness of the verdict

The verdict depends on three proxy knobs. If the result sits near the threshold, sweep them
before trusting it.

- `capacity`, the per-inlet drain rate. Too high drains everything (Tier 1 always much less
  than Tier 0, looking important for the wrong reason); too low does nothing. Ballpark it from
  a design storm divided by inlet count, or an assumed pipe full-flow.
- `outfall_invert`, the drown threshold. Lower means the network shuts off earlier and longer
  under tide, amplifying backup. Anchor it to the real outfall inverts and boundary datum.
- `backup_frac`: 0 tests loss of drainage only, above 0 tests active surcharge. Run both
  ends. If even `backup_frac=0` moves the flood a lot, the case for SWMM is robust.

A defensible protocol is to run the corners of {capacity, outfall_invert, backup_frac} and
report the range of Δ, not a single number.
