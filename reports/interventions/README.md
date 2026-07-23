# Intervention siting: which GIS datasets, and how

Maps context GIS layers to the interventions in `src/coral/interventions/generate.py`
(`INTERVENTIONS` registry, `suitability_mask`). Today `suitability_mask` uses only
elevation (DEM vs. `sea_level`), proximity-to-water (binary dilation of the sea mask),
an optional NLCD grid, and an optional Pin Point `focus_region`. Everything below is
either already wired in, or a concrete next step to wire in.

## Figures

- **`intervention_dataset_matrix.png`** — matrix of intervention x dataset, marking
  each cell PRIMARY (would gate/replace the current siting logic) or context (useful
  as a secondary check / emulator feature, not a hard mask).
- **`pinpoint_siting_context.png`** — two-panel Pin Point map (DEM + NWI wetlands +
  SAGIS drainage basins, 2.5 km focus region) showing the current `suitability_mask`
  output for `marsh` (left, intertidal + near-water + focus, 22.6% of the focus area)
  and `depave` (right, elevation-fallback since no NLCD grid was loaded here, 52.6% of
  focus — overbroad, see note below) with SSURGO soil boundaries overlaid on the
  de-pave panel.

## Mapping table

| intervention | dataset(s) | how it informs siting |
|---|---|---|
| seawall / barrier | (DEM + drainage basins only) | shoreline-following now; basin outfall locations are a natural place to align a targeted levee segment (not yet used) |
| marsh restoration | **NWI wetlands** (primary), NLCD (primary), SSURGO soils, drainage basins | NWI `WETLAND_TYPE`/`ATTRIBUTE` (Cowardin code, e.g. `E2EM*` = estuarine emergent) gives a real "existing marsh" mask instead of relying on NLCD 90/95 alone — more accurate boundary + wetland subtype (emergent vs. forested/shrub) to pick `n_target`/`ksat_add`/`awc_add` magnitudes |
| mangrove | same as marsh | same NWI layer; in Georgia there's no natural mangrove so this is really "expand estuarine wetland" — same siting logic as marsh restoration |
| permeable surface / de-pave | **SSURGO soils** (primary), **SAGIS green infra** (primary where present), NLCD impervious classes | SSURGO `MUSYM`/`MUSYM_DESC` -> hydrologic soil group / infiltration rate should set the *achievable* `ksat_rate` per cell (sandy soils like Lakeland/Chipley/Ocilla sand support higher ksat than clayey Meggett/Cape Fear); SAGIS green infra polygons mark sites *already converted* — exclude from de-pave targeting, or treat as as-built calibration points for `ksat_add`/`awc_add` magnitudes |
| managed retreat | **Building footprints** (primary, not yet fetched) | retreat should target actual structures, not just "developed NLCD land" — footprint centroids/polygons -> real removal targets, with lot regrading to natural grade only under building footprints + a buffer |

## Dataset status at Pin Point clip (`data/raw/sagis_pinpoint/`, `data/raw/sagis_savannah/`)

- `sagis_wetlands_nwi.geojson`: 62 features clipped to bbox — `WETLAND_TYPE` has
  Estuarine/Marine (31), Freshwater Forested/Shrub (13), Riverine (9), Freshwater Pond
  (4), Estuarine/Marine Deepwater (4), Freshwater Emergent (1). Use directly.
- `sagis_soils_nrcs.geojson`: 77 features — `MUSYM_DESC` gives named soil series
  (Ogeechee loamy fine sand, Lakeland sand, Capers soils/tidal marsh, etc.). No
  hydrologic group or Ksat attribute in this SAGIS extract — need an SSURGO tabular
  join (via `MUKEY` -> `chorizon`/`comonth` Ksat, or NRCS Soil Data Access) for
  quantitative infiltration rates; the polygon boundaries alone are usable now for
  qualitative de-pave suitability (sandy vs. clayey series).
- `sagis_green_infra_chatham.geojson`: 29 features county-wide, but **empty at the
  tight Pin Point clip** (`data/raw/sagis_pinpoint` doesn't have this layer at all —
  it's in the `stormwater` group, fetched only for the wider Savannah/Chatham extent).
  Re-fetch with a Pin Point bbox if a local check is needed; for now treat Pin Point as
  "no existing GI" and rely on SSURGO + NLCD impervious for de-pave siting.
- `sagis_drainage_basin_sav.geojson`: 4 basins intersect the Pin Point clip — useful as
  a coarse catchment context (closer to the D8-catchment refinement noted in
  `MANNING_AND_INTERVENTIONS.md` than the current circular `focus_region`).
- `sagis_structures_chatham.geojson` (14,188 features): **not building footprints** —
  these are SAGIS drainage *structures* (manholes, catch basins: `STR_TYPE`,
  `INVERT_ELV`, etc.). Real building footprints (FEMA USA Structures / Microsoft /
  OSM) are still not fetched, matching the "not yet" note in
  `docs/MANNING_AND_INTERVENTIONS.md` for managed retreat and burn_buildings.py.

## Concrete wiring notes (for a follow-up, non-analysis PR)

- `suitability_mask(kind="marsh", ...)`: add an optional `wetlands` polygon/raster
  arg; `wet = wetlands_mask | np.isin(classes, NLCD_WETLAND)` — union NWI with NLCD
  rather than NLCD alone.
- `suitability_mask(kind="depave"/"permeable", ...)`: add an optional `soil_ksat`
  raster (rasterized SSURGO Ksat via MUKEY join) and use it to scale `knobs["ksat_rate"]`
  per cell (`ksat[m] = np.maximum(ksat[m], np.minimum(knobs["ksat_rate"], soil_ksat[m]))`)
  instead of a flat rate — respects soil-limited achievable infiltration.
- `suitability_mask(kind="retreat", ...)`: needs a `buildings` mask (rasterized
  footprints) intersected with `classes==NLCD_DEVELOPED`, once footprints are fetched.
- De-pave: subtract existing SAGIS green-infra polygons from the candidate zone
  (`z = z & ~green_infra_mask`) once refetched for the Pin Point bbox, so scenarios
  don't "convert" land that's already converted.
