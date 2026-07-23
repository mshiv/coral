# SAGIS stormwater network — Pin Point, GA exploratory figures

Data: SAGIS clip to the Pin Point bbox (`/tmp/sagis_pinpoint/sagis_*.geojson`), EPSG:4326.
Script: see `sagis_explore.py` (run with the `coral` conda env). Empty layers in the clip
(green infrastructure, pump stations, reservoir areas) were skipped automatically.

## Figures

1. **`01_network_overview_map.png`** — Composite map of the full Pin Point drainage
   network: open channels (canals/ditches, solid lines) vs. subsurface pipes/conduits
   (dashed lines), all point assets (manholes, inlets, headwalls, structures, outfalls,
   tide gate), and the drainage-basin polygons, with the Pin Point reference point
   marked. The "what's there" overview.
2. **`02_compound_backup_pathway.png`** — Same network greyed out except the open
   channels and pipes/conduits (recolored), with the two outfalls and the single tide
   gate emphasized plus a ~150 m buffer around them — the segment of the network where
   tidal/surge backup into the drainage system is the dominant compound-failure risk.
3. **`03_channel_bed_invert_slope.png`** — Left: histogram of ditch invert elevations
   (`INV_ELV_1`/`INV_ELV_2`, ditches_chatham) plus a summary of `SLOPE_BOTTOM`
   (ditches_maint_sav). Right: per-segment invert drop (`INV_ELV_1 - INV_ELV_2`),
   sorted, as a proxy for local bed slope/flow direction — this is the attribute data
   that will parameterize LISFLOOD-FP sub-grid channel (SGC) bed geometry.
4. **`04_inventory_by_layer.png`** — Horizontal bar chart of feature counts by layer
   (Pin Point clip), colored consistently with figures 1–2, for a quick sense of
   relative data density (e.g., conduit_pipes_sav and inlets_sav dominate by count).

## Layer profile summary (Pin Point clip)

| Layer | n | geom | useful attrs |
|---|---|---|---|
| canals_chatham | 8 | Line | none obvious |
| ditches_chatham | 26 | Line | INV_ELV_1, INV_ELV_2, ELV_U_D |
| ditches_maint_sav | 26 | (Multi)Line | BED_MAT_D, SLOPE_BOTTOM, SLOPE_U_D |
| pipes_chatham | 103 | Line | INV_ELV_1, INV_ELV_2, ELV_U_D, SLOPE_BOT |
| conduit_pipes_sav | 447 | Line | MATERIAL, DIAMETER, PIPE_WIDTH, INVERT_IN/OUT, SLOPE_BOTTOM |
| structures_chatham | 138 | Point | INVERT_ELV, THROAT_ELEV, WEIR_ELV, BOTTOM_ELEV, TOP_ELEV |
| manholes_sav | 122 | Point | MATERIAL, MH_DEPTH, AB_TOP_ELEV, INVERT_OUT_ELEV + directional inverts (N/E/S/W) |
| inlets_sav | 276 | Point | MATERIAL, AB_TOP_ELEV, AB_THROAT_ELEV, INVERT_ELEV_N/E/S/W, MEASURED_DEPTH |
| headwalls_sav | 18 | Point | top_elv, invert_elv, INVENTORY |
| outfalls_chatham | 2 | Point | Elevation |
| tide_gates_sav | 1 | Point | MATERIAL, WIDTH, ELEVATION, AB_TOP_ELV |
| drainage_basin_sav | 3 | Polygon | HIGHEST_ELEV, LOWEST_ELEV |
| green_infra_chatham, pump_stations_sav, reservoir_area_chatham | 0 | — | empty in this clip |

Notable: only **2 outfalls** and **1 tide gate** mediate the entire Pin Point network's
connection to tidewater — a very small, high-leverage set of features for the compound
backup mechanism. `conduit_pipes_sav` and `inlets_sav` are the largest layers by far,
suggesting the subsurface network is much denser than the open-channel network in this
clip (26 ditch + 8 canal segments vs. 447 conduit + 103 pipe segments).

## Ideas for additional visualizations (not yet built)

1. **DEM/bathymetry + flood-extent overlay** — drape the network (especially outfalls,
   tide gate, low-invert ditch segments) over the LISFLOOD-FP DEM and a modeled/observed
   flood extent to see which drainage assets sit below expected surge/tide stage —
   directly motivates which SGC channels need explicit tidal boundary conditions.
2. **Network connectivity graph** — build a directed graph (pipes/ditches as edges,
   manholes/inlets/structures as nodes) using endpoint snapping, to identify the
   drainage tree(s) feeding each of the 2 outfalls and the tide gate. This clarifies
   what's hydraulically "upstream" of each compound-risk point and is a prerequisite
   for any 1-D SWMM-style routing model.
3. **Invert elevation vs. local ground/DEM elevation ("freeboard") map** — for
   manholes/inlets/structures, subtract invert or rim elevation from the DEM to map
   where the pipe network is already close to grade (flood-prone / surcharge-prone
   locations) — a proxy for backup susceptibility without running the model.
4. **Inlet/manhole density heatmap** — kernel density or hex-bin of inlets_sav +
   manholes_sav to show which sub-areas are most intensively drained, useful for
   sanity-checking runoff source-term concentration vs. the drainage_basin polygons.
5. **Along-channel long profile plots** — for each ditch/canal, order vertices along
   the flow path and plot invert elevation vs. distance (a true longitudinal profile,
   not just a segment-level scatter) to directly visualize SGC bed slope discontinuities
   or pinch points.
6. **Contributing-area vs. peak-stage/flood-depth scatter** — once model runs exist,
   join drainage_basin_sav polygon areas (and/or land-use) against simulated peak
   depth near each basin's outfall to see whether runoff volume or tidal backup
   dominates compound flood response by basin.
7. **Pipe material/diameter map** — color-code conduit_pipes_sav/pipes_chatham by
   MATERIAL or DIAMETER/PIPE_WIDTH to flag capacity bottlenecks (e.g., narrow pipes
   feeding into the tide-gate-controlled outlet) before building the SWMM proxy.
