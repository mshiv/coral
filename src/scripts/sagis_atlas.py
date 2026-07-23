"""
SAGIS stormwater/soils/wetlands atlas for the CORAL compound-flood project.

Read-only figure generation script. Does NOT modify any source data or
existing figures. Writes PNGs + README to reports/sagis/atlas/.

Run with:
    /Users/smurugan9/miniforge3/envs/coral/bin/python src/scripts/sagis_atlas.py
(from /Users/smurugan9/research/coral)
"""
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

ROOT = "/Users/smurugan9/research/coral"
RAW_COUNTY = os.path.join(ROOT, "data/raw/sagis_savannah")
RAW_PP = os.path.join(ROOT, "data/raw/sagis_pinpoint")
OUT = os.path.join(ROOT, "reports/sagis/atlas")
os.makedirs(OUT, exist_ok=True)

METRIC_CRS = "EPSG:32617"
PP_BBOX = (-81.1557, -81.1181, 31.9278, 31.9601)  # minx, maxx, miny, maxy

captions = []  # (filename, caption) collected as we go


def log(msg):
    print(f"[atlas] {msg}")


def load(path, layer_name):
    if not os.path.exists(path):
        log(f"MISSING layer, skipping: {layer_name} ({path})")
        return None
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        log(f"ERROR reading {layer_name}: {e}")
        return None
    if gdf.empty:
        log(f"EMPTY layer, skipping: {layer_name}")
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def to_metric(gdf):
    if gdf is None:
        return None
    return gdf.to_crs(METRIC_CRS)


def clip_bbox(gdf, bbox=PP_BBOX):
    """Clip a county-wide (EPSG:4326) gdf to the Pin Point bbox."""
    if gdf is None:
        return None
    minx, maxx, miny, maxy = bbox
    try:
        clipped = gdf.cx[minx:maxx, miny:maxy]
    except Exception as e:
        log(f"clip failed: {e}")
        return None
    if clipped.empty:
        return None
    return clipped


# ---------------------------------------------------------------------------
# Load county-wide layers
# ---------------------------------------------------------------------------
county = {}
county_files = {
    "pipes_chatham": "sagis_pipes_chatham.geojson",
    "conduit_pipes_sav": "sagis_conduit_pipes_sav.geojson",
    "ditches_chatham": "sagis_ditches_chatham.geojson",
    "ditches_maint_sav": "sagis_ditches_maint_sav.geojson",
    "canals_chatham": "sagis_canals_chatham.geojson",
    "manholes_sav": "sagis_manholes_sav.geojson",
    "inlets_sav": "sagis_inlets_sav.geojson",
    "headwalls_sav": "sagis_headwalls_sav.geojson",
    "outfalls_chatham": "sagis_outfalls_chatham.geojson",
    "tide_gates_sav": "sagis_tide_gates_sav.geojson",
    "structures_chatham": "sagis_structures_chatham.geojson",
    "reservoir_area_chatham": "sagis_reservoir_area_chatham.geojson",
    "drainage_basin_sav": "sagis_drainage_basin_sav.geojson",
    "green_infra_chatham": "sagis_green_infra_chatham.geojson",
    "pump_stations_sav": "sagis_pump_stations_sav.geojson",
}
for name, fname in county_files.items():
    county[name] = load(os.path.join(RAW_COUNTY, fname), name)

pp = {}
pp_files = {
    "soils_nrcs": "sagis_soils_nrcs.geojson",
    "wetlands_nwi": "sagis_wetlands_nwi.geojson",
    "buildings_chatham": "sagis_buildings_chatham.geojson",
}
for name, fname in pp_files.items():
    pp[name] = load(os.path.join(RAW_PP, fname), name)

# Re-clip stormwater layers to Pin Point bbox from county files
pp_stormwater = {}
for name in ["pipes_chatham", "conduit_pipes_sav", "ditches_chatham",
             "ditches_maint_sav", "canals_chatham", "manholes_sav",
             "inlets_sav", "headwalls_sav", "outfalls_chatham",
             "tide_gates_sav", "structures_chatham"]:
    pp_stormwater[name] = clip_bbox(county.get(name))
    n = 0 if pp_stormwater[name] is None else len(pp_stormwater[name])
    log(f"Pin Point clip {name}: {n} features")


def savefig(fig, fname, caption):
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    captions.append((fname, caption))
    log(f"wrote {fname}")


# ---------------------------------------------------------------------------
# FIGURE 1: Storm-drain network map (Pin Point) with county inset
# ---------------------------------------------------------------------------
def fig1_network_map():
    conduits = to_metric(pp_stormwater.get("conduit_pipes_sav"))
    pipes = to_metric(pp_stormwater.get("pipes_chatham"))
    inlets = to_metric(pp_stormwater.get("inlets_sav"))
    manholes = to_metric(pp_stormwater.get("manholes_sav"))
    outfalls = to_metric(pp_stormwater.get("outfalls_chatham"))
    tide_gates = to_metric(pp_stormwater.get("tide_gates_sav"))
    headwalls = to_metric(pp_stormwater.get("headwalls_sav"))

    if conduits is None and pipes is None:
        log("No pipe data for Pin Point network map, skipping fig1")
        return

    fig, ax = plt.subplots(figsize=(10, 9))

    # combine pipe-like lines, color by diameter (graduated)
    def diam_col(gdf, col_candidates):
        for c in col_candidates:
            if c in gdf.columns:
                return pd.to_numeric(gdf[c], errors="coerce")
        return None

    all_lines = []
    if conduits is not None:
        d = diam_col(conduits, ["DIAMETER"])
        conduits = conduits.assign(_diam=d)
        all_lines.append(conduits[["geometry", "_diam"]])
    if pipes is not None:
        d = diam_col(pipes, ["PIPE_HT", "PIPE_WT"])
        pipes = pipes.assign(_diam=d)
        all_lines.append(pipes[["geometry", "_diam"]])

    lines = gpd.GeoDataFrame(pd.concat(all_lines, ignore_index=True), crs=METRIC_CRS)
    lines = lines[lines.geometry.notna()]
    vmax = lines["_diam"].quantile(0.98) if lines["_diam"].notna().any() else None
    lines.plot(ax=ax, column="_diam", cmap="viridis", linewidth=1.3,
               legend=True, missing_kwds={"color": "lightgray", "linewidth": 0.6},
               legend_kwds={"label": "Pipe diameter (in)", "shrink": 0.6},
               vmax=vmax)

    if manholes is not None:
        manholes.plot(ax=ax, color="gray", markersize=6, marker="o",
                      label="Manholes", zorder=3, alpha=0.7)
    if inlets is not None:
        inlets.plot(ax=ax, color="steelblue", markersize=6, marker="^",
                    label="Inlets", zorder=3, alpha=0.7)
    if headwalls is not None:
        headwalls.plot(ax=ax, color="saddlebrown", markersize=25, marker="s",
                       label="Headwalls", zorder=4)
    if outfalls is not None:
        outfalls.plot(ax=ax, color="red", markersize=60, marker="*",
                     label="Outfalls (tidewater)", zorder=5)
    if tide_gates is not None:
        tide_gates.plot(ax=ax, color="black", markersize=90, marker="X",
                        label="Tide gates", zorder=6)

    ax.set_title("Storm-drain network, Pin Point, Chatham County GA\n"
                  "(pipe color = diameter; nodes = inlets/manholes; "
                  "tidewater edge = outfalls & tide gates)", fontsize=12)
    ax.set_xlabel("Easting (m, UTM 17N)")
    ax.set_ylabel("Northing (m, UTM 17N)")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="Manholes"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="steelblue", markersize=6, label="Inlets"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="saddlebrown", markersize=8, label="Headwalls"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=12, label="Outfalls"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="black", markersize=10, label="Tide gates"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")

    # County-wide inset for context
    conduits_c = to_metric(county.get("conduit_pipes_sav"))
    pipes_c = to_metric(county.get("pipes_chatham"))
    axins = fig.add_axes([0.68, 0.68, 0.28, 0.28])
    if conduits_c is not None:
        conduits_c.plot(ax=axins, color="lightsteelblue", linewidth=0.3)
    if pipes_c is not None:
        pipes_c.plot(ax=axins, color="lightsteelblue", linewidth=0.3)
    # highlight Pin Point extent box
    if conduits is not None or pipes is not None:
        bnd = lines.total_bounds
        from matplotlib.patches import Rectangle
        axins.add_patch(Rectangle((bnd[0], bnd[1]), bnd[2]-bnd[0], bnd[3]-bnd[1],
                                    fill=False, edgecolor="red", linewidth=1.5))
    axins.set_title("Chatham Co. context", fontsize=7)
    axins.set_xticks([]); axins.set_yticks([])

    savefig(fig, "01_network_map_pinpoint.png",
            "Pin Point storm-drain network: pipes/conduits colored by diameter, "
            "inlets/manholes as nodes, outfalls and tide gates emphasized at the "
            "tidewater edge; inset shows county-wide context.")


# ---------------------------------------------------------------------------
# FIGURE 2: Pipe diameter distribution + material breakdown (county-wide)
# ---------------------------------------------------------------------------
def fig2_diameter_material():
    conduits = county.get("conduit_pipes_sav")
    pipes = county.get("pipes_chatham")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Conduit diameter histogram
    ax = axes[0, 0]
    if conduits is not None and "DIAMETER" in conduits.columns:
        d = pd.to_numeric(conduits["DIAMETER"], errors="coerce").dropna()
        d = d[(d > 0) & (d < 200)]
        ax.hist(d, bins=40, color="steelblue", edgecolor="white")
        ax.set_title(f"Conduit pipe diameter (n={len(d)})")
        ax.set_xlabel("Diameter (in)")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "no DIAMETER data", ha="center")

    # Conduit material bar
    ax = axes[0, 1]
    if conduits is not None and "MATERIAL" in conduits.columns:
        vc = conduits["MATERIAL"].fillna("UNKNOWN").astype(str).value_counts().head(12)
        ax.barh(vc.index[::-1], vc.values[::-1], color="darkorange")
        ax.set_title("Conduit pipe material (top 12)")
        ax.set_xlabel("Count")
    else:
        ax.text(0.5, 0.5, "no MATERIAL data", ha="center")

    # Pipes_chatham height/width as diameter proxy
    ax = axes[1, 0]
    if pipes is not None and "PIPE_HT" in pipes.columns:
        d = pd.to_numeric(pipes["PIPE_HT"], errors="coerce").dropna()
        d = d[(d > 0) & (d < 200)]
        ax.hist(d, bins=40, color="seagreen", edgecolor="white")
        ax.set_title(f"pipes_chatham PIPE_HT (n={len(d)})")
        ax.set_xlabel("Height (in)")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "no PIPE_HT data", ha="center")

    # Pipes_chatham material bar
    ax = axes[1, 1]
    if pipes is not None and "PIPE_MAT" in pipes.columns:
        vc = pipes["PIPE_MAT"].fillna("UNKNOWN").astype(str).value_counts().head(12)
        ax.barh(vc.index[::-1], vc.values[::-1], color="mediumpurple")
        ax.set_title("pipes_chatham material (top 12)")
        ax.set_xlabel("Count")
    else:
        ax.text(0.5, 0.5, "no PIPE_MAT data", ha="center")

    fig.suptitle("Storm-sewer pipe inventory: diameter distributions & material breakdown "
                 "(county-wide)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "02_pipe_diameter_material_inventory.png",
            "County-wide pipe-diameter histograms and material-type bar charts for "
            "conduit_pipes_sav and pipes_chatham -- the standard network-inventory summary.")


# ---------------------------------------------------------------------------
# FIGURE 3: Open-channel + drainage-basin map (catchment plan)
# ---------------------------------------------------------------------------
def fig3_channels_basins():
    basins = to_metric(county.get("drainage_basin_sav"))
    canals = to_metric(county.get("canals_chatham"))
    ditches = to_metric(county.get("ditches_chatham"))
    ditches_maint = to_metric(county.get("ditches_maint_sav"))

    if basins is None:
        log("No drainage_basin data, skipping fig3")
        return

    fig, ax = plt.subplots(figsize=(11, 10))
    elev_col = None
    for c in ["HIGHEST_ELEV"]:
        if c in basins.columns:
            elev_col = c
    if elev_col:
        vals = pd.to_numeric(basins[elev_col], errors="coerce")
        basins = basins.assign(_elev=vals)
        basins.plot(ax=ax, column="_elev", cmap="terrain", edgecolor="black",
                    linewidth=0.6, alpha=0.6, legend=True,
                    legend_kwds={"label": "Highest elevation (ft)", "shrink": 0.6})
    else:
        basins.plot(ax=ax, color="wheat", edgecolor="black", linewidth=0.6, alpha=0.6)

    if canals is not None:
        canals.plot(ax=ax, color="navy", linewidth=1.2, label="Canals")
    if ditches is not None:
        ditches.plot(ax=ax, color="teal", linewidth=0.5, alpha=0.8, label="Ditches (Chatham)")
    if ditches_maint is not None:
        ditches_maint.plot(ax=ax, color="crimson", linewidth=0.6, alpha=0.6,
                           linestyle="--", label="Maintained ditches")

    ax.set_title("Open-channel drainage network and drainage basins (catchments)\n"
                 "Chatham County, GA", fontsize=12)
    ax.set_xlabel("Easting (m, UTM 17N)")
    ax.set_ylabel("Northing (m, UTM 17N)")
    handles = [
        Line2D([0], [0], color="navy", lw=1.5, label="Canals"),
        Line2D([0], [0], color="teal", lw=1.2, label="Ditches (Chatham)"),
        Line2D([0], [0], color="crimson", lw=1.2, ls="--", label="Maintained ditches"),
        Patch(facecolor="wheat", edgecolor="black", label="Drainage basins"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    savefig(fig, "03_channels_and_drainage_basins.png",
            "Open-channel network (canals + ditches) draped over drainage-basin "
            "catchments shaded by highest elevation -- the standard catchment/drainage plan.")


# ---------------------------------------------------------------------------
# FIGURE 4a: Wetlands NWI map by Cowardin class
# ---------------------------------------------------------------------------
def fig4a_wetlands():
    wet = to_metric(pp.get("wetlands_nwi"))
    if wet is None:
        log("No wetlands data, skipping fig4a")
        return
    fig, ax = plt.subplots(figsize=(9, 9))
    col = "WETLAND_TYPE" if "WETLAND_TYPE" in wet.columns else "ATTRIBUTE"
    cats = wet[col].fillna("Unknown").astype(str)
    wet = wet.assign(_cat=cats)
    n_cat = wet["_cat"].nunique()
    cmap = "tab20" if n_cat > 10 else "tab10"
    wet.plot(ax=ax, column="_cat", cmap=cmap, legend=True, edgecolor="black",
             linewidth=0.3, categorical=True,
             legend_kwds={"loc": "upper left", "bbox_to_anchor": (1.02, 1),
                          "fontsize": 7, "title": col})
    bldgs = to_metric(pp.get("buildings_chatham"))
    if bldgs is not None:
        bldgs.plot(ax=ax, color="gray", edgecolor="black", linewidth=0.3, alpha=0.5)
    ax.set_title(f"National Wetlands Inventory (NWI), Pin Point\ncolored by {col} (Cowardin class)",
                fontsize=11)
    ax.set_xlabel("Easting (m, UTM 17N)"); ax.set_ylabel("Northing (m, UTM 17N)")
    ax.set_aspect("equal")
    savefig(fig, "04a_wetlands_nwi_cowardin.png",
            "NWI wetlands at Pin Point colored by Cowardin wetland-type classification "
            "(standard land-cover survey style); building footprints shown in gray.")


# ---------------------------------------------------------------------------
# FIGURE 4b: Soils SSURGO map by MUSYM
# ---------------------------------------------------------------------------
def fig4b_soils():
    soils = to_metric(pp.get("soils_nrcs"))
    if soils is None:
        log("No soils data, skipping fig4b")
        return
    fig, ax = plt.subplots(figsize=(9, 9))
    col = "MUSYM" if "MUSYM" in soils.columns else "MUKEY"
    cats = soils[col].fillna("Unknown").astype(str)
    soils = soils.assign(_cat=cats)
    n_cat = soils["_cat"].nunique()
    cmap = "tab20" if n_cat <= 20 else "nipy_spectral"
    soils.plot(ax=ax, column="_cat", cmap=cmap, legend=True, edgecolor="black",
               linewidth=0.3, categorical=True,
               legend_kwds={"loc": "upper left", "bbox_to_anchor": (1.02, 1),
                            "fontsize": 6, "title": col, "ncol": 1})
    ax.set_title(f"SSURGO soil map units, Pin Point\ncolored by {col} ({n_cat} map units)",
                fontsize=11)
    ax.set_xlabel("Easting (m, UTM 17N)"); ax.set_ylabel("Northing (m, UTM 17N)")
    ax.set_aspect("equal")
    savefig(fig, "04b_soils_ssurgo_musym.png",
            "SSURGO soil-survey map units at Pin Point colored categorically by MUSYM "
            "(standard soil-survey cartography style).")


# ---------------------------------------------------------------------------
# FIGURE 5: Inlet/manhole density heatmap
# ---------------------------------------------------------------------------
def fig5_density():
    inlets = to_metric(county.get("inlets_sav"))
    manholes = to_metric(county.get("manholes_sav"))
    parts = [g for g in [inlets, manholes] if g is not None]
    if not parts:
        log("No inlet/manhole data, skipping fig5")
        return
    pts = gpd.GeoDataFrame(pd.concat([p[["geometry"]] for p in parts], ignore_index=True), crs=METRIC_CRS)
    x = pts.geometry.x.values
    y = pts.geometry.y.values

    fig, ax = plt.subplots(figsize=(10, 9))
    cell = 300.0  # meters
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xbins = np.arange(xmin, xmax + cell, cell)
    ybins = np.arange(ymin, ymax + cell, cell)
    H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
    density = H.T / (cell * cell) * 1e6  # per km^2
    density_masked = np.ma.masked_where(density == 0, density)
    im = ax.pcolormesh(xedges, yedges, density_masked, cmap="inferno", shading="auto")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Inlets+manholes per km$^2$")
    ax.set_title("Drainage-infrastructure density: inlets + manholes\n"
                 f"({cell:.0f} m grid, Chatham County)", fontsize=12)
    ax.set_xlabel("Easting (m, UTM 17N)"); ax.set_ylabel("Northing (m, UTM 17N)")
    ax.set_aspect("equal")
    savefig(fig, "05_inlet_manhole_density_heatmap.png",
            "Point density of storm-drain inlets and manholes per km^2 on a 300 m grid, "
            "highlighting where drainage infrastructure concentrates (urbanized areas).")


# ---------------------------------------------------------------------------
# FIGURE 6a: Along-network invert long-profile (conduit pipes)
# ---------------------------------------------------------------------------
def fig6a_long_profile():
    conduits = county.get("conduit_pipes_sav")
    if conduits is None or "INVERT_IN" not in conduits.columns:
        log("No invert data, skipping fig6a")
        return
    inv_in = pd.to_numeric(conduits["INVERT_IN"], errors="coerce")
    inv_out = pd.to_numeric(conduits["INVERT_OUT"], errors="coerce")
    length = pd.to_numeric(conduits.get("PIPE_LENGTH"), errors="coerce")
    df = pd.DataFrame({"inv_in": inv_in, "inv_out": inv_out, "length": length})
    df = df[(df.inv_in.between(-50, 200)) & (df.inv_out.between(-50, 200)) & (df.length > 0)]
    df = df.sort_values("inv_out", ascending=False).reset_index(drop=True)
    # sample if huge
    if len(df) > 3000:
        df = df.sample(3000, random_state=0).sort_values("inv_out", ascending=False).reset_index(drop=True)
    df["cum_len"] = df["length"].cumsum()

    fig, ax = plt.subplots(figsize=(11, 6))
    for _, row in df.iterrows():
        x0 = row["cum_len"] - row["length"]
        x1 = row["cum_len"]
        ax.plot([x0, x1], [row["inv_in"], row["inv_out"]], color="steelblue", linewidth=0.4, alpha=0.5)
    ax.set_title("Conduit-pipe invert long-profile (pseudo-network, sorted by outlet invert)\n"
                 "each segment: upstream invert-in to downstream invert-out", fontsize=11)
    ax.set_xlabel("Cumulative pipe length (ft, arbitrary ordering)")
    ax.set_ylabel("Invert elevation (ft)")
    savefig(fig, "06a_invert_long_profile.png",
            "Pseudo long-profile of conduit-pipe invert elevations (invert-in to invert-out per "
            "segment) sorted by outlet elevation, showing the overall hydraulic-grade envelope "
            "of the network rather than a single traced path.")


# ---------------------------------------------------------------------------
# FIGURE 6b: Pipe slope distribution
# ---------------------------------------------------------------------------
def fig6b_slope():
    conduits = county.get("conduit_pipes_sav")
    ditches = county.get("ditches_chatham")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if conduits is not None and "SLOPE_BOTTOM" in conduits.columns:
        s = pd.to_numeric(conduits["SLOPE_BOTTOM"], errors="coerce").dropna()
        s = s[s.between(-0.2, 0.2)]
        axes[0].hist(s, bins=60, color="teal", edgecolor="white")
        axes[0].axvline(0, color="black", lw=0.8)
        axes[0].set_title(f"Conduit pipe slope (n={len(s)})")
        axes[0].set_xlabel("Slope (ft/ft)")
    else:
        axes[0].text(0.5, 0.5, "no slope data", ha="center")

    if ditches is not None and "INV_ELV_1" in ditches.columns and "INV_ELV_2" in ditches.columns:
        e1 = pd.to_numeric(ditches["INV_ELV_1"], errors="coerce")
        e2 = pd.to_numeric(ditches["INV_ELV_2"], errors="coerce")
        valid = (e1 != 999) & (e2 != 999) & e1.notna() & e2.notna()
        drop = (e1 - e2).where(valid)
        drop = drop[drop.abs() < 30]
        axes[1].hist(drop.dropna(), bins=60, color="darkgoldenrod", edgecolor="white")
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].set_title(f"Ditch invert drop INV_ELV_1 - INV_ELV_2 (n={drop.dropna().shape[0]})")
        axes[1].set_xlabel("Elevation drop (ft, datum unknown, 999=missing excluded)")
    else:
        axes[1].text(0.5, 0.5, "no ditch invert data", ha="center")

    fig.suptitle("Hydraulic gradient inventory: conduit slope & ditch invert drop", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "06b_slope_distribution.png",
            "Distribution of conduit-pipe bottom slope and ditch invert-elevation drop -- "
            "a quick check on hydraulic gradients and possible flat/adverse-slope segments.")


# ---------------------------------------------------------------------------
# FIGURE 6c: Basin contributing area vs lowest elevation
# ---------------------------------------------------------------------------
def fig6c_area_vs_elev():
    basins = county.get("drainage_basin_sav")
    if basins is None:
        log("No basin data, skipping fig6c")
        return
    basins_m = to_metric(basins)
    area_km2 = basins_m.geometry.area / 1e6
    lowest = pd.to_numeric(basins.get("LOWEST_ELEV"), errors="coerce")
    highest = pd.to_numeric(basins.get("HIGHEST_ELEV"), errors="coerce")
    df = pd.DataFrame({"area_km2": area_km2.values, "lowest": lowest.values,
                       "highest": highest.values,
                       "basin_id": basins.get("BASIN_ID", pd.Series(range(len(basins)))).values})
    df = df.dropna(subset=["area_km2", "lowest"])
    if df.empty:
        log("No usable elevation data for basins, skipping fig6c")
        return
    fig, ax = plt.subplots(figsize=(8, 6.5))
    relief = (df["highest"] - df["lowest"]) if df["highest"].notna().any() else None
    sc = ax.scatter(df["area_km2"], df["lowest"],
                    c=relief if relief is not None else "steelblue",
                    cmap="viridis", s=60, edgecolor="black")
    if relief is not None:
        fig.colorbar(sc, ax=ax, label="Basin relief (ft, highest-lowest)")
    for _, r in df.iterrows():
        ax.annotate(str(r["basin_id"]), (r["area_km2"], r["lowest"]), fontsize=6, alpha=0.7)
    ax.set_xlabel("Basin contributing area (km$^2$)")
    ax.set_ylabel("Basin lowest elevation (ft)")
    ax.set_title("Drainage basins: contributing area vs. lowest elevation\n"
                "(color = basin relief) -- proxy for compound-flood susceptibility", fontsize=11)
    savefig(fig, "06c_basin_area_vs_lowest_elevation.png",
            "Scatter of drainage-basin contributing area vs. lowest elevation, colored by "
            "basin relief -- larger, low-lying, low-relief basins are the ones most exposed "
            "to compound tidal/pluvial backup.")


def write_readme():
    lines = ["# SAGIS Atlas\n",
             "Figures generated from SAGIS stormwater, soils, and wetlands GeoJSONs for the "
             "CORAL compound-flood project. Generated by `src/scripts/sagis_atlas.py`. "
             "Read-only outputs; no source data or code was modified.\n",
             "| Figure | Caption |",
             "|---|---|"]
    for fname, cap in captions:
        lines.append(f"| `{fname}` | {cap} |")
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("wrote README.md")


if __name__ == "__main__":
    fig1_network_map()
    fig2_diameter_material()
    fig3_channels_basins()
    fig4a_wetlands()
    fig4b_soils()
    fig5_density()
    fig6a_long_profile()
    fig6b_slope()
    fig6c_area_vs_elev()
    write_readme()
    log("DONE")
