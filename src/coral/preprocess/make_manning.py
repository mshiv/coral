#!/usr/bin/env python3
"""
Build a LISFLOOD-FP Manning's-n grid from NLCD land cover, snapped to the EXACT
DEM grid (same ncols/nrows/extent/cellsize -- LISFLOOD requires the headers match).

Workflow:
  1. read the target DEM (defines grid + CRS, EPSG:4326)
  2. reproject the NLCD land-cover raster (native EPSG:5070 Albers, 30 m) onto
     that exact grid with nearest-neighbour (preserves class codes)
  3. map each NLCD class -> Manning's n via the lookup below
  4. write Manning_SAV.asc with a header identical to the DEM

Where to get NLCD:
  https://www.mrlc.gov/data  (NLCD 2016 Land Cover, CONUS) -- download the tile/
  AOI covering lon[-81.22,-80.82] lat[31.82,32.08], save as nlcd_savannah.tif.
  (Or use the full CONUS GeoTIFF; this script windows it via the DEM bounds.)

NLCD -> Manning's n  (Kalyanapu et al. 2009, widely used for flood models;
edit to your preferred reference). Savannah is marsh-heavy, so classes 90/95
(wetlands) and 11 (open water) dominate the seaward part.

Deps: numpy, rasterio
Usage:
  python make_manning.py --dem inputs/SUB_DEM_SAV.asc \
      --nlcd nlcd_savannah.tif --out inputs/Manning_SAV.asc
"""
import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

# NLCD class code -> Manning's n  (Kalyanapu et al., 2009)
NLCD_N = {
    11: 0.025,   # Open Water
    12: 0.025,   # Perennial Ice/Snow
    21: 0.0404,  # Developed, Open Space
    22: 0.0678,  # Developed, Low Intensity
    23: 0.0678,  # Developed, Medium Intensity
    24: 0.0404,  # Developed, High Intensity
    31: 0.0113,  # Barren Land
    41: 0.36,    # Deciduous Forest
    42: 0.32,    # Evergreen Forest
    43: 0.40,    # Mixed Forest
    51: 0.40,    # Dwarf Scrub
    52: 0.40,    # Shrub/Scrub
    71: 0.368,   # Grassland/Herbaceous
    72: 0.34,    # Sedge/Herbaceous
    81: 0.325,   # Pasture/Hay
    82: 0.037,   # Cultivated Crops
    90: 0.086,   # Woody Wetlands
    95: 0.061,   # Emergent Herbaceous Wetlands
}
DEFAULT_N = 0.06   # fallback for unmapped LAND cells
WATER_N = 0.025    # fallback for unmapped cells that are below sea level (ocean)

# NWI WETLAND_TYPE -> Manning's n (literature ranges for marsh/wetland roughness;
# emergent marsh ~0.10-0.15, forested/shrub wetland ~0.12-0.20 -- see e.g.
# Chow (1959) Open-Channel Hydraulics Table 5-6, and Kalyanapu et al. (2009)
# for the NLCD-analogue values these are meant to refine on top of).
# These OVERRIDE the coarser NLCD-derived n where NWI marsh/wetland polygons
# are present, because NWI classifies actual wetland type/vegetation density
# rather than NLCD's generic 30 m land-cover class.
NWI_N = {
    "Estuarine and Marine Wetland":        0.11,  # emergent tidal marsh (Spartina)
    "Freshwater Emergent Wetland":         0.12,  # emergent marsh, denser stems
    "Freshwater Forested/Shrub Wetland":   0.16,  # forested/shrub wetland, mid of 0.12-0.20
    "Freshwater Pond":                     0.035, # open water, pond-like
    "Estuarine and Marine Deepwater":      0.030, # subtidal open water
    "Riverine":                            0.045, # channelized open water
}


def write_ascii(path, arr, dem_path, nodata=-9999):
    # copy the DEM's 6 header lines VERBATIM so headers are byte-identical
    # (LISFLOOD requires DEM and Manning grids to align exactly).
    with open(dem_path) as f:
        header_lines = [f.readline() for _ in range(6)]
    a = np.where(np.isnan(arr), nodata, arr)
    with open(path, "w") as f:
        f.writelines(header_lines)
        np.savetxt(f, a, fmt="%.4f")


def classes_on_dem(nlcd_path, dem_path):
    """Reproject NLCD class codes onto the exact DEM grid (nearest -> keep codes).
    Returns (classes[H,W] int16, z[H,W] float32 elevations with nan nodata).
    Reproject once, reuse for many n-mappings (see calibrate_manning)."""
    with rasterio.open(dem_path) as dem:
        dem_tr = dem.transform
        dem_crs = dem.crs or CRS.from_epsg(4326)
        H, W = dem.height, dem.width
        z = dem.read(1).astype("float32")
        dem_nod = dem.nodata if dem.nodata is not None else -9999
    z = np.where((z == dem_nod) | (z <= -9990), np.nan, z)
    classes = np.zeros((H, W), dtype="int16")
    with rasterio.open(nlcd_path) as nlcd:
        reproject(source=rasterio.band(nlcd, 1), destination=classes,
                  src_transform=nlcd.transform, src_crs=nlcd.crs,
                  dst_transform=dem_tr, dst_crs=dem_crs,
                  resampling=Resampling.nearest)
    return classes, z


def classes_to_n(classes, z, class_n=None, *, default_n=DEFAULT_N,
                 water_n=WATER_N, water_level=0.81):
    """Map class codes -> Manning's n using `class_n` (defaults to NLCD_N).
    Pass a custom {code: n} to override values (the calibration knob)."""
    class_n = NLCD_N if class_n is None else class_n
    mapped = np.isin(classes, list(class_n))
    n = np.where((~mapped) & (z < water_level), water_n, default_n).astype("float32")
    for code, val in class_n.items():
        m = classes == code
        if m.any():
            n[m] = val
    return n


def nwi_types_on_dem(nwi_geojson_path, dem_path):
    """Rasterize an NWI wetlands GeoJSON (WETLAND_TYPE attribute per polygon) onto
    the exact DEM grid. Returns a string array (H,W) with '' where no wetland polygon
    covers the cell. Optional companion to classes_on_dem (NLCD)."""
    import json
    from rasterio.features import rasterize
    with rasterio.open(dem_path) as dem:
        dem_tr = dem.transform
        H, W = dem.height, dem.width
    d = json.load(open(nwi_geojson_path))
    # stable int code per distinct WETLAND_TYPE string, 0 = no wetland
    types = sorted({f["properties"].get("WETLAND_TYPE") for f in d["features"]
                    if f["properties"].get("WETLAND_TYPE")})
    code_of = {t: i + 1 for i, t in enumerate(types)}
    shapes = [(f["geometry"], code_of[f["properties"]["WETLAND_TYPE"]])
              for f in d["features"] if f["properties"].get("WETLAND_TYPE")]
    if not shapes:
        return np.full((H, W), "", dtype=object)
    codes = rasterize(shapes, out_shape=(H, W), transform=dem_tr, fill=0, dtype="int32")
    inv = {v: k for k, v in code_of.items()}
    out = np.full((H, W), "", dtype=object)
    for code, name in inv.items():
        out[codes == code] = name
    return out


def apply_nwi_overlay(n, nwi_types, nwi_n=None):
    """Overlay NWI-derived Manning's n on top of an existing NLCD-derived grid `n`.
    Cells covered by an NWI wetland polygon get the marsh/wetland n (nwi_n, defaults
    to NWI_N) instead of the NLCD value -- this is the calibration knob for the
    optional --nwi flag. Returns (n_new, count_changed)."""
    nwi_n = NWI_N if nwi_n is None else nwi_n
    n2 = n.copy()
    changed = np.zeros(n.shape, dtype=bool)
    for wtype, val in nwi_n.items():
        m = nwi_types == wtype
        if m.any():
            n2[m] = val
            changed |= m
    return n2, int(changed.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True)
    ap.add_argument("--nlcd", required=True, help="NLCD land-cover GeoTIFF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--default-n", type=float, default=DEFAULT_N)
    ap.add_argument("--water-n", type=float, default=WATER_N,
                    help="n for unmapped cells below --water-level (ocean)")
    ap.add_argument("--water-level", type=float, default=None,
                    help="DEM elevation (m) below which unmapped cells are water "
                         "(default from config manning.sea_level)")
    ap.add_argument("--config", help="scenario YAML (provides water-level)")
    ap.add_argument("--nwi", default=None,
                    help="optional NWI wetlands GeoJSON (e.g. from fetch_sagis.py "
                         "--group physics); overlays marsh/wetland n on top of NLCD")
    args = ap.parse_args()
    if args.water_level is None:
        if args.config:
            from coral import config
            args.water_level = config.load(args.config).manning.sea_level
        else:
            args.water_level = 0.81

    classes, z = classes_on_dem(args.nlcd, args.dem)
    H, W = classes.shape
    n = classes_to_n(classes, z, default_n=args.default_n,
                     water_n=args.water_n, water_level=args.water_level)
    if args.nwi:
        nwi_types = nwi_types_on_dem(args.nwi, args.dem)
        n, n_changed = apply_nwi_overlay(n, nwi_types)
        print(f"NWI overlay ({args.nwi}): {n_changed:,} cells refined to marsh/wetland n")
    found = {c: int((classes == c).sum()) for c in NLCD_N if (classes == c).any()}
    write_ascii(args.out, n, args.dem, nodata=-9999)
    mapped = np.isin(classes, list(NLCD_N))

    print(f"wrote {args.out}  ({W}x{H}, n range {n.min():.3f}-{n.max():.3f})")
    print("class coverage (NLCD code: cells):")
    for c in sorted(found):
        print(f"  {c:>3} (n={NLCD_N[c]:.4f}): {found[c]:,}")
    um = ~mapped
    um_water = int((um & (z < args.water_level)).sum())
    um_land = int((um & ~(z < args.water_level)).sum())
    if um_water:
        print(f"  unmapped + below sea level -> water n={args.water_n}: {um_water:,} cells")
    if um_land:
        print(f"  unmapped land -> default n={args.default_n}: {um_land:,} cells")
    print("\nVERIFY header matches the DEM exactly:")
    print(f"  head -6 {args.out}; head -6 {args.dem}")


if __name__ == "__main__":
    main()
