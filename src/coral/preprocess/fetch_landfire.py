"""Fetch LANDFIRE Existing Vegetation Height (EVH) and put it on the run DEM grid.

Supplies the canopy height the 30 m domain needs for within-marsh-class roughness modulation
(marsh_corrections.modulate_marsh_n). The 4 m clip uses a measured 3DEP CHM instead
(preprocess.make_chm); EVH is used at 30 m because it is 30 m native, whereas 3DEP over the whole
Savannah domain would be ~40x the Pin Point download for no gain at this cell size.

The two products are not equivalent:

  3DEP CHM   measured, continuous, sub-metre vertical detail. Marsh median 0.48 m at Pin Point,
             cleanly separated from ~19 m woody classes.
  LANDFIRE   modelled from Landsat plus biophysical gradients, calibrated to field plots, and
             delivered as BINNED height classes. National vegetation typing discriminates tidal
             emergent marsh less sharply than lidar does.

EVH is delivered as class codes as opposed to metres. `evh_code_to_m` maps the herbaceous and shrub codes
that matter for marsh; tree codes are mapped coarsely because marsh roughness is what this feeds.

How to use:
  fetch   LANDFIRE Product Service (LFPS). The API changes between LANDFIRE releases; if it
          fails, download the EVH tile for your area from landfire.gov and use --from-file.
  grid    reproject a downloaded EVH raster onto the run DEM (nearest, to preserve codes) and
          write height in metres.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

import numpy as np

LFPS = "https://lfps.usgs.gov/api"

# EVH class codes -> representative height (m). LANDFIRE encodes herbaceous height as 101-104,
# shrub as 105-108, tree as 109-programmatic. Values are bin midpoints.
EVH_CODE_M = {
    101: 0.25, 102: 0.75, 103: 1.5, 104: 3.0,          # herbaceous: <0.5, 0.5-1, 1-2, >2 m
    105: 0.25, 106: 0.75, 107: 1.5, 108: 3.0,          # shrub: same bins
}
# Tree classes 109-160 encode height in metres directly as (code - 100) in most releases.
TREE_LO, TREE_HI = 109, 160


def evh_code_to_m(codes):
    """Map EVH class codes to representative heights in metres. Unknown codes -> 0."""
    out = np.zeros(codes.shape, dtype="float32")
    for c, h in EVH_CODE_M.items():
        out[codes == c] = h
    tree = (codes >= TREE_LO) & (codes <= TREE_HI)
    out[tree] = (codes[tree] - 100).astype("float32")
    return out


def fetch(bbox, out_zip, layer="240EVH"):
    """Submit an LFPS job and download the result. bbox = (W, S, E, N) in WGS84."""
    q = urllib.parse.urlencode({
        "Layer_List": layer,
        "Area_of_Interest": " ".join(str(v) for v in bbox)})
    with urllib.request.urlopen(f"{LFPS}/job/submit?{q}", timeout=60) as r:
        job = json.load(r)
    jid = job.get("jobId") or job.get("JobId")
    if not jid:
        raise SystemExit(f"LFPS did not return a job id: {job}\n"
                         "The API changes between LANDFIRE releases. Download EVH for your area "
                         "from landfire.gov and use --from-file instead.")
    print(f"LFPS job {jid}")
    for _ in range(120):
        with urllib.request.urlopen(f"{LFPS}/job/status?JobId={jid}", timeout=60) as r:
            st = json.load(r)
        status = str(st.get("status", st.get("Status", ""))).lower()
        print(f"  {status}")
        if "succ" in status:
            url = st.get("outputFile") or st.get("OutputFile")
            urllib.request.urlretrieve(url, out_zip)
            print(f"-> {out_zip}")
            return out_zip
        if "fail" in status:
            raise SystemExit(f"LFPS job failed: {st}")
        time.sleep(15)
    raise SystemExit("LFPS job did not finish in 30 minutes")


def grid(evh_raster, dem_asc, out_asc, nodata=-9999.0):
    """Reproject an EVH raster onto the DEM grid and write height in metres."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import reproject, Resampling

    with rasterio.open(dem_asc) as dem:
        H, W = dem.height, dem.width
        tr, crs = dem.transform, (dem.crs or CRS.from_epsg(4326))
        hdr = dict(ncols=W, nrows=H, cellsize=dem.transform[0],
                   xllcorner=dem.bounds.left, yllcorner=dem.bounds.bottom)
    codes = np.zeros((H, W), dtype="int32")
    with rasterio.open(evh_raster) as src:
        # Nearest, not bilinear: these are class codes, and interpolating between them produces
        # codes that do not exist.
        reproject(source=rasterio.band(src, 1), destination=codes,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest)
    hm = evh_code_to_m(codes)
    with open(out_asc, "w") as f:
        f.write(f"ncols        {W}\nnrows        {H}\n"
                f"xllcorner    {hdr['xllcorner']:.12f}\nyllcorner    {hdr['yllcorner']:.12f}\n"
                f"cellsize     {hdr['cellsize']:.12f}\nNODATA_value {nodata:.0f}\n")
        np.savetxt(f, hm, fmt="%.2f")
    u, c = np.unique(codes, return_counts=True)
    print(f"EVH -> {out_asc}: {hm.min():.2f}-{hm.max():.2f} m, median {np.median(hm):.2f}")
    print("  top codes:", [(int(k), int(v)) for k, v in sorted(zip(u, c), key=lambda x: -x[1])[:6]])
    if (hm == 0).mean() > 0.5:
        print("  WARNING: over half the domain mapped to 0 m. Check the EVH codes against the "
              "release you downloaded; EVH_CODE_M may need updating.")
    return out_asc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="cmd", required=True)
    f = s.add_parser("fetch")
    f.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    f.add_argument("--out", required=True); f.add_argument("--layer", default="240EVH")
    g = s.add_parser("grid")
    g.add_argument("--evh", required=True, help="EVH GeoTIFF (fetched or downloaded manually)")
    g.add_argument("--dem", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.bbox, a.out, a.layer)
    else:
        grid(a.evh, a.dem, a.out)


if __name__ == "__main__":
    main()
