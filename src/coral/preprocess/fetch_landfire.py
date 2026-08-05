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
import os
import time
import re
import struct
import urllib.parse
import urllib.request

import numpy as np

LFPS = "https://lfps.usgs.gov/api"

# Layer names are LF<version>_EVH; the bare acronym is rejected ("Invalid products: 240EVH").
# LF2016 is the default because Matthew is October 2016 and the vegetation state at the event is
# what the roughness field should represent. Later versions are available and differ mainly in
# disturbance updates. `list_products` prints what the service currently offers.
DEFAULT_LAYER = "LF2016_EVH"

# EVH encodes height in the class NAME, and the encoding differs between releases. LF2016:
#   1xx  Tree  height = code - 100 metres          (101 = 1 m ... 126 = 26 m)
#   2xx  Shrub height = (code - 200) * 0.1 metres  (203 = 0.3 m)
#   3xx  Herb  height = (code - 300) * 0.1 metres  (306 = 0.6 m)
# Everything else (11 Open Water, 22-25 Developed, 31 Barren, 64 crops...) is land cover with no
# height and maps to 0.
#
# Prefer the raster's own .vat.dbf, which carries CLASSNAMES like "Tree Height = 14 meters", over
# these rules: it is authoritative for whatever release was downloaded. The rules are the
# fallback when no VAT ships alongside.
TREE_LO, SHRUB_LO, HERB_LO = 100, 200, 300
_HEIGHT_RE = re.compile(r"Height\s*=\s*([0-9.]+)\s*meter", re.I)


def heights_from_vat(vat_dbf):
    """Parse a .vat.dbf into {code: height_m}. Returns {} if it cannot be read."""
    try:
        b = open(vat_dbf, "rb").read()
        nrec, hlen, rlen = struct.unpack("<IHH", b[4:12])
        flds = [(b[i:i + 11].split(b"\0")[0].decode(), b[i + 16]) for i in range(32, hlen - 1, 32)]
        names = [f[0] for f in flds]
        iv, ic = names.index("Value"), names.index("CLASSNAMES")
        out, off = {}, hlen
        for _ in range(nrec):
            rec, off, pos, vals = b[off:off + rlen], off + rlen, 1, []
            for _, sz in flds:
                vals.append(rec[pos:pos + sz].decode("latin1").strip()); pos += sz
            m = _HEIGHT_RE.search(vals[ic])
            out[int(float(vals[iv]))] = float(m.group(1)) if m else 0.0
        return out
    except Exception as e:
        print(f"  could not read VAT ({e}); falling back to code arithmetic")
        return {}


def evh_code_to_m(codes, lookup=None):
    """Map EVH class codes to heights in metres. Uses `lookup` (from the VAT) when given."""
    out = np.zeros(codes.shape, dtype="float32")
    if lookup:
        for c, h in lookup.items():
            out[codes == c] = h
        return out
    tree = (codes > TREE_LO) & (codes < SHRUB_LO)
    shrub = (codes > SHRUB_LO) & (codes < HERB_LO)
    herb = (codes > HERB_LO) & (codes < HERB_LO + 100)
    out[tree] = (codes[tree] - TREE_LO).astype("float32")
    out[shrub] = (codes[shrub] - SHRUB_LO).astype("float32") * 0.1
    out[herb] = (codes[herb] - HERB_LO).astype("float32") * 0.1
    return out


def fetch(bbox, out_zip, layer=DEFAULT_LAYER, email=None):
    """Submit an LFPS job and download the result. bbox = (W, S, E, N) in WGS84.

    LFPS requires an Email parameter; it is used to notify on completion and the request is
    rejected without it (HTTP 400).
    """
    if not email:
        raise SystemExit("LFPS requires --email")
    q = urllib.parse.urlencode({
        "Layer_List": layer,
        "Area_of_Interest": " ".join(str(v) for v in bbox),
        "Email": email})
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
    vat = evh_raster + ".vat.dbf"
    lookup = heights_from_vat(vat) if os.path.exists(vat) else {}
    if lookup:
        print(f"  using VAT class names ({len(lookup)} codes)")
    hm = evh_code_to_m(codes, lookup)
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
    f.add_argument("--out", required=True); f.add_argument("--layer", default=DEFAULT_LAYER)
    f.add_argument("--email", required=True, help="LFPS requires this; used for job notification")
    s.add_parser("products", help="list available CONUS layer names")
    g = s.add_parser("grid")
    g.add_argument("--evh", required=True, help="EVH GeoTIFF (fetched or downloaded manually)")
    g.add_argument("--dem", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "products":
        list_products()
    elif a.cmd == "fetch":
        fetch(a.bbox, a.out, a.layer, a.email)
    else:
        grid(a.evh, a.dem, a.out)


if __name__ == "__main__":
    main()
