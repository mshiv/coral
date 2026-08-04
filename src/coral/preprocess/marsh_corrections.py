"""Two marsh corrections that must be applied before the 4 m decision ensemble.

1. ZERO INFILTRATION ON THE MARSH PLATFORM (`mask_marsh_infiltration`)

   CORAL applies POLARIS/SSURGO storage-limited infiltration everywhere, including intertidal
   marsh. Intertidal marsh soil is saturated: during an event there is no storage to fill, so
   infiltration is negligible. Leaving it on lets the marsh absorb water it cannot absorb,
   which suppresses inundation where marsh-restoration and living-shoreline
   interventions are sited.

2. LIDAR VEGETATION BIAS IN THE DEM (`correct_marsh_dem`)

   Bare-earth lidar over Spartina does not reach the ground: dense canopy returns are
   classified as ground, so the DEM sits above true marsh elevation. Hladik and Alber (2012)
   measure this for Georgia salt marsh and correct it against species/height. An
   uncorrected DEM shrinks inundation extent, and a systematic elevation bias
   contaminates every matched-pair adaptation delta.

   Two correction modes:
     - `--offset`      a single constant subtracted from marsh cells (Hladik and Alber report
                       order 0.1-0.3 m for Georgia Spartina).
     - `--chm`         subtract a fraction of a canopy height model, cell by cell. Preferred
                       when a CHM exists, since bias scales with canopy height.

   Every edited cell is logged to a sidecar CSV. The 4 m runs are the decision product and
   reviewers will ask exactly which cells were altered and by how much.

NOTE: The Barinas (2024) closed-form biomass-to-roughness form needs depth and velocity at 
every timestep, which LISFLOOD-FP 8.0.3 cannot supply without solver changes.
"""
import argparse
import csv
import numpy as np

# NLCD classes that are marsh/wetland on this coast. 95 = emergent herbaceous wetland
# (Spartina platform), 90 = woody wetland. Open water (11) is not marsh and is excluded:
# it has no canopy and no soil storage question.
MARSH_NLCD = (95,)
WETLAND_NLCD = (90, 95)

# Arefin et al. (2026) saltmarsh interquartile range. Modulation is clamped to this so a tall
# canopy cannot push n above what the observational record supports.
AREFIN_N_LO, AREFIN_N_HI = 0.045, 0.145


def read_asc(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    h["ncols"], h["nrows"] = int(h["ncols"]), int(h["nrows"])
    return np.loadtxt(path, skiprows=6), h


def write_asc(path, arr, h, nodata=-9999.0, fmt="%.4f"):
    with open(path, "w") as f:
        f.write(f"ncols        {h['ncols']}\nnrows        {h['nrows']}\n"
                f"xllcorner    {h['xllcorner']:.12f}\nyllcorner    {h['yllcorner']:.12f}\n"
                f"cellsize     {h['cellsize']:.12f}\nNODATA_value {nodata:.0f}\n")
        np.savetxt(f, arr, fmt=fmt)


def _marsh_mask(classes, h_cls, target_shape, codes):
    if classes.shape != target_shape:
        raise SystemExit(f"class raster {classes.shape} does not match target {target_shape}; "
                         "reproject it onto the run DEM first (make_manning.classes_on_dem)")
    return np.isin(classes.astype(int), codes)


def mask_marsh_infiltration(infil_asc, infilcap_asc, classes_asc, out_infil, out_infilcap,
                            codes=WETLAND_NLCD, nodata=-9999.0):
    """Set infiltration rate and capacity to zero on marsh and woody wetland."""
    inf, hi = read_asc(infil_asc)
    cap, hc = read_asc(infilcap_asc)
    cls, _ = read_asc(classes_asc)
    m = _marsh_mask(cls, _, inf.shape, codes)
    keep = inf != nodata
    inf_out = np.where(m & keep, 0.0, inf)
    cap_out = np.where(m & keep, 0.0, cap)
    write_asc(out_infil, inf_out, hi, nodata)
    write_asc(out_infilcap, cap_out, hc, nodata)
    n = int((m & keep).sum())
    print(f"marsh infiltration zeroed on {n} cells ({100*n/keep.sum():.1f}% of valid)")
    print(f"  mean Ksat off-marsh {inf_out[keep & ~m].mean():.1f}, on-marsh 0.0")
    print(f"  -> {out_infil}, {out_infilcap}")
    return out_infil, out_infilcap


def correct_marsh_dem(dem_asc, classes_asc, out_dem, offset=None, chm_asc=None,
                      chm_fraction=0.5, codes=MARSH_NLCD, nodata=-9999.0, log_csv=None):
    """Lower marsh cells to remove lidar canopy bias. Returns the edited DEM path.

    `offset` subtracts a constant. `chm_asc` subtracts `chm_fraction` x canopy height, which is
    the better-founded form: the bias is a fraction of canopy, not a fixed number. Hladik and
    Alber found the residual scales with height and species, so a uniform offset is a
    first-order stand-in for a CHM, not an equivalent.
    """
    dem, h = read_asc(dem_asc)
    cls, _ = read_asc(classes_asc)
    m = _marsh_mask(cls, _, dem.shape, codes) & (dem != nodata)

    if chm_asc:
        chm, _ = read_asc(chm_asc)
        if chm.shape != dem.shape:
            raise SystemExit("CHM does not match the DEM grid; reproject it first")
        drop = np.where(m, np.clip(chm, 0, None) * chm_fraction, 0.0)
    elif offset is not None:
        drop = np.where(m, float(offset), 0.0)
    else:
        raise SystemExit("supply either --offset or --chm")

    out = np.where(m, dem - drop, dem)
    write_asc(out_dem, out, h, nodata)
    print(f"marsh DEM lowered on {int(m.sum())} cells; mean drop {drop[m].mean():.3f} m, "
          f"max {drop[m].max():.3f} m -> {out_dem}")

    # Auditable record of every edited cell. Cheap now; the alternative is being unable to
    # answer "which cells did you change" about the headline result.
    if log_csv:
        rows = np.argwhere(m)
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "col", "dem_before_m", "drop_m", "dem_after_m"])
            for r, c in rows:
                w.writerow([r, c, f"{dem[r,c]:.4f}", f"{drop[r,c]:.4f}", f"{out[r,c]:.4f}"])
        print(f"  edit log ({len(rows)} cells) -> {log_csv}")
    return out_dem


def modulate_marsh_n(manning_asc, classes_asc, chm_asc, out_manning,
                     codes=MARSH_NLCD, lo=AREFIN_N_LO, hi=AREFIN_N_HI, nodata=-9999.0):
    """Vary n within the marsh class by canopy height, clamped to the Arefin saltmarsh IQR.

    Keeps the class map as the spatial structure and lets height set where a cell sits inside
    the observed range, rather than inventing values outside it. Height is rescaled by its own
    marsh-cell 5th-95th percentiles so the mapping is relative to this site's canopy, not to an
    absolute height that would not transfer.
    """
    n_arr, hm = read_asc(manning_asc)
    cls, _ = read_asc(classes_asc)
    chm, _ = read_asc(chm_asc)
    m = _marsh_mask(cls, _, n_arr.shape, codes) & (n_arr != nodata)
    if not m.any():
        raise SystemExit("no marsh cells found; check the class raster and codes")
    hgt = np.clip(chm, 0, None)
    p5, p95 = np.percentile(hgt[m], [5, 95])
    frac = np.clip((hgt - p5) / max(p95 - p5, 1e-6), 0, 1)
    out = np.where(m, lo + frac * (hi - lo), n_arr)
    write_asc(out_manning, out, hm, nodata)
    print(f"marsh n modulated on {int(m.sum())} cells: {out[m].min():.3f}-{out[m].max():.3f} "
          f"(canopy {p5:.2f}-{p95:.2f} m mapped onto Arefin IQR {lo}-{hi}) -> {out_manning}")
    return out_manning


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("infil", help="zero infiltration on marsh/wetland")
    a.add_argument("--infil", required=True); a.add_argument("--infilcap", required=True)
    a.add_argument("--classes", required=True)
    a.add_argument("--out-infil", required=True); a.add_argument("--out-infilcap", required=True)

    b = sub.add_parser("dem", help="remove lidar canopy bias from marsh elevations")
    b.add_argument("--dem", required=True); b.add_argument("--classes", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--offset", type=float, default=None)
    b.add_argument("--chm", default=None); b.add_argument("--chm-fraction", type=float, default=0.5)
    b.add_argument("--log", default=None)

    c = sub.add_parser("manning", help="modulate marsh n by canopy height")
    c.add_argument("--manning", required=True); c.add_argument("--classes", required=True)
    c.add_argument("--chm", required=True); c.add_argument("--out", required=True)

    v = ap.parse_args()
    if v.cmd == "infil":
        mask_marsh_infiltration(v.infil, v.infilcap, v.classes, v.out_infil, v.out_infilcap)
    elif v.cmd == "dem":
        correct_marsh_dem(v.dem, v.classes, v.out, offset=v.offset, chm_asc=v.chm,
                          chm_fraction=v.chm_fraction, log_csv=v.log)
    else:
        modulate_marsh_n(v.manning, v.classes, v.chm, v.out)


if __name__ == "__main__":
    main()
