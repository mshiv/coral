"""Land-surface corrections applied before the decision ensembles.

Two are corrections to wrong assumptions, and belong in every run:

  mask_marsh_infiltration  Tidal marsh soil is saturated; there is no storage to fill during an
                           event. Applying storage-limited infiltration there lets the marsh
                           absorb water it cannot, suppressing inundation exactly where marsh
                           and living-shoreline interventions are sited.

  correct_marsh_dem        Bare-earth lidar over Spartina does not reach the ground; canopy
                           returns are classified as ground and the DEM sits high. Hladik and
                           Alber (2012) measured this for Georgia marsh. Uncorrected, it
                           shrinks inundation extent and biases every adaptation delta.

One is a refinement, and is better reported as a separate sensitivity than folded into a
baseline:

  modulate_marsh_n         Vary n within the marsh class by canopy height, clamped to the
                           Arefin et al. (2026) saltmarsh IQR.

Not implemented: the Barinas (2024) biomass-to-roughness form needs depth and velocity every
timestep, which LISFLOOD-FP 8.0.3 cannot supply, and was fitted on fluvial floodplains.
"""
import argparse
import csv
import numpy as np

# NLCD classes that are marsh/wetland on this coast. 95 = emergent herbaceous wetland
# (Spartina platform), 90 = woody wetland. Open water (11) is not marsh and is excluded:
# it has no canopy and no soil storage question.
MARSH_NLCD = (95,)
WETLAND_NLCD = (90, 95)

# Arefin et al. (2026), Est. Coast. Shelf Sci. 334, 109791: saltmarsh Manning's n interquartile
# range, from a meta-analysis of 36 studies. Modulation is clamped to this so a tall canopy
# cannot push n past what the observational record supports.
#
# This is the SALTMARSH range. An earlier version used 0.045-0.145, which spans saltmarsh and
# mangrove (mangrove IQR is 0.10-0.14) and would have assigned mangrove roughness to tall
# Spartina. The intervention registry in interventions/generate.py keeps the same distinction.
AREFIN_N_LO, AREFIN_N_HI = 0.04, 0.08

# Ceiling on how far a marsh cell may be lowered. Hladik and Alber report Georgia Spartina
# offsets of order 0.1-0.3 m, and the observed marsh canopy p90 at Pin Point is 1.39 m, so a
# correction beyond a metre is not vegetation bias -- it is an NLCD class-95 cell that actually
# contains trees. Uncapped, those cells were being lowered by up to 22.5 m, gouging pits into
# the marsh platform that would act as artificial sinks and destabilise the timestep.
MAX_MARSH_DROP_M = 1.0

# Baseline n above this is not vegetation. Buildings are represented as roughness (n = 2.0)
# rather than as raised DEM blocks, because 4 m vertical steps collapsed the timestep. Any
# building footprint falling inside NLCD class 95 would otherwise be overwritten by the marsh
# modulation and silently deleted from the model.
STRUCTURE_N_FLOOR = 0.2

# NWI WETLAND_TYPE values that are tidal, and therefore saturated during an event. This is the
# classification the infiltration question actually turns on. NLCD 90 ("woody wetland") mixes
# estuarine forested wetland with palustrine freshwater forest, which drains between events and
# does have storage; masking it wholesale over-masks. NLCD 95 has the same problem in miniature,
# lumping regularly-flooded low marsh with irregularly-flooded high marsh.
NWI_TIDAL = ("Estuarine and Marine Wetland",)


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
                            codes=WETLAND_NLCD, nodata=-9999.0, nwi_geojson=None, dem_asc=None):
    """Zero infiltration rate and capacity on tidal wetland.

    Both are zeroed, not just the rate: with capacity zero the storage-limited scheme has
    nothing to give whatever Ksat says.

    Default mask is NLCD codes 90 and 95. Pass `nwi_geojson` (with `dem_asc`) to mask on NWI
    tidal types instead, which is the physically correct criterion -- saturation follows tidal
    regime, not land cover.
    """
    inf, hi = read_asc(infil_asc)
    cap, hc = read_asc(infilcap_asc)
    if nwi_geojson:
        if not dem_asc:
            raise SystemExit("--nwi needs --dem to rasterize onto")
        from .make_manning import nwi_types_on_dem
        types = nwi_types_on_dem(nwi_geojson, dem_asc)
        m = np.isin(types.astype(str), NWI_TIDAL)
        print(f"NWI mask: {sorted(set(types.ravel().tolist()) - {''})}")
    else:
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
                      chm_fraction=0.5, codes=MARSH_NLCD, nodata=-9999.0, log_csv=None,
                      max_drop=MAX_MARSH_DROP_M):
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

    n_capped = int((drop > max_drop).sum())
    drop = np.minimum(drop, max_drop)
    out = np.where(m, dem - drop, dem)
    if n_capped:
        print(f"  {n_capped} cells capped at {max_drop:.2f} m "
              f"({100*n_capped/max(int(m.sum()),1):.2f}% of marsh; these are class-95 cells "
              "containing trees, not marsh canopy)")
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


# Absolute height range for the height -> n map, matching interventions.generate. Anchors the
# mapping to physical canopy height rather than to each raster's own distribution.
VEG_H_LO, VEG_H_HI = 0.2, 1.4


def modulate_marsh_n(manning_asc, classes_asc, chm_asc, out_manning,
                     codes=MARSH_NLCD, lo=AREFIN_N_LO, hi=AREFIN_N_HI, nodata=-9999.0,
                     structure_floor=STRUCTURE_N_FLOOR, absolute=True):
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
    struct = m & (n_arr > structure_floor)
    if struct.any():
        print(f"  preserving {int(struct.sum())} structure cells (n > {structure_floor}) "
              "inside the marsh class; these are buildings encoded as roughness")
        m = m & ~struct
    if not m.any():
        raise SystemExit("no marsh cells found; check the class raster and codes")
    hgt = np.clip(chm, 0, None)
    # Cells with no canopy signal carry no vegetation information, so they keep their baseline n.
    # Mapping them through the percentile would put them at the bottom of the range -- the
    # smoothest value available -- purely because the canopy product has nothing there. At 30 m
    # these are open-water and developed EVH codes falling inside the NLCD marsh class.
    nodata_veg = m & (hgt <= 0)
    if nodata_veg.any():
        print(f"  {int(nodata_veg.sum())} marsh cells have zero canopy; left at baseline n "
              "(no vegetation signal, so not mapped to the smooth end)")
        m = m & ~nodata_veg
    if not m.any():
        raise SystemExit("no marsh cells with canopy data")
    if absolute:
        # Fixed 0.2-1.4 m scale. Percentile rescaling normalises each raster to its own spread,
        # which makes 4 m and 30 m fields incomparable and, worse, inflates a narrow one: the
        # LANDFIRE herbaceous bins give a 0.60-0.90 m range over marsh, and stretching 0.3 m
        # across the whole IQR turns a single 0.1 m bin step into a large roughness change.
        lo_h, hi_h = VEG_H_LO, VEG_H_HI
    else:
        lo_h, hi_h = np.percentile(hgt[m], [5, 95])
    frac = np.clip((hgt - lo_h) / max(hi_h - lo_h, 1e-6), 0, 1)
    out = np.where(m, lo + frac * (hi - lo), n_arr)
    write_asc(out_manning, out, hm, nodata)
    print(f"marsh n modulated on {int(m.sum())} cells: {out[m].min():.3f}-{out[m].max():.3f} "
          f"(canopy {hgt[m].min():.2f}-{hgt[m].max():.2f} m on a "
          f"{'fixed ' + format(lo_h, '.1f') + '-' + format(hi_h, '.1f') if absolute else 'percentile'} m "
          f"scale -> Arefin IQR {lo}-{hi}) -> {out_manning}")
    return out_manning


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("infil", help="zero infiltration on tidal wetland")
    a.add_argument("--infil", required=True); a.add_argument("--infilcap", required=True)
    a.add_argument("--classes", default=None, help="NLCD-on-DEM raster (default mask)")
    a.add_argument("--nwi", default=None, help="NWI GeoJSON; masks tidal types instead of NLCD")
    a.add_argument("--dem", default=None, help="DEM to rasterize the NWI polygons onto")
    a.add_argument("--out-infil", required=True); a.add_argument("--out-infilcap", required=True)

    b = sub.add_parser("dem", help="remove lidar canopy bias from marsh elevations")
    b.add_argument("--dem", required=True); b.add_argument("--classes", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--offset", type=float, default=None)
    b.add_argument("--chm", default=None); b.add_argument("--chm-fraction", type=float, default=0.5)
    b.add_argument("--log", default=None)
    b.add_argument("--max-drop", type=float, default=MAX_MARSH_DROP_M,
                   help="ceiling on the marsh lowering (m); guards against class-95 cells "
                        "that actually contain trees")

    c = sub.add_parser("manning", help="modulate marsh n by canopy height")
    c.add_argument("--manning", required=True); c.add_argument("--classes", required=True)
    c.add_argument("--chm", required=True); c.add_argument("--out", required=True)
    c.add_argument("--percentile", action="store_true",
                   help="rescale by this raster's own 5-95 percentile instead of the fixed "
                        "0.2-1.4 m scale; not comparable across resolutions")

    v = ap.parse_args()
    if v.cmd == "infil":
        if not (v.classes or v.nwi):
            raise SystemExit("supply --classes or --nwi")
        mask_marsh_infiltration(v.infil, v.infilcap, v.classes, v.out_infil, v.out_infilcap,
                                nwi_geojson=v.nwi, dem_asc=v.dem)
    elif v.cmd == "dem":
        correct_marsh_dem(v.dem, v.classes, v.out, offset=v.offset, chm_asc=v.chm,
                          chm_fraction=v.chm_fraction, log_csv=v.log, max_drop=v.max_drop)
    else:
        modulate_marsh_n(v.manning, v.classes, v.chm, v.out, absolute=not v.percentile)


if __name__ == "__main__":
    main()
