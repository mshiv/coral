#!/usr/bin/env python3
"""Comparison figures: current physics inputs (NLCD Manning's n, POLARIS Ksat/AWC)
vs the new NWI-wetlands / SSURGO-soils refinements, over the Pin Point, GA domain.

Run: /Users/smurugan9/miniforge3/envs/coral/bin/python src/scripts/physics_input_comparison.py
Outputs: reports/physics/*.png
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coral" / "preprocess"))
from make_manning import NLCD_N, NWI_N, classes_to_n, classes_on_dem  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "physics"
OUT.mkdir(parents=True, exist_ok=True)

DEM = "/Users/smurugan9/research/coastalFlood/savannah_matthew_workflow/inputs/SUB_DEM_SAV.asc"
NLCD_TIF = "/Users/smurugan9/research/coastalFlood/savannah_matthew_workflow/inputs/nlcd_savannah.tif"
NWI_GEOJSON = REPO / "data/raw/sagis_pinpoint/sagis_wetlands_nwi.geojson"
SOILS_GEOJSON = REPO / "data/raw/sagis_pinpoint/sagis_soils_nrcs.geojson"
SSURGO_TABLE = REPO / "data/raw/ssurgo_pinpoint/ksat_awc_by_mukey.json"
CUR_INFIL = "/Users/smurugan9/research/coastalFlood/savannah_matthew_workflow/lisflood_compound_infilcap_run/infil_matthew_compound.asc"
CUR_INFILCAP = "/Users/smurugan9/research/coastalFlood/savannah_matthew_workflow/lisflood_compound_infilcap_run/infilcap_matthew_compound.asc"

# Pin Point focus bbox (native domain is wider Savannah bbox; NWI/SSURGO fetches
# were clipped here, so we crop all maps to this window for a fair, focused figure)
PP_BBOX = (-81.1557, -81.1181, 31.9278, 31.9601)  # W, E, S, N

# Full-domain (whole DEM extent) inputs, fetched separately over the entire
# Savannah compound-flood domain (see data/raw/sagis_savannah, ssurgo_savannah).
FULL_NWI_GEOJSON = REPO / "data/raw/sagis_savannah/sagis_wetlands_nwi.geojson"
FULL_SOILS_GEOJSON = REPO / "data/raw/sagis_savannah/sagis_soils_nrcs.geojson"
FULL_SSURGO_TABLE = REPO / "data/raw/ssurgo_savannah/ksat_awc_by_mukey.json"
OUT_FULL = REPO / "reports" / "physics" / "full_domain"
OUT_FULL.mkdir(parents=True, exist_ok=True)

# max dimension (pixels) for plotted images; full res is always used for stats
PLOT_MAX_DIM = 1200


def downsample_for_plot(arr, max_dim=PLOT_MAX_DIM):
    """Block-average downsample a 2D array (via reshape) for fast plotting.
    Only downsamples if a dimension exceeds max_dim; returns the array unchanged
    otherwise. Stats should always be computed on the full-res array, not this."""
    h, w = arr.shape
    fy = max(1, h // max_dim + (1 if h % max_dim else 0))
    fx = max(1, w // max_dim + (1 if w % max_dim else 0))
    if fy == 1 and fx == 1:
        return arr
    hh, ww = (h // fy) * fy, (w // fx) * fx
    a = arr[:hh, :ww]
    is_bool = a.dtype == bool
    a = a.astype("float64")
    a = a.reshape(hh // fy, fy, ww // fx, fx)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(a, axis=(1, 3))
    return out > 0.5 if is_bool else out


def crop_to_bbox(arr, transform, bbox, nodata=-9999):
    """Crop a (H,W) array + its affine transform to a lon/lat bbox. Returns (arr, extent)."""
    W, E, S, N = bbox
    inv = ~transform
    col0, row0 = inv * (W, N)
    col1, row1 = inv * (E, S)
    c0, c1 = sorted((int(col0), int(col1)))
    r0, r1 = sorted((int(row0), int(row1)))
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, arr.shape[1]), min(r1, arr.shape[0])
    sub = arr[r0:r1, c0:c1]
    return sub, (W, E, S, N)


def read_dem():
    with rasterio.open(DEM) as d:
        z = d.read(1).astype("float32")
        nod = d.nodata if d.nodata is not None else -9999
        z = np.where((z == nod) | (z <= -9990), np.nan, z)
        return z, d.transform, d.crs


def nwi_types_on_grid(nwi_geojson, transform, shape):
    d = json.load(open(nwi_geojson))
    types = sorted({f["properties"].get("WETLAND_TYPE") for f in d["features"]
                    if f["properties"].get("WETLAND_TYPE")})
    code_of = {t: i + 1 for i, t in enumerate(types)}
    shapes = [(f["geometry"], code_of[f["properties"]["WETLAND_TYPE"]])
              for f in d["features"] if f["properties"].get("WETLAND_TYPE")]
    codes = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")
    inv = {v: k for k, v in code_of.items()}
    out = np.full(shape, "", dtype=object)
    for code, name in inv.items():
        out[codes == code] = name
    return out


def fig_manning(full_domain=False):
    z, tr, crs = read_dem()
    classes, _ = classes_on_dem(NLCD_TIF, DEM)
    n_current = classes_to_n(classes, z)
    nwi_path = FULL_NWI_GEOJSON if full_domain else NWI_GEOJSON
    nwi = nwi_types_on_grid(nwi_path, tr, classes.shape)
    n_new = n_current.copy()
    changed = np.zeros(classes.shape, dtype=bool)
    for wtype, val in NWI_N.items():
        m = nwi == wtype
        if m.any():
            n_new[m] = val
            changed |= m

    if full_domain:
        h, w = z.shape
        n_current_c, n_new_c, changed_c, z_c = n_current, n_new, changed, z
        ext = (tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f)  # W, E, S, N
        out_dir, tag, region = OUT_FULL, "full_domain", "full domain (Savannah)"
    else:
        n_current_c, ext = crop_to_bbox(n_current, tr, PP_BBOX)
        n_new_c, _ = crop_to_bbox(n_new, tr, PP_BBOX)
        changed_c, _ = crop_to_bbox(changed, tr, PP_BBOX)
        z_c, _ = crop_to_bbox(z, tr, PP_BBOX)
        out_dir, tag, region = OUT, "current_vs_nwi", "Pin Point, GA"
    land = ~np.isnan(z_c)
    n_frac_changed = changed_c[land].mean() * 100 if land.any() else np.nan

    def plotarr(a):
        return downsample_for_plot(np.where(land, a, np.nan)) if full_domain else np.where(land, a, np.nan)

    vmin, vmax = 0.0, 0.45
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(np.ma.masked_invalid(plotarr(n_current_c)),
                          extent=ext, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Current: NLCD-based Manning's n")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="n")

    im1 = axes[1].imshow(np.ma.masked_invalid(plotarr(n_new_c)),
                          extent=ext, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("NWI-refined Manning's n\n(marsh/wetland overlay)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="n")

    diff_full = np.where(land, n_new_c - n_current_c, np.nan)
    diff = downsample_for_plot(diff_full) if full_domain else diff_full
    dmax = np.nanmax(np.abs(diff_full)) if np.any(~np.isnan(diff_full)) else 0.1
    im2 = axes[2].imshow(diff, extent=ext, vmin=-dmax, vmax=dmax, cmap="RdBu_r")
    axes[2].set_title(f"Difference (NWI - current)\n{n_frac_changed:.1f}% of land cells changed")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="Δn")
    for ax in axes:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.suptitle(f"Manning's n: current (NLCD) vs NWI-refined, {region}")
    fig.tight_layout()
    fig.savefig(out_dir / f"manning_n_maps_{tag}.png", dpi=150)
    plt.close(fig)

    # histogram / violin of value distributions
    fig, ax = plt.subplots(figsize=(7, 5))
    cur_vals = n_current_c[land]
    new_vals = n_new_c[land]
    ax.hist(cur_vals, bins=40, alpha=0.5, label="current (NLCD)", color="tab:blue", density=True)
    ax.hist(new_vals, bins=40, alpha=0.5, label="NWI-refined", color="tab:orange", density=True)
    ax.set_xlabel("Manning's n"); ax.set_ylabel("density")
    ax.set_title(f"Manning's n distribution over land cells, {region}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"manning_n_histogram_{tag}.png", dpi=150)
    plt.close(fig)

    print(f"Manning ({region}): {n_frac_changed:.1f}% of land cells changed by NWI overlay; "
          f"current median n={np.nanmedian(cur_vals):.3f}, new median n={np.nanmedian(new_vals):.3f}")
    return n_frac_changed, np.nanmedian(cur_vals), np.nanmedian(new_vals)


def read_asc(path):
    with rasterio.open(path) as d:
        arr = d.read(1).astype("float64")
        nod = d.nodata if d.nodata is not None else -9999
        arr = np.where(arr <= -9990, np.nan, arr)
        return arr, d.transform


def fig_ksat_awc(full_domain=False):
    ksat_cur, tr = read_asc(CUR_INFIL)     # mm/hr, POLARIS
    awc_cur_mm, _ = read_asc(CUR_INFILCAP)  # mm (depth-integrated capacity), POLARIS

    soils_path = FULL_SOILS_GEOJSON if full_domain else SOILS_GEOJSON
    ssurgo_path = FULL_SSURGO_TABLE if full_domain else SSURGO_TABLE
    soils = json.load(open(soils_path))
    table = json.load(open(ssurgo_path))
    shapes_ksat = []
    shapes_awc = []       # plant-available water (awc_r), NOT comparable to infilcap - kept only as an annotation series
    shapes_satstor = []    # sat_storage_mm = (theta_s - theta_1500kPa)*50cm, the apples-to-apples match to POLARIS infilcap
    n_missing_satstor = 0
    for f in soils["features"]:
        rec = table.get(f["properties"].get("MUKEY"))
        if rec is None:
            continue
        shapes_ksat.append((f["geometry"], rec["ksat_r_mm_hr"]))
        shapes_awc.append((f["geometry"], rec["awc_r"]))  # fraction; plant-available water only
        if rec.get("sat_storage_mm") is not None:
            shapes_satstor.append((f["geometry"], rec["sat_storage_mm"]))
        else:
            n_missing_satstor += 1

    ksat_ssurgo = rasterize(shapes_ksat, out_shape=ksat_cur.shape, transform=tr,
                             fill=np.nan, dtype="float64") if shapes_ksat else None
    awc_frac_ssurgo = rasterize(shapes_awc, out_shape=ksat_cur.shape, transform=tr,
                                 fill=np.nan, dtype="float64") if shapes_awc else None
    # convert SSURGO AWC fraction -> mm over the same ~50cm profile depth used by
    # POLARIS's make_capacity (thickness-weighted 0-50cm). NOTE: awc_r is
    # plant-available water (field_capacity - wilting_point), a much smaller
    # quantity than infilcap's (theta_s - theta_r) drainable-porosity storage.
    # It is retained below only as a separate, clearly-labeled reference series.
    awc_mm_ssurgo = awc_frac_ssurgo * 500.0 if awc_frac_ssurgo is not None else None  # 500 mm = 50 cm

    # sat_storage_mm: the corrected, apples-to-apples SSURGO analog of POLARIS infilcap
    satstor_ssurgo = rasterize(shapes_satstor, out_shape=ksat_cur.shape, transform=tr,
                                fill=np.nan, dtype="float64") if shapes_satstor else None
    if n_missing_satstor:
        print(f"AWC: {n_missing_satstor} MUKEYs missing wsatiated_r/wfifteenbar_r; "
              f"sat_storage_mm left as nodata there (fallback: excluded from raster)")

    if full_domain:
        ext = (tr.c, tr.c + ksat_cur.shape[1] * tr.a, tr.f + ksat_cur.shape[0] * tr.e, tr.f)
        ksat_cur_c, awc_cur_c = ksat_cur, awc_cur_mm
        ksat_ssurgo_c, awc_ssurgo_c, satstor_ssurgo_c = ksat_ssurgo, awc_mm_ssurgo, satstor_ssurgo
        out_dir, tag, region = OUT_FULL, "full_domain", "full domain (Savannah)"
    else:
        ksat_cur_c, ext = crop_to_bbox(ksat_cur, tr, PP_BBOX)
        awc_cur_c, _ = crop_to_bbox(awc_cur_mm, tr, PP_BBOX)
        ksat_ssurgo_c, _ = crop_to_bbox(ksat_ssurgo, tr, PP_BBOX) if ksat_ssurgo is not None else (None, ext)
        awc_ssurgo_c, _ = crop_to_bbox(awc_mm_ssurgo, tr, PP_BBOX) if awc_mm_ssurgo is not None else (None, ext)
        satstor_ssurgo_c, _ = crop_to_bbox(satstor_ssurgo, tr, PP_BBOX) if satstor_ssurgo is not None else (None, ext)
        out_dir, tag, region = OUT, "current_vs_ssurgo", "Pin Point, GA"

    def plotarr(a):
        if a is None:
            return a
        return downsample_for_plot(a) if full_domain else a

    # --- Ksat maps + histogram ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = np.nanpercentile(ksat_cur_c, 98)
    im0 = axes[0].imshow(plotarr(ksat_cur_c), extent=ext, vmin=0, vmax=vmax, cmap="YlGnBu")
    axes[0].set_title("Current: POLARIS Ksat (mm/hr)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(plotarr(ksat_ssurgo_c), extent=ext, vmin=0, vmax=vmax, cmap="YlGnBu")
    axes[1].set_title("SSURGO Ksat (mm/hr, ksat_r)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    diff_full = ksat_ssurgo_c - ksat_cur_c
    dmax = np.nanpercentile(np.abs(diff_full), 98) if np.any(~np.isnan(diff_full)) else 1
    im2 = axes[2].imshow(plotarr(diff_full), extent=ext, vmin=-dmax, vmax=dmax, cmap="RdBu_r")
    axes[2].set_title("Difference (SSURGO - POLARIS)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.suptitle(f"Ksat: current (POLARIS) vs SSURGO, {region}")
    fig.tight_layout()
    fig.savefig(out_dir / f"ksat_maps_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ksat_cur_c[~np.isnan(ksat_cur_c)].ravel(), bins=40, alpha=0.5,
            label="current (POLARIS)", color="tab:blue", density=True)
    ax.hist(ksat_ssurgo_c[~np.isnan(ksat_ssurgo_c)].ravel(), bins=40, alpha=0.5,
            label="SSURGO", color="tab:green", density=True)
    ax.set_xlabel("Ksat (mm/hr)"); ax.set_ylabel("density")
    ax.set_title(f"Ksat distribution, {region}: POLARIS vs SSURGO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"ksat_histogram_{tag}.png", dpi=150)
    plt.close(fig)

    # --- AWC / storage-capacity maps + histogram ---
    # Headline comparison: POLARIS infilcap (theta_s - theta_r)*50cm vs the
    # CORRECTED SSURGO analog sat_storage_mm = (wsatiated_r - wfifteenbar_r)*50cm.
    # The old awc_r-based comparison mixed two different soil properties (see
    # reports/physics/README.md) and is retained below only as a separate,
    # clearly-labeled reference series (plant-available water, not storage capacity).
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = np.nanpercentile(awc_cur_c, 98)
    im0 = axes[0].imshow(plotarr(awc_cur_c), extent=ext, vmin=0, vmax=vmax, cmap="BuPu")
    axes[0].set_title("Current: POLARIS infilcap\n(theta_s - theta_r) x 50cm, mm")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(plotarr(satstor_ssurgo_c), extent=ext, vmin=0, vmax=vmax, cmap="BuPu")
    axes[1].set_title("SSURGO sat_storage_mm (corrected)\n(wsatiated_r - wfifteenbar_r) x 50cm")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    diffA_full = satstor_ssurgo_c - awc_cur_c
    dmaxA = np.nanpercentile(np.abs(diffA_full), 98) if np.any(~np.isnan(diffA_full)) else 1
    im2 = axes[2].imshow(plotarr(diffA_full), extent=ext, vmin=-dmaxA, vmax=dmaxA, cmap="RdBu_r")
    axes[2].set_title("Difference (SSURGO sat_storage - POLARIS infilcap)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.suptitle(f"Soil storage capacity: POLARIS infilcap vs corrected SSURGO sat_storage_mm, {region}")
    fig.tight_layout()
    fig.savefig(out_dir / f"awc_maps_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(awc_cur_c[~np.isnan(awc_cur_c)].ravel(), bins=40, alpha=0.5,
            label="POLARIS infilcap (theta_s - theta_r)", color="tab:blue", density=True)
    ax.hist(satstor_ssurgo_c[~np.isnan(satstor_ssurgo_c)].ravel(), bins=40, alpha=0.5,
            label="SSURGO sat_storage_mm (corrected)", color="tab:purple", density=True)
    ax.hist(awc_ssurgo_c[~np.isnan(awc_ssurgo_c)].ravel(), bins=40, alpha=0.35,
            label="SSURGO awc_r (plant-available water, NOT comparable)", color="tab:red",
            density=True, hatch="//")
    ax.set_xlabel("Storage capacity (mm, 0-50cm)"); ax.set_ylabel("density")
    ax.set_title(f"Storage capacity distribution, {region}:\nPOLARIS infilcap vs SSURGO (corrected sat_storage_mm and old awc_r)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"awc_histogram_{tag}.png", dpi=150)
    plt.close(fig)

    def s(a):
        v = a[~np.isnan(a)]
        return (np.nanmedian(v), np.nanmin(v), np.nanmax(v)) if v.size else (np.nan,) * 3

    ksat_cur_med = s(ksat_cur_c); ksat_new_med = s(ksat_ssurgo_c)
    awc_cur_med = s(awc_cur_c); awc_new_med = s(satstor_ssurgo_c)
    awc_r_med = s(awc_ssurgo_c)
    print(f"Ksat ({region}): current median {ksat_cur_med[0]:.1f} mm/hr [{ksat_cur_med[1]:.1f}-{ksat_cur_med[2]:.1f}]; "
          f"SSURGO median {ksat_new_med[0]:.1f} mm/hr [{ksat_new_med[1]:.1f}-{ksat_new_med[2]:.1f}]")
    print(f"AWC/storage ({region}): POLARIS infilcap median {awc_cur_med[0]:.1f} mm [{awc_cur_med[1]:.1f}-{awc_cur_med[2]:.1f}]; "
          f"SSURGO sat_storage_mm (corrected) median {awc_new_med[0]:.1f} mm [{awc_new_med[1]:.1f}-{awc_new_med[2]:.1f}]; "
          f"(old, non-comparable SSURGO awc_r median {awc_r_med[0]:.1f} mm plant-available water)")
    return ksat_cur_med, ksat_new_med, awc_cur_med, awc_new_med


if __name__ == "__main__":
    import sys as _sys
    full = "--full-domain" in _sys.argv or "--full" in _sys.argv
    pin_point = "--full-domain" not in _sys.argv and "--full" not in _sys.argv or "--both" in _sys.argv

    if pin_point:
        print("=== Manning's n (Pin Point) ===")
        fig_manning(full_domain=False)
        print("\n=== Ksat / AWC (Pin Point) ===")
        fig_ksat_awc(full_domain=False)
        print(f"\nPin Point figures written to {OUT}")

    if full:
        print("\n=== Manning's n (full domain) ===")
        fig_manning(full_domain=True)
        print("\n=== Ksat / AWC (full domain) ===")
        fig_ksat_awc(full_domain=True)
        print(f"\nfull-domain figures written to {OUT_FULL}")
