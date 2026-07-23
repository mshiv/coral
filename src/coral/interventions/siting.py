"""Targeted adaptation siting: rank cells by dataset-driven suitability.

The random placement in generate.patch_mask gives emulator-training variety; suitability_score
instead gives a continuous 0-1 score per cell within each adaptation's zone for a realistic
decision scenario, so placement can target where the adaptation would actually be built and be
effective (flood exposure, tidal frame, what is being protected) rather than a random field.
generate.apply_intervention(place="targeted") places at the top-scoring cells.

Drivers per kind (see the plan): marsh uses migration space adjacent to NWI marsh within the
tidal frame; living_shoreline uses the marsh-water edge; depave uses flooded/upstream impervious
on permeable soil; retreat uses buildings with the deepest modeled flooding in the flood zone;
seawall uses shoreline seaward of flooded buildings at low crest.

Reuses hydrology (flow_accum/catchment/hydraulic_connectivity) and scipy.ndimage. Tidal-frame
elevations are in the DEM datum (NAVD88): pass mhw/mlw from NOAA datums (make_tide.fetch_datums).
"""
from __future__ import annotations
import numpy as np


def _norm(a):
    """Scale finite values to [0,1]; non-finite -> 0."""
    a = np.where(np.isfinite(a), a, np.nan)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or hi <= lo:
        return np.nan_to_num(a * 0.0)
    return np.nan_to_num((a - lo) / (hi - lo))


def suitability_score(dem, kind, *, sea_level=0.81, wetlands=None, buildings=None,
                      flood_depth=None, flood_zone=None, classes=None, soil_ksat=None,
                      focus=None, mhw=0.94, mlw=-1.17, slr_buffer=0.5):
    """Per-cell suitability in [0,1] for `kind`; 0 outside its zone. Higher = better target."""
    from scipy import ndimage
    land = np.isfinite(dem) & (dem > sea_level)
    sea = np.isfinite(dem) & (dem <= sea_level)
    s = np.zeros(dem.shape, "float64")
    NLCD_DEV = (21, 22, 23, 24); NLCD_IMP = (22, 23, 24)

    if kind == "marsh":
        # migration space: land landward-adjacent to existing marsh, within [MHW, MHW+SLR]
        band = land & (dem >= mhw) & (dem <= mhw + slr_buffer)
        if classes is not None:
            band = band & ~np.isin(classes, NLCD_DEV)
        if buildings is not None:
            band = band & ~buildings
        if wetlands is not None and wetlands.any():
            dist = ndimage.distance_transform_edt(~wetlands)      # cells from existing marsh
            prox = 1.0 / (1.0 + dist)                             # nearer marsh = higher
        else:
            prox = np.ones(dem.shape)
        low = _norm(-(dem))                                       # lower land preferred
        s = np.where(band, 0.6 * _norm(prox) + 0.4 * low, 0.0)

    elif kind == "living_shoreline":
        # the marsh-water edge: NWI marsh cells touching open water
        if wetlands is None:
            return s
        edge = wetlands & ndimage.binary_dilation(sea, iterations=1)
        exposure = ndimage.uniform_filter(sea.astype(float), size=5)  # open-water frontage
        s = np.where(edge, _norm(exposure) + 0.1, 0.0)

    elif kind == "depave":
        zone = (land & np.isin(classes, NLCD_IMP)) if classes is not None else (land & (dem > sea_level + 0.5))
        drive = np.zeros(dem.shape)
        if flood_depth is not None:
            drive = _norm(flood_depth) + _norm(ndimage.maximum_filter(flood_depth, size=9))  # floods or borders flooding
        if soil_ksat is not None:
            drive = drive * (0.5 + 0.5 * _norm(np.where(np.isfinite(soil_ksat), soil_ksat, 0)))
        s = np.where(zone, _norm(drive) + 0.05, 0.0)

    elif kind == "retreat":
        zone = (land & buildings) if buildings is not None else (land & np.isin(classes, NLCD_DEV) if classes is not None else land)
        if flood_zone is not None:
            zone = zone & flood_zone
        if flood_depth is not None:
            s = np.where(zone, _norm(flood_depth) + 0.05, 0.0)    # deepest-flooded buildings first
        else:
            s = np.where(zone, _norm(-dem), 0.0)

    elif kind == "seawall":
        shore = ndimage.binary_dilation(sea, iterations=2) & land
        landward_flood = ndimage.maximum_filter(flood_depth, size=15) if flood_depth is not None else np.zeros(dem.shape)
        lowcrest = _norm(-(dem))                                  # low points overtop first
        s = np.where(shore, 0.6 * _norm(landward_flood) + 0.4 * lowcrest, 0.0)

    else:
        s = land.astype(float)

    if focus is not None:
        s = np.where(focus, s, 0.0)
    return s


def targeted_mask(dem, kind, area_frac, *, close_iter=1, **drivers):
    """Boolean placement mask: the top `area_frac` of suitability_score, smoothed for
    contiguity. Drop-in alternative to generate.patch_mask for realistic scenarios."""
    from scipy import ndimage
    s = suitability_score(dem, kind, **drivers)
    pos = s[s > 0]
    if pos.size == 0:
        return np.zeros(dem.shape, bool)
    thr = np.quantile(pos, max(0.0, 1.0 - area_frac))
    m = s >= max(thr, np.nextafter(0, 1))
    if close_iter:
        m = ndimage.binary_closing(m, iterations=int(close_iter))
    return m & (s > 0)
