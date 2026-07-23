"""Geostatistical intervention generator: low-dim knobs -> grid edits.

random_field / patch_mask draw a spatially-correlated field (Gaussian random field
via spectral synthesis) and threshold it to a patchy placement mask. Each
intervention consumes a mask + a few parameters and edits (DEM, Manning n, Ksat,
AWC) in place-safe copies. sample_intervention draws random-but-realistic configs
for the training sweep; apply_intervention expands one config onto the grids.

Deps: numpy, scipy (ndimage).
"""
from __future__ import annotations
import numpy as np


def random_field(shape, corr_len=30.0, seed=0):
    """Unit-variance Gaussian random field with ~`corr_len`-cell correlation length,
    via FFT spectral synthesis (isotropic Gaussian covariance)."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    white = rng.standard_normal((ny, nx))
    ky = np.fft.fftfreq(ny)[:, None]; kx = np.fft.fftfreq(nx)[None, :]
    k2 = kx**2 + ky**2
    filt = np.exp(-2 * (np.pi * corr_len) ** 2 * k2)      # Gaussian spectral filter
    f = np.fft.ifft2(np.fft.fft2(white) * np.sqrt(filt)).real
    return (f - f.mean()) / (f.std() + 1e-9)


# NLCD class groups for land-cover-conditioned siting (from make_manning.NLCD_N)
NLCD_WETLAND = (90, 95)                 # woody + emergent wetland (existing marsh)
NLCD_DEVELOPED = (21, 22, 23, 24)       # developed (open -> high intensity)
NLCD_IMPERVIOUS = (22, 23, 24)          # parking lots / roads / dense development


def suitability_mask(dem, sea_level=0.81, *, kind="marsh", classes=None, focus=None,
                     elev_band=2.0, near_water_cells=15, wetlands=None, buildings=None):
    """Physically- and contextually-suitable zone for an intervention.

    - marsh/mangrove: real marsh footprint. If a `wetlands` mask (from NWI, via
      context_rasters.wetlands_mask) is given, restrict to it (marsh cells within real
      wetlands); marsh is intertidal so the elevation/land test is dropped there.
      Else fall back to low intertidal land near water, optionally adjacent to NLCD
      wetland classes.
    - retreat: built land. Real building footprints (`buildings` mask from FEMA USA
      Structures) intersect developed, else NLCD developed, else elevation fallback.
    - permeable: developed land; depave: impervious/parking land.
    `focus` (bool mask, e.g. the Pin Point neighbourhood from focus_region) intersects
    the result so interventions sit around the focal community (up/downstream).
    Without `classes`/`wetlands`/`buildings`, falls back to elevation-only heuristics.
    """
    from scipy import ndimage
    land = np.isfinite(dem) & (dem > sea_level)
    sea = np.isfinite(dem) & (dem <= sea_level)
    if kind in ("marsh", "mangrove"):
        if wetlands is not None:                         # real NWI marsh footprint
            z = np.isfinite(dem) & wetlands
        else:
            z = (land & (dem <= sea_level + elev_band)
                 & ndimage.binary_dilation(sea, iterations=int(near_water_cells)))
            if classes is not None:                      # extend existing marsh
                wet = np.isin(classes, NLCD_WETLAND)
                z = z & ndimage.binary_dilation(wet, iterations=int(near_water_cells))
    elif kind == "retreat":
        if buildings is not None:                        # real footprints (FEMA)
            z = land & buildings
            if classes is not None:
                z = z & np.isin(classes, NLCD_DEVELOPED)
        elif classes is not None:
            z = land & np.isin(classes, NLCD_DEVELOPED)
        else:
            z = land & (dem > sea_level + 0.5)
    elif kind == "permeable":
        z = (land & np.isin(classes, NLCD_DEVELOPED)) if classes is not None \
            else (land & (dem > sea_level + 0.5))
    elif kind == "depave":
        z = (land & np.isin(classes, NLCD_IMPERVIOUS)) if classes is not None \
            else (land & (dem > sea_level + 0.5))
    else:
        z = land
    if focus is not None:
        z = z & focus
    return z


def focus_region(dem_shape, ext, center_lonlat, radius_km):
    """Circular ~radius_km focus mask around center_lonlat (e.g. Pin Point) on the DEM
    grid — restricts interventions to the focal community's neighbourhood. `ext` is
    [W,E,S,N]. A first cut for 'up/downstream of Pin Point'; a catchment-based version
    (D8 routing on the DEM) is the hydrological refinement."""
    ny, nx = dem_shape
    W, E, S, N = ext
    lon = W + (np.arange(nx) + 0.5) * (E - W) / nx
    lat = N - (np.arange(ny) + 0.5) * (N - S) / ny        # row 0 = north
    clon, clat = center_lonlat
    dx = (lon[None, :] - clon) * 111.0 * np.cos(np.radians(clat))
    dy = (lat[:, None] - clat) * 111.0
    return (dx ** 2 + dy ** 2) <= radius_km ** 2


def patch_mask(shape, corr_len=30.0, area_frac=0.15, seed=0, restrict=None):
    """Boolean placement mask covering ~`area_frac` of `restrict` (or the whole
    grid), as spatially-coherent patches from a thresholded random field."""
    f = random_field(shape, corr_len, seed)
    region = np.ones(shape, bool) if restrict is None else restrict
    vals = f[region]
    if vals.size == 0:
        return np.zeros(shape, bool)
    thr = np.quantile(vals, 1.0 - area_frac)              # top area_frac of the field
    return (f >= thr) & region


# intervention registry: kind -> (knob ranges) used by sample_intervention
INTERVENTIONS = {
    "seawall":   {"crest_m": (2.0, 4.5), "buffer_cells": (1, 3)},
    "marsh":     {"area_frac": (0.05, 0.30), "n_target": (0.10, 0.15),
                  "ksat_add": (10, 40), "awc_add": (50, 150), "corr_len": (15, 50)},
    "mangrove":  {"area_frac": (0.03, 0.15), "n_target": (0.15, 0.30),
                  "ksat_add": (5, 20), "awc_add": (30, 100), "corr_len": (10, 40)},
    "permeable": {"area_frac": (0.05, 0.25), "ksat_rate": (20, 60), "corr_len": (10, 40)},
    "retreat":   {"area_frac": (0.02, 0.12), "natural_n": (0.035, 0.05), "corr_len": (10, 30)},
    "depave":    {"area_frac": (0.10, 0.50), "n_target": (0.06, 0.12),   # parking/impervious
                  "ksat_rate": (20, 50), "awc_add": (30, 100), "corr_len": (8, 25)},  # -> vegetated
}


def sample_intervention(kind, rng):
    """Draw one config (dict of scalar knobs) for `kind` from its ranges."""
    if kind not in INTERVENTIONS:
        raise ValueError(f"unknown intervention {kind!r}; choose {list(INTERVENTIONS)}")
    knobs = {"kind": kind, "seed": int(rng.integers(1 << 30))}
    for k, (lo, hi) in INTERVENTIONS[kind].items():
        knobs[k] = (int(rng.integers(lo, hi + 1)) if isinstance(lo, int)
                    else round(float(rng.uniform(lo, hi)), 4))
    return knobs


def apply_intervention(knobs, dem, manning, ksat, awc, *, sea_level=0.81,
                       classes=None, focus=None, wetlands=None, soil_ksat=None,
                       buildings=None):
    """Expand one config onto copies of the grids. `classes` = optional NLCD grid.
    SAGIS-conditioned siting (context_rasters.py): `wetlands` (NWI mask, so marsh sits
    on real marsh), `buildings` (FEMA footprints, so retreat acts on real structures),
    `soil_ksat` (SSURGO Ksat grid, mm/hr, so de-pave/permeable can't exceed what the soil
    physically allows). `focus` = optional bool mask (Pin Point neighbourhood). Returns
    (dem, manning, ksat, awc, intensity); intensity marks the edited cells."""
    from scipy import ndimage
    dem, manning = dem.copy(), manning.copy()
    ksat, awc = ksat.copy(), awc.copy()
    land = np.isfinite(dem) & (dem > sea_level)
    sea = np.isfinite(dem) & (dem <= sea_level)
    kind = knobs["kind"]
    intensity = np.zeros(dem.shape, "float32")

    def soil_capped(rate):
        """Achievable Ksat = min(target rate, local SSURGO Ksat); uncapped where no soil."""
        if soil_ksat is None:
            return np.full(dem.shape, rate, "float64")
        cap = np.where(np.isfinite(soil_ksat), soil_ksat, rate)
        return np.minimum(rate, cap)

    def zone(k):
        return suitability_mask(dem, sea_level, kind=k, classes=classes, focus=focus,
                                elev_band=knobs.get("elev_band", 2.0),
                                near_water_cells=knobs.get("near_water_cells", 15),
                                wetlands=wetlands, buildings=buildings)

    def patch(k):
        return patch_mask(dem.shape, knobs["corr_len"], knobs["area_frac"],
                          knobs["seed"], restrict=zone(k))

    if kind == "seawall":
        shore = ndimage.binary_dilation(sea, iterations=int(knobs["buffer_cells"])) & land
        if focus is not None:
            shore = shore & focus
        dem[shore] = np.maximum(dem[shore], knobs["crest_m"])
        intensity[shore] = 1.0

    elif kind in ("marsh", "mangrove"):
        m = patch(kind)                                  # intertidal + adjacent to existing marsh
        manning[m] = np.maximum(manning[m], knobs["n_target"])
        ksat[m] = ksat[m] + knobs["ksat_add"]; awc[m] = awc[m] + knobs["awc_add"]
        intensity[m] = 1.0

    elif kind == "depave":                               # parking/impervious -> vegetated
        m = patch("depave")
        cap = soil_capped(knobs["ksat_rate"])            # SSURGO-limited achievable rate
        manning[m] = np.maximum(manning[m], knobs["n_target"])
        ksat[m] = np.maximum(ksat[m], cap[m]); awc[m] = awc[m] + knobs["awc_add"]
        intensity[m] = 1.0

    elif kind == "permeable":
        m = patch("permeable")
        cap = soil_capped(knobs["ksat_rate"])            # SSURGO-limited achievable rate
        ksat[m] = np.maximum(ksat[m], cap[m])
        intensity[m] = 1.0

    elif kind == "retreat":
        m = patch("retreat")
        base = np.where(land & ~m, dem, np.nan)
        med = np.nanmedian(base) if np.isfinite(base).any() else np.nanmedian(dem[land])
        dem[m] = np.minimum(dem[m], med); manning[m] = knobs["natural_n"]
        intensity[m] = 1.0

    else:
        raise ValueError(f"unknown intervention {kind!r}")

    return dem, manning, ksat, awc, intensity
