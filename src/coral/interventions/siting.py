"""Targeted adaptation siting: rank cells by dataset-driven suitability.

The random placement in generate.patch_mask gives emulator-training variety; suitability_score
instead gives a continuous 0-1 score per cell within each adaptation's zone for a realistic
decision scenario, so placement can target where the adaptation would actually be built and be
effective (flood exposure, tidal frame, what is being protected) rather than a random field.
generate.apply_intervention(place="targeted") places at the top-scoring cells.

Drivers per kind (see the plan): marsh uses migration space adjacent to NWI marsh within the
tidal frame; living_shoreline uses the marsh-water edge; depave uses flooded/upstream impervious
on suitable soil; retreat uses buildings with the deepest modeled flooding in the flood zone;
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
    # A relative tolerance, not hi <= lo. Once the water level passes MHW the whole restoration
    # band is sea, so uniform_filter returns a near-constant exposure and hi - lo is float64
    # rounding residue. Dividing by it stretched that residue across [0,1], and because the
    # filter accumulates along rows first the residue carries row structure, which rank selection
    # then followed: horizontal banding in the coverage figure, 8.5x row-to-column anisotropy at
    # High2050 against 1.85 at slr0.0. A driver this flat carries no information, so return a
    # constant and let the seeded tiebreak place the cells.
    if not np.isfinite(lo) or (hi - lo) <= 1e-12 * max(1.0, abs(hi), abs(lo)):
        return np.nan_to_num(a * 0.0)
    return np.nan_to_num((a - lo) / (hi - lo))


def suitability_score(dem, kind, *, sea_level=None, wetlands=None, buildings=None,
                      roads=None, flood_depth=None, flood_zone=None, classes=None,
                      soil_ksat=None, focus=None, mhw=None, mlw=None, slr_buffer=0.5,
                      res_m=30.0):
    """Per-cell suitability in [0,1] for `kind`; 0 outside its zone. Higher = better target.

    sea_level, mhw and mlw have no defaults on purpose. They previously defaulted to 0.81 and the
    published 1983-2001 datums, both superseded: 0.81 has no derivation on record and the
    published epoch is centred on 1992. Pass the scenario's datums.
    """
    if sea_level is None or mhw is None or mlw is None:
        raise ValueError("suitability_score needs sea_level, mhw and mlw from the scenario "
                         "datums; they no longer fall back to the superseded epoch")
    from scipy import ndimage
    def _win(metres, minimum=3):        # metres to an odd cell window for the filters below
        c = max(minimum, int(round(metres / res_m)))
        return c + 1 - c % 2
    land = np.isfinite(dem) & (dem > sea_level)
    sea = np.isfinite(dem) & (dem <= sea_level)
    s = np.zeros(dem.shape, "float64")
    NLCD_DEV = (21, 22, 23, 24); NLCD_IMP = (22, 23, 24)

    if kind == "marsh_restoration":
        # The existing platform, MLW to MHW. Rank by exposure to open water, since the marsh that
        # does the most work against a surge is the marsh the surge has to cross.
        band = np.isfinite(dem) & (dem >= mlw) & (dem <= mhw)
        if wetlands is not None:
            band = band & wetlands
        exposure = ndimage.uniform_filter(sea.astype(float), size=_win(300.0))
        s = np.where(band, _norm(exposure) + 0.1, 0.0)

    elif kind == "marsh_migration":
        # Migration space: land landward-adjacent to existing marsh, within [MHW, MHW+SLR].
        # slr_buffer must be the member's own rise, not a constant. Migration space is defined
        # by how far the tide advances, so a fixed 0.5 m gave a Low2050 member and a High2100
        # member the same corridor.
        #
        # No `land` gate. The corridor runs from present MHW to future MHW, so cells inside it
        # are meant to sit below the member's raised water level; that is what makes them
        # migration space. Gating on dem > sea_level put the floor at the raised waterline while
        # the ceiling stayed at mhw + rise, which is the same elevation, and the band closed to a
        # few millimetres. Every targeted member above slr0.0 sited nothing. suitability_mask,
        # which the random path uses, never had the gate, so only targeted members were affected.
        band = np.isfinite(dem) & (dem >= mhw) & (dem <= mhw + slr_buffer)
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
        exposure = ndimage.uniform_filter(sea.astype(float), size=_win(150.0))  # open-water frontage
        s = np.where(edge, _norm(exposure) + 0.1, 0.0)

    elif kind == "depave":
        zone = (land & np.isin(classes, NLCD_IMP)) if classes is not None else (land & (dem > sea_level + 0.5))
        drive = np.zeros(dem.shape)
        if flood_depth is not None:
            drive = _norm(flood_depth) + _norm(ndimage.maximum_filter(flood_depth, size=_win(270.0)))  # floods or borders flooding
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

    elif kind == "road_raise":
        # Road cells the modelled peak reaches. Raising a road that never floods buys nothing,
        # and without a road layer there is no alignment to raise, so score nothing.
        if roads is None:
            return s
        band = roads & np.isfinite(dem)
        if flood_depth is not None:
            wet = np.isfinite(flood_depth) & (flood_depth > 0.05)
            band = band & wet
            s = np.where(band, _norm(np.where(band, flood_depth, 0.0)), 0.0)
        else:
            s = np.where(band, _norm(-(dem)), 0.0)      # lowest road first

    elif kind in ("floodwall", "seawall"):
        shore = ndimage.binary_dilation(sea, iterations=max(1,int(round(60.0/res_m)))) & land
        landward_flood = ndimage.maximum_filter(flood_depth, size=_win(450.0)) if flood_depth is not None else np.zeros(dem.shape)
        lowcrest = _norm(-(dem))                                  # low points overtop first
        s = np.where(shore, 0.6 * _norm(landward_flood) + 0.4 * lowcrest, 0.0)

    else:
        s = land.astype(float)

    if focus is not None:
        s = np.where(focus, s, 0.0)
    return s


def intertidal_mask(dem, *, mlw=None, mhw=None, slr_buffer=0.5, nlcd=None, nwi=None):
    """Intertidal band from the tidal frame. mlw and mhw are required; see suitability_score."""
    if mlw is None or mhw is None:
        raise ValueError("intertidal_mask needs mlw and mhw from the scenario datums")
    """Marsh footprint from the tidal frame, filtered by land cover.

    Neither land-cover source works alone here. NWI maps 2.4% of the Pin Point clip as
    estuarine wetland and NLCD maps 52.5%, a twentyfold disagreement, so NWI under-maps and
    NLCD over-maps. The DEM is the better measurement at 4 m, so the tidal band sets the extent
    and land cover only says which of those cells carry vegetation.
    """
    band = np.isfinite(dem) & (dem >= mlw) & (dem <= mhw + slr_buffer)
    veg = None
    if nlcd is not None:
        veg = np.isin(np.round(nlcd), (90, 95))          # woody and emergent herbaceous wetland
    if nwi is not None:
        veg = nwi if veg is None else (veg | nwi)
    return band if veg is None else (band & veg)


def targeted_mask(dem, kind, area_frac, *, close_iter=1, seed=0, **drivers):
    """Boolean placement mask: the top `area_frac` of suitability_score, smoothed for
    contiguity. Drop-in alternative to generate.patch_mask for realistic scenarios.

    Selection takes the highest-scoring cells by rank, with a seeded tiebreak. A quantile
    threshold does not work here because several scores are flat over most of their zone: the
    de-pave driver comes from baseline peak depth, and the impervious footprint is mostly dry,
    so the score sits at its constant floor. Every cell then ties, the threshold lands on that
    constant, and `s >= thr` returns the whole zone whatever area_frac says. Measured across 86
    targeted members, footprint against area_frac gave Spearman +0.007: the knob controlled
    nothing. Ties also made the selection follow score level-sets, which is the horizontal
    banding visible in the coverage figure.

    Closing runs after selection and can add a few percent to the count, so the footprint is
    close to `area_frac` of the zone rather than exactly it.
    """
    from scipy import ndimage
    s = suitability_score(dem, kind, **drivers)
    flat = s.ravel()
    pos = np.flatnonzero(flat > 0)
    if pos.size == 0:
        return np.zeros(dem.shape, bool)
    n = int(min(pos.size, max(1, round(float(area_frac) * pos.size))))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(pos.size), -flat[pos]))   # score first, random breaks ties
    m = np.zeros(flat.size, bool)
    m[pos[order[:n]]] = True
    m = m.reshape(s.shape)
    if close_iter:
        m = ndimage.binary_closing(m, iterations=int(close_iter))
    return m & (s > 0)
