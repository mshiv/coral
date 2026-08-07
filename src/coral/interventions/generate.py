"""Geostatistical intervention generator: turns low-dim knobs into grid edits.

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
NLCD_DEVELOPED = (21, 22, 23, 24)       # developed, open to high intensity
NLCD_IMPERVIOUS = (22, 23, 24)          # parking lots / roads / dense development


def suitability_mask(dem, sea_level=0.81, *, kind="marsh", classes=None, focus=None,
                     elev_band=2.0, near_water_m=450.0, wetlands=None, buildings=None,
                     res_m=30.0):
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
    Falls back to elevation-only heuristics when classes, wetlands, and buildings are all unset.
    """
    from scipy import ndimage
    near_water_cells = max(1, int(round(near_water_m / res_m)))   # metres to cells
    land = np.isfinite(dem) & (dem > sea_level)
    sea = np.isfinite(dem) & (dem <= sea_level)
    if kind in ("marsh", "mangrove"):
        if wetlands is not None:                         # real NWI marsh footprint
            z = np.isfinite(dem) & wetlands
        else:
            z = (land & (dem <= sea_level + elev_band)
                 & ndimage.binary_dilation(sea, iterations=near_water_cells))
            if classes is not None:                      # extend existing marsh
                wet = np.isin(classes, NLCD_WETLAND)
                z = z & ndimage.binary_dilation(wet, iterations=near_water_cells)
    elif kind == "retreat":
        # Developed land is the constraint; footprints only refine within it. Gating on FEMA
        # coverage made a missing polygon permanently ineligible, which is an error that never
        # shows up in the output. Buildings stay a metric (structures removed), not a gate.
        if classes is not None:
            z = land & np.isin(classes, NLCD_DEVELOPED)
            if buildings is not None:
                zb = z & ndimage.binary_dilation(buildings, iterations=2)
                if zb.sum() >= 0.05 * max(z.sum(), 1):   # only if it leaves a usable zone
                    z = zb
        elif buildings is not None:
            z = land & buildings
        else:
            z = land & (dem > sea_level + 0.5)
    elif kind == "living_shoreline":                     # marsh-water edge
        z = (wetlands & ndimage.binary_dilation(sea, iterations=1)) if wetlands is not None \
            else (sea & ndimage.binary_dilation(land, iterations=1))
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


# intervention registry: kind maps to knob ranges used by sample_intervention

def shoreline_contour(dem, sea_level=0.81, min_len=50):
    """Shoreline polylines: contours of the DEM at the water level, longest first.

    A seawall is a linear structure on an alignment, not a band of raised ground. Extracting the
    zero-level contour gives that alignment from the terrain itself, and it is also what a user
    draws in the tool: a line along the shore, of some length, at some crest height.

    Returns a list of (row, col) float arrays in grid coordinates.
    """
    from scipy import ndimage
    z = np.where(np.isfinite(dem), dem, sea_level - 10.0)
    try:
        from skimage import measure
        cs = [c for c in measure.find_contours(z, sea_level) if len(c) >= min_len]
    except ImportError:
        # Fallback with no scikit-image: the land/sea boundary cells, ordered by walking
        # nearest-neighbour. Coarser than marching squares but it gives the same alignment.
        land = z > sea_level
        edge = land & ~ndimage.binary_erosion(land)
        pts = np.argwhere(edge).astype(float)
        if len(pts) < min_len:
            return []
        order, used = [0], {0}
        while len(order) < len(pts):
            d = np.linalg.norm(pts - pts[order[-1]], axis=1)
            d[list(used)] = np.inf
            k = int(np.argmin(d))
            if not np.isfinite(d[k]):
                break
            order.append(k); used.add(k)
        cs = [pts[order]]
    return sorted(cs, key=len, reverse=True)


def _extend_to_tie_in(arc, run, dem, tie_elev, max_extend_cells=200):
    """Grow an alignment along its contour until both ends reach ground above `tie_elev`.

    A floodwall only works if it ties into high ground at each end. With open ends water flanks it
    and floods the same area slightly later, however long the wall is. This is why length is the
    wrong criterion: a 30 m closure between two rises works, a 500 m wall with both ends in the
    marsh does not.

    Extending rather than rejecting matches what an engineer does -- run the wall until it meets
    high ground -- and it keeps short closures, which are the cheap interventions worth finding.
    Returns (arc, tied) so a caller can discard alignments with no tie-in within reach.
    """
    if len(run) < 2:
        return arc, False
    nr, nc = dem.shape
    def high(pt, rad=6):
        """High ground NEAR the endpoint, not at it.

        The alignment follows an isoline, so every point on it sits at the same elevation and a
        test on the point itself can never pass. What matters is whether the end abuts ground the
        wall can tie into, which is a question about the neighbourhood.
        """
        r = int(np.clip(round(pt[0]), 0, nr - 1)); c = int(np.clip(round(pt[1]), 0, nc - 1))
        w = dem[max(r - rad, 0):r + rad + 1, max(c - rad, 0):c + rad + 1]
        return w.size > 0 and float(np.nanmax(w)) >= tie_elev
    # locate the arc inside its parent run
    i0 = int(np.argmin(np.hypot(*(run - arc[0]).T)))
    i1 = int(np.argmin(np.hypot(*(run - arc[-1]).T)))
    if i1 < i0:
        i0, i1 = i1, i0
    a, b, n = i0, i1, 0
    while a > 0 and not high(run[a]) and n < max_extend_cells:
        a -= 1; n += 1
    n = 0
    while b < len(run) - 1 and not high(run[b]) and n < max_extend_cells:
        b += 1; n += 1
    return run[a:b + 1], (high(run[a]) and high(run[b]))


def _span_cells(c):
    """End-to-end distance of a polyline, in cells.

    Span, not traced length, is what makes a wall useful: a closed islet outline can trace a long
    way and return to where it started, and simplification then collapses it to a stub. Filtering
    on span rejects loops and doubling-back, and keeps stretches that actually reach across
    something."""
    return float(np.hypot(*(c[-1] - c[0]))) if len(c) > 1 else 0.0


def _contiguous_runs(c, max_gap=2.0, min_len=20):
    """Split a contour into runs of consecutive, adjacent points.

    Filtering a contour by a mask keeps points but drops the ones between them, so what is left is
    not one stretch of shoreline: taking an arc from it joins segments that are far apart and
    produces walls running seaward across the marsh. Splitting on gaps keeps arcs physical.
    """
    if len(c) < 2:
        return []
    gaps = np.hypot(*np.diff(c, axis=0).T)
    idx = np.flatnonzero(gaps > max_gap) + 1
    return [r for r in np.split(c, idx) if len(r) >= min_len]


def shelters_floodable_land(arc, dem, shape, *, res_m, sea_level=0.81, reach_m=200.0,
                            band_m=2.0, min_cells=200):
    """Does this alignment have floodable land behind it?

    A wall must shelter something. This is the terrain-based analogue of the constraints the other
    interventions already use -- marsh on NWI wetland, depave on NLCD impervious -- and it keeps
    building footprints as a metric rather than a gate, so an incomplete structures inventory
    cannot make an area permanently ineligible.

    Floodable means above the waterline but within `band_m` of it: ground the surge can actually
    reach. Land well above that is not at risk and a wall in front of it protects nothing.
    """
    from scipy import ndimage
    m = np.zeros(shape, dtype=bool)
    rr = np.clip(arc[:, 0].round().astype(int), 0, shape[0] - 1)
    cc = np.clip(arc[:, 1].round().astype(int), 0, shape[1] - 1)
    m[rr, cc] = True
    reach = ndimage.binary_dilation(m, iterations=max(1, int(reach_m / res_m)))
    floodable = np.isfinite(dem) & (dem > sea_level) & (dem <= sea_level + band_m)
    return int((reach & floodable).sum()) >= min_cells, int((reach & floodable).sum())


def _simplify(arc, tol_cells=5.0):
    """Reduce a contour to a buildable polyline.

    Following a contour exactly produces alignments that double back into every creek inlet, which
    no one would build. Douglas-Peucker keeps the shape and drops the wiggle; the fallback drops
    vertices that reverse direction, which removes the worst of it without scikit-image.
    """
    if len(arc) < 3:
        return arc
    try:
        from skimage.measure import approximate_polygon
        out = approximate_polygon(arc, tolerance=tol_cells)
        return out if len(out) >= 2 else arc
    except ImportError:
        keep = [arc[0]]
        for i in range(1, len(arc) - 1):
            a = arc[i] - keep[-1]; b = arc[i + 1] - arc[i]
            if np.dot(a, b) > 0 and np.hypot(*a) >= tol_cells:
                keep.append(arc[i])
        keep.append(arc[-1])
        return np.array(keep)


def seawall_line_mask(dem, shape, *, crest_m, length_m, res_m, sea_level=0.81,
                      width_m=6.0, start_frac=None, rng=None, near_mask=None,
                      protect_radius_m=150.0, min_length_m=150.0,
                      tie_freeboard_m=1.0, require_tie_in=True,
                      weight_by_score=False, alignment_index=None):
    """Rasterise a seawall of given length along the shoreline contour.

    Picks a contour, takes an arc of `length_m` from it, and dilates to `width_m`. Unlike the
    buffer form this raises a line rather than every near-shore cell, so the wall has an alignment,
    a length and two ends that water can go around -- which is what makes siting matter.

    `near_mask` restricts which shoreline is eligible, and without it the placement is wrong in a
    way that is easy to miss. A marsh has many contours -- 177 over the Pin Point 2 m clip -- and
    the longest ones are islet outlines, not the shore in front of the community. Ranking by length
    alone therefore sites walls on uninhabited islands: mechanically valid, practically pointless.
    Pass the focus region or the building footprints, so only shoreline near the assets being
    protected is eligible. That is also what a user does in the tool: draw a wall in front of their
    own neighbourhood.
    """
    from scipy import ndimage
    rng = rng or np.random.default_rng()
    cs = shoreline_contour(dem, sea_level)
    if near_mask is None:
        # Terrain criterion: keep alignments that shelter floodable land, rank by how much.
        # Reject stretches too short to be a structure. Without this, small marsh islets pass the
        # floodable-land test and produce 40-cell blobs rather than walls.
        # Only a degenerate floor here. The tie-in test below is the physical criterion, and a
        # length threshold would wrongly reject short closures across narrow gaps.
        runs = [r for c in cs for r in _contiguous_runs(c) if _span_cells(r) >= 3]
        sc = [shelters_floodable_land(r, dem, shape, res_m=res_m, sea_level=sea_level)
              for r in runs]
        cs = [r for r, (ok, _) in zip(runs, sc) if ok]
        w = np.array([v for (ok, v), in zip(sc) if ok], dtype=float) if False else \
            np.array([v for (ok, v) in sc if ok], dtype=float)
    else:
        # Optional asset conditioning, for the targeted decision ensemble only. Training wants
        # breadth, so the terrain criterion above is the default.
        nm = ndimage.binary_dilation(near_mask, iterations=max(1, int(200.0 / res_m)))
        def touches(c):
            rr = np.clip(c[:, 0].round().astype(int), 0, shape[0] - 1)
            cc = np.clip(c[:, 1].round().astype(int), 0, shape[1] - 1)
            return nm[rr, cc]
        cs = [r for c in cs for r in _contiguous_runs(c[touches(c)]) if _span_cells(r) >= 3]
        # Rank by how many structures the alignment sits in front of, not by length. "Longest
        # eligible contour" is not "most people protected": it favours whichever cluster happens
        # to have the most intricate shoreline.
        prot = max(int(round(protect_radius_m / res_m)), 1)
        def score(c):
            rr = np.clip(c[:, 0].round().astype(int), 0, shape[0] - 1)
            cc = np.clip(c[:, 1].round().astype(int), 0, shape[1] - 1)
            s = np.zeros(shape, dtype=bool); s[rr, cc] = True
            return int((ndimage.binary_dilation(s, iterations=prot) & near_mask).sum())
        w = np.array([score(c) for c in cs], dtype=float)
    if not len(cs):
        return np.zeros(shape, dtype=bool)
    # Two sampling modes, for two jobs.
    #
    # Training wants breadth: every alignment that passes the tie-in test is a plausible structure,
    # so drawing uniformly and WITHOUT REPLACEMENT across an ensemble guarantees N members get N
    # distinct walls rather than N independent draws that repeat. Score-weighting here would
    # concentrate members on the few alignments with the largest sheltered area and waste the
    # ensemble on near-duplicates.
    #
    # The targeted decision set wants the good alignments, so it weights by sheltered floodable
    # area. A wall protecting nothing is a wasted run there.
    #
    # `alignment_index` is how without-replacement works without shared state: candidates are
    # shuffled with a fixed seed, so member i deterministically takes the i-th, and different
    # members cannot collide.
    if alignment_index is not None:
        order = np.random.default_rng(0).permutation(len(cs))
        c = cs[int(order[int(alignment_index) % len(cs)])]
    elif weight_by_score:
        w = np.clip(w, 0, None)
        prob = w / w.sum() if w.sum() > 0 else np.full(len(cs), 1.0 / len(cs))
        c = cs[int(rng.choice(len(cs), p=prob))]
    else:
        c = cs[int(rng.integers(0, len(cs)))]            # flat over accepted alignments
    step = float(np.hypot(*np.diff(c, axis=0).T).mean()) * res_m
    n = max(2, int(length_m / max(step, 1e-6)))
    if n >= len(c):
        arc = c
    else:
        i0 = int((start_frac if start_frac is not None else rng.random()) * (len(c) - n))
        arc = c[i0:i0 + n]
    arc, tied = _extend_to_tie_in(arc, c, dem, sea_level + tie_freeboard_m,
                                  max_extend_cells=int(600.0 / res_m))
    if require_tie_in and not tied:
        return np.zeros(shape, dtype=bool)      # no high ground within reach: flanked anyway
    arc = _simplify(arc, tol_cells=max(2.0, 20.0 / res_m))
    m = np.zeros(shape, dtype=bool)
    # Draw straight segments between simplified vertices, so the wall is a buildable polyline
    # rather than an exact tracing of the contour.
    for (r0, c0), (r1, c1) in zip(arc[:-1], arc[1:]):
        n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rr = np.clip(np.linspace(r0, r1, n).round().astype(int), 0, shape[0] - 1)
        cc = np.clip(np.linspace(c0, c1, n).round().astype(int), 0, shape[1] - 1)
        m[rr, cc] = True
    w = max(1, int(round(width_m / res_m / 2)))
    return ndimage.binary_dilation(m, iterations=w)


INTERVENTIONS = {
    # Spatial knobs are in METRES and are divided by the grid cell size at apply time, so the
    # same registry gives the same physical intervention at 30 m and at 4 m. Values below
    # reproduce the original cell-based ranges at 30 m.
    # crest_m is supported: Charleston's Low Battery seawall was raised to about 2.74 m NAVD88
    # and the USACE peninsula proposal is about 3.66 m. buffer_m is UNSOURCED.
    # length_m is the alignment length along the shoreline contour. buffer_m is retained for the
    # legacy band form (seawall_line=False) and is UNSOURCED; length is the physically meaningful
    # knob and the one a user sets in the tool.
    "seawall":   {"crest_m": (2.0, 4.5), "buffer_m": (30, 90), "length_m": (200, 1500)},
    # Roughness bands widened DOWNWARD on 2026-08-03 to cover the observed distribution.
    # Arefin et al. (2026), Est. Coast. Shelf Sci. 334, 109791, meta-analyse 36 studies and
    # report a saltmarsh Manning's n interquartile range of 0.04-0.08 and a mangrove IQR of
    # 0.10-0.14. The previous bands (marsh 0.08-0.16, mangrove 0.15-0.30, living shoreline
    # 0.10-0.20) began at or above the literature's upper quartile, so the sampled range
    # excluded realistic values entirely rather than merely being wide. Upper bounds are kept
    # above the IQR deliberately: Arefin et al. report n increasing landward, so high marsh can
    # legitimately exceed the pooled range, and a training ensemble wants the tails.
    #
    # NOTE: the completed 1506-member 30 m ensemble was sampled with the OLD bands. Its
    # nature-based effect sizes are therefore upper bounds. Seawall and retreat are unaffected.
    "marsh":     {"area_frac": (0.05, 0.30), "n_target": (0.03, 0.16),
                  "ksat_add": (10, 40), "awc_add": (50, 150), "corr_len_m": (450, 1500)},
    #            Chow (1959) puts the 0.16 end at "medium to dense brush, summer"; the 0.03 end
    #            is young or sparse Spartina, below the saltmarsh IQR floor.
    # Mangrove is a NO-ANALOGUE scenario at this site. Pin Point is 31.95 N and the poleward
    # Georgia mangrove records sit near 30.73 N, so this is a range-expansion hypothetical
    # rather than a plantable option, and must be labelled as such wherever it is reported.
    "mangrove":  {"area_frac": (0.03, 0.15), "n_target": (0.08, 0.16),
                  "ksat_add": (5, 20), "awc_add": (30, 100), "corr_len_m": (300, 1200)},
    "living_shoreline": {"area_frac": (0.3, 0.7), "n_target": (0.04, 0.16),  # marsh-water edge
                  "sill_m": (0.15, 0.40), "corr_len_m": (240, 750)},
    #            A sill with planted marsh behind it is a marsh surface, so the saltmarsh IQR
    #            applies. sill_m is UNSOURCED as an absolute rise: NOAA and Georgia guidance set
    #            the sill crest relative to MHW rather than as a fixed height above bed.

    "permeable": {"area_frac": (0.05, 0.25), "ksat_rate": (20, 60), "corr_len_m": (300, 1200)},
    # natural_n is supported by Chow's pasture and cleared-land classes, 0.025-0.050.
    "retreat":   {"area_frac": (0.02, 0.12), "natural_n": (0.035, 0.05), "corr_len_m": (300, 900)},
    # Chow (1959) floodplain classes: short-grass pasture 0.025-0.035, high grass 0.030-0.050,
    # scattered brush and heavy weeds 0.035-0.070. The previous 0.06-0.12 band sat above all of
    # them; only dense summer brush reaches it. Baseline asphalt is 0.013-0.016 for reference.
    "depave":    {"area_frac": (0.10, 0.50), "n_target": (0.03, 0.10),   # parking/impervious
                  "ksat_rate": (20, 50), "awc_add": (30, 100), "corr_len_m": (240, 750)},  # to vegetated
}


# Height-to-roughness map for vegetation-specified interventions. Linear in height across the
# Arefin saltmarsh IQR, anchored at the canopy heights actually observed on the Pin Point marsh
# in the 3DEP CHM (median 0.48 m, p90 1.39 m).
#
# The linearity is an assumption, not a result: Arefin et al. give the range, not a functional
# form. It is used because it is monotone, stays inside the observed bounds, and makes the
# scenario statable in practitioner terms. Anything more elaborate would be unconstrained by the
# data behind it.
VEG_H_LO, VEG_H_HI = 0.2, 1.4          # m, sparse young Spartina -> mature stand
VEG_N_LO, VEG_N_HI = 0.04, 0.08        # Arefin saltmarsh IQR


def n_from_height(h_m):
    """Manning's n for a target marsh canopy height, clamped to the saltmarsh IQR."""
    f = (float(h_m) - VEG_H_LO) / (VEG_H_HI - VEG_H_LO)
    f = min(max(f, 0.0), 1.0)
    return round(VEG_N_LO + f * (VEG_N_HI - VEG_N_LO), 4)


# Interventions whose roughness can be stated as a restoration target height instead of an n.
# "Restore marsh to 0.8 m canopy" is actionable; "set n to 0.06" is not.
HEIGHT_SPECIFIED = ("marsh", "living_shoreline")


def sample_intervention(kind, rng, *, by_height=False):
    """Draw one config (dict of scalar knobs) for `kind` from its ranges.

    `by_height=True` samples a target canopy height for marsh-type interventions and derives
    n_target from it, so the scenario is specified as a restoration target rather than as a
    friction coefficient. The sampled n then lies inside the Arefin saltmarsh IQR by
    construction, which is narrower than the registry band -- the registry keeps wider tails
    deliberately, so the two modes are not interchangeable and should not be mixed in one
    ensemble.
    """
    if kind not in INTERVENTIONS:
        raise ValueError(f"unknown intervention {kind!r}; choose {list(INTERVENTIONS)}")
    knobs = {"kind": kind, "seed": int(rng.integers(1 << 30))}
    for k, (lo, hi) in INTERVENTIONS[kind].items():
        knobs[k] = (int(rng.integers(lo, hi + 1)) if isinstance(lo, int)
                    else round(float(rng.uniform(lo, hi)), 4))
    if by_height and kind in HEIGHT_SPECIFIED:
        knobs["veg_height_m"] = round(float(rng.uniform(VEG_H_LO, VEG_H_HI)), 3)
        knobs["n_target"] = n_from_height(knobs["veg_height_m"])
        knobs["n_from_height"] = True
    return knobs


def apply_intervention(knobs, dem, manning, ksat, awc, *, sea_level=0.81,
                       classes=None, focus=None, wetlands=None, soil_ksat=None,
                       buildings=None, place="random", flood_depth=None, flood_zone=None,
                       mhw=0.94, mlw=-1.17, slr_buffer=0.5, res_m=30.0):
    """Expand one config onto copies of the grids. `classes` = optional NLCD grid.
    SAGIS-conditioned siting (context_rasters.py): `wetlands` (NWI mask, so marsh sits
    on real marsh), `buildings` (FEMA footprints, so retreat acts on real structures),
    `soil_ksat` (SSURGO Ksat grid, mm/hr, so de-pave/permeable can't exceed what the soil
    physically allows). `focus` = optional bool mask (Pin Point neighbourhood).

    `place`: "random" = Gaussian-field patches within the zone (training variety); "targeted"
    = rank by siting.suitability_score using `flood_depth` (baseline .max), `flood_zone`, and
    the tidal frame (`mhw`/`mlw`/`slr_buffer`), for realistic decision scenarios. Returns
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

    def m_to_cells(metres, minimum=1):
        return max(minimum, int(round(metres / res_m)))

    def zone(k):
        return suitability_mask(dem, sea_level, kind=k, classes=classes, focus=focus,
                                elev_band=knobs.get("elev_band", 2.0),
                                near_water_m=knobs.get("near_water_m", 450.0),
                                wetlands=wetlands, buildings=buildings, res_m=res_m)

    def place_mask(k):
        """Placement cells for kind k: random Gaussian patches within the zone, or the
        top suitability-scored cells when place='targeted'. Spatial knobs arrive in metres
        and are converted to cells here, so the same knobs give the same physical footprint
        at any grid resolution."""
        if place == "targeted":
            from .siting import targeted_mask
            return targeted_mask(dem, k, knobs.get("area_frac", 0.1), sea_level=sea_level,
                                 wetlands=wetlands, buildings=buildings, flood_depth=flood_depth,
                                 flood_zone=flood_zone, classes=classes, soil_ksat=soil_ksat,
                                 focus=focus, mhw=mhw, mlw=mlw, slr_buffer=slr_buffer,
                                 res_m=res_m)
        if k == "seawall":
            s = ndimage.binary_dilation(sea, iterations=m_to_cells(knobs.get("buffer_m", 60.0))) & land
            return (s & focus) if focus is not None else s
        return patch_mask(dem.shape, m_to_cells(knobs.get("corr_len_m", 900.0)),
                          knobs.get("area_frac", 0.1), knobs.get("seed", 0), restrict=zone(k))

    if kind == "seawall":
        if knobs.get("seawall_line", True):
            # Eligible shoreline: buildings if available, else the focus region. Without one of
            # these the wall lands on whichever islet has the longest outline.
            # Terrain siting by default; assets only for the targeted decision set.
            near = buildings if (place == "targeted" and buildings is not None) else None
            # About half of candidate alignments have no tie-in within reach and return an empty
            # mask. Retry with the next alignment so the ensemble fills; without this a member
            # would silently run with no intervention at all.
            base_idx = knobs.get("alignment_index", knobs.get("seed", 0))
            m = np.zeros(dem.shape, dtype=bool)
            for attempt in range(12):
                m = seawall_line_mask(dem, dem.shape, crest_m=knobs["crest_m"],
                                      length_m=knobs.get("length_m", 600.0), res_m=res_m,
                                      sea_level=sea_level, near_mask=near,
                                      alignment_index=base_idx + attempt,
                                      rng=np.random.default_rng(knobs.get("seed", 0) + attempt))
                if m.any():
                    break
            if not m.any():
                print(f"  no tie-in alignment found after 12 attempts (seed "
                      f"{knobs.get('seed')}); member has no floodwall")
        else:
            m = place_mask("seawall")                     # legacy near-shore band
        dem[m] = np.maximum(dem[m], knobs["crest_m"])
        # A wall is impervious and hydraulically smooth relative to marsh or vegetated ground.
        # Leaving n at its land-cover value would let a concrete structure retain marsh roughness.
        manning[m] = 0.015
        intensity[m] = 1.0

    elif kind in ("marsh", "mangrove"):
        m = place_mask(kind)                             # intertidal + adjacent to existing marsh
        manning[m] = np.maximum(manning[m], knobs["n_target"])
        ksat[m] = ksat[m] + knobs["ksat_add"]; awc[m] = awc[m] + knobs["awc_add"]
        intensity[m] = 1.0

    elif kind == "living_shoreline":                     # marsh-water edge sill + roughness
        m = place_mask("living_shoreline")
        manning[m] = np.maximum(manning[m], knobs["n_target"])
        dem[m] = dem[m] + knobs.get("sill_m", 0.2)
        intensity[m] = 1.0

    elif kind == "depave":                               # parking/impervious to vegetated
        m = place_mask("depave")
        cap = soil_capped(knobs["ksat_rate"])            # SSURGO-limited achievable rate
        manning[m] = np.maximum(manning[m], knobs["n_target"])
        ksat[m] = np.maximum(ksat[m], cap[m]); awc[m] = awc[m] + knobs["awc_add"]
        intensity[m] = 1.0

    elif kind == "permeable":
        m = place_mask("permeable")
        cap = soil_capped(knobs["ksat_rate"])            # SSURGO-limited achievable rate
        ksat[m] = np.maximum(ksat[m], cap[m])
        intensity[m] = 1.0

    elif kind == "retreat":
        m = place_mask("retreat")
        base = np.where(land & ~m, dem, np.nan)
        med = np.nanmedian(base) if np.isfinite(base).any() else np.nanmedian(dem[land])
        dem[m] = np.minimum(dem[m], med); manning[m] = knobs["natural_n"]
        intensity[m] = 1.0

    else:
        raise ValueError(f"unknown intervention {kind!r}")

    return dem, manning, ksat, awc, intensity
