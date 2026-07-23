"""coral.interventions — adaptation strategies as edits to the LISFLOOD input grids.

Every intervention reduces to editing three fields the flood model already reads:
DEM elevation, Manning's n, infiltration (Ksat + AWC capacity). A seawall raises DEM
cells; a marsh raises n and infiltration; managed retreat regrades footprints. So the
same channel stack encodes baseline AND intervention scenarios (see the emulator).

Placement is geostatistical: rather than hand-drawn polygons, an intervention is
generated from a few low-dimensional knobs (location, extent, intensity, correlation
length) expanded into a spatially-correlated field via a Gaussian random field. This
gives realistic patchy structure, a principled sampler for the training design-of-
experiments, and a small human-facing control space for CHORUS.

See generate.py for the field generators and per-intervention edits.
"""
from .generate import (
    random_field, patch_mask, suitability_mask, focus_region, INTERVENTIONS,
    apply_intervention, sample_intervention,
)
