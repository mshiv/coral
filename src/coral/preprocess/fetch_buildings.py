"""Fetch FEMA USA Structures building footprints (ArcGIS REST -> GeoJSON), clipped to a bbox.

The SAGIS DPLAT buildings layer is sparse (unincorporated, plat-derived, no heights). FEMA
USA Structures is the comprehensive footprint source (>450 sq ft, occupancy class, address),
used for managed-retreat siting (interventions/context_rasters.buildings_mask) and for
burn_buildings.py at 2 m. Reuses the paginated ArcGIS query in fetch_sagis.fetch_layer.

Deps: stdlib (via fetch_sagis).
"""
from __future__ import annotations
from pathlib import Path
from .fetch_sagis import fetch_layer

# FEMA USA Structures (hosted feature service, polygons: OCC_CLS/PRIM_OCC/PROP_ADDR/HEIGHT)
FEMA_USA_STRUCTURES = ("https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/"
                       "services/USA_Structures_View/FeatureServer/0")


def fetch_buildings(bbox, out_path, url=FEMA_USA_STRUCTURES):
    """Fetch building footprints intersecting bbox=[W,E,S,N] -> GeoJSON. Returns count."""
    return fetch_layer(url, bbox, out_path)


def from_config(cfg, out_dir=None, url=FEMA_USA_STRUCTURES):
    out_dir = out_dir or f"data/raw/buildings_{cfg.name}"
    out = Path(out_dir) / f"fema_structures_{cfg.name}.geojson"
    return fetch_buildings(cfg.domain.bbox, str(out), url)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch FEMA USA Structures footprints for a bbox")
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "E", "S", "N"))
    ap.add_argument("--out", default="data/raw/fema_structures.geojson")
    ap.add_argument("--url", default=FEMA_USA_STRUCTURES)
    a = ap.parse_args()
    fetch_buildings(a.bbox, a.out, a.url)
