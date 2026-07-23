"""Fetch Savannah Area GIS (SAGIS) open-data layers, clipped to a bbox, as GeoJSON.

SAGIS publishes Chatham County / City of Savannah GIS as ArcGIS REST services
(pub.sagis.org). The stormwater network (pipes, ditches, canals, manholes, inlets,
outfalls, tide gates, pump stations, reservoirs, drainage basins) is the input for
the drainage model (burn_drainage.py + SWMM coupling); soils/wetlands/buildings feed
the physics + building-scale clip. This queries each layer over an envelope and writes
a GeoJSON FeatureCollection (WGS84), paginating past the service transfer limit.

Two-tier per repo convention: fetch_layer(url, bbox, out) worker; from_config / CLI
that pull a named GROUP (e.g. stormwater) for the scenario bbox.

Deps: stdlib (urllib + json).
"""
from __future__ import annotations
import json, urllib.parse, urllib.request
from pathlib import Path

BASE = "https://pub.sagis.org/arcgis/rest/services/OpenData"

# name -> service/layer URL. Groups drive bulk pulls.
LAYERS = {
    # --- Tier 1: stormwater network (Hydrology) ---
    "pipes_chatham":        f"{BASE}/Hydrology/MapServer/21",
    "conduit_pipes_sav":    f"{BASE}/Hydrology/MapServer/29",
    "ditches_maint_sav":    f"{BASE}/Hydrology/MapServer/17",
    "ditches_chatham":      f"{BASE}/Hydrology/MapServer/22",
    "canals_chatham":       f"{BASE}/Hydrology/MapServer/26",
    "structures_chatham":   f"{BASE}/Hydrology/MapServer/20",
    "manholes_sav":         f"{BASE}/Hydrology/MapServer/16",
    "inlets_sav":           f"{BASE}/Hydrology/MapServer/18",
    "headwalls_sav":        f"{BASE}/Hydrology/MapServer/19",
    "outfalls_chatham":     f"{BASE}/Hydrology/MapServer/27",
    "tide_gates_sav":       f"{BASE}/Hydrology/MapServer/14",
    "pump_stations_sav":    f"{BASE}/Hydrology/MapServer/15",
    "reservoir_area_chatham": f"{BASE}/Hydrology/MapServer/25",
    "drainage_basin_sav":   f"{BASE}/Hydrology/MapServer/30",
    "green_infra_chatham":  f"{BASE}/Hydrology/MapServer/28",
    "flood_zones_2018":     f"{BASE}/Hydrology/MapServer/31",
    # --- Tier 2: physics refinement ---
    "soils_nrcs":           f"{BASE}/Environment/MapServer/5",
    "wetlands_nwi":         f"{BASE}/Environment/MapServer/2",
    # --- Tier 3: buildings ---
    "buildings_chatham":    f"{BASE}/DPLAT_Chatham/MapServer/1",
    "benchmarks_chatham":   f"{BASE}/DPLAT_Chatham/MapServer/0",
}
GROUPS = {
    "stormwater": ["pipes_chatham", "conduit_pipes_sav", "ditches_maint_sav",
                   "ditches_chatham", "canals_chatham", "structures_chatham",
                   "manholes_sav", "inlets_sav", "headwalls_sav", "outfalls_chatham",
                   "tide_gates_sav", "pump_stations_sav", "reservoir_area_chatham",
                   "drainage_basin_sav", "green_infra_chatham"],
    "physics":   ["soils_nrcs", "wetlands_nwi"],
    "buildings": ["buildings_chatham"],
    "context":   ["flood_zones_2018", "benchmarks_chatham"],
}


def _page(url, bbox, offset, count=1000):
    W, E, S, N = bbox
    q = {"where": "1=1", "geometry": f"{W},{S},{E},{N}",
         "geometryType": "esriGeometryEnvelope", "inSR": "4326", "outSR": "4326",
         "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "f": "geojson",
         "returnGeometry": "true", "resultOffset": offset, "resultRecordCount": count}
    with urllib.request.urlopen(f"{url}/query?{urllib.parse.urlencode(q)}", timeout=90) as r:
        return json.loads(r.read())


def fetch_layer(url, bbox, out_path, count=1000):
    """Query one ArcGIS layer over bbox=[W,E,S,N]; write a GeoJSON FeatureCollection."""
    feats, offset = [], 0
    while True:
        d = _page(url, bbox, offset, count)
        if "error" in d:
            raise SystemExit(f"SAGIS error {url}: {d['error'].get('message')}")
        f = d.get("features", [])
        feats.extend(f)
        if len(f) < count and not d.get("exceededTransferLimit"):
            break
        offset += len(f)
        if len(f) == 0:
            break
    fc = {"type": "FeatureCollection", "features": feats}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(fc, open(out_path, "w"))
    print(f"  {Path(out_path).stem}: {len(feats)} features -> {out_path}")
    return len(feats)


def fetch_group(group, bbox, out_dir):
    """Pull every layer in a GROUP (or a single layer name) for bbox -> out_dir/*.geojson."""
    names = GROUPS.get(group, [group] if group in LAYERS else None)
    if names is None:
        raise SystemExit(f"unknown group/layer {group!r}; groups={list(GROUPS)}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"SAGIS '{group}' over bbox {bbox} -> {out}")
    return {n: fetch_layer(LAYERS[n], bbox, out / f"sagis_{n}.geojson") for n in names}


def from_config(cfg, group="stormwater", out_dir=None):
    out_dir = out_dir or f"data/raw/sagis_{cfg.name}"
    return fetch_group(group, cfg.domain.bbox, out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch SAGIS layers clipped to a bbox")
    ap.add_argument("--group", default="stormwater",
                    help="group (stormwater|physics|buildings|context) or a single layer name")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "E", "S", "N"),
                    default=[-81.22, -80.82, 31.82, 32.08])
    ap.add_argument("--out", default="data/raw/sagis")
    a = ap.parse_args()
    fetch_group(a.group, a.bbox, a.out)
