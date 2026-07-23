"""Fetch SSURGO representative Ksat + AWC per map unit from USDA Soil Data Access (SDA),
as a comparison/alternative infiltration source to POLARIS (make_infil.py).

Given the MUKEYs present in a SAGIS soils_nrcs clip (data/raw/sagis_<name>/sagis_soils_nrcs.geojson),
this queries mapunit -> component (major components only) -> chorizon (horizons within the
top --depth-cm, default 50 cm, matching POLARIS's 0-50cm topsoil convention used elsewhere in
this repo) and returns a thickness-weighted representative ksat_r (micrometers/sec, SSURGO's
native unit) and awc_r (fraction, water fraction per inch of soil) per map unit, weighted by
component percent for multi-component map units.

Deps: stdlib only (urllib). SDA endpoint:
  https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest
"""
from __future__ import annotations
import json, urllib.request
from pathlib import Path

SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"


def _query(sql: str) -> list[list]:
    data = json.dumps({"query": sql, "format": "JSON"}).encode()
    req = urllib.request.Request(SDA_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
    return d.get("Table", [])


def mukeys_from_geojson(path) -> list[str]:
    d = json.load(open(path))
    keys = sorted({f["properties"].get("MUKEY") for f in d["features"] if f["properties"].get("MUKEY")})
    return keys


def fetch_ssurgo_ksat_awc(mukeys: list[str], depth_cm: float = 50.0) -> dict:
    """Query SDA for horizons in the top `depth_cm`, aggregate to one
    (comppct-weighted x thickness-weighted) ksat_r [um/s], awc_r [frac],
    wsatiated_r [% by volume, ~theta_s] and wfifteenbar_r [% by volume, ~wilting
    point] per mukey.

    Also derives `sat_storage_mm`, the SSURGO analog of the POLARIS infiltration
    capacity computed in make_infil.py (depth-weighted (theta_s - theta_r) x
    layer thickness): sat_storage_mm = (wsatiated_r/100 - wfifteenbar_r/100) * depth_cm*10.
    This is directly comparable to POLARIS infilcap. awc_r (plant-available
    water = field_capacity - wilting_point) is a different, much smaller
    quantity; do not compare it against infilcap.

    Returns {mukey: {"ksat_r_um_s":..., "ksat_r_mm_hr":..., "awc_r":...,
    "wsatiated_r":..., "wfifteenbar_r":..., "sat_storage_mm":..., "muname":...}}."""
    if not mukeys:
        return {}
    mk = ",".join(mukeys)
    sql = f"""
    SELECT mu.mukey, mu.muname, c.compname, c.comppct_r,
           ch.hzdept_r, ch.hzdepb_r, ch.ksat_r, ch.awc_r,
           ch.wsatiated_r, ch.wfifteenbar_r
    FROM mapunit mu
    INNER JOIN component c ON c.mukey = mu.mukey
    INNER JOIN chorizon ch ON ch.cokey = c.cokey
    WHERE mu.mukey IN ({mk}) AND c.majcompflag = 'Yes' AND ch.hzdept_r < {depth_cm}
    ORDER BY mu.mukey, c.comppct_r DESC, ch.hzdept_r
    """
    rows = _query(sql)
    # rows: mukey, muname, compname, comppct_r, hzdept_r, hzdepb_r, ksat_r, awc_r, wsatiated_r, wfifteenbar_r
    by_mu: dict[str, dict] = {}
    for mukey, muname, compname, pct, top, bot, ksat, awc, wsat, wfb in rows:
        top, bot = float(top), float(bot)
        bot = min(bot, depth_cm)
        thick = max(bot - float(top), 0.0)
        if thick <= 0:
            continue
        pct = float(pct) if pct is not None else 100.0
        wt = thick * pct
        rec = by_mu.setdefault(mukey, {
            "muname": muname, "ksat_num": 0.0, "awc_num": 0.0,
            "wsat_num": 0.0, "wsat_wsum": 0.0,
            "wfb_num": 0.0, "wfb_wsum": 0.0,
            "wsum": 0.0,
        })
        if ksat is not None:
            rec["ksat_num"] += wt * float(ksat)
        if awc is not None:
            rec["awc_num"] += wt * float(awc)
        if wsat is not None:
            rec["wsat_num"] += wt * float(wsat)
            rec["wsat_wsum"] += wt
        if wfb is not None:
            rec["wfb_num"] += wt * float(wfb)
            rec["wfb_wsum"] += wt
        rec["wsum"] += wt
    out = {}
    for mukey, rec in by_mu.items():
        if rec["wsum"] <= 0:
            continue
        ksat_um_s = rec["ksat_num"] / rec["wsum"]
        awc_r = rec["awc_num"] / rec["wsum"]
        wsatiated_r = rec["wsat_num"] / rec["wsat_wsum"] if rec["wsat_wsum"] > 0 else None
        wfifteenbar_r = rec["wfb_num"] / rec["wfb_wsum"] if rec["wfb_wsum"] > 0 else None
        if wsatiated_r is not None and wfifteenbar_r is not None:
            sat_storage_mm = (wsatiated_r / 100.0 - wfifteenbar_r / 100.0) * depth_cm * 10.0
        else:
            sat_storage_mm = None  # missing wsatiated_r/wfifteenbar_r for this mukey
        out[mukey] = {
            "muname": rec["muname"],
            "ksat_r_um_s": ksat_um_s,
            "ksat_r_mm_hr": ksat_um_s * 3.6,   # um/s -> mm/hr (1 um/s = 3.6 mm/hr)
            "awc_r": awc_r,                    # fraction (cm water / cm soil) - PLANT-AVAILABLE water, not comparable to infilcap
            "wsatiated_r": wsatiated_r,        # % by volume, ~theta_s (satiated water content)
            "wfifteenbar_r": wfifteenbar_r,    # % by volume, ~wilting point (water content at 15 bar)
            "sat_storage_mm": sat_storage_mm,  # (theta_s - theta_1500kPa) * depth_cm*10 -> mm; comparable to POLARIS infilcap
        }
    return out


def from_sagis(sagis_soils_geojson, out_json, depth_cm: float = 50.0):
    mukeys = mukeys_from_geojson(sagis_soils_geojson)
    result = fetch_ssurgo_ksat_awc(mukeys, depth_cm=depth_cm)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_json, "w"), indent=2)
    print(f"SSURGO: {len(result)}/{len(mukeys)} mukeys resolved -> {out_json}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch SSURGO ksat_r/awc_r for MUKEYs in a SAGIS soils clip")
    ap.add_argument("--soils-geojson", required=True, help="sagis_soils_nrcs.geojson (has MUKEY)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-cm", type=float, default=50.0)
    a = ap.parse_args()
    from_sagis(a.soils_geojson, a.out, depth_cm=a.depth_cm)
