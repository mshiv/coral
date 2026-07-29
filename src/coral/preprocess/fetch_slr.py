"""Fetch NOAA/NASA relative sea-level-rise scenario values for a tide gauge.

The interventions sweep (emulator.sweep) needs SLR levels to run interventions
against. Round numbers (0.0, 0.5, 1.0 m) are arbitrary; this module resolves
SLR levels from the 2022 Interagency Sea Level Rise Technical Report scenarios
instead, so the sweep's SLR axis is projection-anchored to a real gauge and
year.

Two-tier per repo convention:
  fetch_slr(...)    -- pure worker (station, years, scenarios), tries network
                        first, falls back to a bundled table
  slr_levels(...)    -- adapter: resolves (scenario, year) targets to metres
                         for the sweep/config layer

Deps: stdlib only (urllib + json).
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import urllib.request
import urllib.error

# NOAA 2022 Interagency Sea Level Rise Technical Report scenario names.
SCENARIOS = ("Low", "IntLow", "Int", "IntHigh", "High")

# Fort Pulaski, GA (same gauge as make_tide.py).
DEFAULT_STATION = "8670870"

# Median (quantile 50) scenario values in metres for PSMSL id 395 (Fort
# Pulaski GA, NOAA CO-OPS station 8670870), 2022 Interagency Sea Level Rise
# Task Force report, via the NASA sea level projection tool. Values are
# relative to a 2000 baseline; the source spreadsheet reports mm, converted
# here to metres. Archived at data/raw/slr/sl_taskforce_scenarios_psmsl_id_395.xlsx.
# Generated programmatically from the "Total" sheet (process=total, quantile=50).
FORT_PULASKI = {
    "Low": {2030: 0.1836, 2040: 0.2516, 2050: 0.3136, 2060: 0.3716, 2070: 0.4176,
            2080: 0.4546, 2090: 0.4896, 2100: 0.5416, 2110: 0.5886, 2120: 0.6326,
            2130: 0.6776, 2140: 0.7186, 2150: 0.7626},
    "IntLow": {2030: 0.2016, 2040: 0.2796, 2050: 0.3556, 2060: 0.4326, 2070: 0.5086,
               2080: 0.5836, 2090: 0.6566, 2100: 0.7316, 2110: 0.8156, 2120: 0.9006,
               2130: 0.9846, 2140: 1.0706, 2150: 1.1536},
    "Int": {2030: 0.2086, 2040: 0.2956, 2050: 0.3956, 2060: 0.5056, 2070: 0.6346,
            2080: 0.7876, 2090: 0.9736, 2100: 1.1926, 2110: 1.4216, 2120: 1.6306,
            2130: 1.8136, 2140: 1.9936, 2150: 2.1696},
    "IntHigh": {2030: 0.2196, 2040: 0.3256, 2050: 0.4556, 2060: 0.6286, 2070: 0.8356,
                2080: 1.0726, 2090: 1.3466, 2100: 1.6396, 2110: 1.9156, 2120: 2.1416,
                2130: 2.3416, 2140: 2.5356, 2150: 2.7376},
    "High": {2030: 0.2236, 2040: 0.3486, 2050: 0.5146, 2060: 0.7456, 2070: 1.0346,
             2080: 1.3846, 2090: 1.7496, 2100: 2.1376, 2110: 2.5426, 2120: 2.9136,
             2130: 3.2076, 2140: 3.4926, 2150: 3.7886},
}

# Observed rise from the scenario's 2000 baseline to 2016 (Hurricane Matthew
# surge boundary year used throughout this repo), from the "Observations"
# sheet, "Obs Extrap. 50th" column at Year 2016: 95.1 mm = 0.0951 m. The
# model's surge boundary already sits at the 2016 sea level, which already
# contains this 2000-to-2016 rise; adding a raw scenario value on top of it
# double counts that rise, so fetch_slr/slr_levels subtract this offset by
# default (see baseline_year / subtract_baseline below).
BASELINE_OFFSET_M = 0.0951


def _try_network_fetch(station, years, scenarios):
    """Attempt one or two documented NOAA/NASA sea level projection endpoints.
    Returns a values dict {scenario: {year: metres}} on a clean parse, or None
    if the fetch fails or the response shape is not recognized. Does not raise;
    callers fall back to the bundled table on None."""
    candidates = [
        f"https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product/scenarios.json?station={station}",
        f"https://cdn.tidesandcurrents.noaa.gov/sltrends/data/{station}_meantrend.csv",
    ]
    for url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                raw = r.read()
            if url.endswith(".json"):
                payload = json.loads(raw)
                values = {}
                for rec in payload.get("scenarios", []):
                    scen, year, m = rec.get("scenario"), rec.get("year"), rec.get("value_m")
                    if scen in scenarios and year in years and m is not None:
                        values.setdefault(scen, {})[year] = float(m)
                if values:
                    return values
            # CSV mean-trend response is a historical trend line, not scenario
            # projections; it cannot be parsed into scenario values, so skip it.
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, ValueError, KeyError):
            continue
    return None


def fetch_slr(station=DEFAULT_STATION, years=(2050,), scenarios=SCENARIOS, out_json=None,
              baseline_year=2016, subtract_baseline=True):
    """Return relative SLR projections for `station`, `years`, `scenarios`.

    Tries a network fetch first (NOAA/NASA sea level projection services);
    falls back to the bundled FORT_PULASKI table if the fetch does not clearly
    succeed and parse for this station. Only station 8670870 has a bundled
    fallback; other stations require a working network fetch.

    The bundled table is relative to the report's 2000 baseline. This repo's
    surge boundary is Hurricane Matthew (2016), whose water level already
    contains the 2000-to-2016 rise, so by default (`subtract_baseline=True`)
    each value has BASELINE_OFFSET_M subtracted, re-basing it to `baseline_year`
    (2016) instead of 2000. Pass `subtract_baseline=False` to get the raw
    scenario value as published (still relative to 2000).

    Returns {"station":..., "source":..., "retrieved":..., "datum": "relative
    to 2000 baseline (see provenance)", "provenance": {...}, "values":
    {scenario: {year: metres}}} where "values" holds the (optionally
    baseline-corrected) numbers and "provenance" records both raw and applied.
    """
    values = _try_network_fetch(station, years, scenarios)
    if values:
        source = "NOAA/NASA sea level projection service"
    else:
        if station != DEFAULT_STATION:
            raise SystemExit(
                f"no network fetch succeeded for station {station} and no bundled "
                f"fallback table exists for it (only {DEFAULT_STATION} is bundled)")
        table = FORT_PULASKI
        values = {}
        for scen in scenarios:
            if scen not in table:
                continue
            for year in years:
                if year not in table[scen]:
                    continue
                values.setdefault(scen, {})[year] = table[scen][year]
        source = ("2022 Interagency Sea Level Rise Task Force report, via the NASA sea "
                  "level projection tool (archived data/raw/slr/sl_taskforce_scenarios_"
                  "psmsl_id_395.xlsx)")

    raw_values = {scen: dict(yr) for scen, yr in values.items()}
    if subtract_baseline:
        for scen, yr in values.items():
            for year in yr:
                yr[year] = round(yr[year] - BASELINE_OFFSET_M, 4)

    provenance = {
        "scenario_baseline_year": 2000,
        "effective_baseline_year": baseline_year if subtract_baseline else 2000,
        "baseline_offset_m": BASELINE_OFFSET_M if subtract_baseline else 0.0,
        "subtract_baseline": subtract_baseline,
        "note": ("values are the 2000-baseline scenario minus the observed 2000->2016 "
                "rise (BASELINE_OFFSET_M), so they are relative to the 2016 Matthew "
                "surge boundary" if subtract_baseline else
                "raw scenario values, relative to the report's 2000 baseline"),
    }
    result = {
        "station": str(station),
        "source": source,
        "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datum": "relative to 2000 baseline (see provenance)",
        "provenance": provenance,
        "raw_values": raw_values,
        "values": values,
    }
    out_json = out_json or f"data/raw/slr_{station}.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_json, "w"), indent=2)
    print(f"SLR: station {station}, {source} -> {out_json}")
    return result


def slr_levels(targets, station=DEFAULT_STATION, baseline_year=2016,
               subtract_baseline=True, **kw):
    """Resolve a list of (scenario, year) tuples to [(label, rise_m), ...],
    e.g. [("Int", 2050)] -> [("Int2050", 0.3005)]. Label format:
    f"{scenario}{year}". Values are baseline-corrected (2016) by default; see
    `fetch_slr` for `subtract_baseline`."""
    scenarios = sorted({s for s, _ in targets})
    years = sorted({y for _, y in targets})
    result = fetch_slr(station=station, years=years, scenarios=scenarios,
                       baseline_year=baseline_year, subtract_baseline=subtract_baseline, **kw)
    out = []
    for scen, year in targets:
        try:
            m = result["values"][scen][year]
        except KeyError:
            raise SystemExit(f"no SLR value resolved for {scen}{year} at station {station}")
        out.append((f"{scen}{year}", m))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch NOAA/NASA relative SLR scenario values")
    ap.add_argument("--station", default=DEFAULT_STATION)
    ap.add_argument("--years", nargs="+", type=int, default=[2050])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    result = fetch_slr(station=a.station, years=a.years, scenarios=a.scenarios, out_json=a.out)
    prov = result["provenance"]
    print(f"station {result['station']}  source: {result['source']}")
    print(f"baseline: scenario 2000 -> effective {prov['effective_baseline_year']}"
         f"  (offset {prov['baseline_offset_m']:.4f} m subtracted: {prov['subtract_baseline']})")
    print("raw (2000 baseline):")
    print(f"{'scenario':10s}" + "".join(f"{y:>8d}" for y in a.years))
    for scen in a.scenarios:
        row = result["raw_values"].get(scen, {})
        print(f"{scen:10s}" + "".join(f"{row.get(y, float('nan')):8.4f}" for y in a.years))
    print(f"applied (baseline-corrected to {prov['effective_baseline_year']}):")
    print(f"{'scenario':10s}" + "".join(f"{y:>8d}" for y in a.years))
    for scen in a.scenarios:
        row = result["values"].get(scen, {})
        print(f"{scen:10s}" + "".join(f"{row.get(y, float('nan')):8.4f}" for y in a.years))
