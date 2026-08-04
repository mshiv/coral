"""Validate modelled flood EXTENT against Sentinel-1 SAR, where high-water marks do not reach.

WHY: The 23 quality-filtered Matthew high-water marks all fall outside the Pin Point clip, 
so `08_validate_hwm.py` reports "no HWMs in domain" and the 4 m run can currently only be 
checked for consistency against its 30 m parent. Sentinel-1 gives an independent observation 
at 10 m, comparable to the model grid, over the whole clip.

NOTE: SAR observes open water extent nly at overpass times, so this validates the wet/dry 
field. The two are complementary:

  HWMs   -> depth, at points, at the storm maximum
  SAR    -> extent, everywhere, at an arbitrary time that is probably not the maximum

CAVEATS:
  - Emergent vegetation. Over Spartina, C-band backscatter can INCREASE through double-bounce
    off flooded stems rather than fall, so simple low-backscatter thresholding misses flooded
    marsh. This is the dominant error source at Pin Point
  - Wind roughening of open water raises backscatter and shrinks apparent extent.
  - Buildings and layover make urban SAR flood mapping unreliable; developed cells are
    reported separately for the same reason.
  - Overpass timing. Sentinel-1 revisit is ~6 days for the constellation; the nearest scene to
    Matthew's peak may be hours off, during which extent changes a great deal. Compare against
    the model AT THE SCENE TIME, never against the run maximum.

The comparison is therefore scored as agreement in wet/dry classification (CSI, POD, FAR),
stratified by land cover, and read as a check that the model floods roughly the right places at
roughly the right time.

Inputs are a model depth grid and a pre-classified SAR water mask on the same grid (1 water,
0 land, nodata elsewhere). 
    
    python -m coral.validate.sar_flood_extent --model-depth run/res.max \\
        --sar-water s1_water_20161009.asc --classes nlcd_on_dem.asc --thresh 0.05
"""
import argparse
import numpy as np

WATER, LAND = 1, 0
MARSH_CODES, DEVELOPED_CODES, WATER_CODES = (90, 95), (21, 22, 23, 24), (11,)


def read_asc(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            h[k.lower()] = float(v)
    h["ncols"], h["nrows"] = int(h["ncols"]), int(h["nrows"])
    return np.loadtxt(path, skiprows=6), h


def contingency(model_wet, obs_wet, valid):
    """CSI / POD / FAR over `valid` cells. CSI is the headline: it ignores the true negatives
    that dominate any flood map and would otherwise make every model look excellent."""
    m, o = model_wet & valid, obs_wet & valid
    hits = int((m & o).sum()); miss = int((~m & o).sum()); false = int((m & ~o).sum())
    denom = hits + miss + false
    return {
        "n_valid": int(valid.sum()), "hits": hits, "misses": miss, "false_alarms": false,
        "CSI": hits / denom if denom else float("nan"),
        "POD": hits / (hits + miss) if (hits + miss) else float("nan"),
        "FAR": false / (hits + false) if (hits + false) else float("nan"),
        "bias": (hits + false) / (hits + miss) if (hits + miss) else float("nan"),
    }


def compare(model_depth_asc, sar_water_asc, classes_asc=None, thresh=0.05, nodata=-9999.0):
    dep, _ = read_asc(model_depth_asc)
    sar, _ = read_asc(sar_water_asc)
    if dep.shape != sar.shape:
        raise SystemExit(f"model {dep.shape} and SAR {sar.shape} differ; coregister first")
    valid = (dep != nodata) & (sar != nodata)
    model_wet = dep > thresh
    obs_wet = sar == WATER

    out = {"all": contingency(model_wet, obs_wet, valid)}
    if classes_asc:
        cls, _ = read_asc(classes_asc)
        cls = cls.astype(int)
        # Permanent open water is excluded from "all": both model and SAR call it wet, which
        # inflates CSI without saying anything about flood skill.
        perm = np.isin(cls, WATER_CODES)
        out["excl_permanent_water"] = contingency(model_wet, obs_wet, valid & ~perm)
        out["marsh"] = contingency(model_wet, obs_wet, valid & np.isin(cls, MARSH_CODES))
        out["developed"] = contingency(model_wet, obs_wet, valid & np.isin(cls, DEVELOPED_CODES))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-depth", required=True, help="model depth grid AT THE SCENE TIME")
    ap.add_argument("--sar-water", required=True, help="binary water mask, same grid")
    ap.add_argument("--classes", default=None, help="land-cover raster for stratification")
    ap.add_argument("--thresh", type=float, default=0.05, help="model wet depth threshold (m)")
    a = ap.parse_args()
    res = compare(a.model_depth, a.sar_water, a.classes, a.thresh)
    for k, r in res.items():
        print(f"\n[{k}]  n={r['n_valid']}")
        print(f"  CSI {r['CSI']:.3f}   POD {r['POD']:.3f}   FAR {r['FAR']:.3f}   bias {r['bias']:.3f}")
        print(f"  hits {r['hits']}  misses {r['misses']}  false alarms {r['false_alarms']}")
    print("\nSAR sees extent, not depth, and only at the overpass. Over Spartina, double-bounce")
    print("can raise backscatter so flooded marsh reads as dry: treat the marsh row as a lower")
    print("bound and report it separately from open terrain.")


if __name__ == "__main__":
    main()
