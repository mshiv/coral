"""Scenario configuration: load + validate the YAML knobs that vary per run.

A scenario config holds everything that differs between runs (domain, storm,
surge, datum, SLR, rainfall, resolution, ...). Every script reads values from
here instead of hardcoding them, and the jinja templates render setrun.py / .par
from these fields. See configs/scenarios/*.yaml.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yaml

# --- Closed-set registries: the only legal values for branch-selecting knobs. ---
# Add a value here (and wire the branch that reads it) to extend an option.
# `storm` is not a closed set, it's an extensible label, so new
# storms (dorian, florence, ...) don't require editing this file.
DRAG_LAWS = {"garratt", "powell"}
MANNING_SOURCES = {"nlcd2016", "constant"}
RAIN_SOURCES = {"aorc", "mrms"}    # AORC (1km hourly) | MRMS (radar QPE, ~1km hourly)
RAIN_MODES = {"uniform", "dynamic"}  # LISFLOOD: rainfile time series | dynamicrainfile netCDF
INFIL_SOURCES = {"polaris"}        # distributed-infiltration sources (make_infil.py)
TIDE_SOURCES = {"noaa"}            # tidal boundary sources (make_tide.py; NOAA CO-OPS)


def _require(value, allowed: set, field_name: str) -> None:
    if value not in allowed:
        raise ValueError(
            f"{field_name}={value!r} is not a valid option; "
            f"choose one of {sorted(allowed)}"
        )


@dataclass
class Domain:
    dem: str
    bbox: list[float]                 # [W, E, S, N]
    res_m: float = 30.0
    ref_point: list[float] = field(default_factory=lambda: [-81.137, 31.944])
    focus_radius_km: Optional[float] = None  # if set, interventions are restricted to
    #                                          within this radius of ref_point (Pin Point)

@dataclass
class GeoClaw:
    storm: str = "matthew"            # open label (extensible), validated non-empty
    drag: str = "garratt"             # one of DRAG_LAWS
    sea_level: float = 0.81           # static high-tide stage (m, MSL)
    amr_max: int = 6
    refine_box: list[float] = field(default_factory=list)   # [W,E,S,N]

    def __post_init__(self):
        if not self.storm:
            raise ValueError("geoclaw.storm must be a non-empty storm label")
        _require(self.drag, DRAG_LAWS, "geoclaw.drag")

@dataclass
class Coupling:
    gauge_spacing_m: float = 400.0
    seaward_cells: int = 3
    datum_offset_m: float = 0.071     # MSL -> NAVD88
    dry_thresh: float = 0.05
    landfall_s: float = 172800.0      # model-clock time of landfall (bdy/par origin)
    landfall_utc: Optional[str] = None  # calendar anchor for model t=0 (ISO8601),
    #                                     e.g. "2016-10-08T12:00:00"; maps model
    #                                     seconds <-> real time for rainfall fetch

@dataclass
class Manning:
    source: str = "nlcd2016"          # one of MANNING_SOURCES
    constant_n: float = 0.06
    sea_level: float = 0.81

    def __post_init__(self):
        _require(self.source, MANNING_SOURCES, "manning.source")

@dataclass
class LisfloodCfg:
    saveint: float = 1800.0
    sim_window_h: list[float] = field(default_factory=lambda: [-24, 24])
    initial_tstep: float = 10.0

@dataclass
class Forcing:
    rainfall: Optional[str] = None    # None = surge-only; else a RAIN_SOURCES name
    rain_mode: str = "uniform"        # one of RAIN_MODES (only used if rainfall set)
    infiltration: Optional[str] = None  # None = off; else an INFIL_SOURCES name (polaris)
    infil_capped: bool = True         # True = storage-limited (AWC capacity, infilcapfile);
    #                                   False = constant-rate Ksat only (over-drains surge)
    tide: Optional[str] = None        # None = static sea_level; else a TIDE_SOURCES name.
    #                                   When set, GeoClaw runs at MSL and the time-varying
    #                                   tide is added onto the surge in the .bdy (make_tide.py).
    tide_station: str = "8670870"     # NOAA CO-OPS station id (Fort Pulaski, GA)
    slr_m: float = 0.0

    def __post_init__(self):
        if self.rainfall is not None:
            _require(self.rainfall, RAIN_SOURCES, "forcing.rainfall")
            _require(self.rain_mode, RAIN_MODES, "forcing.rain_mode")
        if self.infiltration is not None:
            _require(self.infiltration, INFIL_SOURCES, "forcing.infiltration")
        if self.tide is not None:
            _require(self.tide, TIDE_SOURCES, "forcing.tide")

@dataclass
class HPC:
    account: str = "gts-arobel3-atlas"
    partition: str = "cpu-medium"
    lisflood_bin: str = "/path/to/LISFLOOD-FP-trunk/lisflood"

@dataclass
class Interventions:
    """Defines the SLR x intervention ensemble for emulator.sweep (first-class + reproducible).
    Knob ranges stay in interventions.generate.INTERVENTIONS; this selects which kinds/levels
    to sample and the SAGIS context data that conditions siting (Phase 2)."""
    kinds: list[str] = field(default_factory=lambda:
        ["seawall", "marsh", "living_shoreline", "permeable", "retreat", "depave"])
    siting: str = "random"                    # random: training variety; targeted: realistic
    flood_depth: Optional[str] = None         # baseline .max path, drives targeted siting
    flood_zone: Optional[str] = None          # flood-zone polygon geojson, targets retreat/depave
    slr_levels: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 1.0, 1.5])
    n_per_kind: int = 4
    include_combos: bool = True
    focus_radius_km: Optional[float] = None   # restrict siting to this radius of ref_point
    seed: int = 0
    # SAGIS context data (paths) for conditioned siting; None = generic elevation/NLCD siting
    wetlands: Optional[str] = None            # NWI geojson -> marsh/mangrove footprint
    soils_geojson: Optional[str] = None       # SSURGO soils geojson (MUKEY) -> de-pave Ksat cap
    ssurgo_table: Optional[str] = None        # fetch_ssurgo ksat table json (with soils_geojson)
    buildings: Optional[str] = None           # FEMA footprints geojson -> retreat siting

    def __post_init__(self):
        from .interventions.generate import INTERVENTIONS
        bad = [k for k in self.kinds if k not in INTERVENTIONS]
        if bad:
            raise ValueError(f"interventions.kinds {bad} not in {sorted(INTERVENTIONS)}")

@dataclass
class Scenario:
    name: str
    domain: Domain
    geoclaw: GeoClaw = field(default_factory=GeoClaw)
    coupling: Coupling = field(default_factory=Coupling)
    manning: Manning = field(default_factory=Manning)
    lisflood: LisfloodCfg = field(default_factory=LisfloodCfg)
    forcing: Forcing = field(default_factory=Forcing)
    hpc: HPC = field(default_factory=HPC)
    interventions: Optional[Interventions] = None  # None = no ensemble sweep for this scenario

    # convenient derived values used across the workflow
    @property
    def tstart(self) -> float:
        return self.landfall_plus(self.lisflood.sim_window_h[0])

    @property
    def sim_time(self) -> float:
        return self.landfall_plus(self.lisflood.sim_window_h[1])

    def landfall_plus(self, hours: float) -> float:
        return self.coupling.landfall_s + hours * 3600.0

    @property
    def landfall_dt(self):
        """Calendar UTC datetime of landfall (model t=0), or None if unset."""
        if self.coupling.landfall_utc is None:
            return None
        return datetime.fromisoformat(self.coupling.landfall_utc)

    def rain_window_utc(self):
        """(start, end) real UTC datetimes spanning the sim window, the rainfall
        fetch interval. Derived from landfall_utc + lisflood.sim_window_h so the
        rain, surge, and sim clocks all share the landfall origin."""
        lf = self.landfall_dt
        if lf is None:
            raise ValueError(
                "coupling.landfall_utc is required to fetch rainfall (maps model "
                "seconds to calendar time)")
        return (lf + timedelta(hours=self.lisflood.sim_window_h[0]),
                lf + timedelta(hours=self.lisflood.sim_window_h[1]))


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(scenario_path: str | Path,
         base: str | Path = "configs/base.yaml") -> Scenario:
    """Load base.yaml then overlay the scenario YAML, return a Scenario."""
    data = yaml.safe_load(open(base)) if Path(base).exists() else {}
    data = _merge(data, yaml.safe_load(open(scenario_path)))
    sub = {k: globals()[cls](**data[k]) for k, cls in {
        "domain": "Domain", "geoclaw": "GeoClaw", "coupling": "Coupling",
        "manning": "Manning", "lisflood": "LisfloodCfg", "forcing": "Forcing",
        "hpc": "HPC", "interventions": "Interventions"}.items() if k in data}
    return Scenario(name=data["name"], **sub)
