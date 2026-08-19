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
    ref_point: list[float] = field(default_factory=lambda: [-81.0903, 31.9522])
    focus_radius_km: Optional[float] = None  # if set, interventions are restricted to
    #                                          within this radius of ref_point (Pin Point)

@dataclass
class GeoClaw:
    storm: str = "matthew"            # open label (extensible), validated non-empty
    drag: str = "garratt"             # one of DRAG_LAWS
    sea_level: float = 0.0            # run datum; the tide enters through the .bdy
    amr_max: int = 7
    refine_box: list[float] = field(default_factory=list)   # [W,E,S,N]

    # Forcing the finest level over the whole refine_box costs more than it looks. Each level
    # refines in x, y AND t, so level 7 is 64x level 6 over the same area. Measured: level 7
    # across the 33 x 63 km gauge box ran at 1.5% of the 72 h simulation in 5 wall hours,
    # projecting to about 13.6 days.
    #
    # The coupling gauges lie on a curve, not a filled box. refine_front_km replaces the box
    # region with a ruled rectangle hugging that curve, half-width in km. refine_t1_h starts
    # the forced refinement late (hours relative to landfall, negative = before), because the
    # .bdy only consumes the window in coupling.sim_window_h and the ocean ahead of that is
    # quiescent. refine_t2_h ends it early for the same reason. Leave them None to force
    # amr_max across the whole refine_box for the whole run, which is what cost 13.6 days.
    refine_front_km: Optional[float] = None
    refine_t1_h: Optional[float] = None
    refine_t2_h: Optional[float] = None

    # Override the end of the run, hours relative to landfall. None keeps setrun's +24 h.
    # Mainly for short timing tests: set 2-3 h, confirm the gauges report the level you asked
    # for, and measure the real rate before committing a long job.
    tfinal_h: Optional[float] = None

    def __post_init__(self):
        if not self.storm:
            raise ValueError("geoclaw.storm must be a non-empty storm label")
        _require(self.drag, DRAG_LAWS, "geoclaw.drag")
        if self.refine_front_km is not None and self.refine_front_km <= 0:
            raise ValueError("geoclaw.refine_front_km must be positive")

@dataclass
class Datums:
    """Tidal datums for the forcing station, m NAVD88.

    One place for the waterline, because three different roles were using three different
    numbers: roughness classification, the shoreline contour the seaward front follows, and the
    marsh migration band. `sea_level` in the geoclaw block is a different quantity -- the run
    datum -- and is 0.0.
    """
    station: str = "8670870"          # Fort Pulaski
    mhw: float = 1.114
    msl: float = 0.074
    mlw: float = -1.091
    source: str = "2016 observed, CO-OPS 6-min, Matthew week excluded"


@dataclass
class Coupling:
    gauge_spacing_m: float = 400.0
    seaward_cells: int = 3
    datum_offset_m: float = 0.071     # MSL -> NAVD88
    # Mean water level GeoClaw's barotropic surge does not reproduce: seasonal swelling plus
    # the post-storm mean. Calibrated against Fort Pulaski, not assumed. The tide-explicit
    # boundary is eta = surge - z_base + tide, and build_bdy's surge_baseline IS z_base:
    #     z_base = geoclaw.sea_level - mean_offset_m
    # With the old sea_level 0.81 this gives 0.42, the value derived in methods 2.4.6, which
    # brought the boundary to a mean bias of -0.001 m. With sea_level now 0.0 it gives -0.39,
    # so the offset is added rather than removed. Passing sea_level alone as the baseline, as
    # the code did, leaves the boundary 0.39 m low.
    mean_offset_m: float = 0.39
    dry_thresh: float = 0.05
    landfall_s: float = 172800.0      # model-clock time of landfall (bdy/par origin)
    landfall_utc: Optional[str] = None  # calendar anchor for model t=0 (ISO8601),
    #                                     e.g. "2016-10-08T12:00:00"; maps model
    #                                     seconds <-> real time for rainfall fetch

@dataclass
class Manning:
    source: str = "nlcd2016"          # one of MANNING_SOURCES
    constant_n: float = 0.06
    # Land/water threshold for classifying cells, NOT a datum. Cells below it that land cover
    # leaves unmapped get water roughness, and it is the contour gen_boundary_points follows to
    # find the shoreline. Named sea_level until 2026-08-17, which collided with
    # geoclaw.sea_level -- a different quantity that is now 0.0 while this stays near MHW.
    water_level: float | None = None   # None -> datums.mhw

    def __post_init__(self):
        _require(self.source, MANNING_SOURCES, "manning.source")

@dataclass
class LisfloodCfg:
    saveint: float = 1800.0
    # Primed initial water surface, required by any run using edge boundary conditions: edge
    # flux is guarded by `if (H > DepthThresh)` (sgc.cpp:1354), so an edge on a dry domain is
    # inert forever. Build it with couple.make_startfile --level. Rendered alongside a bare
    # `startelev`, which tells LISFLOOD the file is water-surface elevation, not depth.
    startfile: Optional[str] = None
    # Hours between restart files. Off by default: on a resubmit or a timeout LISFLOOD reloads
    # the stale checkpoint and can sit at the first step, and the .mass then opens with a
    # "Checkpoint restart" banner on what looks like a fresh run.
    checkpoint_h: Optional[float] = None
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
    # Salt marsh soil is at or near saturation through the tidal cycle, so the conductivity and
    # storage a soil survey reports for it are unavailable when a surge arrives. Setting this
    # zeroes both over the NWI wetland footprint. Off by default: it changes results, and the
    # 30 m compound run was calibrated against high-water marks without it.
    mask_saturated_wetland: bool = False

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
    # The binary is dynamically linked against these, so a compute node with a clean module
    # environment cannot load it without them.
    modules: str = "gcc/12.3.0 netcdf-c/4.9.2-gszew36"
    # Only the main flux loop in fp_acc.cpp is parallel and one section sits in `omp critical`,
    # so threading saturates well before 24. Throughput comes from concurrent array members
    # instead. Benchmark and set this at the knee.
    cpus_per_task: int = 8
    # A 1356x882 domain needs a few hundred MB of field arrays. This is per task, not per cpu:
    # --mem-per-cpu=12G on 24 cpus asks for 288 GB and never schedules.
    mem: str = "4G"
    walltime: str = "04:00:00"
    throttle: int = 50        # concurrent array members; the main lever on ensemble wall time

@dataclass
class Interventions:
    """Defines the SLR x intervention ensemble for emulator.sweep (first-class + reproducible).
    Knob ranges stay in interventions.generate.INTERVENTIONS; this selects which kinds/levels
    to sample and the SAGIS context data that conditions siting (Phase 2)."""
    # Must be keys of interventions.generate.INTERVENTIONS. "marsh" was renamed to
    # "marsh_migration" and __post_init__ rejects the old name, so the stale default made
    # Interventions(...) fail for any config that set one of the other fields.
    kinds: list[str] = field(default_factory=lambda:
        ["seawall", "marsh_migration", "living_shoreline", "retreat", "depave", "road_raise"])
    siting: str = "random"                    # random: training variety; targeted: realistic
    flood_depth: Optional[str] = None         # baseline .max path, drives targeted siting
    flood_zone: Optional[str] = None          # flood-zone polygon geojson, targets retreat/depave
    slr_levels: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.6, 1.0, 1.5])
    slr_scenarios: Optional[list] = None      # e.g. [["Int",2050],["IntHigh",2050]]; resolved
    #                                            to metres via fetch_slr, additive to slr_levels
    n_per_kind: int = 4
    include_combos: bool = True
    focus_radius_km: Optional[float] = None   # restrict siting to this radius of ref_point
    seed: int = 0
    seawall_walls: Optional[object] = None    # None = one wall per seawall member (classic
    #                                            single-structure ensemble); int N samples
    #                                            1..N walls, weighted toward few, so the
    #                                           ensemble teaches the emulator wall systems.
    #                                           A LIST, e.g. [1, 3, 5, 10], draws uniformly from
    #                                           those counts, so each system size is equally
    #                                           represented.
    # SAGIS context data (paths) for conditioned siting; None = generic elevation/NLCD siting
    wetlands: Optional[str] = None            # NWI geojson -> marsh footprint
    soils_geojson: Optional[str] = None       # SSURGO soils geojson (MUKEY) -> de-pave Ksat cap
    ssurgo_table: Optional[str] = None        # fetch_ssurgo ksat table json (with soils_geojson)
    buildings: Optional[str] = None           # FEMA footprints geojson -> retreat siting
    roads: Optional[str] = None               # Overture segments geojson -> road_raise siting

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
    datums: Datums = field(default_factory=Datums)
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
        """Calendar UTC datetime of landfall (model t=0), or None if unset.

        Checked against the measured model epoch. `landfall_utc - landfall_s` is model
        time zero, and `build_bdy.MODEL_T0_UTC` holds the same quantity measured by
        cross-correlating the boundary surge against the Fort Pulaski record. Nothing
        used to compare them, so when the epoch was measured the configs kept an earlier
        guess and every series placed on the model clock was shifted by 7 hours: the
        tide by a near half-cycle, and the rainfall relative to the surge.
        """
        if self.coupling.landfall_utc is None:
            return None
        lf = datetime.fromisoformat(self.coupling.landfall_utc)
        self._check_epoch(lf)
        return lf

    def _check_epoch(self, landfall_dt, tol_s=60.0):
        """Warn if the config epoch disagrees with the measured one."""
        from datetime import timedelta
        try:
            from .couple.build_bdy import MODEL_T0_UTC
        except ImportError:
            return
        measured = datetime.fromisoformat(MODEL_T0_UTC)
        implied = landfall_dt - timedelta(seconds=self.coupling.landfall_s)
        off = (implied - measured).total_seconds()
        if abs(off) > tol_s:
            import warnings
            warnings.warn(
                f"model epoch mismatch: landfall_utc - landfall_s = {implied.isoformat()}, "
                f"but the measured epoch is {measured.isoformat()} ({off/3600:+.2f} h). "
                "Tide and rainfall series built from this config will be shifted by that "
                "amount relative to the surge.", stacklevel=2)

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
    """Load base.yaml then overlay the scenario YAML, return a Scenario.

    base is resolved next to the scenario file when the given path does not exist, so a tool
    run from a subdirectory still gets the shared defaults. Silently falling back to dataclass
    defaults is worse than failing: a run from the wrong directory used amr_max 6 while the
    config said 7, and nothing said so.
    """
    base = Path(base)
    if not base.exists():
        near = Path(scenario_path).resolve().parent.parent / "base.yaml"
        if near.exists():
            base = near
        else:
            raise SystemExit(
                f"base config not found at {base} or {near}. Run from the repo root, or pass "
                f"base= explicitly; without it every shared default falls back silently.")
    data = yaml.safe_load(open(base))
    data = _merge(data, yaml.safe_load(open(scenario_path)))
    sub = {k: globals()[cls](**data[k]) for k, cls in {
        "domain": "Domain", "geoclaw": "GeoClaw", "coupling": "Coupling",
        "manning": "Manning", "lisflood": "LisfloodCfg", "forcing": "Forcing",
        "datums": "Datums",
        "hpc": "HPC", "interventions": "Interventions"}.items() if k in data}
    return Scenario(name=data["name"], **sub)
