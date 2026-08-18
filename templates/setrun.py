# encoding: utf-8
"""
Module to set up run time parameters for Clawpack.

The values set in the function setrun are then written out to data files
that will be read in by the Fortran code.

"""

from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
import gzip

import numpy as np

# Scenario config, so sea level, AMR depth and the refine box come from the same yaml the
# LISFLOOD side reads. Set CORAL_SCENARIO to point at a different one. If coral is not on the
# path -- setrun runs inside the clawpack environment, which may not have it -- the run
# continues on the defaults below and says so, rather than dying at make .data.
_CFG = None
try:
    from coral import config as _coral_config
    # CORAL_SCENARIO wins. Otherwise fall back to a path under $CORAL, because setrun runs
    # from the run directory and a relative default resolves against that instead of the repo.
    _sc = os.environ.get("CORAL_SCENARIO")
    if not _sc:
        _root = os.environ.get("CORAL", "")
        _sc = os.path.join(_root, "configs", "scenarios",
                           "savannah_matthew_compound.yaml")
    _CFG = _coral_config.load(_sc)
    print("setrun: scenario %s, sea_level %.2f, amr_max %d"
          % (_CFG.name, _CFG.geoclaw.sea_level, _CFG.geoclaw.amr_max))
except Exception as _e:
    print("setrun: no scenario config (%s); using the values written in this file" % _e)

from clawpack.geoclaw.surge.storm import Storm
from clawpack.geoclaw import topotools
import clawpack.clawutil as clawutil

from clawpack.amrclaw import region_tools
from clawpack.amrclaw.data import FlagRegion


# Time Conversions
def days2seconds(days):
    return days * 60.0 ** 2 * 24.0


# Scratch directory for storing topo and storm files:
# scratch_dir = os.path.join(os.environ["CLAW"], 'geoclaw', 'scratch')

# Second option for location of scratch directory
scratch_dir = os.path.join(os.getcwd(), 'scratch')
if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)


# ------------------------------
def setrun(claw_pkg='geoclaw'):
    """
    Define the parameters used for running Clawpack.

    INPUT:
        claw_pkg expected to be "geoclaw" for this setrun.

    OUTPUT:
        rundata - object of class ClawRunData

    """

    from clawpack.clawutil import data

    assert claw_pkg.lower() == 'geoclaw', "Expected claw_pkg = 'geoclaw'"

    num_dim = 2
    rundata = data.ClawRunData(claw_pkg, num_dim)

    # ------------------------------------------------------------------
    # Standard Clawpack parameters to be written to claw.data:
    #   (or to amr2ez.data for AMR)
    # ------------------------------------------------------------------
    clawdata = rundata.clawdata  # initialized when rundata instantiated

    # Set single grid parameters first.
    # See below for AMR parameters.

    # ---------------
    # Spatial domain:
    # ---------------

    # Number of space dimensions:
    clawdata.num_dim = num_dim

    # Lower and upper edge of computational domain:
    clawdata.lower[0] = -85  # west longitude
    clawdata.upper[0] = -60  # east longitude

    clawdata.lower[1] = 20  # south latitude
    clawdata.upper[1] = 45  # north latitude

    # Number of grid cells:
    degree_factor = 4  # (0.25º,0.25º) ~ (25237.5 m, 27693.2 m) resolution
    clawdata.num_cells[0] = int(clawdata.upper[0] - clawdata.lower[0]) * degree_factor
    clawdata.num_cells[1] = int(clawdata.upper[1] - clawdata.lower[1]) * degree_factor

    # ---------------
    # Size of system:
    # ---------------

    # Number of equations in the system:
    clawdata.num_eqn = 3

    # Number of auxiliary variables in the aux array (initialized in setaux)
    # First three are from shallow GeoClaw, fourth is friction and last 3 are
    # storm fields
    clawdata.num_aux = 3 + 1 + 3

    # Index of aux array corresponding to capacity function, if there is one:
    clawdata.capa_index = 2

    # -------------
    # Initial time:
    # -------------
    clawdata.t0 = -days2seconds(2)

    # Restart from checkpoint file of a previous run?
    # If restarting, t0 above should be from original run, and the
    # restart_file 'fort.chkNNNNN' specified below should be in
    # the OUTDIR indicated in Makefile.

    clawdata.restart = False  # True to restart from prior results
    clawdata.restart_file = 'fort.chk00006'  # File to use for restart data

    # -------------
    # Output times:
    # --------------

    # Specify at what times the results should be written to fort.q files.
    # Note that the time integration stops after the final output time.
    # The solution at initial time t0 is always written in addition.

    clawdata.output_style = 1

    if clawdata.output_style == 1:
        # Output nout frames at equally spaced times up to tfinal:
        # geoclaw.tfinal_h shortens the run for timing tests. Landfall is t = 0, so +24 h is
        # the production value and 2-3 h is enough to read the gauge level and the real rate.
        clawdata.tfinal = (_CFG.geoclaw.tfinal_h * 3600.0
                           if (_CFG and _CFG.geoclaw.tfinal_h is not None)
                           else days2seconds(1))
        recurrence = 4
        clawdata.num_output_times = int((clawdata.tfinal - clawdata.t0) *
                                        recurrence / (60 ** 2 * 24))

        clawdata.output_t0 = True  # output at initial (or restart) time?

    elif clawdata.output_style == 2:
        # Specify a list of output times.
        clawdata.output_times = [0.5, 1.0]

    elif clawdata.output_style == 3:
        # Output every iout timesteps with a total of ntot time steps:
        clawdata.output_step_interval = 1
        clawdata.total_steps = 1
        clawdata.output_t0 = True

    clawdata.output_format = 'ascii'  # 'ascii' or 'binary'
    clawdata.output_q_components = 'all'  # could be list such as [True,True]
    clawdata.output_aux_components = 'all'
    clawdata.output_aux_onlyonce = False  # output aux arrays only at t0

    # ---------------------------------------------------
    # Verbosity of messages to screen during integration:
    # ---------------------------------------------------

    # The current t, dt, and cfl will be printed every time step
    # at AMR levels <= verbosity.  Set verbosity = 0 for no printing.
    #   (E.g. verbosity == 2 means print only on levels 1 and 2.)
    clawdata.verbosity = 0

    # --------------
    # Time stepping:
    # --------------

    # if dt_variable==1: variable time steps used based on cfl_desired,
    # if dt_variable==0: fixed time steps dt = dt_initial will always be used.
    clawdata.dt_variable = True

    # Initial time step for variable dt.
    # If dt_variable==0 then dt=dt_initial for all steps:
    clawdata.dt_initial = 0.016

    # Max time step to be allowed if variable dt used:
    clawdata.dt_max = 1e+99

    # Desired Courant number if variable dt used, and max to allow without
    # retaking step with a smaller dt:
    clawdata.cfl_desired = 0.75
    clawdata.cfl_max = 1.0

    # Maximum number of time steps to allow between output times:
    clawdata.steps_max = 10000  # changed from 5000

    # ------------------
    # Method to be used:
    # ------------------

    # Order of accuracy:  1 => Godunov,  2 => Lax-Wendroff plus limiters
    clawdata.order = 1  # changed from 1

    # Use dimensional splitting? (not yet available for AMR)
    clawdata.dimensional_split = 'unsplit'

    # For unsplit method, transverse_waves can be
    #  0 or 'none'      ==> donor cell (only normal solver used)
    #  1 or 'increment' ==> corner transport of waves
    #  2 or 'all'       ==> corner transport of 2nd order corrections too
    clawdata.transverse_waves = 1  # changed from 2

    # Number of waves in the Riemann solution:
    clawdata.num_waves = 3

    # List of limiters to use for each wave family:
    # Required:  len(limiter) == num_waves
    # Some options:
    #   0 or 'none'     ==> no limiter (Lax-Wendroff)
    #   1 or 'minmod'   ==> minmod
    #   2 or 'superbee' ==> superbee
    #   3 or 'mc'       ==> MC limiter
    #   4 or 'vanleer'  ==> van Leer
    clawdata.limiter = ['mc', 'mc', 'mc']

    clawdata.use_fwaves = True  # True ==> use f-wave version of algorithms

    # Source terms splitting:
    #   src_split == 0 or 'none'
    #      ==> no source term (src routine never called)
    #   src_split == 1 or 'godunov'
    #      ==> Godunov (1st order) splitting used,
    #   src_split == 2 or 'strang'
    #      ==> Strang (2nd order) splitting used,  not recommended.
    clawdata.source_split = 'godunov'

    # --------------------
    # Boundary conditions:
    # --------------------

    # Number of ghost cells (usually 2)
    clawdata.num_ghost = 2

    # Choice of BCs at xlower and xupper:
    #   0 => user specified (must modify bcN.f to use this option)
    #   1 => extrapolation (non-reflecting outflow)
    #   2 => periodic (must specify this at both boundaries)
    #   3 => solid wall for systems where q(2) is normal velocity

    clawdata.bc_lower[0] = 'extrap'
    clawdata.bc_upper[0] = 'extrap'

    clawdata.bc_lower[1] = 'extrap'
    clawdata.bc_upper[1] = 'extrap'

    # Specify when checkpoint files should be created that can be
    # used to restart a computation.

    clawdata.checkpt_style = 0

    if clawdata.checkpt_style == 0:
        # Do not checkpoint at all
        pass

    elif np.abs(clawdata.checkpt_style) == 1:
        # Checkpoint only at tfinal.
        pass

    elif np.abs(clawdata.checkpt_style) == 2:
        # Specify a list of checkpoint times.
        clawdata.checkpt_times = [0.1, 0.15]

    elif np.abs(clawdata.checkpt_style) == 3:
        # Checkpoint every checkpt_interval timesteps (on Level 1)
        # and at the final time.
        clawdata.checkpt_interval = 5

    # ---------------
    # AMR parameters:
    # ---------------
    amrdata = rundata.amrdata

    # allocate memory before running the simulation to prevent crash at high refinement levels
    # amrdata.memsize = 16777212

    # max number of refinement levels:
    # Level 6 is about 145 m, coarse for a boundary feeding a 4 m nest. A seventh at ratio 4
    # gives about 36 m, and applies only inside refine_box; level 7 across the domain costs a
    # great deal and improves nothing offshore.
    amrdata.amr_levels_max = _CFG.geoclaw.amr_max if _CFG else 7

    # List of refinement ratios at each level (length at least amr_max_levels-1)
    # ratios start at level 2 (ratio 4 is to get from level 5 to level 6)
    # One ratio per step, so the list has to be amr_max - 1 long. Sliced from the full ladder
    # rather than written out, so changing amr_max in the scenario is enough to run the
    # convergence test at level 6 against level 7.
    #   0.25 deg -> 13.9 km -> 6.9 km -> 3.5 km -> 578 m -> 145 m -> 36 m
    _RATIOS = [2, 2, 2, 6, 4, 4]
    _r = _RATIOS[:amrdata.amr_levels_max - 1]
    amrdata.refinement_ratios_x = list(_r)
    amrdata.refinement_ratios_y = list(_r)
    amrdata.refinement_ratios_t = list(_r)

    # Specify type of each aux variable in amrdata.auxtype.
    # This must be a list of length maux, each element of which is one of:
    #   'center',  'capacity', 'xleft', or 'yleft'  (see documentation).

    amrdata.aux_type = ['center', 'capacity', 'yleft', 'center', 'center',
                        'center', 'center']

    # Flag using refinement routine flag2refine rather than richardson error
    amrdata.flag_richardson = False  # use Richardson?
    amrdata.flag2refine = True

    # steps to take on each level L between regriddings of level L+1:
    amrdata.regrid_interval = 3

    # width of buffer zone around flagged points:
    # (typically the same as regrid_interval so waves don't escape):
    amrdata.regrid_buffer_width = 2

    # clustering alg. cutoff for (# flagged pts) / (total # of cells refined)
    # (closer to 1.0 => more small grids may be needed to cover flagged cells)
    amrdata.clustering_cutoff = 0.700000

    # print info about each regridding up to this level:
    amrdata.verbosity_regrid = 0

    #  ----- For developers -----
    # Toggle debugging print statements:
    amrdata.dprint = False  # print domain flags
    amrdata.eprint = False  # print err est flags
    amrdata.edebug = False  # even more err est flags
    amrdata.gprint = False  # grid bisection/clustering
    amrdata.nprint = False  # proper nesting output
    amrdata.pprint = False  # proj. of tagged points
    amrdata.rprint = False  # print regridding summary
    amrdata.sprint = False  # space/memory output
    amrdata.tprint = False  # time step reporting each level
    amrdata.uprint = False  # update/upbnd reporting

    # More AMR parameters can be set -- see the defaults in pyclaw/data.py

    # == setregions.data values ==
    regions = rundata.regiondata.regions
    # to specify regions of refinement append lines of the form
    #  [minlevel,maxlevel,t1,t2,x1,x2,y1,y2]
    # Entire domain region - to decrease run time
    regions.append([1, 3, rundata.clawdata.t0, rundata.clawdata.tfinal, clawdata.lower[0], clawdata.upper[0],
                    clawdata.lower[1], clawdata.upper[1]])

    # Pin Point / Savannah coupling box -- force the finest level ONLY here.
    # Everything else stays coarse; the surge wave self-refines en route (wave_tolerance=1.0).
    _lvl = _CFG.geoclaw.amr_max if _CFG else 7
    _box = list(_CFG.geoclaw.refine_box) if (_CFG and _CFG.geoclaw.refine_box) \
        else [-81.111, -80.819, 31.804, 32.100]

    # Forced refinement can start late. The .bdy only consumes coupling.sim_window_h and the
    # ocean before that is quiescent, so holding the box at the finest level from t0 pays for
    # a day and a half of model time that is thrown away.
    _t1 = (_CFG.geoclaw.refine_t1_h * 3600.0
           if (_CFG and _CFG.geoclaw.refine_t1_h is not None)
           else rundata.clawdata.t0)
    if _t1 < rundata.clawdata.t0:
        _t1 = rundata.clawdata.t0
    _t2 = (_CFG.geoclaw.refine_t2_h * 3600.0
           if (_CFG and _CFG.geoclaw.refine_t2_h is not None)
           else rundata.clawdata.tfinal)
    if _t2 > rundata.clawdata.tfinal:
        _t2 = rundata.clawdata.tfinal

    _front_km = _CFG.geoclaw.refine_front_km if _CFG else None
    if _front_km:
        # The gauges lie on a curve. Refine a band around the curve instead of its bounding
        # box, as a ruled rectangle -- the same shape kml2slu builds for the other regions.
        from coral.geoclaw.refine_front import write as _write_front
        _front_file = os.path.abspath('RuledRectangle_front.data')
        _csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'boundary_points.csv')
        _slu = _write_front(_csv, _front_file, _front_km)
        _fr = FlagRegion(num_dim=2)
        _fr.name = 'Region_front'
        _fr.minlevel = _lvl
        _fr.maxlevel = _lvl
        _fr.t1 = _t1
        _fr.t2 = _t2
        _fr.spatial_region_type = 2          # Ruled Rectangle
        _fr.spatial_region_file = _front_file
        rundata.flagregiondata.flagregions.append(_fr)
        print('setrun: level %d forced in a %.1f km band on %d gauge latitudes, '
              't = %.0f to %.0f s' % (_lvl, _front_km, _slu.shape[0] - 2, _t1, _t2))
        # The box still caps refinement so nothing outside the band reaches _lvl by flagging.
        regions.append([1, max(_lvl - 1, 1), rundata.clawdata.t0, rundata.clawdata.tfinal,
                        *_box])
    else:
        regions.append([_lvl, _lvl, _t1, _t2, *_box])

    # append as many flagregions as desired to this list:
    flagregions = rundata.flagregiondata.flagregions

    from kml2slu import kml2slu
    # Coverts .kml file with polygons drawn in Google Earth to slu format
    slus = kml2slu("regions.kml")

    flag_regions = {"mayport": {"levels": (6, 6),
                                "slu": slus.get("mayport")},
                    "pulaski": {"levels": (4, 6),
                                "slu": slus.get("pulaski")},
                    "charleston": {"levels": (4, 6),
                                   "slu": slus.get("charleston")},
                    "wilmington": {"levels": (6, 6),
                                   "slu": slus.get("wilmington")}}

    for (name, region_dict) in flag_regions.items():
        # write RuledRectangle .data file
        rr = region_tools.RuledRectangle(slu=region_dict["slu"])
        rr.ixy = 'y'
        rr.method = 1
        rr.write('RuledRectangle_%s.data' % name)

        # use RuledRectangle .data file and desired refinement levels to append to flagregions
        flagregion = FlagRegion(num_dim=2)
        flagregion.name = 'Region_' + name
        flagregion.minlevel = region_dict["levels"][0]
        flagregion.maxlevel = region_dict["levels"][1]
        flagregion.t1 = rundata.clawdata.t0
        flagregion.t2 = rundata.clawdata.tfinal
        flagregion.spatial_region_type = 2  # Ruled Rectangle
        flagregion.spatial_region_file = os.path.abspath('RuledRectangle_%s.data' % name)
        flagregions.append(flagregion)

    # == setgauges.data values ==
    #
    # Coupling gauges come from boundary_points.csv, written by gen_boundary_points alongside
    # the .bci. They were inlined until the spacing went from 400 m to 100 m and the count from
    # 63 to 436, at which point pasting them in was no longer reasonable -- and an inlined block
    # that drifts from the .bci is a boundary that silently does not match its own forcing.
    #
    #   cd inputs && python -m coral.preprocess.gen_boundary_points \
    #       --dem <30 m DEM> --config configs/scenarios/savannah_matthew_tide.yaml \
    #       --spacing-m 100
    #
    # IDs are 1..N and must stay in the file's order, because build_bdy matches gauge N to bcN.
    gauge_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'boundary_points.csv')
    n_coupling = 0
    if os.path.exists(gauge_csv):
        rows = np.atleast_2d(np.genfromtxt(gauge_csv, delimiter=',', skip_header=1,
                                           usecols=(0, 2, 3)))
        for gid, glon, glat in rows:
            rundata.gaugedata.gauges.append([int(gid), float(glon), float(glat),
                                             rundata.clawdata.t0, rundata.clawdata.tfinal])
        n_coupling = len(rows)
        print('setrun: %d coupling gauges from %s' % (n_coupling, gauge_csv))
    else:
        raise SystemExit('setrun: %s not found. Run gen_boundary_points first; without the '
                         'coupling gauges there is no boundary.' % gauge_csv)

    # NOAA validation stations, at 9001+ so they cannot collide with the coupling range however
    # many gauges that becomes. Not in the .bci, so they show as unreferenced and are ignored.
    for gid, glon, glat, label in (
            (9001, -81.427915, 30.398600, 'Mayport FL 8720218'),
            (9002, -80.903052, 32.034668, 'Fort Pulaski GA 8670870'),
            (9003, -79.923646, 32.780783, 'Charleston SC 8665530'),
            (9004, -77.786275, 34.213270, 'Wrightsville Beach NC 8658163'),
            (9005, -77.953000, 34.226667, 'Wilmington NC 8658120')):
        rundata.gaugedata.gauges.append([gid, glon, glat,
                                         rundata.clawdata.t0, rundata.clawdata.tfinal])
    print('setrun: 5 NOAA stations at 9001-9005 (Fort Pulaski is 9002)')

    # Force the gauges to also record the wind and pressure fields
    # rundata.gaugedata.aux_out_fields = [4, 5, 6]

    # ------------------------------------------------------------------
    # GeoClaw specific parameters:
    # ------------------------------------------------------------------
    rundata = setgeo(rundata)

    return rundata
    # end of function setrun
    # ----------------------


# -------------------
def setgeo(rundata):
    """
    Set GeoClaw specific runtime parameters.
    For documentation see ....
    """

    geo_data = rundata.geo_data

    # == Physics ==
    geo_data.gravity = 9.81
    geo_data.coordinate_system = 2
    geo_data.earth_radius = 6367.5e3
    geo_data.rho = 1025.0
    geo_data.rho_air = 1.15
    geo_data.ambient_pressure = 101.3e3

    # == Forcing Options
    geo_data.coriolis_forcing = True
    geo_data.friction_forcing = True
    geo_data.friction_depth = 1e10

    # == Algorithm and Initial Conditions ==
    # Note that in the original paper due to gulf summer swelling this was set
    # to 0.28
    # From the scenario, so this cannot drift from what the LISFLOOD side was told. 0.0 puts
    # the tide in the boundary series rather than the datum; with it in the datum, eta in every
    # fort.q carries the same offset and the surge cannot be read as an anomaly.
    geo_data.sea_level = _CFG.geoclaw.sea_level if _CFG else 0.0
    geo_data.dry_tolerance = 1.e-2

    # Refinement Criteria
    refine_data = rundata.refinement_data
    refine_data.wave_tolerance = 1.0
    refine_data.speed_tolerance = [1.0, 2.0, 3.0, 4.0]
    refine_data.deep_depth = 300.0
    refine_data.max_level_deep = 4
    refine_data.variable_dt_refinement_ratios = True

    # == settopo.data values ==
    topo_data = rundata.topo_data
    topo_data.topofiles = []
    topo_data.topo_missing = -32767
    # for topography, append lines of the form
    #   [topotype, fname]
    # See regions for control over these regions, need better bathy data for
    # the smaller domains
    clawutil.data.get_remote_file(
        "https://www.dropbox.com/s/s58bi1l45tw9uka/gebco_2020_n50.0_s10.0_w-90.0_e-60.0.asc?dl=1", scratch_dir,
        file_name="gebco_2020_n50.0_s10.0_w-90.0_e-60.0.asc", verbose=True)
    full_topo_path = os.path.join(scratch_dir, 'gebco_2020_n50.0_s10.0_w-90.0_e-60.0.asc')
    topo_data.topofiles.append([3, full_topo_path])

    clawutil.data.get_remote_file("https://www.ngdc.noaa.gov/thredds/fileServer/crm/crm_vol2.nc", scratch_dir,
                                  file_name="crm_vol2_se_atl.nc", verbose=True)
    southeast_topo_path = os.path.join(scratch_dir, 'crm_vol2_se_atl')
    topotools.read_netcdf((southeast_topo_path + ".nc"), extent=[-81.5, -77.0, 31.5, 34.8], verbose=True).write(
        (southeast_topo_path + ".asc"), topo_type=3, no_data_value=-32767, header_style="asc", Z_format='%.0f')
    topo_data.topofiles.append([3, (southeast_topo_path + ".asc")])

    # == setfixedgrids.data values ==
    rundata.fgout_data.fgout_grids = []
    # rundata.fixed_grid_data.fixedgrids = []
    # for fixed grids append lines of the form
    # [t1,t2,noutput,x1,x2,y1,y2,xpoints,ypoints,\
    #  ioutarrivaltimes,ioutsurfacemax]

    # ================
    #  Set Surge Data
    # ================
    data = rundata.surge_data

    # Source term controls
    data.wind_forcing = True
    data.drag_law = 2
    data.pressure_forcing = True

    data.display_landfall_time = True

    # AMR parameters, m/s and m respectively
    data.wind_refine = [20.0, 40.0, 60.0]
    data.R_refine = [60.0e3, 40e3, 20e3]

    # Storm parameters - Parameterized storm (Holland 1980)
    data.storm_specification_type = 'holland80'  # (type 1)
    data.storm_file = os.path.expandvars(os.path.join(os.getcwd(),
                                                      'matthew.storm'))

    # Convert ATCF data to GeoClaw format
    clawutil.data.get_remote_file(
        "https://ftp.nhc.noaa.gov/atcf/archive/2016/bal142016.dat.gz", scratch_dir)
    atcf_path = os.path.join(scratch_dir, "bal142016.dat")
    # Note that the get_remote_file function does not support gzip files which
    # are not also tar files.  The following code handles this
    with gzip.open(".".join((atcf_path, 'gz')), 'rb') as atcf_file, \
            open(atcf_path, 'w') as atcf_unzipped_file:
        atcf_unzipped_file.write(atcf_file.read().decode('ascii'))

    matthew = Storm(path=atcf_path, file_format="ATCF")

    # Calculate landfall time - Need to specify as the file above does not include (10/8/2016 ~ 12 UTC)
    matthew.time_offset = datetime.datetime(2016, 10, 8, 12)

    matthew.write(data.storm_file, file_format='geoclaw')

    # =======================
    #  Set Variable Friction
    # =======================
    data = rundata.friction_data

    # Variable friction
    data.variable_friction = True

    # Region based friction
    # Entire domain
    data.friction_regions.append([rundata.clawdata.lower,
                                  rundata.clawdata.upper,
                                  [np.inf, 0.0, -np.inf],
                                  [0.030, 0.022]])

    # Louisiana-Texas Shelf (abnormally smooth)
    data.friction_regions.append([(-98, 25.25), (-90, 30),
                                  [np.inf, -10.0, -200.0, -np.inf],
                                  [0.030, 0.012, 0.022]])

    return rundata
    # end of function setgeo
    # ----------------------


if __name__ == '__main__':
    # Set up run-time parameters and write all data files.
    import sys

    if len(sys.argv) == 2:
        rundata = setrun(sys.argv[1])
    else:
        rundata = setrun()

    rundata.write()
