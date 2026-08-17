"""
CORAL Project — Study Domain Maps for Chapman 2026 Poster
Produces TWO separate figures:
  Fig 1: Regional context (SE US + GeoClaw domain + Matthew + Dorian tracks)
  Fig 2: Local Savannah/Pin Point domain (topobathy + SSLS sensors + gauges + tracks)

Tracks use real ATCF/HURDAT2 coordinates parsed directly from bal142016.dat
(already downloaded by setrun.py). Dorian uses hardcoded HURDAT2 coords (no
local ATCF file) but trimmed to Bahamas start.

Run with:  conda run -n aislens python3 src/figures/make_domain_map.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import json, glob, os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.windows import from_bounds

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEM_PATH    = os.path.join(REPO_ROOT, '..', 'coastalFlood',
                           'data', 'GoeClaw - Dorian', 'ga_cudem_30m.tif')
SENSOR_DIR  = os.path.join(REPO_ROOT, 'archive', 'quick-prototype',
                           'coral-quick', 'data', 'processed',
                           'features', 'most-reliable-sensors')
ATCF_MATTHEW = os.path.join(REPO_ROOT, '..', 'coastalFlood',
                             'matthew_2016', 'scratch', 'bal142016.dat')
OUT_DIR     = os.path.join(REPO_ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

proj = ccrs.PlateCarree()

# Poster colour palette (matches poster header #1a2c52 and accent coral)
C_MATTHEW = '#d62728'    # red
C_DORIAN  = '#1f77b4'    # blue
C_ACCENT  = '#e05c00'    # orange/coral — study area
C_DOMAIN  = '#2d4373'    # dark navy — GeoClaw domain
C_GAUGE   = '#1a2c52'    # poster header navy — Fort Pulaski marker
C_SSLS    = '#ff7f0e'    # orange triangles

# ── SSLS sensor coordinates ────────────────────────────────────────────────
sensor_files = glob.glob(os.path.join(SENSOR_DIR, '*.json'))
sensors = []
for f in sorted(sensor_files):
    try:
        d = json.load(open(f))
        name = (os.path.basename(f)
                .replace('most_reliable_sensors_subset_', '')
                .replace('_processed.json', ''))
        if 'lat' in d and 'lon' in d and name != 'processed':
            sensors.append({'name': name, 'lat': d['lat'], 'lon': d['lon']})
    except Exception:
        pass
sensor_lats = np.array([s['lat'] for s in sensors])
sensor_lons = np.array([s['lon'] for s in sensors])
print(f"Loaded {len(sensors)} SSLS sensors")

# ── Key locations ──────────────────────────────────────────────────────────
FORT_PULASKI = (-80.9017, 32.0347)
PIN_POINT    = (-81.060,  31.965)
SAVANNAH     = (-81.100,  32.083)
TYBEE        = (-80.845,  32.003)

# ── Parse REAL Matthew track from downloaded ATCF file ─────────────────────
def parse_atcf(path):
    """Parse ATCF best-track file → unique 6-hourly positions."""
    lons, lats = [], []
    seen = set()
    with open(path) as f:
        for line in f:
            parts = line.split(',')
            if len(parts) < 8:
                continue
            date = parts[2].strip()
            if date in seen:
                continue
            seen.add(date)
            lat_s = parts[6].strip()   # e.g. '316N'
            lon_s = parts[7].strip()   # e.g. '806W'
            lat = float(lat_s[:-1]) / 10.0 * (1 if lat_s[-1] == 'N' else -1)
            lon = float(lon_s[:-1]) / 10.0 * (-1 if lon_s[-1] == 'W' else 1)
            lons.append(lon)
            lats.append(lat)
    return np.array(lons), np.array(lats)

matthew_lon_full, matthew_lat_full = parse_atcf(ATCF_MATTHEW)
# Trim to Bahamas start (~23°N) for cleaner visual
trim = matthew_lat_full >= 23.0
matthew_lon_full = matthew_lon_full[trim]
matthew_lat_full = matthew_lat_full[trim]
print(f"Matthew track: {len(matthew_lon_full)} points  "
      f"lon=[{matthew_lon_full.min():.1f}, {matthew_lon_full.max():.1f}]  "
      f"lat=[{matthew_lat_full.min():.1f}, {matthew_lat_full.max():.1f}]")

# Closest approach to Savannah/GA coast from real data
# Oct 7 06Z: 31.6N, -80.6W  — offshore of Jekyll/Cumberland Islands
matthew = dict(
    lon=matthew_lon_full, lat=matthew_lat_full,
    approach_lon=-80.6, approach_lat=31.6,   # offshore, not landfall
    label='Matthew (Oct 2016)', color=C_MATTHEW, ls='-',
)

# ── Dorian 2019 track (HURDAT2 coords, trimmed to Bahamas) ────────────────
# Dorian's famous stall over the Bahamas, then NE up the US coast
dorian = dict(
    lon=np.array([-76.5, -77.5, -77.8, -77.6, -77.3, -77.5, -78.0, -78.5,
                  -79.0, -79.4, -79.8, -80.1, -80.3, -80.5, -80.6, -80.6,
                  -80.5, -80.2, -79.5, -78.5, -77.0, -75.5, -73.5]),
    lat=np.array([23.5,  25.0,  26.3,  26.5,  26.5,  26.5,  26.6,  26.6,
                   26.6,  26.7,  27.2,  28.0,  29.0,  29.9,  30.8,  31.7,
                   32.5,  33.3,  34.2,  35.0,  35.5,  36.0,  37.0]),
    approach_lon=-80.55, approach_lat=31.7,   # closest approach to GA, offshore
    label='Dorian (Sep 2019)', color=C_DORIAN, ls='--',
)

# ── Custom topo-bathy colormap ─────────────────────────────────────────────
topo_bathy_colors = [
    (0.00, '#08306b'),
    (0.20, '#2171b5'),
    (0.38, '#9ecae1'),
    (0.47, '#deebf7'),
    (0.50, '#f7fcf5'),
    (0.54, '#c7e9c0'),
    (0.62, '#74c476'),
    (0.75, '#d9b072'),
    (0.88, '#a07040'),
    (1.00, '#f0f0f0'),
]
topo_cmap = LinearSegmentedColormap.from_list(
    'topobathy', [(v, c) for v, c in topo_bathy_colors])

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Regional Context Map  (portrait, good for poster inset)
# ══════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(6.5, 8), subplot_kw={'projection': proj})
fig1.patch.set_facecolor('white')

ax.set_extent([-90, -60, 18, 46], crs=proj)
ax.add_feature(cfeature.OCEAN,     facecolor='#cde7f7', zorder=0)
ax.add_feature(cfeature.LAND,      facecolor='#e8e4da', zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6,        zorder=2)
ax.add_feature(cfeature.STATES,    linewidth=0.35, edgecolor='#aaa', zorder=2)
ax.add_feature(cfeature.BORDERS,   linewidth=0.6,        zorder=2)

# GeoClaw domain box
domain_rect = mpatches.Rectangle(
    (-85, 20), 25, 25,
    linewidth=1.8, edgecolor=C_DOMAIN, facecolor=C_DOMAIN + '08',
    linestyle='--', zorder=3, transform=proj)
ax.add_patch(domain_rect)
ax.text(-84.5, 20.6, 'GeoClaw domain', fontsize=7,
        color=C_DOMAIN, fontstyle='italic', transform=proj)

# Storm tracks
for storm in [matthew, dorian]:
    lons, lats = storm['lon'], storm['lat']
    ax.plot(lons, lats, color=storm['color'], lw=1.8, ls=storm['ls'],
            zorder=4, transform=proj, label=storm['label'])
    ax.plot(lons[::4], lats[::4], 'o', color=storm['color'],
            ms=3.5, zorder=4, transform=proj)
    # closest-approach marker (not labelled as "landfall" for GA)
    ax.plot(storm['approach_lon'], storm['approach_lat'],
            '*', color=storm['color'], ms=13, zorder=5,
            markeredgecolor='white', markeredgewidth=0.6, transform=proj)

# Study area box
study_rect = mpatches.Rectangle(
    (-81.4, 31.7), 0.8, 0.65,
    linewidth=2, edgecolor=C_ACCENT, facecolor=C_ACCENT + '20',
    zorder=5, transform=proj)
ax.add_patch(study_rect)
ax.annotate('Study area\n(Savannah / Pin Point)',
            xy=(-81.0, 32.35), xytext=(-85.0, 35.5),
            fontsize=8, color=C_ACCENT, fontweight='bold', transform=proj,
            arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=1.2))

# Legend — storms + boxes, no duplicates
storm_handles, _ = ax.get_legend_handles_labels()
extra = [
    mpatches.Patch(facecolor=C_DOMAIN + '08', edgecolor=C_DOMAIN,
                   linestyle='--', linewidth=1.5, label='GeoClaw domain'),
    mpatches.Patch(facecolor=C_ACCENT + '20', edgecolor=C_ACCENT,
                   linewidth=1.5, label='Study area'),
]
ax.legend(handles=storm_handles + extra, loc='lower right',
          fontsize=8, framealpha=0.9)

ax.set_title('GeoClaw Model Domain — SE US Atlantic Coast',
             fontsize=10, fontweight='bold', pad=8)
# Caption
fig1.text(0.5, 0.01,
    'Fig. 1  SE US Atlantic domain for GeoClaw storm surge simulations.\n'
    'Best-track positions (HURDAT2/ATCF) shown at 6-hourly intervals.\n'
    '★ = closest approach to study area.',
    ha='center', fontsize=7, color='#444', style='italic',
    wrap=True)

gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
gl.top_labels = gl.right_labels = False
plt.tight_layout(rect=[0, 0.06, 1, 1])
out1 = os.path.join(OUT_DIR, 'fig_domain_regional.png')
fig1.savefig(out1, dpi=300, bbox_inches='tight', facecolor=fig1.get_facecolor())
print(f"Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Local Savannah / Pin Point Domain (portrait)
# ══════════════════════════════════════════════════════════════════════════
LOCAL = dict(x0=-81.35, x1=-80.78, y0=31.75, y1=32.28)

fig2, ax2 = plt.subplots(figsize=(6.5, 7), subplot_kw={'projection': proj})
fig2.patch.set_facecolor('white')
ax2.set_extent([LOCAL['x0'], LOCAL['x1'], LOCAL['y0'], LOCAL['y1']], crs=proj)

# ── DEM ────────────────────────────────────────────────────────────────────
if os.path.exists(DEM_PATH):
    with rasterio.open(DEM_PATH) as ds:
        window = from_bounds(LOCAL['x0'], LOCAL['y0'],
                             LOCAL['x1'], LOCAL['y1'], ds.transform)
        Z = ds.read(1, window=window).astype(float)
        nodata = ds.nodata
        if nodata is not None:
            Z[Z == nodata] = np.nan

    Z_display = Z.copy()
    Z_display[np.isnan(Z_display)] = -20.0   # offshore → ocean blue

    z_min = np.nanpercentile(Z, 1)
    z_max = np.nanpercentile(Z, 99)
    norm  = TwoSlopeNorm(vmin=max(z_min, -30), vcenter=0, vmax=min(z_max, 12))

    extent = [LOCAL['x0'], LOCAL['x1'], LOCAL['y0'], LOCAL['y1']]
    im = ax2.imshow(np.flipud(Z_display), extent=extent, origin='upper',
                    cmap=topo_cmap, norm=norm, zorder=1, aspect='auto',
                    transform=proj)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.55, pad=0.02, extend='both')
    cbar.set_label('Elevation / Depth (m, NAVD88)', fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
else:
    ax2.set_facecolor('#cde7f7')

ax2.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=3, edgecolor='#333')
ax2.add_feature(cfeature.RIVERS,    linewidth=0.8, zorder=3,
                edgecolor='#4a90d9', facecolor='none')

# Storm tracks are shown on the regional inset only — omitted from local map for clarity

# ── SSLS sensors ──────────────────────────────────────────────────────────
ax2.scatter(sensor_lons, sensor_lats,
            marker='^', s=50, c=C_SSLS, edgecolors='white',
            linewidths=0.7, zorder=7, transform=proj,
            label=f'SSLS sensors — Dorian cal. (TBD, n={len(sensors)})')

# ── Fort Pulaski gauge ─────────────────────────────────────────────────────
ax2.plot(*FORT_PULASKI, 'D', color=C_GAUGE, ms=9,
         markeredgecolor='white', markeredgewidth=0.9,
         zorder=8, transform=proj, label='Fort Pulaski NOAA (8670870)')

# ── Labels ────────────────────────────────────────────────────────────────
stroke = [pe.withStroke(linewidth=2.5, foreground='white')]
lkw = dict(transform=proj, zorder=9, fontsize=8.5, fontweight='bold',
           path_effects=stroke)

ax2.annotate('Fort Pulaski\nNOAA Gauge',
             xy=FORT_PULASKI,
             xytext=(FORT_PULASKI[0] - 0.10, FORT_PULASKI[1] + 0.08),
             fontsize=7, color=C_GAUGE, fontweight='bold',
             transform=proj, zorder=9,
             arrowprops=dict(arrowstyle='->', color=C_GAUGE, lw=0.9),
             path_effects=stroke)
ax2.text(SAVANNAH[0],  SAVANNAH[1]  + 0.02, 'Savannah',
         ha='center', color='#111', **lkw)
ax2.text(PIN_POINT[0], PIN_POINT[1] - 0.025, 'Pin Point',
         ha='center', color='#222', **lkw)
ax2.text(TYBEE[0],     TYBEE[1]     + 0.02, 'Tybee Is.',
         ha='center', color='#333', fontsize=7.5,
         path_effects=stroke, transform=proj, zorder=9)
ax2.text(-81.04, 32.085, 'Savannah River', ha='center',
         color='#2255aa', fontstyle='italic', fontsize=7,
         transform=proj, zorder=9, path_effects=stroke)

ax2.set_title('Study Domain: Savannah / Pin Point, GA',
              fontsize=10, fontweight='bold', pad=8)
ax2.legend(loc='upper left', fontsize=7, framealpha=0.92)

fig2.text(0.5, 0.01,
    'Fig. 2  Study domain topobathy (NOAA CUDEM, ~30 m, NAVD88). '
    'Track positions at 6-hourly intervals.\n'
    'SSLS sensor network (2019–present) shown for ongoing Dorian-based calibration (TBD). '
    'Fort Pulaski NOAA gauge (8670870) used for model validation.',
    ha='center', fontsize=7, color='#444', style='italic', wrap=True)

gl2 = ax2.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
gl2.top_labels = gl2.right_labels = False
plt.tight_layout(rect=[0, 0.06, 1, 1])
out2 = os.path.join(OUT_DIR, 'fig_domain_local.png')
fig2.savefig(out2, dpi=300, bbox_inches='tight', facecolor=fig2.get_facecolor())
print(f"Saved: {out2}")
print("Done. Figures saved to results/figures/")
