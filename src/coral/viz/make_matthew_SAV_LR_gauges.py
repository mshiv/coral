"""
Overlay a selection of the matthew_SAV_LR coastline coupling gauges (1-63) to
see alongshore variation in surge. Same style as the Fort Pulaski progress plot.

Modelled surge = eta - sea_level (0.81). Lines colored by latitude (N->S) so the
spatial gradient is visible; colorbar shows latitude.

Run: MPLBACKEND=Agg ~/miniforge3/envs/ml-env/bin/python src/figures/make_matthew_SAV_LR_gauges.py
     [--every 9]  [--ids 1,16,32,48,63]
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import datetime, os, re, argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CF_ROOT   = os.path.join(REPO_ROOT, '..', 'coastalFlood')
OUT_DIR   = os.path.join(REPO_ROOT, 'results', 'figures')
OUTDIR_GA = os.path.join(CF_ROOT, 'matthew_SAV_LR', '_output')
os.makedirs(OUT_DIR, exist_ok=True)

SEA_LEVEL = 0.81
setrun_t0 = datetime.datetime(2016, 10, 8, 12)
ga_t0     = datetime.datetime(2016, 10, 8,  6)
shift_h   = (setrun_t0 - ga_t0).total_seconds() / 3600

C_MATTHEW, C_LANDFALL = '#d62728', '#666'
plt.rcParams.update({'font.family': 'Lato, DejaVu Sans, sans-serif', 'font.size': 16,
    'axes.titlesize': 22, 'axes.labelsize': 18, 'legend.fontsize': 13})


def read_gauge(path):
    lat = lon = None
    t, eta = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                m = re.search(r'location=\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)', line)
                if m:
                    lon, lat = float(m.group(1)), float(m.group(2))
                continue
            p = line.split()
            if len(p) >= 6:
                try:
                    t.append(float(p[1])); eta.append(float(p[5]))
                except ValueError:
                    pass
    return np.array(t), np.array(eta), lon, lat


ap = argparse.ArgumentParser()
ap.add_argument('--every', type=int, default=9, help='select every Nth gauge from 1..63')
ap.add_argument('--ids', default=None, help='explicit comma list, e.g. 1,16,32,48,63')
args = ap.parse_args()

if args.ids:
    ids = [int(x) for x in args.ids.split(',')]
else:
    ids = list(range(1, 64, args.every))

# load selected gauges
gs = []
for i in ids:
    p = os.path.join(OUTDIR_GA, f'gauge{i:05d}.txt')
    if not os.path.exists(p):
        continue
    t, eta, lon, lat = read_gauge(p)
    if len(t):
        gs.append((i, t, eta, lon, lat))
if not gs:
    raise SystemExit('no gauges loaded')

lats = [g[4] for g in gs if g[4] is not None]
norm = Normalize(vmin=min(lats), vmax=max(lats))
cmap = cm.viridis

# DEM for the inset map
def _read_ascii(path):
    h = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split(); h[k.lower()] = float(v)
        a = np.loadtxt(f)
    a = np.where((a == h.get('nodata_value', -9999)) | (a <= -9990), np.nan, a)
    nx, ny, cs = int(h['ncols']), int(h['nrows']), h['cellsize']
    return a, [h['xllcorner'], h['xllcorner'] + nx * cs,
               h['yllcorner'], h['yllcorner'] + ny * cs]

DEM_PATH = os.path.join(CF_ROOT, 'savannah_matthew_workflow', 'inputs', 'SUB_DEM_SAV.asc')
dem, dem_ext = (_read_ascii(DEM_PATH) if os.path.exists(DEM_PATH) else (None, None))

fig, ax = plt.subplots(figsize=(14, 5.2))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.30, linewidth=0.7)

reached = -1e9
for i, t, eta, lon, lat in gs:
    surge = eta - SEA_LEVEL
    th = t / 3600.0 + shift_h
    ax.plot(th, surge, color=cmap(norm(lat)), lw=1.8, alpha=0.9,
            label=f'g{i} ({lat:.3f}°N)')
    reached = max(reached, th.max())

ax.axvline(0, color=C_LANDFALL, lw=1.8, ls='--', alpha=0.85)
ax.axvline(reached, color='k', lw=1.2, ls=':', alpha=0.6)
ax.text(reached, 0.92, f'  reached {reached:+.0f} h', transform=ax.get_xaxis_transform(),
        fontsize=13, va='top')
ax.text(0.5, 0.97, 'GA closest approach · Oct 8, 06Z', transform=ax.transAxes,
        ha='center', va='top', fontsize=13, color=C_LANDFALL,
        bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.85))

sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.85); cb.set_label('gauge latitude (°N)')

ax.set_xlim([-48, 36]); ax.set_ylim([-1.8, 3.2])
ax.set_xlabel('Hours from closest approach', fontweight='bold')
ax.set_ylabel('Surge height (m)', fontweight='bold')
ax.set_title(f'Matthew 2016 — {len(gs)} coastline gauges (matthew_SAV_LR, in progress)',
             fontweight='bold', color=C_MATTHEW, pad=12)
ax.legend(loc='lower left', ncol=2, framealpha=0.92, edgecolor='#aaa', fontsize=11)

# inset map: Savannah DEM with gauge locations in their line colors
if dem is not None:
    from matplotlib.colors import TwoSlopeNorm
    iax = ax.inset_axes([0.76, 0.54, 0.24, 0.38])
    iax.imshow(dem, extent=dem_ext, origin='upper', cmap='Greys',
               norm=TwoSlopeNorm(vmin=max(np.nanmin(dem), -6), vcenter=0,
                                 vmax=min(np.nanmax(dem), 8)), alpha=0.45, zorder=0)
    for i, t, eta, lon, lat in gs:
        iax.scatter(lon, lat, s=60, color=cmap(norm(lat)), ec='k', lw=0.7, zorder=3)
        iax.annotate(f'{i}', (lon, lat), fontsize=6, zorder=4,
                     xytext=(2, 2), textcoords='offset points')
    iax.set_xlim(dem_ext[0], dem_ext[1]); iax.set_ylim(dem_ext[2], dem_ext[3])
    iax.set_title('gauge locations', fontsize=9)
    iax.tick_params(labelsize=6)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig_matthew_SAV_LR_gauges.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"loaded {len(gs)} gauges: {[g[0] for g in gs]}; reached {reached:+.1f} h")
print(f"Saved: {out}")
