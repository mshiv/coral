"""
AMR resolution map for a GeoClaw run: draws every grid patch in a frame,
colored by AMR level, so I can see what resolution is used where.

Reads fort.qNNNN patch headers directly (no clawpack needed): each patch gives
AMR_level, mx, my, xlow, ylow, dx, dy. Cell size in metres is computed from dx
at the patch latitude.

Two panels: full domain + zoom to the Savannah/Pin Point box.

Run: MPLBACKEND=Agg ~/miniforge3/envs/ml-env/bin/python src/figures/make_domain_amr_map.py
     [--run matthew_SAV_LR] [--frame 5]   (frame default = latest)
"""
import glob, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
try:
    import contextily as ctx
    HAVE_CTX = True
except ImportError:
    HAVE_CTX = False


def read_dem_decimated(path, max_dim=1800):
    """Read a (possibly large) DEM at reduced resolution for plotting."""
    import rasterio
    with rasterio.open(path) as s:
        scale = max(1, int(max(s.width, s.height) / max_dim))
        a = s.read(1, out_shape=(s.height // scale, s.width // scale)).astype(float)
        b = s.bounds
    a = np.where(a <= -9990, np.nan, a)
    return a, [b.left, b.right, b.bottom, b.top]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CF_ROOT   = os.path.join(REPO_ROOT, '..', 'coastalFlood')
OUT_DIR   = os.path.join(REPO_ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Savannah / Pin Point coupling box (the L6 flagregion)
SAV_BOX = (-81.111, -80.819, 31.804, 32.100)


def read_patches(path):
    """Return list of dicts: level, x0, y0, x1, y1, dx (deg)."""
    lines = open(path).read().splitlines()
    out = []
    for i, l in enumerate(lines):
        if 'grid_number' in l:
            level = int(lines[i + 1].split()[0])
            mx    = int(lines[i + 2].split()[0])
            my    = int(lines[i + 3].split()[0])
            x0    = float(lines[i + 4].split()[0])
            y0    = float(lines[i + 5].split()[0])
            dx    = float(lines[i + 6].split()[0])
            dy    = float(lines[i + 7].split()[0])
            out.append(dict(level=level, x0=x0, y0=y0,
                            x1=x0 + mx * dx, y1=y0 + my * dy, dx=dx))
    return out


def dx_to_m(dx_deg, lat):
    return dx_deg * 111320.0 * np.cos(np.radians(lat))


ap = argparse.ArgumentParser()
ap.add_argument('--run', default='matthew_SAV_LR')
ap.add_argument('--frame', type=int, default=None)
ap.add_argument('--no-basemap', action='store_true')
ap.add_argument('--alpha', type=float, default=0.30,
                help='patch transparency over the basemap/DEM')
args = ap.parse_args()
use_basemap = HAVE_CTX and not args.no_basemap

# DEM for the zoom-panel underlay: full regional CUDEM (decimated for plotting)
DEM_PATH = os.path.join(CF_ROOT, 'ForShiva', 'GoeClaw - Dorian', 'ga_cudem_30m.asc')
dem, dem_ext = (read_dem_decimated(DEM_PATH) if os.path.exists(DEM_PATH) else (None, None))

odir = os.path.join(CF_ROOT, args.run, '_output')
frames = sorted(glob.glob(os.path.join(odir, 'fort.q*')))
fq = (os.path.join(odir, f'fort.q{args.frame:04d}') if args.frame is not None
      else frames[-1])
patches = read_patches(fq)
maxlevel = max(p['level'] for p in patches)
print(f"{os.path.basename(fq)}: {len(patches)} patches, levels 1..{maxlevel}")

# discrete colormap by level
base = plt.cm.viridis(np.linspace(0, 1, maxlevel))
cmap = ListedColormap(base)
norm = BoundaryNorm(np.arange(0.5, maxlevel + 1.5), maxlevel)

# representative cell size (m) per level, for the legend
lvl_m = {}
for p in patches:
    lat = 0.5 * (p['y0'] + p['y1'])
    lvl_m.setdefault(p['level'], dx_to_m(p['dx'], lat))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, zoom in zip(axes, [False, True]):
    # coarse first so fine patches draw on top; semi-transparent over basemap
    for p in sorted(patches, key=lambda d: d['level']):
        c = cmap(norm(p['level']))
        ax.add_patch(mpatches.Rectangle(
            (p['x0'], p['y0']), p['x1'] - p['x0'], p['y1'] - p['y0'],
            facecolor=c, edgecolor='k', lw=0.15, alpha=args.alpha, zorder=2))
    ax.add_patch(mpatches.Rectangle(
        (SAV_BOX[0], SAV_BOX[2]), SAV_BOX[1] - SAV_BOX[0], SAV_BOX[3] - SAV_BOX[2],
        fill=False, ec='red', lw=1.6, ls='--', zorder=3))
    if zoom:
        if dem is not None:
            ax.set_xlim(dem_ext[0], dem_ext[1]); ax.set_ylim(dem_ext[2], dem_ext[3])
        else:
            m = 0.15
            ax.set_xlim(SAV_BOX[0] - m, SAV_BOX[1] + m)
            ax.set_ylim(SAV_BOX[2] - m, SAV_BOX[3] + m)
        ax.set_title('Zoom: Savannah / Pin Point box (DEM)', fontsize=15)
    else:
        xs = [p['x0'] for p in patches] + [p['x1'] for p in patches]
        ys = [p['y0'] for p in patches] + [p['y1'] for p in patches]
        ax.set_xlim(min(xs), max(xs)); ax.set_ylim(min(ys), max(ys))
        ax.set_title('Full domain (satellite)', fontsize=15)
    ax.set_xlabel('lon'); ax.set_ylabel('lat'); ax.set_aspect('auto')

    # underlay: DEM on the zoom panel, satellite on the full domain
    if zoom and dem is not None:
        ax.imshow(dem, extent=dem_ext, origin='upper', cmap='BrBG_r',
                  norm=TwoSlopeNorm(vmin=max(np.nanmin(dem), -6), vcenter=0,
                                    vmax=min(np.nanmax(dem), 8)), zorder=0)
    elif use_basemap:
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', zorder=0,
                            source=ctx.providers.Esri.WorldImagery)
        except Exception as e:
            print(f"  basemap skipped ({e})")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cb = fig.colorbar(sm, ax=axes, shrink=0.7, ticks=range(1, maxlevel + 1))
cb.set_label('AMR level')
cb.ax.set_yticklabels([f'L{L}  (~{lvl_m[L]:.0f} m)' if L in lvl_m else f'L{L}'
                       for L in range(1, maxlevel + 1)])

fig.suptitle(f'GeoClaw AMR resolution — {args.run}, {os.path.basename(fq)} '
             f'(red dashed = L6 flagregion)', fontsize=18, fontweight='bold')
out = os.path.join(OUT_DIR, f'fig_amr_map_{args.run}.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print("cell size by level:", {L: round(v) for L, v in sorted(lvl_m.items())})
print(f"Saved: {out}")
