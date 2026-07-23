"""
GeoClaw spatial surge field (eta) at a chosen output frame, drawn patch-by-patch
(coarse first, fine on top). Default frame = landfall (t=0). Shows the regional
surge pattern the coastline gauges sample.

fort.q patch: header (grid_number, AMR_level, mx, my, xlow, ylow, dx, dy) then
mx*my data rows of [h, hu, hv, eta]; eta is column index 3.

Run: MPLBACKEND=Agg ~/miniforge3/envs/ml-env/bin/python src/figures/make_matthew_SAV_LR_surgefield.py [--frame 8]
"""
import numpy as np, matplotlib.pyplot as plt, os, argparse
import matplotlib.patches as mpatches

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ODIR = os.path.join(REPO, '..', 'coastalFlood', 'matthew_SAV_LR', '_output')
OUT  = os.path.join(REPO, 'results', 'figures'); os.makedirs(OUT, exist_ok=True)
SEA_LEVEL = 0.81
SAV_BOX = (-81.111, -80.819, 31.804, 32.100)

ap = argparse.ArgumentParser()
ap.add_argument('--frame', type=int, default=8)   # 8 = t=0 landfall
ap.add_argument('--var', default='surge', choices=['surge', 'eta'])
args = ap.parse_args()

fq = os.path.join(ODIR, f'fort.q{args.frame:04d}')
ft = os.path.join(ODIR, f'fort.t{args.frame:04d}')
tsec = float(open(ft).readline().split()[0]) if os.path.exists(ft) else None

# parse patches: list of (level, x0,y0,dx,dy,mx,my, eta 2D array)
patches = []
lines = open(fq).read().splitlines()
i = 0
while i < len(lines):
    if 'grid_number' in lines[i]:
        lvl = int(lines[i+1].split()[0]); mx = int(lines[i+2].split()[0])
        my = int(lines[i+3].split()[0]); x0 = float(lines[i+4].split()[0])
        y0 = float(lines[i+5].split()[0]); dx = float(lines[i+6].split()[0])
        dy = float(lines[i+7].split()[0])
        j = i + 8
        while j < len(lines) and lines[j].strip() == '':
            j += 1
        vals = np.empty((my, mx))
        for r in range(my):
            for c in range(mx):
                f = lines[j].split()
                h = float(f[0]); eta = float(f[3])   # col0=h(depth), col3=eta(surface)
                vals[r, c] = eta if h > 0.01 else np.nan   # mask DRY cells
                j += 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
        patches.append((lvl, x0, y0, dx, dy, mx, my, vals))
        i = j
    else:
        i += 1

allv = np.concatenate([p[7].ravel() for p in patches])
allv = allv if args.var == 'eta' else allv - SEA_LEVEL
allv = allv[np.isfinite(allv)]
vmax = np.nanpercentile(allv, 99.5); vmin = -vmax   # symmetric about 0 for diverging

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, zoom in zip(axes, [False, True]):
    for lvl, x0, y0, dx, dy, mx, my, vals in sorted(patches, key=lambda d: d[0]):
        z = vals if args.var == 'eta' else vals - SEA_LEVEL
        xe = x0 + np.arange(mx + 1) * dx
        ye = y0 + np.arange(my + 1) * dy
        ax.pcolormesh(xe, ye, z, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                      shading='flat', zorder=lvl)
    ax.add_patch(mpatches.Rectangle((SAV_BOX[0], SAV_BOX[2]),
                 SAV_BOX[1]-SAV_BOX[0], SAV_BOX[3]-SAV_BOX[2],
                 fill=False, ec='k', lw=1.5, ls='--', zorder=99))
    if zoom:
        ax.set_xlim(SAV_BOX[0]-0.4, SAV_BOX[1]+0.4)
        ax.set_ylim(SAV_BOX[2]-0.4, SAV_BOX[3]+0.4); ax.set_title('Savannah region')
    else:
        xs = [p[1] for p in patches]+[p[1]+p[5]*p[3] for p in patches]
        ys = [p[2] for p in patches]+[p[2]+p[6]*p[4] for p in patches]
        ax.set_xlim(min(xs), max(xs)); ax.set_ylim(min(ys), max(ys))
        ax.set_title('Full domain')
    ax.set_xlabel('lon'); ax.set_ylabel('lat')

sm = plt.cm.ScalarMappable(cmap='RdBu_r',
        norm=plt.Normalize(vmin=vmin, vmax=vmax)); sm.set_array([])
cb = fig.colorbar(sm, ax=axes, shrink=0.7)
cb.set_label('surge (m)' if args.var == 'surge' else 'water surface eta (m)')
th = (tsec/3600+6) if tsec is not None else args.frame
fig.suptitle(f'GeoClaw {args.var} field — frame {args.frame} ({th:+.0f} h from closest approach)',
             fontsize=16, fontweight='bold')
p = os.path.join(OUT, f'fig_matthew_SAV_LR_surgefield_f{args.frame}.png')
fig.savefig(p, dpi=160, bbox_inches='tight'); print('Saved:', p)
