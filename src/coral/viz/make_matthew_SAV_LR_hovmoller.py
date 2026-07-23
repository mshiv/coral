"""
Hovmoller (time x alongshore) of surge for the 63 coupling gauges: a 2D colormap
showing the surge propagating along the coast. Compact view of all gauges at once.

Run: MPLBACKEND=Agg ~/miniforge3/envs/ml-env/bin/python src/figures/make_matthew_SAV_LR_hovmoller.py
"""
import numpy as np, matplotlib.pyplot as plt, os, re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ODIR = os.path.join(REPO, '..', 'coastalFlood', 'matthew_SAV_LR', '_output')
OUT  = os.path.join(REPO, 'results', 'figures'); os.makedirs(OUT, exist_ok=True)
SEA_LEVEL = 0.81
shift_h = 6.0  # gauge t=0 (12Z) -> closest approach (06Z)


def read_gauge(path):
    lat = None; t = []; eta = []
    for line in open(path):
        if line.startswith('#'):
            m = re.search(r'location=\(\s*[\d.eE+-]+\s+([\d.eE+-]+)', line)
            if m: lat = float(m.group(1))
            continue
        p = line.split()
        if len(p) >= 6:
            try: t.append(float(p[1])); eta.append(float(p[5]))
            except ValueError: pass
    return lat, np.array(t), np.array(eta)


# common time axis (hours from closest approach)
tgrid = np.arange(-42, 31, 0.25)
rows = []
lats = []
for i in range(1, 64):
    p = os.path.join(ODIR, f'gauge{i:05d}.txt')
    if not os.path.exists(p):
        continue
    lat, t, eta = read_gauge(p)
    if lat is None or len(t) < 2:
        continue
    th = t / 3600 + shift_h
    surge = np.interp(tgrid, th, eta - SEA_LEVEL, left=np.nan, right=np.nan)
    rows.append(surge); lats.append(lat)

lats = np.array(lats); rows = np.array(rows)
order = np.argsort(lats)
lats = lats[order]; rows = rows[order]

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.pcolormesh(tgrid, lats, rows, shading='auto', cmap='viridis',
                   vmin=np.nanmin(rows), vmax=np.nanmax(rows))
ax.axvline(0, color='w', ls='--', lw=1.3, alpha=0.8)
cb = fig.colorbar(im, ax=ax); cb.set_label('surge height (m)')
ax.set_xlabel('hours from closest approach', fontweight='bold')
ax.set_ylabel('gauge latitude (°N)', fontweight='bold')
ax.set_title('Alongshore surge evolution (Hovmoller) — 63 coupling gauges, matthew_SAV_LR',
             fontweight='bold', color='#d62728', pad=10)
fig.tight_layout()
p = os.path.join(OUT, 'fig_matthew_SAV_LR_hovmoller.png')
fig.savefig(p, dpi=160, bbox_inches='tight'); print('Saved:', p)
