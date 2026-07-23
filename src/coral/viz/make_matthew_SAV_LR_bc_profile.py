"""
Alongshore profile of the LISFLOOD boundary forcing: peak total water level (eta)
and peak surge (eta-0.81) at each of the 63 coupling gauges vs latitude. This is
exactly what LISFLOOD will be forced with -- a sanity check on the BC.

Run: MPLBACKEND=Agg ~/miniforge3/envs/ml-env/bin/python src/figures/make_matthew_SAV_LR_bc_profile.py
"""
import numpy as np, matplotlib.pyplot as plt, os, re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ODIR = os.path.join(REPO, '..', 'coastalFlood', 'matthew_SAV_LR', '_output')
OUT  = os.path.join(REPO, 'results', 'figures'); os.makedirs(OUT, exist_ok=True)
SEA_LEVEL = 0.81


def read_gauge(path):
    lat = None; eta = []
    for line in open(path):
        if line.startswith('#'):
            m = re.search(r'location=\(\s*[\d.eE+-]+\s+([\d.eE+-]+)', line)
            if m: lat = float(m.group(1))
            continue
        p = line.split()
        if len(p) >= 6:
            try: eta.append(float(p[5]))
            except ValueError: pass
    return lat, (max(eta) if eta else np.nan)


lats, peaks, ids = [], [], []
for i in range(1, 64):
    p = os.path.join(ODIR, f'gauge{i:05d}.txt')
    if os.path.exists(p):
        lat, pk = read_gauge(p)
        if lat is not None:
            lats.append(lat); peaks.append(pk); ids.append(i)
order = np.argsort(lats)
lats = np.array(lats)[order]; peaks = np.array(peaks)[order]
ids = np.array(ids)[order]
surge = peaks - SEA_LEVEL

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(lats, peaks, '-o', color='#1a2c52', lw=1.8, ms=4, label='peak total WL (eta)')
ax.plot(lats, surge, '-o', color='#d62728', lw=1.8, ms=4, label='peak surge (eta-0.81)')
ax.axhline(SEA_LEVEL, color='#888', ls=':', lw=1, label='static tide baseline 0.81 m')
ax.fill_between(lats, surge, peaks, color='#1a2c52', alpha=0.06)
ax.set_xlabel('gauge latitude (°N)  — south → north', fontweight='bold')
ax.set_ylabel('peak water level (m, MSL)', fontweight='bold')
ax.set_title('LISFLOOD boundary forcing — alongshore peak profile (63 coupling gauges)',
             fontweight='bold', color='#d62728', pad=10)
ax.grid(alpha=0.3); ax.legend(loc='upper left')
ax.annotate(f'mean peak WL {peaks.mean():.2f} m\nrange {peaks.min():.2f}–{peaks.max():.2f} m',
            xy=(0.985, 0.04), xycoords='axes fraction', ha='right', va='bottom',
            fontsize=11, bbox=dict(facecolor='#f5f5f5', edgecolor='#aaa', boxstyle='round'))
fig.tight_layout()
p = os.path.join(OUT, 'fig_matthew_SAV_LR_bc_profile.png')
fig.savefig(p, dpi=160, bbox_inches='tight')
print(f"peak WL: min {peaks.min():.2f}, max {peaks.max():.2f}, mean {peaks.mean():.2f} m")
print('Saved:', p)
