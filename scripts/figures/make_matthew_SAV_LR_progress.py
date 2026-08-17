"""
Matthew 2016 — Fort Pulaski calibration / progress plot for the IN-PROGRESS
matthew_SAV_LR GeoClaw run (the new Savannah/Pin Point coupling run).

Same single-panel style as make_matthew_calibration.py: GeoClaw modelled surge
vs NOAA observed surge at Fort Pulaski (8670870), x-axis hours from GA closest
approach. Lets me see how far the run has progressed and how the surge tracks
the obs as it fills in.

Difference vs the original calibration run: matthew_SAV_LR uses sea_level = 0.81
(static high-tide offset), so modelled surge = eta - 0.81.

Run:  MPLBACKEND=Agg ~/miniforge3/envs/ml-env/bin/python src/figures/make_matthew_SAV_LR_progress.py
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime, os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CF_ROOT   = os.path.join(REPO_ROOT, '..', 'coastalFlood')
OUT_DIR   = os.path.join(REPO_ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

GAUGE      = os.path.join(CF_ROOT, 'matthew_SAV_LR', '_output', 'gauge00065.txt')
SEA_LEVEL  = 0.81           # static offset baked into eta for this run
STATION    = '8670870'      # Fort Pulaski

C_MODEL, C_OBS, C_MATTHEW, C_LANDFALL = '#1a2c52', '#2ca02c', '#d62728', '#666'
FS_TITLE, FS_AXIS, FS_TICK, FS_LEGEND, FS_PEAK = 22, 18, 16, 16, 14
plt.rcParams.update({'font.family': 'Lato, DejaVu Sans, sans-serif',
    'font.size': FS_TICK, 'axes.titlesize': FS_TITLE, 'axes.labelsize': FS_AXIS,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_LEGEND})


def read_geoclaw_gauge(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            p = line.split()
            if len(p) < 6:
                continue
            try:
                data.append((float(p[1]), float(p[5])))
            except ValueError:
                continue
    if not data:
        return np.array([]), np.array([])
    d = np.array(data)
    return d[:, 0], d[:, 1]


import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--total', action='store_true',
                 help='plot TOTAL water level (eta vs observed WL) instead of '
                      'surge (eta-0.81 vs observed WL-tide)')
ARGS = _ap.parse_args()


def fetch_noaa(station, begin, end, total=False):
    """total=False -> surge (WL - tide pred); total=True -> observed WL (MSL)."""
    import urllib.request, json
    def get(product):
        url = (f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
               f'?begin_date={begin:%Y%m%d}&end_date={end:%Y%m%d}'
               f'&station={station}&product={product}'
               f'&datum=MSL&time_zone=GMT&units=metric&format=json')
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  NOAA fetch error ({product}): {e}"); return {}
    wl = get('water_level').get('data', [])
    if not wl:
        return None, None
    if total:
        t, s = [], []
        for w in wl:
            try:
                t.append(datetime.datetime.strptime(w['t'], '%Y-%m-%d %H:%M'))
                s.append(float(w['v']))
            except (KeyError, ValueError):
                continue
        return (np.array(t), np.array(s)) if t else (None, None)
    pr = {p['t']: float(p['v']) for p in get('predictions').get('predictions', []) if 'v' in p}
    if not pr:
        return None, None
    t, s = [], []
    for w in wl:
        try:
            pv = pr.get(w['t'])
            if pv is not None:
                t.append(datetime.datetime.strptime(w['t'], '%Y-%m-%d %H:%M'))
                s.append(float(w['v']) - pv)
        except (KeyError, ValueError):
            continue
    return (np.array(t), np.array(s)) if t else (None, None)


setrun_t0 = datetime.datetime(2016, 10, 8, 12)   # GeoClaw t=0
ga_t0     = datetime.datetime(2016, 10, 8,  6)   # GA closest approach
shift_h   = (setrun_t0 - ga_t0).total_seconds() / 3600  # +6 h

fig, ax = plt.subplots(figsize=(14, 4.8))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.30, linewidth=0.7)

# modelled: surge (eta - sea_level) or total water level (eta)
t_s, eta = read_geoclaw_gauge(GAUGE)
m_t_hrs = np.array([])
mod_label = ('GeoClaw total WL (eta, in progress)' if ARGS.total
             else 'GeoClaw surge (eta-0.81, in progress)')
if len(t_s):
    surge_mod = eta if ARGS.total else eta - SEA_LEVEL
    m_t_hrs = t_s / 3600.0 + shift_h
    ax.plot(m_t_hrs, surge_mod, color=C_MODEL, lw=2.6, label=mod_label, zorder=3)
    last_h = m_t_hrs.max()
    ax.axvline(last_h, color=C_MODEL, lw=1.5, ls=':', alpha=0.8)
    ax.text(last_h, 0.92, f'  run reached\n  {last_h:+.0f} h', transform=ax.get_xaxis_transform(),
            fontsize=FS_PEAK, color=C_MODEL, va='top')
    print(f"modelled: {len(m_t_hrs)} pts, surge [{surge_mod.min():.2f}, {surge_mod.max():.2f}] m, "
          f"reached {last_h:+.1f} h from closest approach")

obs_t, obs_s = fetch_noaa(STATION, datetime.datetime(2016, 10, 5),
                          datetime.datetime(2016, 10, 10), total=ARGS.total)
obs_label = 'Observed total WL (MSL)' if ARGS.total else 'Observed surge'
if obs_t is not None:
    obs_h = np.array([(t - ga_t0).total_seconds() / 3600 for t in obs_t])
    ax.plot(obs_h, obs_s, color=C_OBS, lw=2.6, label=obs_label, zorder=3)
    print(f"observed: {len(obs_h)} pts, [{obs_s.min():.2f}, {obs_s.max():.2f}] m")

ax.axvline(0, color=C_LANDFALL, lw=1.8, ls='--', alpha=0.85)
ax.text(0.50, 0.97, 'GA closest approach · Oct 8, 06Z',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=13, color=C_LANDFALL,
        bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.85))
ax.set_xlim([-48, 36]); ax.set_ylim([-1.8, 3.2])
ax.set_xlabel('Hours from closest approach', fontweight='bold')
ax.set_ylabel('Total water level (m, MSL)' if ARGS.total else 'Surge height (m)',
              fontweight='bold')
ax.set_title('Matthew 2016 — Fort Pulaski (8670870) — matthew_SAV_LR (in progress)',
             fontweight='bold', color=C_MATTHEW, pad=12)
ax.legend(loc='lower left', framealpha=0.92, edgecolor='#aaa')

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig_matthew_SAV_LR_progress'
                   + ('_total' if ARGS.total else '') + '.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
