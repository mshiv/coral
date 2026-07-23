"""
5-station NOAA validation for matthew_SAV_LR: GeoClaw surge (eta-0.81) vs observed
surge (WL - tidal prediction) at all 5 NOAA gauges in the run (ids 64-68).
Tells whether the Fort Pulaski bias is local or systematic along the track.

Run: MPLBACKEND=Agg ~/miniforge3/envs/ml-env/bin/python src/figures/make_matthew_SAV_LR_validation5.py
"""
import numpy as np, matplotlib.pyplot as plt, datetime, os, urllib.request, json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ODIR = os.path.join(REPO, '..', 'coastalFlood', 'matthew_SAV_LR', '_output')
OUT  = os.path.join(REPO, 'results', 'figures'); os.makedirs(OUT, exist_ok=True)
SEA_LEVEL = 0.81
ga_t0 = datetime.datetime(2016, 10, 8, 6)          # closest approach
shift_h = (datetime.datetime(2016, 10, 8, 12) - ga_t0).total_seconds() / 3600

# gauge id -> (NOAA station, name)
STN = {64: ('8720218', 'Mayport, FL'), 65: ('8670870', 'Fort Pulaski, GA'),
       66: ('8665530', 'Charleston, SC'), 67: ('8658163', 'Wrightsville Bch, NC'),
       68: ('8658120', 'Wilmington, NC')}


def read_gauge(path):
    t, eta = [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        if len(p) >= 6:
            try: t.append(float(p[1])); eta.append(float(p[5]))
            except ValueError: pass
    return np.array(t), np.array(eta)


def noaa_surge(stn):
    def get(prod):
        u = (f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?'
             f'begin_date=20161005&end_date=20161010&station={stn}&product={prod}'
             f'&datum=MSL&time_zone=GMT&units=metric&format=json')
        try:
            with urllib.request.urlopen(u, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception: return {}
    wl = get('water_level').get('data', [])
    pr = {p['t']: float(p['v']) for p in get('predictions').get('predictions', []) if 'v' in p}
    t, s = [], []
    for w in wl:
        pv = pr.get(w['t'])
        if pv is not None:
            try:
                t.append(datetime.datetime.strptime(w['t'], '%Y-%m-%d %H:%M'))
                s.append(float(w['v']) - pv)
            except ValueError: pass
    return (np.array(t), np.array(s)) if t else (None, None)


fig, axes = plt.subplots(3, 2, figsize=(14, 11)); axes = axes.flatten()
for ax, gid in zip(axes, sorted(STN)):
    stn, name = STN[gid]
    t, eta = read_gauge(os.path.join(ODIR, f'gauge{gid:05d}.txt'))
    mod_pk = obs_pk = np.nan
    if len(t):
        th = t / 3600 + shift_h; surge = eta - SEA_LEVEL
        ax.plot(th, surge, color='#1a2c52', lw=2, label='GeoClaw')
        w = (th > -24) & (th < 24); mod_pk = surge[w].max() if w.any() else np.nan
    ot, os_ = noaa_surge(stn)
    if ot is not None:
        oh = np.array([(x - ga_t0).total_seconds() / 3600 for x in ot])
        ax.plot(oh, os_, color='#2ca02c', lw=2, label='Observed')
        w = (oh > -24) & (oh < 24); obs_pk = os_[w].max() if w.any() else np.nan
    ax.axvline(0, color='#666', ls='--', lw=1.2)
    err = f'{(mod_pk-obs_pk)/obs_pk*100:+.0f}%' if obs_pk == obs_pk and obs_pk else 'n/a'
    ax.set_title(f'{name} ({stn})  ·  peak mod {mod_pk:.2f} / obs {obs_pk:.2f} m ({err})',
                 fontsize=12)
    ax.set_xlim(-36, 36); ax.set_ylim(-1.8, 3.2); ax.grid(alpha=0.3)
    ax.set_xlabel('hours from closest approach'); ax.set_ylabel('surge (m)')
    ax.legend(fontsize=9, loc='upper left')
axes[-1].axis('off')
fig.suptitle('Matthew 2016 surge validation — GeoClaw vs NOAA (matthew_SAV_LR)',
             fontsize=16, fontweight='bold', color='#d62728')
fig.tight_layout()
p = os.path.join(OUT, 'fig_matthew_SAV_LR_validation5.png')
fig.savefig(p, dpi=160, bbox_inches='tight'); print('Saved:', p)
