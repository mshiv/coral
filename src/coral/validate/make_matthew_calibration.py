"""
CORAL Project — Matthew 2016 Calibration Figure for Chapman 2026 Poster
Single-panel landscape figure: GeoClaw modelled vs NOAA observed surge
at Fort Pulaski (NOAA 8670870) during Hurricane Matthew 2016.

Larger fonts throughout: titles, axis labels, ticks, legend, annotations —
designed to be legible when placed on a printed poster at full size.

Run with:  conda run -n aislens python3 src/figures/make_matthew_calibration.py
        or MPLBACKEND=Agg conda run -n aislens python3 src/figures/make_matthew_calibration.py
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime, os

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CF_ROOT   = os.path.join(REPO_ROOT, '..', 'coastalFlood')
OUT_DIR   = os.path.join(REPO_ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

MATTHEW_GAUGE = os.path.join(CF_ROOT, 'matthew_2016', '_output', 'gauge00002.txt')

# ── Poster palette ─────────────────────────────────────────────────────────
C_MATTHEW  = '#d62728'
C_MODEL    = '#1a2c52'    # dark navy
C_OBS      = '#2ca02c'    # green
C_LANDFALL = '#666'

# ── Font sizes (poster-scale) ──────────────────────────────────────────────
FS_TITLE     = 22
FS_AXIS      = 18
FS_TICK      = 16
FS_LEGEND    = 16
FS_PEAK      = 14
FS_LANDFALL  = 13
FS_CAPTION   = 12

# Apply globally where possible
plt.rcParams.update({
    'font.family':      'Lato, DejaVu Sans, sans-serif',
    'font.size':        FS_TICK,
    'axes.titlesize':   FS_TITLE,
    'axes.labelsize':   FS_AXIS,
    'xtick.labelsize':  FS_TICK,
    'ytick.labelsize':  FS_TICK,
    'legend.fontsize':  FS_LEGEND,
})

# ── GeoClaw reader ─────────────────────────────────────────────────────────
def read_geoclaw_gauge(path):
    """GeoClaw gauge file → (time_s, eta) arrays.
    Columns: level, time(s), q1, q2, q3, eta, aux..."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                data.append((float(parts[1]), float(parts[5])))
            except ValueError:
                continue
    if not data:
        return np.array([]), np.array([])
    data = np.array(data)
    return data[:, 0], data[:, 1]

# ── NOAA reader ────────────────────────────────────────────────────────────
def fetch_noaa_surge(station_id, begin_date, end_date):
    """NOAA CO-OPS API → (datetime array, surge array). Surge = WL − tidal pred."""
    import urllib.request, json
    fmt_api = '%Y%m%d'

    def get(product):
        url = (f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
               f'?begin_date={begin_date.strftime(fmt_api)}'
               f'&end_date={end_date.strftime(fmt_api)}'
               f'&station={station_id}&product={product}'
               f'&datum=MSL&time_zone=GMT&units=metric&format=json')
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  NOAA fetch error ({product}): {e}")
            return {}

    wl_json   = get('water_level')
    pred_json = get('predictions')
    wl_data   = wl_json.get('data', [])
    pr_data   = pred_json.get('predictions', [])
    if not wl_data or not pr_data:
        print(f"  No data returned for station {station_id}")
        return None, None

    pred_map = {p['t']: float(p['v']) for p in pr_data if 'v' in p}
    times, surge = [], []
    for w in wl_data:
        try:
            t  = datetime.datetime.strptime(w['t'], '%Y-%m-%d %H:%M')
            wl = float(w['v'])
            pv = pred_map.get(w['t'])
            if pv is not None:
                times.append(t)
                surge.append(wl - pv)
        except (KeyError, ValueError):
            continue
    if not times:
        return None, None
    return np.array(times), np.array(surge)

# ── Reference times ────────────────────────────────────────────────────────
matthew_setrun_t0 = datetime.datetime(2016, 10, 8, 12)   # GeoClaw t=0
matthew_ga_t0     = datetime.datetime(2016, 10, 8,  6)   # GA closest approach
matthew_gauge_shift = (matthew_setrun_t0 - matthew_ga_t0).total_seconds() / 3600  # +6h

# ══════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4.8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.30, linewidth=0.7)

# GeoClaw modelled
m_t_mod, m_eta_mod = read_geoclaw_gauge(MATTHEW_GAUGE)
m_t_hrs, m_eta = np.array([]), np.array([])
if len(m_t_mod) > 0:
    valid   = (m_eta_mod > -3.0) & (m_eta_mod < 5.0)
    m_t_hrs = m_t_mod[valid] / 3600.0 + matthew_gauge_shift
    m_eta   = m_eta_mod[valid]
    ax.plot(m_t_hrs, m_eta, color=C_MODEL, lw=2.6,
            label='GeoClaw (modelled)', zorder=3)
    print(f"Matthew GeoClaw: {len(m_t_hrs)} pts, eta [{m_eta.min():.2f}, {m_eta.max():.2f}] m")

# NOAA observed (tides removed)
obs_t, obs_surge = fetch_noaa_surge(
    '8670870',
    datetime.datetime(2016, 10, 5),
    datetime.datetime(2016, 10, 10))

obs_t_hrs = np.array([])
if obs_t is not None and len(obs_t) > 0:
    obs_t_hrs = np.array([(t - matthew_ga_t0).total_seconds() / 3600 for t in obs_t])
    ax.plot(obs_t_hrs, obs_surge, color=C_OBS, lw=2.6,
            label='Observed surge', zorder=3)
    print(f"Matthew NOAA: {len(obs_t_hrs)} pts, surge [{obs_surge.min():.2f}, {obs_surge.max():.2f}] m")
else:
    print("Matthew NOAA: not available")

# Closest-approach reference line
ax.axvline(x=0, color=C_LANDFALL, lw=1.8, ls='--', alpha=0.85, zorder=2)
ax.text(0.50, 0.97,
        'GA closest approach  ·  Oct 8, 06Z  ·  31.6°N, 80.6°W',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=FS_LANDFALL, color=C_LANDFALL,
        bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.85))

# Axes
ax.set_xlim([-36, 36])
ax.set_ylim([-1.8, 3.2])
ax.set_xticks([-36, -24, -12, 0, 12, 24, 36])
ax.set_xticklabels(['-36', '-24', '-12', '0', '+12', '+24', '+36'])
ax.set_xlabel('Hours from closest approach', fontsize=FS_AXIS, fontweight='bold')
ax.set_ylabel('Surge height (m)', fontsize=FS_AXIS, fontweight='bold')
ax.set_title('Hurricane Matthew (2016) — Fort Pulaski, GA   (NOAA 8670870)',
             fontsize=FS_TITLE, fontweight='bold', color=C_MATTHEW, pad=12)

ax.legend(loc='lower left', fontsize=FS_LEGEND, framealpha=0.92,
          edgecolor='#aaa', borderpad=0.7, handlelength=2.4)

# ── Peak annotations ──────────────────────────────────────────────────────
if len(obs_t_hrs) > 0:
    window = (obs_t_hrs > -24) & (obs_t_hrs < 24)
    if window.sum() > 0:
        pk_t = obs_t_hrs[window][np.argmax(obs_surge[window])]
        pk_v = obs_surge[window].max()
        ax.annotate(f'Observed peak\n{pk_v:.2f} m',
                    xy=(pk_t, pk_v), xytext=(pk_t + 6.5, pk_v - 0.55),
                    fontsize=FS_PEAK, color=C_OBS, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_OBS, lw=1.5))

if len(m_t_hrs) > 0:
    win = (m_t_hrs > -24) & (m_t_hrs < 24)
    if win.sum() > 0:
        mod_pk_v = m_eta[win].max()
        mod_pk_t = m_t_hrs[win][np.argmax(m_eta[win])]
        ax.annotate(f'Modelled peak\n{mod_pk_v:.2f} m',
                    xy=(mod_pk_t, mod_pk_v),
                    xytext=(mod_pk_t - 17, mod_pk_v + 0.35),
                    fontsize=FS_PEAK, color=C_MODEL, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_MODEL, lw=1.5))

# Undershoot callout
if len(obs_t_hrs) > 0 and len(m_t_hrs) > 0:
    win_o = (obs_t_hrs > -24) & (obs_t_hrs < 24)
    win_m = (m_t_hrs   > -24) & (m_t_hrs   < 24)
    if win_o.sum() > 0 and win_m.sum() > 0:
        pk_obs = obs_surge[win_o].max()
        pk_mod = m_eta[win_m].max()
        pct = (pk_mod - pk_obs) / pk_obs * 100
        ax.text(0.985, 0.04,
                f'Modelled undershoot:  {pct:+.0f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=FS_PEAK, color=C_MATTHEW, fontweight='bold',
                bbox=dict(facecolor='#fff5f4', edgecolor=C_MATTHEW,
                          boxstyle='round,pad=0.5', linewidth=1.3))

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig_matthew_calibration.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
print("Done.")
