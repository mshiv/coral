"""
CORAL Project — Fort Pulaski Gauge Time Series for Chapman 2026 Poster
Two-panel figure:
  Top:    Matthew 2016 — GeoClaw modelled vs NOAA observed surge at Fort Pulaski
  Bottom: Dorian  2019 — GeoClaw modelled vs NOAA observed surge at Fort Pulaski

Data sources:
  Matthew modelled:  gauge0002fig300.png exists but no local fort.gauge file
                     → reads from PACE _output via local plot PNG if needed
                     → here we re-create from NOAA cached data + note model on PACE
  Dorian modelled:   coastalFlood/data/GoeClaw - Dorian/gauge00005a.txt (local)
  NOAA observed:     matthew scratch cache + dorian noaa_gauge_data.pkl

Run with:  conda run -n aislens python3 src/figures/make_gauge_timeseries.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
import pickle, datetime, os

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CF_ROOT     = os.path.join(REPO_ROOT, '..', 'coastalFlood')
OUT_DIR     = os.path.join(REPO_ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Gauge output files: gauge00002 = Fort Pulaski for both storms
# Matthew: num_var=4, cols: level, time, q1, q2, q3, eta
# Dorian:  num_var=7, cols: level, time, q1, q2, q3, eta, aux5, aux6, aux7
# time is in seconds relative to storm time_offset
# Matthew time_offset: datetime(2016, 10, 8, 12)  [setrun.py line 487]
# Dorian  time_offset: datetime(2019, 9,  4, 12)  [setrun.py line 483]
MATTHEW_GAUGE  = os.path.join(CF_ROOT, 'matthew_2016', '_output', 'gauge00002.txt')
DORIAN_GAUGE   = os.path.join(CF_ROOT, 'dorian_2019',  '_output', 'gauge00005.txt')

# Poster colours
C_MATTHEW  = '#d62728'
C_DORIAN   = '#1f77b4'
C_MODEL    = '#1a2c52'    # dark navy for modelled line
C_OBS      = '#2ca02c'    # green for observed (matches setplot.py convention)
C_LANDFALL = '#888'

# ── Read Dorian GeoClaw gauge output ───────────────────────────────────────
def read_geoclaw_gauge(path):
    """Read GeoClaw fort.gauge or gauge*.txt file.
    Columns: level, time(s), q1, q2, q3, eta, aux...
    Returns (time_s, eta) arrays."""
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
                t   = float(parts[1])   # time in seconds
                eta = float(parts[5])   # water surface elevation (m)
                data.append((t, eta))
            except ValueError:
                continue
    if not data:
        return np.array([]), np.array([])
    data = np.array(data)
    return data[:, 0], data[:, 1]

# ── Read NOAA observed data ────────────────────────────────────────────────
def fetch_noaa_surge(station_id, begin_date, end_date):
    """Fetch water level + tide prediction from NOAA CO-OPS API.
    Returns (datetime array, surge array) where surge = WL - tide."""
    import urllib.request, json

    fmt_in  = '%Y%m%d %H:%M'
    fmt_api = '%Y%m%d'

    def get(product):
        url = (f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
               f'?begin_date={begin_date.strftime(fmt_api)}'
               f'&end_date={end_date.strftime(fmt_api)}'
               f'&station={station_id}&product={product}'
               f'&datum=MSL&time_zone=GMT&units=metric&format=json')
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
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

    # Align on time
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

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
fig.patch.set_facecolor('white')

stroke = [pe.withStroke(linewidth=2, foreground='white')]

for ax in axes:
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5)

# ══════════════════════════════════════════════════════════════════════
# TOP PANEL — Matthew 2016
# ══════════════════════════════════════════════════════════════════════
ax_m = axes[0]

# ── Reference times: GA closest approach (from HURDAT2/ATCF best-track) ──
# Matthew: Oct 8 06Z — storm centre at 31.6°N 80.6°W, ~55 km offshore Savannah
# Dorian:  Sep 5 06Z — storm centre at 31.7°N 80.6°W, ~50 km offshore Savannah
# These differ from setrun time_offset (used by GeoClaw); we correct below.

# setrun time_offsets
matthew_setrun_t0 = datetime.datetime(2016, 10, 8, 12)   # setrun.py line 487
dorian_setrun_t0  = datetime.datetime(2019, 9,  4, 12)   # setrun.py line 483

# GA closest approach reference times
matthew_ga_t0 = datetime.datetime(2016, 10, 8,  6)   # Oct 8 06Z
dorian_ga_t0  = datetime.datetime(2019, 9,  5,  6)   # Sep 5 06Z

# Shift from setrun t=0 to GA t=0 (in hours):
# GeoClaw gauge file times are in seconds relative to setrun time_offset.
# To plot relative to GA closest approach, we offset the gauge times:
matthew_gauge_shift = (matthew_setrun_t0 - matthew_ga_t0).total_seconds() / 3600  # +6h
dorian_gauge_shift  = (dorian_setrun_t0  - dorian_ga_t0 ).total_seconds() / 3600  # -18h

# GeoClaw modelled
m_t_mod, m_eta_mod = read_geoclaw_gauge(MATTHEW_GAUGE)
m_t_hrs = np.array([])
m_eta   = np.array([])
if len(m_t_mod) > 0:
    valid   = (m_eta_mod > -3.0) & (m_eta_mod < 5.0)
    # shift gauge time to GA reference
    m_t_hrs = m_t_mod[valid] / 3600.0 + matthew_gauge_shift
    m_eta   = m_eta_mod[valid]
    ax_m.plot(m_t_hrs, m_eta, color=C_MODEL, lw=1.8,
              label='GeoClaw (modelled)', zorder=3)
    print(f"Matthew GeoClaw: {len(m_t_hrs)} pts, eta [{m_eta.min():.2f}, {m_eta.max():.2f}] m")

# NOAA observed (tides already removed: surge = water_level − NOAA predictions)
matthew_obs_t, matthew_obs_surge = fetch_noaa_surge(
    '8670870',
    datetime.datetime(2016, 10, 5),
    datetime.datetime(2016, 10, 10))

obs_t_hrs_m = np.array([])
if matthew_obs_t is not None and len(matthew_obs_t) > 0:
    obs_t_hrs_m = np.array([(t - matthew_ga_t0).total_seconds() / 3600
                             for t in matthew_obs_t])
    ax_m.plot(obs_t_hrs_m, matthew_obs_surge, color=C_OBS, lw=1.8,
              label='Observed − tidal prediction (NOAA 8670870)', zorder=3)
    print(f"Matthew NOAA: {len(obs_t_hrs_m)} pts, surge range "
          f"[{matthew_obs_surge.min():.2f}, {matthew_obs_surge.max():.2f}] m")
else:
    print("Matthew NOAA: not available (no internet?)")

ax_m.axvline(x=0, color=C_LANDFALL, lw=1.5, ls='--', alpha=0.85, zorder=2)
ax_m.text(0.5, 0.96,
          'GA closest approach (Oct 8 06Z) — 31.6°N, 80.6°W',
          transform=ax_m.transAxes, ha='center', va='top',
          fontsize=6.5, color=C_LANDFALL)

ax_m.set_xlim([-36, 36])
ax_m.set_ylim([-1.6, 3.0])
ax_m.set_xticks([-36, -24, -12, 0, 12, 24, 36])
ax_m.set_xticklabels(['-36', '-24', '-12', '0', '+12', '+24', '+36'])
ax_m.set_ylabel('Surge height (m)', fontsize=9)
ax_m.set_title('Hurricane Matthew 2016 — Fort Pulaski, GA (NOAA 8670870)',
               fontsize=9.5, fontweight='bold', color=C_MATTHEW, pad=4)
ax_m.legend(loc='upper left', fontsize=8, framealpha=0.85)

# Annotate peaks
if len(obs_t_hrs_m) > 0:
    window = (obs_t_hrs_m > -24) & (obs_t_hrs_m < 24)
    if window.sum() > 0:
        pk_t = obs_t_hrs_m[window][np.argmax(matthew_obs_surge[window])]
        pk_v = matthew_obs_surge[window].max()
        ax_m.annotate(f'Observed peak\n{pk_v:.2f} m',
                      xy=(pk_t, pk_v), xytext=(pk_t + 5, pk_v - 0.45),
                      fontsize=7, color=C_OBS,
                      arrowprops=dict(arrowstyle='->', color=C_OBS, lw=0.8))
if len(m_t_hrs) > 0:
    window_m = (m_t_hrs > -24) & (m_t_hrs < 24)
    if window_m.sum() > 0:
        mod_pk   = m_eta[window_m].max()
        mod_pk_t = m_t_hrs[window_m][np.argmax(m_eta[window_m])]
        ax_m.annotate(f'Modelled peak\n{mod_pk:.2f} m',
                      xy=(mod_pk_t, mod_pk),
                      xytext=(mod_pk_t - 14, mod_pk + 0.25),
                      fontsize=7, color=C_MODEL,
                      arrowprops=dict(arrowstyle='->', color=C_MODEL, lw=0.8))

# ══════════════════════════════════════════════════════════════════════
# BOTTOM PANEL — Dorian 2019
# ══════════════════════════════════════════════════════════════════════
ax_d = axes[1]

# GeoClaw modelled
d_t_mod, d_eta_mod = read_geoclaw_gauge(DORIAN_GAUGE)
d_t_hrs = np.array([])
d_eta   = np.array([])
if len(d_t_mod) > 0:
    idx    = np.argsort(d_t_mod)
    d_t_mod, d_eta_mod = d_t_mod[idx], d_eta_mod[idx]
    valid  = (d_eta_mod > -2.0) & (d_eta_mod < 4.0)
    # shift from setrun t=0 (Sep 4 12Z) to GA reference (Sep 5 06Z)
    d_t_hrs = d_t_mod[valid] / 3600.0 + dorian_gauge_shift
    d_eta   = d_eta_mod[valid]
    ax_d.plot(d_t_hrs, d_eta, color=C_MODEL, lw=1.8,
              label='GeoClaw (modelled)', zorder=3)
    print(f"Dorian GeoClaw: {len(d_t_hrs)} pts after filter, eta [{d_eta.min():.2f}, {d_eta.max():.2f}] m")

# NOAA observed
dorian_obs_t, dorian_obs_surge = fetch_noaa_surge(
    '8670870',
    datetime.datetime(2019, 8, 30),
    datetime.datetime(2019, 9, 10))

if dorian_obs_t is not None and len(dorian_obs_t) > 0:
    obs_t_hrs_d = np.array([(t - dorian_ga_t0).total_seconds() / 3600.0
                             for t in dorian_obs_t])
    ax_d.plot(obs_t_hrs_d, dorian_obs_surge, color=C_OBS, lw=1.8,
              label='Observed − tidal prediction (NOAA 8670870)', zorder=3)
    print(f"Dorian NOAA: {len(obs_t_hrs_d)} pts, surge [{dorian_obs_surge.min():.2f}, {dorian_obs_surge.max():.2f}] m")

ax_d.axvline(x=0, color=C_LANDFALL, lw=1.5, ls='--', alpha=0.85, zorder=2)
ax_d.text(0.5, 0.96,
          'GA closest approach (Sep 5 06Z) — 31.7°N, 80.6°W',
          transform=ax_d.transAxes, ha='center', va='top',
          fontsize=6.5, color=C_LANDFALL)

ax_d.set_xlim([-36, 36])
ax_d.set_ylim([-1.0, 2.0])
ax_d.set_xticks([-36, -24, -12, 0, 12, 24, 36])
ax_d.set_xticklabels(['-36', '-24', '-12', '0', '+12', '+24', '+36'])
ax_d.set_xlabel('Hours relative to landfall reference', fontsize=9)
ax_d.set_ylabel('Surge height (m)', fontsize=9)
ax_d.set_title('Hurricane Dorian 2019 — Fort Pulaski, GA (NOAA 8670870)',
               fontsize=9.5, fontweight='bold', color=C_DORIAN, pad=4)
ax_d.legend(loc='upper left', fontsize=8, framealpha=0.85)

# ── Caption ────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
    'Fig. 3  Storm surge at Fort Pulaski, GA (NOAA station 8670870) for Hurricane Matthew 2016 (top) '
    'and Hurricane Dorian 2019 (bottom).\n'
    'Observed surge = water level − tidal prediction. '
    'GeoClaw modelled surge uses Holland (1980) parametric wind forcing, '
    'GEBCO 2020 + NOAA CRM 90 m bathymetry, surge-only (no tides).\n'
    'Dorian: pre-calibration baseline run. Matthew modelled output available from PACE HPC run.',
    ha='center', fontsize=7, color='#444', style='italic')

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.subplots_adjust(hspace=0.38)

out = os.path.join(OUT_DIR, 'fig_gauge_timeseries.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
plt.show()
