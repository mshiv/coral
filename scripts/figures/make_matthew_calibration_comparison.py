"""
CORAL Project — Matthew 2016 Calibration COMPARISON  (2×2 ablation)
Multi-run version of make_matthew_calibration.py.

Experimental design — sensitivity sweep, one variable changed per run vs. baseline.
"Tidal IC offset" = static sea_level initial condition (not time-varying tidal forcing).

  Experiment                  | sea_level | drag_law      | Topo
  ---------------------------+-----------+---------------+----------------------------
  Baseline                    |    0.0    | 1 (Garratt)   | GEBCO + CRM 90 m
  + Hi-res topo (Savannah)    |    0.0    | 1 (Garratt)   | GEBCO + CRM + CUDEM ~30 m
  + Tidal IC                  |   +0.81   | 1 (Garratt)   | GEBCO + CRM 90 m
  + Tidal IC + Powell drag    |   +0.81   | 2 (Powell)    | GEBCO + CRM 90 m

  Note: matthew_2016_Powell is a duplicate of matthew_2016_MSLtide_0.81
  (identical settings) — not plotted.

Plots whichever runs have completed; skips empty/missing gauge files gracefully.

Run with:  MPLBACKEND=Agg conda run -n aislens \
              python3 src/figures/make_matthew_calibration_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime, os

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CF_ROOT   = os.path.join(REPO_ROOT, '..', 'coastalFlood')
OUT_DIR   = os.path.join(REPO_ROOT, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Each run: (long_label, short_label, gauge_path, line_color, line_style)
# Sensitivity sweep — one change per run vs. baseline (matthew_2016).
# Note: matthew_2016_Powell is a duplicate of matthew_2016_MSLtide_0.81 — dropped.
# Order chosen to read as a story: baseline → topo refine → tide IC → tide IC + drag
RUNS = [
    ('Baseline                          (sea=0,       Garratt,  GEBCO+CRM)',
     'Baseline',
     os.path.join(CF_ROOT, 'matthew_2016',                 '_output', 'gauge00002.txt'),
     '#1a2c52', '-'),
    ('+ Hi-res topo (CUDEM)   (sea=0,       Garratt,  + Savannah CUDEM)',
     '+ Hi-res topo',
     os.path.join(CF_ROOT, 'matthew_2016_30m',             '_output', 'gauge00002.txt'),
     '#b6755a', '-'),
    ('+ Tidal IC                       (sea=+0.81,  Garratt,  GEBCO+CRM)',
     '+ Tidal IC',
     os.path.join(CF_ROOT, 'matthew_2016_Garrett',         '_output', 'gauge00002.txt'),
     '#1a5c5c', '-'),
    ('+ Tidal IC + Powell        (sea=+0.81,  Powell,    GEBCO+CRM)',
     '+ Tide + Powell',
     os.path.join(CF_ROOT, 'matthew_2016_MSLtide_0.81',    '_output', 'gauge00002.txt'),
     '#e05c00', '-'),
]

# Observed (NOAA) colour
C_OBS      = '#2ca02c'
C_LANDFALL = '#666'
C_MATTHEW  = '#d62728'

# ── Font sizes (poster / meeting scale) ────────────────────────────────────
FS_TITLE    = 22
FS_AXIS     = 18
FS_TICK     = 16
FS_LEGEND   = 14
FS_PEAK     = 14
FS_LANDFALL = 13

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
    if not os.path.exists(path):
        return np.array([]), np.array([])
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
matthew_setrun_t0   = datetime.datetime(2016, 10, 8, 12)   # GeoClaw t=0
matthew_ga_t0       = datetime.datetime(2016, 10, 8,  6)   # GA closest approach
matthew_gauge_shift = (matthew_setrun_t0 - matthew_ga_t0).total_seconds() / 3600  # +6h

# ══════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5.6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.30, linewidth=0.7)

# ── Load + plot each modelled run ──────────────────────────────────────────
peak_summary = []     # (short_label, color, peak_value or None)

for label, short_label, path, color, ls in RUNS:
    t_raw, eta_raw = read_geoclaw_gauge(path)
    if len(t_raw) == 0:
        print(f"[skip] {short_label:<14s}  ({path})  — no data yet")
        peak_summary.append((short_label, color, None))
        continue
    valid = (eta_raw > -3.0) & (eta_raw < 5.0)
    t_hrs = t_raw[valid] / 3600.0 + matthew_gauge_shift
    eta   = eta_raw[valid]
    ax.plot(t_hrs, eta, color=color, ls=ls, lw=2.6, label=label, zorder=3)
    # Peak in the surge window
    win = (t_hrs > -24) & (t_hrs < 24)
    pk  = eta[win].max() if win.sum() > 0 else np.nan
    peak_summary.append((short_label, color, pk))
    print(f"[ok]   {short_label:<14s}  {len(t_hrs):>6d} pts, "
          f"eta [{eta.min():+.2f}, {eta.max():+.2f}] m,  peak {pk:+.2f} m")

# ── NOAA observed ──────────────────────────────────────────────────────────
obs_t, obs_surge = fetch_noaa_surge(
    '8670870',
    datetime.datetime(2016, 10, 5),
    datetime.datetime(2016, 10, 10))

obs_peak = np.nan
if obs_t is not None and len(obs_t) > 0:
    obs_t_hrs = np.array([(t - matthew_ga_t0).total_seconds() / 3600 for t in obs_t])
    ax.plot(obs_t_hrs, obs_surge, color=C_OBS, lw=2.8,
            label='Observed surge  (NOAA 8670870)', zorder=4)
    win = (obs_t_hrs > -24) & (obs_t_hrs < 24)
    if win.sum() > 0:
        obs_peak = obs_surge[win].max()
    print(f"[ok]  Observed: {len(obs_t_hrs)} pts, peak {obs_peak:.2f} m")
else:
    print("[skip] Observed: NOAA fetch returned nothing")

# ── Closest-approach reference line ───────────────────────────────────────
ax.axvline(x=0, color=C_LANDFALL, lw=1.8, ls='--', alpha=0.85, zorder=2)
ax.text(0.50, 0.97,
        'GA closest approach  ·  Oct 8, 06Z  ·  31.6°N, 80.6°W',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=FS_LANDFALL, color=C_LANDFALL,
        bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.85))

# ── Axes formatting ───────────────────────────────────────────────────────
ax.set_xlim([-36, 36])
ax.set_ylim([-1.8, 3.2])
ax.set_xticks([-36, -24, -12, 0, 12, 24, 36])
ax.set_xticklabels(['-36', '-24', '-12', '0', '+12', '+24', '+36'])
ax.set_xlabel('Hours from closest approach', fontsize=FS_AXIS, fontweight='bold')
ax.set_ylabel('Surge height  (m)', fontsize=FS_AXIS, fontweight='bold')
ax.set_title('Hurricane Matthew (2016) — Calibration sensitivity at Fort Pulaski (NOAA 8670870)',
             fontsize=FS_TITLE, fontweight='bold', color=C_MATTHEW, pad=12)

ax.legend(loc='lower left', fontsize=FS_LEGEND, framealpha=0.92,
          edgecolor='#aaa', borderpad=0.7, handlelength=2.4)

# ── Peak-summary callout (upper right) ────────────────────────────────────
if not np.isnan(obs_peak):
    lines = [f'{"Observed":<13s} {obs_peak:>5.2f} m']
    lines.append('-' * 26)
    for short_label, color, pk in peak_summary:
        if pk is None or np.isnan(pk):
            lines.append(f'{short_label:<13s} {"pending":>5s}')
        else:
            pct = (pk - obs_peak) / obs_peak * 100
            lines.append(f'{short_label:<13s} {pk:>5.2f} m  ({pct:+.0f}%)')
    summary = '\n'.join(lines)
    ax.text(0.985, 0.96, summary,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FS_PEAK, color='#222', family='monospace',
            bbox=dict(facecolor='#fff5f4', edgecolor=C_MATTHEW,
                      boxstyle='round,pad=0.6', linewidth=1.4))

plt.tight_layout()
out = os.path.join(OUT_DIR, 'fig_matthew_calibration_comparison.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {out}")
print("Done.")
