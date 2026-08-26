"""Plot completed paired diagnostics, preserving reporting thresholds and sources."""
from pathlib import Path
import argparse
import csv
import json

CASES = [
    ('native4m_depave', 'De-paving'),
    ('native4m_living_shoreline', 'Living shoreline'),
    ('native4m_road_raise', 'Raised road'),
    ('native4m_marsh_restoration', 'Marsh restoration'),
    ('regional30m_floodwall', 'Floodwall'),
    ('regional30m_road_raise', 'Raised road'),
    ('regional30m_living_shoreline', 'Shoreline: maximum crest'),
    ('regional30m_living_shoreline_fractional', 'Shoreline: fractional rise'),
    ('regional30m_marsh_restoration', 'Marsh restoration'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows, sources = [], []
    for stem, label in CASES:
        path = a.inputs / (stem + '.json')
        data = json.loads(path.read_text())
        sources.append({'case': stem, 'path': str(path.resolve()),
                        'baseline_depth': data.get('baseline_control_depth'),
                        'field_tolerance': data.get('field_tolerance')})
        for r in data['threshold_sensitivity']:
            rows.append(dict(case=stem, label=label, **r))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / 'paired_thresholds.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (a.out_dir / 'paired_sources.json').write_text(json.dumps(sources, indent=2))
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
    for ax, prefix, letter in zip(axes, ['native4m', 'regional30m'], ['a', 'b']):
        selected = [r for r in rows if r['case'].startswith(prefix)
                    and abs(r['threshold_m']-.01) < 1e-8]
        y = np.arange(len(selected))
        ax.barh(y-.18, [r['benefit_m3']/1000 for r in selected], .34,
                color='#287ca4', label='Peak-depth reduction')
        ax.barh(y+.18, [r['adverse_m3']/1000 for r in selected], .34,
                color='#c65b32', label='Peak-depth increase')
        ax.set_yticks(y, [r['label'] for r in selected]); ax.invert_yaxis()
        ax.set_xlabel('Area-integrated peak-depth change ($10^3$ m³)')
        ax.text(0, 1.03, f'({letter})', transform=ax.transAxes, weight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(loc='lower right', fontsize=9)
    for suffix in ['png', 'pdf']:
        fig.savefig(a.out_dir / ('coral_paired_summary.'+suffix), dpi=250)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout='constrained')
    for ax, (stem, label), letter in zip(axes.flat, CASES, 'abcdefghi'):
        rr = sorted([r for r in rows if r['case'] == stem], key=lambda r:r['threshold_m'])
        x = [r['threshold_m'] for r in rr]
        reduction=np.asarray([r['benefit_m3'] for r in rr])
        increase=np.asarray([r['adverse_m3'] for r in rr])
        ax.plot(x, reduction/(reduction[0] or 1), 'o-', color='#287ca4')
        ax.plot(x, increase/(increase[0] or 1), 's--', color='#c65b32')
        ax.set(xscale='log', xlabel='Threshold (m)',ylim=(-.03,1.30),
               ylabel='Fraction retained from 0.005 m')
        ax.set_yticks([0,.25,.5,.75,1])
        ax.text(.02,.98,f'({letter})',transform=ax.transAxes,va='top',weight='bold')
        ax.text(.98,.98,label+'\n'+('4 m' if stem.startswith('native') else '30 m'),
                transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.grid(alpha=.2)
    for suffix in ['png', 'pdf']:
        fig.savefig(a.out_dir / ('coral_paired_thresholds.'+suffix), dpi=250)
    plt.close(fig)
    print(f'wrote summary, threshold figure and source tables -> {a.out_dir}')


if __name__ == '__main__':
    main()
