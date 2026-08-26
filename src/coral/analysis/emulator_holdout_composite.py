"""Difference-only holdout comparison from exported fields, with a shared scale.

Repeat --case 'Label=/path/to/export.npz'. Selection is worst within each report,
not a common member or an unbiased estimate of each holdout's average accuracy.
"""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=float, default=.7, help='Shared symmetric colour limit (m)')
    p.add_argument('--selection', choices=['best','median','worst'], default='worst')
    a = p.parse_args()
    if a.limit <= 0:
        p.error('--limit must be positive')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(a.case)
    fig = plt.figure(figsize=(max(7,3.4*n), 7.3), layout='constrained')
    grid = fig.add_gridspec(2,n,height_ratios=[1.4,1])
    axes = [fig.add_subplot(grid[0,i]) for i in range(n)]
    metric_ax = fig.add_subplot(grid[1,:])
    provenance = []
    shape = None
    for i,(ax, case) in enumerate(zip(axes,a.case)):
        label, path = case.split('=',1)
        with np.load(path, allow_pickle=False) as data:
            err = data[a.selection+'_error']
            meta = json.loads(str(data['metadata_json']))
        if shape is not None and shape != err.shape:
            raise ValueError('Holdout maps do not share raster dimensions')
        shape = err.shape
        chosen = next(r for r in meta['members'] if r['selection']==a.selection)
        rmse = np.asarray(meta.get('holdout_rmse_m',[]),dtype=float)
        if not rmse.size or np.any(~np.isfinite(rmse)) or np.any(rmse<0):
            raise ValueError('Export lacks valid holdout metrics; regenerate with the current emulator_vs_physics script')
        rmse.sort()
        metric_ax.step(rmse, np.arange(1,len(rmse)+1)/len(rmse), where='post',
                       label=f'{label} (n={len(rmse)})')
        im = ax.imshow(err, cmap='RdBu_r',vmin=-a.limit,vmax=a.limit, interpolation='none')
        ax.set(xticks=[],yticks=[])
        ax.text(.02,.98,f'({chr(97+i)})',transform=ax.transAxes,va='top',weight='bold')
        ax.set_xlabel(f'{label}\nRMSE {chosen["rmse_m"]:.3f} m',fontsize=10)
        valid = err[np.isfinite(err)]
        provenance.append(dict(label=label, export=str(Path(path).resolve()),
            selected=chosen, clipped_fraction=float(np.mean(np.abs(valid)>a.limit)),source=meta))
    fig.colorbar(im, ax=axes, shrink=.8, label='Emulator − physics peak depth (m)',extend='both')
    metric_ax.set(xlabel='Held-out member RMSE (m)',ylabel='Cumulative fraction of members',ylim=(0,1.02))
    metric_ax.text(.01,.98,f'({chr(97+n)})',transform=metric_ax.transAxes,va='top',weight='bold')
    metric_ax.legend(loc='lower right',frameon=False)
    metric_ax.grid(alpha=.2)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,dpi=250)
    fig.savefig(Path(a.out).with_suffix('.pdf'))
    Path(a.out).with_suffix('.json').write_text(json.dumps(dict(
        selection=a.selection,shared_limit_m=a.limit,cases=provenance),indent=2))
    plt.close(fig)


if __name__ == '__main__':
    main()
