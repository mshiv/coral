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
    fig, axes = plt.subplots(1, n, figsize=(3.4*n, 4.5), squeeze=False, layout='constrained')
    provenance = []
    shape = None
    for i,(ax, case) in enumerate(zip(axes.flat,a.case)):
        label, path = case.split('=',1)
        with np.load(path, allow_pickle=False) as data:
            err = data[a.selection+'_error']
            meta = json.loads(str(data['metadata_json']))
        if shape is not None and shape != err.shape:
            raise ValueError('Holdout maps do not share raster dimensions')
        shape = err.shape
        chosen = next(r for r in meta['members'] if r['selection']==a.selection)
        im = ax.imshow(err, cmap='RdBu_r',vmin=-a.limit,vmax=a.limit, interpolation='none')
        ax.set(xticks=[],yticks=[])
        ax.text(.02,.98,f'({chr(97+i)})',transform=ax.transAxes,va='top',weight='bold')
        ax.set_xlabel(f'{label}\nRMSE {chosen["rmse_m"]:.3f} m',fontsize=10)
        valid = err[np.isfinite(err)]
        provenance.append(dict(label=label, export=str(Path(path).resolve()),
            selected=chosen, clipped_fraction=float(np.mean(np.abs(valid)>a.limit)),source=meta))
    fig.colorbar(im, ax=list(axes.flat), shrink=.8, label='Emulator − physics peak depth (m)',extend='both')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,dpi=250)
    Path(a.out).with_suffix('.json').write_text(json.dumps(dict(
        selection=a.selection,shared_limit_m=a.limit,cases=provenance),indent=2))
    plt.close(fig)


if __name__ == '__main__':
    main()
