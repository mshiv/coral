"""Export an explicit baseline and a bounded inventory of missing chapter sources.

No simulations, no recursive ensemble search, and no inferred factorial-arm labels.
Snapshot times are nominal output times, not exact solver timestamps or UTC.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import numpy as np
from coral.analysis.chapter_figure_bundle import file_record, previous_inputs, par_inputs
from coral.emulator.dataset import read_asc


def same_grid(a, b):
    for key in ['nrows', 'ncols', 'xllcorner', 'yllcorner', 'cellsize']:
        if not np.isclose(a[key], b[key], rtol=0, atol=1e-10):
            raise ValueError('Grid mismatch: '+key)


def frame_sequence(files, start, stop, interval):
    if not 0 < interval <= stop-start:
        raise ValueError('Parameter file does not support a snapshot sequence')
    numbered = []
    for path in files:
        match = re.search(r'-(\d+)\.wd$', Path(path).name)
        if not match:
            raise ValueError('Unrecognised snapshot name: '+str(path))
        numbered.append((int(match[1]), Path(path)))
    numbered.sort()
    expected = list(range(int(np.floor((stop-start)/interval))+1))
    if [i for i, _ in numbered] != expected:
        raise ValueError('Missing, duplicate or unexpected snapshot indices')
    return [p for _, p in numbered], start+interval*np.asarray(expected)


def collect(manifest, runs_root, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Use a fresh export directory: '+str(out))
    explicit = previous_inputs(manifest)
    fields, record = par_inputs(explicit['par30'])
    par = record['parameters']; run = explicit['par30'].parent
    result = Path(par['dirroot'][0])
    if not result.is_absolute():
        result = run/result
    stem = par['resroot'][0]
    maximum = result/(stem+'.max')
    dem, header = read_asc(fields['demfile'])
    peak, hp = read_asc(maximum); same_grid(header, hp)
    frames, times = frame_sequence(result.glob(stem+'-*.wd'),
        float(par['tstart'][0]), float(par['sim_time'][0]), float(par['saveint'][0]))
    chosen = np.unique(np.rint(np.linspace(0, len(frames)-1, 6)).astype(int))
    selected = []; wet_area = []; sources = []
    land = np.isfinite(dem) & (dem > 1.114)
    for index, path in enumerate(frames):
        depth, h = read_asc(path); same_grid(header, h)
        if np.any(land & ~np.isfinite(depth)):
            raise ValueError('Missing depth over valid baseline land: '+str(path))
        wet_area.append(int(np.count_nonzero(land & (depth > .1))))
        if index in chosen:
            selected.append(depth.astype('float32'))
        sources.append(file_record(path))
    out.mkdir(parents=True)
    meta = dict(source_manifest=file_record(manifest), baseline=record,
        maximum=file_record(maximum), snapshots=sources, header=header,
        selected_indices=chosen.tolist(), nominal_model_seconds=times.tolist(),
        clock='Nominal tstart + index*saveint; actual writes follow solver timesteps. No UTC conversion.',
        provenance_caveat='Current archived par matches the bundle; generating job log still requires review.',
        waterline_m=1.114, wet_depth_threshold_m=.1,
        wet_area_note='Counts of cells, avoiding an unverified constant area for a geographic grid.')
    np.savez_compressed(out/'baseline_event.npz', dem=dem.astype('float32'),
        peak=peak.astype('float32'), frames=np.stack(selected),
        nominal_seconds=times, selected_indices=chosen, wet_land_cells=np.asarray(wet_area),
        metadata_json=np.asarray(json.dumps(meta)))
    (out/'baseline_event.json').write_text(json.dumps(meta, indent=2))
    # Small evidence files and maxima; never copy a multi-GB frame directory.
    roots = Path(runs_root)
    candidates = {roots/name for name in ['statictide_30m','compound_tide_30m',
        'statictide_norain_30m','tide_norain_30m']}
    # Only exact scenario filenames, one directory level beneath the runs root.
    for name in ['savannah_matthew_rain_only','savannah_matthew_tide_baseline']:
        candidates.update(p.parent for p in roots.glob('*/'+name+'.par'))
    inventory = []
    for directory in sorted(candidates):
        entry = dict(directory=str(directory), exists=directory.is_dir(), parameters=[], outputs=[])
        if directory.is_dir():
            dest = out/'validation_sources'/directory.name
            dest.mkdir(parents=True)
            for parameter in sorted(directory.glob('*.par')):
                try:
                    _, info = par_inputs(parameter)
                    entry['parameters'].append(info)
                    shutil.copy2(parameter, dest/parameter.name, follow_symlinks=True)
                    params = info['parameters']
                    folder = Path(params['dirroot'][0])
                    if not folder.is_absolute():
                        folder = directory/folder
                    prefix = params['resroot'][0]
                    for suffix in ['.max','.mxe','.mass']:
                        source = folder/(prefix+suffix)
                        if source.is_file():
                            target = dest/parameter.stem/source.name
                            target.parent.mkdir(exist_ok=True)
                            shutil.copy2(source, target)
                            entry['outputs'].append(dict(source=file_record(source),
                                relative=str(target.relative_to(out))))
                except (OSError, ValueError, KeyError) as error:
                    entry.setdefault('errors',[]).append(str(error))
        inventory.append(entry)
    (out/'validation_inventory.json').write_text(json.dumps(inventory, indent=2))
    print('Exported baseline event and candidate validation sources:', out, flush=True)


def plot(path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(path, allow_pickle=False) as data:
        frames=data['frames']; seconds=data['nominal_seconds']; indices=data['selected_indices']
        dem=data['dem']; counts=data['wet_land_cells']; meta=json.loads(str(data['metadata_json']))
    land=np.isfinite(dem)&(dem>meta['waterline_m'])
    fig=plt.figure(figsize=(11,9),layout='constrained')
    gs=fig.add_gridspec(3,3,height_ratios=[1,1,.5]); axes=[]
    for i,depth in enumerate(frames):
        ax=fig.add_subplot(gs[i//3,i%3]); axes.append(ax)
        im=ax.imshow(np.where(land & (depth>.1),depth,np.nan),cmap='Blues',vmin=0,vmax=2)
        ax.set(xticks=[],yticks=[],xlabel=f'Nominal model hour {seconds[indices[i]]/3600:g}')
        ax.text(.02,.98,f'({chr(97+i)})',transform=ax.transAxes,va='top',weight='bold')
    fig.colorbar(im,ax=axes,label='Depth on baseline land (m)',extend='max',shrink=.7)
    ax=fig.add_subplot(gs[2,:]); ax.plot(seconds/3600,100*counts/land.sum(),color='#0072B2')
    ax.set(xlabel='Nominal model time (hours)',ylabel='Flooded baseline land (%)')
    ax.text(.02,.98,'(g)',transform=ax.transAxes,va='top',weight='bold'); ax.grid(alpha=.2)
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['.png','.pdf']:
        fig.savefig(out.with_suffix(extension),dpi=250)
    plt.close(fig)
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path); p.add_argument('--runs-root',type=Path)
    p.add_argument('--out',required=True,type=Path); p.add_argument('--plot',type=Path)
    a=p.parse_args()
    if a.plot:
        plot(a.plot,a.out)
    elif a.manifest and a.runs_root:
        collect(a.manifest,a.runs_root,a.out)
    else:
        p.error('Supply --plot, or both --manifest and --runs-root')


if __name__=='__main__':
    main()
