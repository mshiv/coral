"""Postprocess chapter figures from explicit inputs; never launch physics runs.

Paths inside a symlinked .par resolve against its RUN directory, as LISFLOOD does,
not the directory containing the symlink target. Sources and hashes are recorded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import subprocess
import sys


def file_record(path):
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return dict(path=str(path), resolved=str(path.resolve()),
                bytes=path.stat().st_size, sha256=digest.hexdigest())


def par_inputs(path):
    path = Path(path).absolute()  # deliberately do not resolve the .par symlink
    values = {}
    for line in path.read_text().splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens:
            key = tokens[0].lower()
            if key in values:
                raise ValueError(f'duplicate parameter {key}: {path}')
            values[key] = tokens[1:]
    fields, files = {}, {'par': file_record(path)}
    for key in ['demfile','manningfile','infilfile','infilcapfile','bdyfile','bcifile',
                'startfile','rainfall']:
        if key not in values:
            continue
        if len(values[key]) != 1:
            raise ValueError(f'expected one file for {key}: {path}')
        f = Path(values[key][0])
        f = f if f.is_absolute() else path.parent / f
        fields[key] = str(f.absolute())
        files[key] = file_record(f)
    if 'demfile' not in fields:
        raise ValueError(f'no DEMfile in {path}')
    return fields, dict(parameters=values, files=files)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--par4', type=Path)
    p.add_argument('--par30', type=Path)
    p.add_argument('--geoclaw-data', type=Path)
    p.add_argument('--track', type=Path)
    p.add_argument('--coastline', type=Path)
    p.add_argument('--paired', type=Path, default=Path('reports/adapt/paired'))
    p.add_argument('--amr-csv', type=Path)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.out_dir.exists() and not a.resume:
        p.error('output directory exists; choose a new folder or explicitly --resume')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    inventory = {'runs': {}, 'commands': [], 'omitted': []}
    def run(module, args):
        cmd = [sys.executable, '-m', module, *map(str,args)]
        inventory['commands'].append(cmd)
        with (a.out_dir / (module.rsplit('.',1)[-1]+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('OK',module,flush=True)
    try:
        grids = {}
        for name, par in [('4m',a.par4),('30m',a.par30)]:
            if par is None:
                inventory['omitted'].append(name+' input maps: no explicit parameter file')
                continue
            fields, record = par_inputs(par)
            inventory['runs'][name] = record
            grids[name] = fields['demfile']
            args = ['--dem',fields['demfile'],'--publication','--out',
                    a.out_dir / ('coral_model_inputs.png' if name=='4m' else 'appendix_inputs_30m.png')]
            for key, flag in [('manningfile','--manning'),('infilfile','--infil'),('infilcapfile','--infilcap')]:
                if key not in fields:
                    raise ValueError(f'{par}: missing {key}; inspect production configuration before plotting')
                args += [flag,fields[key]]
            run('coral.viz.fig_model_inputs',args)
        domain_args = [a.geoclaw_data,a.track,a.par4,a.par30]
        if all(domain_args):
            inventory['geoclaw'] = file_record(a.geoclaw_data)
            inventory['track'] = file_record(a.track)
            args=['--dem30',grids['30m'],'--dem4',grids['4m'],
                '--geoclaw-data',a.geoclaw_data,'--track',a.track,
                '--out',a.out_dir/'coral_domain_context.png']
            if a.coastline:
                args += ['--coastline',a.coastline]
            run('coral.viz.fig_chapter_domains',args)
        else:
            inventory['omitted'].append('Domain map needs --par4 --par30 --geoclaw-data --track')
        run('coral.viz.fig_model_chain',['--out',a.out_dir/'coral_model_chain.png'])
        if a.paired.is_dir():
            run('coral.analysis.paired_chapter_summary',['--inputs',a.paired,'--out-dir',a.out_dir])
        else:
            inventory['omitted'].append('paired summary: JSON directory missing')
        if a.amr_csv:
            inventory['amr_csv'] = file_record(a.amr_csv)
            run('coral.viz.fig_amr_convergence',['--from-csv',a.amr_csv,'--publication',
                '--out',a.out_dir/'coral_amr_convergence.png'])
    finally:
        (a.out_dir/'bundle_manifest.json').write_text(json.dumps(inventory,indent=2))
    print('Bundle:',a.out_dir)
    for why in inventory['omitted']:
        print('NOT BUILT:',why)


if __name__ == '__main__':
    main()
