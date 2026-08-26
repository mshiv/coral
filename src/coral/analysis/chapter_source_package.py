"""Copy only verified context-figure inputs into a portable, hash-checked package.

Run on HPC with --from-manifest OLD/bundle_manifest.json --out-dir NEW_PACKAGE.
After rsync, --render-package NEW_PACKAGE --out-dir NEW_FIGURES renders locally.
No simulation, source raster edits, or overwriting of existing directories.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

from coral.analysis.chapter_figure_bundle import file_record, previous_inputs


def collect(manifest, out):
    old = json.loads(Path(manifest).read_text())
    explicit = previous_inputs(manifest)  # fail before copying if a source changed
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    sources = {}
    results_inventory = {}
    for name, run in old['runs'].items():
        for key in ['par', 'demfile', 'manningfile', 'infilfile', 'infilcapfile',
                    'bdyfile', 'bcifile', 'rainfall']:
            if key in run['files']:
                info = run['files'][key]
                sources[name+'/'+key] = (Path(info['path']), Path(name)/Path(info['path']).name)
        run_dir = Path(run['files']['par']['path']).parent
        # Inventory only: never substitute an SLR-shifted diagnostic for the baseline.
        results_inventory[name] = dict(run_directory=str(run_dir), parameters=run['parameters'],
            outputs={suffix: sorted(str(p) for p in run_dir.glob('results_*/*'+suffix))
                     for suffix in ['.max','.mxe','.mass','.wd']},
            clock_note='UTC origin and snapshot-index timing still require verification')
    for flag in ['geoclaw_data', 'track', 'coastline']:
        if flag not in explicit:
            raise ValueError('Missing context source: '+flag)
        p = explicit[flag]
        sources[flag] = (p, Path('context')/p.name)
    prj = explicit['coastline'].with_suffix('.prj')
    sources['coastline_prj'] = (prj, Path('context')/prj.name)
    targets = [str(target) for _,target in sources.values()]
    if len(set(targets)) != len(targets):
        raise ValueError('Source basenames collide; use distinct package paths')
    records = {key: dict(source=file_record(src), relative=str(target))
               for key,(src,target) in sources.items()}
    out.mkdir(parents=True)
    for key,(src,target) in sources.items():
        dest=out/target; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src,dest,follow_symlinks=True)
        if file_record(dest)['sha256'] != records[key]['source']['sha256']:
            raise ValueError('Copied file failed checksum: '+str(dest))
    (out/'package.json').write_text(json.dumps(dict(
        source_manifest=file_record(manifest), files=records, results_inventory=results_inventory,
        note='Parameter files are archived unchanged, not runnable local configurations'), indent=2))
    print('Source package:',out)


def render(package, out):
    root=Path(package); out=Path(out)
    data=json.loads((root/'package.json').read_text())
    files={}
    for key,info in data['files'].items():
        p=(root/info['relative']).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError('Package path escapes root')
        if file_record(p)['sha256'] != info['source']['sha256']:
            raise ValueError('Package checksum mismatch: '+str(p))
        files[key]=p
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    commands=[]
    for name,stem in [('4m','coral_model_inputs'), ('30m','appendix_inputs_30m')]:
        args=['--publication','--out',out/(stem+'.png')]
        for key,flag in [('demfile','--dem'),('manningfile','--manning'),
                         ('infilfile','--infil'),('infilcapfile','--infilcap')]:
            args += [flag,files[name+'/'+key]]
        commands.append(['coral.viz.fig_model_inputs',*args])
    commands.append(['coral.viz.fig_chapter_domains','--dem4',files['4m/demfile'],
        '--dem30',files['30m/demfile'],'--geoclaw-data',files['geoclaw_data'],
        '--track',files['track'],'--coastline',files['coastline'],
        '--out',out/'coral_domain_context.png'])
    for module,*args in commands:
        subprocess.run([sys.executable,'-m',module,*map(str,args)],check=True)
    (out/'package_provenance.json').write_text(json.dumps(data,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--from-manifest',type=Path)
    mode.add_argument('--render-package',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    if a.from_manifest:
        collect(a.from_manifest,a.out_dir)
    else:
        render(a.render_package,a.out_dir)


if __name__=='__main__':
    main()
