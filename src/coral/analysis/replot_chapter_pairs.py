"""Replot explicit staged pairs, checking metrics against the copied prior reports.

No restaging or simulations. --pair CASE CONTROL_DIRECTORY INTERVENTION_DIRECTORY
may be repeated. Inputs are resolved relative to each run, never its par symlink target.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
from coral.analysis.chapter_figure_bundle import par_inputs, file_record


def staged_inputs(run):
    run=Path(run)
    pars=list(run.glob('*.par'))
    if len(pars) != 1:
        raise ValueError(f'{run}: expected exactly one .par; found {len(pars)}')
    fields,record=par_inputs(pars[0])
    maxima=list(run.glob('results_*/*.max'))
    if len(maxima) != 1:
        raise ValueError(f'{run}: expected exactly one completed maximum raster; found {len(maxima)}')
    record['maximum']=file_record(maxima[0])
    return fields, maxima[0].parent, record


def compare_metrics(old, new):
    for key in ['wet_cells','footprint_cells','improved_cells','worsened_cells','benefit_m3','adverse_m3']:
        if not np.isclose(old[key],new[key],rtol=1e-6,atol=1e-5):
            raise ValueError(f'Replot changed {key}: {old[key]} -> {new[key]}; review sources before use')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair',action='append',nargs=3,required=True,metavar=('CASE','CONTROL','INTERVENTION'))
    p.add_argument('--previous-reports',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--cell-m',type=float,required=True)
    p.add_argument('--waterline',type=float,required=True)
    p.add_argument('--error-limit',type=float,default=.1)
    a=p.parse_args()
    if a.out_dir.exists():
        p.error('Choose a new output directory; existing results are never overwritten')
    jobs=[]; records={}
    for case,control,intervention in a.pair:
        if Path(case).name != case or case in records:
            p.error('Case labels must be unique plain filenames')
        old_path=a.previous_reports/(case+'.json')
        old=json.loads(old_path.read_text())
        c,cr,cm=staged_inputs(control); i,ir,im=staged_inputs(intervention)
        pairs=[]
        for key in ['demfile','manningfile','infilfile','infilcapfile']:
            if key not in c or key not in i:
                raise ValueError('Missing required field '+key)
            pairs.append(c[key]+':'+i[key])
        args=['--control-results',cr,'--intervention-results',ir,'--control-dem',c['demfile'],
              '--fields',*pairs,'--waterline',a.waterline,'--cell-m',a.cell_m,
              '--tol-m',old['depth_tolerance_m'],'--field-tol',old['field_tolerance'],
              '--wet-m',old['baseline_control_depth']['wet_threshold_m'],
              '--publication','--error-limit',a.error_limit,
              '--out-json',a.out_dir/(case+'.json'),'--out-fig',a.out_dir/(case+'.png'),
              '--export-npz',a.out_dir/(case+'.npz')]
        jobs.append((case,old,[sys.executable,'-m','coral.analysis.paired_redistribution',*map(str,args)]))
        records[case]=dict(control=cm,intervention=im,previous_report=file_record(old_path))
    a.out_dir.mkdir(parents=True)
    try:
        for case,old,cmd in jobs:
            records[case]['command']=cmd
            with (a.out_dir/(case+'.log')).open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            new=json.loads((a.out_dir/(case+'.json')).read_text())
            compare_metrics(old,new)
            records[case]['metrics_match_previous']=True
            print('Verified unchanged metrics:',case,flush=True)
    finally:
        (a.out_dir/'replot_manifest.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
