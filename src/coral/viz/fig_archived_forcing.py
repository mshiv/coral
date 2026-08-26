"""Plot actual archived rainfall and HVAR boundaries on their native model clock.

Streaming BDY parsing avoids loading a multi-hundred-MB token list. Boundary
types come from BCI, not from whether a series happens to look like stage.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from coral.analysis.chapter_figure_bundle import file_record


def hvar_points(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        v=line.split()
        if len(v)>=5 and v[0]=='P' and v[3]=='HVAR':
            result.append((v[4],float(v[1]),float(v[2])))
    if not result:
        raise ValueError('No point HVAR boundaries in BCI')
    return result


def boundary_series(path, wanted):
    result={}
    with Path(path).open() as f:
        next(f, None)  # LISFLOOD comment line
        while True:
            line=f.readline()
            if not line:
                break
            name=line.strip()
            if not name:
                continue
            info=f.readline().split()
            if len(info)!=2 or info[1].lower()!='seconds':
                raise ValueError(f'Expected counted seconds block after {name}')
            n=int(info[0]); rows=[]
            for _ in range(n):
                line=f.readline()
                if not line:
                    raise ValueError('Truncated BDY block '+name)
                if name in wanted:
                    rows.append([float(v) for v in line.split()])
            if name in wanted:
                arr=np.asarray(rows)
                if arr.shape!=(n,2) or not np.isfinite(arr).all() or np.any(np.diff(arr[:,1])<=0):
                    raise ValueError('Invalid or nonmonotonic BDY series '+name)
                result[name]=arr
    if set(result)!=set(wanted):
        raise ValueError('Missing requested HVAR series')
    return result


def build(package,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from coral.emulator.dataset import read_asc
    from coral.viz.fig_chapter_domains import bounds
    package=Path(package); out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    meta=json.loads((package/'package.json').read_text())
    def source(key):
        info=meta['files']['30m/'+key]; p=package/info['relative']
        if file_record(p)['sha256']!=info['source']['sha256']:
            raise ValueError('Changed archived input: '+str(p))
        return p
    bci,bdy,rain,dem=[source(k) for k in ['bcifile','bdyfile','rainfall','demfile']]
    points=sorted(hvar_points(bci),key=lambda v:v[2])
    chosen=[points[i] for i in [0,len(points)//2,len(points)-1]]
    names=[p[0] for p in chosen]; series=boundary_series(bdy,names)
    rain_data=np.loadtxt(rain,skiprows=2)
    if rain_data.ndim!=2 or rain_data.shape[1]!=2 or np.any(np.diff(rain_data[:,1])<=0):
        raise ValueError('Expected increasing (rate, model seconds) rainfall')
    par=meta['results_inventory']['30m']['parameters']
    start=float(par['tstart'][0]); stop=float(par['sim_time'][0])
    fig=plt.figure(figsize=(12,6.5),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=[1,1.8]); axmap=fig.add_subplot(gs[:,0])
    axrain=fig.add_subplot(gs[0,1]); axstage=fig.add_subplot(gs[1,1],sharex=axrain)
    z,h=read_asc(dem); ext=bounds(h)
    axmap.imshow(z,extent=ext,origin='upper',cmap='Greys',vmin=-5,vmax=8,alpha=.45)
    axmap.scatter([p[1] for p in points],[p[2] for p in points],s=3,c='.55')
    colors=['#0072B2','#D55E00','#009E73']
    labels=['Southern boundary','Central boundary','Northern boundary']
    summaries=[]
    for (name,x,y),color,label in zip(chosen,colors,labels):
        axmap.scatter(x,y,c=color,s=35,zorder=4)
        arr=series[name]; take=(arr[:,1]>=start)&(arr[:,1]<=stop); arr=arr[take]
        axstage.plot(arr[:,1]/3600,arr[:,0],c=color,label=label,lw=1.2)
        summaries.append(dict(name=name,label=label,longitude=x,latitude=y,
                              peak_stage_m=float(arr[:,0].max()),
                              peak_time_model_s=float(arr[np.argmax(arr[:,0]),1])))
    axmap.set(xlabel='Longitude (°)',ylabel='Latitude (°)')
    axmap.set_aspect(1/np.cos(np.deg2rad(32)))
    axmap.tick_params(labelsize=8)
    axrain.plot(rain_data[:,1]/3600,rain_data[:,0],color='#0072B2')
    axrain.set(ylabel='Applied rainfall (mm h$^{-1}$)')
    axstage.set(xlabel='Model time (hours)',ylabel='Applied boundary stage (m)',xlim=(start/3600,stop/3600))
    axstage.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,fontsize=9,frameon=False)
    axmap.xaxis.set_major_locator(MaxNLocator(4))
    for letter,ax in zip('abc',[axmap,axrain,axstage]):
        ax.text(.02,.98,f'({letter})',transform=ax.transAxes,va='top',weight='bold',
                bbox=dict(fc='white',ec='none',alpha=.85))
    for ax in [axrain,axstage]:
        ax.grid(alpha=.2)
    for extension in ['.png','.pdf']:
        fig.savefig(out.with_suffix(extension),dpi=300)
    plt.close(fig)
    out.with_suffix('.json').write_text(json.dumps(dict(
        sources={k:file_record(p) for k,p in [('bci',bci),('bdy',bdy),('rain',rain),('dem',dem)]},
        boundaries=summaries,tstart_model_s=start,tfinal_model_s=stop,
        clock='Native model seconds; no UTC conversion inferred',
        interpretation='Applied HVAR stages include coupling transformations; not pure surge or observations'),indent=2))
    with out.with_suffix('.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['model_s','rain_mm_h',*names])
        for rate,t in rain_data:
            writer.writerow([t,rate,*[np.interp(t,series[n][:,1],series[n][:,0]) for n in names]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(); build(a.package,a.out)


if __name__=='__main__':
    main()
