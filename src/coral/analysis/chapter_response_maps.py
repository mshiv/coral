"""Clean paired peak-depth differences from verified NPZ exports, not new simulations."""
from pathlib import Path
import argparse
import json
import numpy as np
from coral.analysis.chapter_figure_bundle import file_record


def build(root, out, limit=.1, threshold=.01, package=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    groups = {
        'native': [('depave','De-paving'), ('living_shoreline','Living shoreline'),
                   ('road_raise','Raised road'), ('marsh_restoration','Marsh restoration')],
        'regional': [('floodwall','Floodwall'), ('road_raise','Raised road'),
                     ('living_shoreline','Shoreline: maximum crest'),
                     ('living_shoreline_fractional','Shoreline: fractional rise'),
                     ('marsh_restoration','Marsh restoration')]}
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    for group,cases in groups.items():
        prefix='native4m_' if group=='native' else 'regional30m_'
        ncol=2 if group=='native' else 3
        fig,axes=plt.subplots(2,ncol,figsize=(4.2*ncol,8),layout='constrained',squeeze=False)
        records=[]
        loaded=[]
        for kind,label in cases:
            path=Path(root)/group/(prefix+kind+'.npz')
            with np.load(path,allow_pickle=False) as data:
                loaded.append((kind,label,path,data['delta'],json.loads(str(data['metadata_json']))))
        union=np.logical_or.reduce([np.isfinite(d)&(np.abs(d)>threshold) for _,_,_,d,_ in loaded])
        yy,xx=np.where(union)
        if group=='regional' and len(yy):
            y0,y1=max(0,yy.min()-20),min(union.shape[0],yy.max()+21)
            x0,x1=max(0,xx.min()-20),min(union.shape[1],xx.max()+21)
        else:
            y0,y1,x0,x1=0,union.shape[0],0,union.shape[1]
        view=np.s_[y0:y1,x0:x1]
        for index,(kind,label,path,delta,meta) in enumerate(loaded):
            material=np.isfinite(delta)&(np.abs(delta)>threshold)
            ax=axes.flat[index]
            # No terrain shading or input-mask overlays: only the response is coloured.
            display=np.where(material,delta,np.nan)
            im=ax.imshow(display[view],cmap='RdBu_r',vmin=-limit,vmax=limit,
                         interpolation='nearest')
            ax.set(xticks=[],yticks=[])
            ax.text(.02,.98,f'({chr(97+index)})',transform=ax.transAxes,va='top',weight='bold')
            ax.set_xlabel(label,fontsize=11)
            if not material.any():
                ax.text(.5,.5,f'No changes > {threshold*100:g} cm',transform=ax.transAxes,
                        ha='center',va='center',color='.4',fontsize=10)
            elif group=='native' and kind=='road_raise':
                cy,cx=np.unravel_index(np.nanargmax(np.abs(delta)),delta.shape)
                radius=65
                iy0,iy1=max(0,cy-radius),min(delta.shape[0],cy+radius+1)
                ix0,ix1=max(0,cx-radius),min(delta.shape[1],cx+radius+1)
                inset=ax.inset_axes([.04,.04,.43,.43])
                inset.imshow(display[iy0:iy1,ix0:ix1],cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
                inset.set(xticks=[],yticks=[])
                ax.add_patch(Rectangle((ix0,iy0),ix1-ix0,iy1-iy0,fill=False,ec='.4',lw=.8))
                inset.text(.03,.97,'Detail',transform=inset.transAxes,va='top',fontsize=8)
            records.append(dict(case=prefix+kind,source=file_record(path),
                material_cells=int(material.sum()),
                clipped_fraction=float(np.mean(np.abs(delta[material])>limit)) if material.any() else 0.,
                metrics=meta))
        for ax in list(axes.flat)[len(cases):]:
            ax.axis('off')
        if group=='regional':
            ax=axes.flat[-1]; ax.axis('on'); ax.set(xticks=[],yticks=[])
            if package:
                from coral.emulator.dataset import read_asc
                info=json.loads((Path(package)/'package.json').read_text())
                dem,_=read_asc(Path(package)/info['files']['30m/demfile']['relative'])
                ax.imshow(dem,cmap='Greys',vmin=-5,vmax=8,alpha=.45)
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#0072B2',lw=1.8))
            ax.set(xlim=(0,union.shape[1]),ylim=(union.shape[0],0),xlabel='Full regional domain; blue box = shared detail')
            ax.text(.02,.98,'(f)',transform=ax.transAxes,va='top',weight='bold')
        fig.colorbar(im,ax=list(axes.flat),orientation='horizontal',fraction=.035,pad=.035,
                     label='Intervention − control peak depth (m)',extend='both')
        stem=out/('coral_response_'+group)
        for extension in ['.png','.pdf']:
            fig.savefig(stem.with_suffix(extension),dpi=300)
        plt.close(fig)
        stem.with_suffix('.json').write_text(json.dumps(dict(
            threshold_m=threshold,color_limit_m=limit,
            extent='Native maps full extent with selected detail insets; regional maps shared response envelope plus full-domain locator',
            displayed_rows=[int(y0),int(y1)],displayed_columns=[int(x0),int(x1)],
            all_material_cells_retained=bool(union[view].sum()==union.sum()),
            interpretation='Selected cases, not rankings. Depth changes can reflect edited bed elevation.',
            cases=records),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--limit',type=float,default=.1)
    p.add_argument('--threshold',type=float,default=.01)
    p.add_argument('--package',type=Path)
    a=p.parse_args()
    if not 0 < a.threshold < a.limit:
        p.error('Require 0 < threshold < limit')
    build(a.pairs,a.out_dir,a.limit,a.threshold,a.package)


if __name__=='__main__':
    main()
