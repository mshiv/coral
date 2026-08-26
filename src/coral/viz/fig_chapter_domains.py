"""Matthew regional context and the exact staged 30 m / 4 m domain extents.

Uses local Natural Earth coastline data only. No tile services or silent downloads.
"""
from pathlib import Path
import argparse
import gzip
import json
import struct
import numpy as np
from coral.emulator.dataset import read_asc
from coral.analysis.chapter_figure_bundle import file_record


def bounds(h):
    x,y,d = h['xllcorner'],h['yllcorner'],h['cellsize']
    return [x,x+h['ncols']*d,y,y+h['nrows']*d]


def geoclaw_bounds(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if '=:' not in line:
            continue
        value, key = line.split('=:',1)
        values[key.strip().split()[0]] = value.split()
    lo,hi = [float(v) for v in values['lower']], [float(v) for v in values['upper']]
    return [lo[0],hi[0],lo[1],hi[1]]


def track_points(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    rows = {}
    with opener(path,'rt') as f:
        for line in f:
            v = [s.strip() for s in line.split(',')]
            if len(v)<8 or v[4]!='BEST':
                continue
            if v[0]!='AL' or v[1]!='14' or not v[2].startswith('2016'):
                raise ValueError('Expected Matthew AL14 2016 BEST track, not another storm')
            def coord(s):
                return float(s[:-1])/10 * (-1 if s[-1] in 'SW' else 1)
            rows[v[2]] = (coord(v[7]),coord(v[6]))
    if len(rows)<2:
        raise ValueError('No Matthew BEST track found')
    return np.asarray([rows[k] for k in sorted(rows)])


def coastline_lines(path):
    """Read local WGS84 Natural Earth polyline geometry without a GIS dependency."""
    prj = Path(path).with_suffix('.prj').read_text().upper()
    if 'GEOGCS' not in prj or 'PROJCS' in prj or not ('WGS_1984' in prj or 'WGS 84' in prj):
        raise ValueError('Coastline must have a WGS84 geographic .prj file')
    with Path(path).open('rb') as f:
        header=f.read(100)
        if len(header)!=100 or struct.unpack('>i',header[:4])[0]!=9994:
            raise ValueError('Invalid shapefile header')
        while record := f.read(8):
            if len(record)!=8:
                raise ValueError('Truncated shapefile record')
            _,words=struct.unpack('>ii',record)
            content=f.read(words*2)
            if len(content)!=words*2:
                raise ValueError('Truncated shapefile content')
            kind=struct.unpack('<i',content[:4])[0]
            if kind==0:
                continue
            if kind not in (3,13,23):
                raise ValueError('Expected polyline coastline shapefile')
            n_parts,n_points=struct.unpack('<ii',content[36:44])
            starts=np.frombuffer(content,dtype='<i4',count=n_parts,offset=44)
            xy=np.frombuffer(content,dtype='<f8',count=2*n_points,offset=44+4*n_parts).reshape(-1,2)
            for begin,end in zip(starts,np.r_[starts[1:],n_points]):
                yield xy[begin:end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['dem30','dem4','geoclaw-data','track','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--coastline',type=Path,default=Path.home()/'.local/share/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp')
    a = p.parse_args()
    if not a.coastline.is_file():
        p.error('Local coastline missing; supply --coastline PATH to a WGS84 coastline shapefile')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    dem,h30 = read_asc(a.dem30)
    _,h4 = read_asc(a.dem4)
    b30,b4,bgeo = bounds(h30),bounds(h4),geoclaw_bounds(a.geoclaw_data)
    if not (b30[0]<=b4[0]<b4[1]<=b30[1] and b30[2]<=b4[2]<b4[3]<=b30[3]):
        raise ValueError('4 m DEM is not contained in the 30 m DEM')
    track = track_points(a.track)
    fig,axes = plt.subplots(1,2,figsize=(12,7),layout='constrained')
    def rect(ax,b,color,label):
        ax.add_patch(Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],fill=False,
                              ec=color,lw=2,label=label))
    ax = axes[0]
    for xy in coastline_lines(a.coastline):
        ax.plot(xy[:,0],xy[:,1],color='.55',lw=.5)
    ax.plot(track[:,0],track[:,1],color='#9c3f5d',lw=1.7,label='Matthew BEST track')
    rect(ax,bgeo,'#41677e','GeoClaw domain')
    rect(ax,b30,'#d27819','30 m Savannah domain')
    ax.set(xlim=(bgeo[0]-1,bgeo[1]+1),ylim=(bgeo[2]-1,bgeo[3]+1))
    ax.legend(loc='lower left',fontsize=9)
    ax = axes[1]
    dem = np.where(dem<=-9990,np.nan,dem)
    lo,hi = np.nanpercentile(dem,[2,98])
    im=ax.imshow(dem,extent=b30,origin='upper',cmap='terrain',vmin=lo,vmax=hi)
    rect(ax,b4,'#842f5b','4 m Pin Point nest')
    for name,x,y,dx,dy in [('Pin Point',-81.0903,31.9522,9,-14),
                          ('Fort Pulaski',-80.9017,32.0347,-68,9),
                          ('Savannah',-81.0998,32.0835,7,7)]:
        ax.plot(x,y,'o',color='#202b33',ms=3)
        ax.annotate(name,(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=10,
                    bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
    ax.set(xlim=b30[:2],ylim=b30[2:]); ax.legend(loc='lower left',fontsize=9)
    fig.colorbar(im,ax=ax,shrink=.65,label='Elevation (m NAVD88)',extend='both')
    for i,ax in enumerate(axes):
        ax.set(xlabel='Longitude (°)',ylabel='Latitude (°)')
        ax.set_aspect(1/np.cos(np.deg2rad(32)))
        ax.text(.02,.98,f'({chr(97+i)})',transform=ax.transAxes,va='top',weight='bold',
                bbox=dict(fc='white',ec='none',alpha=.85))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    for path in [Path(a.out),Path(a.out).with_suffix('.pdf')]:
        fig.savefig(path,dpi=250)
    Path(a.out).with_suffix('.json').write_text(json.dumps(dict(
        bounds30=b30,bounds4=b4,geoclaw_bounds=bgeo,dem_display_percentiles=[2,98],
        sources=[file_record(v) for v in [a.dem30,a.dem4,a.geoclaw_data,a.track,a.coastline]]),indent=2))
    plt.close(fig)


if __name__=='__main__':
    main()
