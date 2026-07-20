"""Shared dense-fill fibre routine used by all regenerated figures.
Direction field = longitudinal harmonic (internal z-caps). Fibres = full streamlines
from a uniform cross-section grid, cut into contiguous Lf segments -> the belly fills."""
import numpy as np, json
from scipy.ndimage import gaussian_filter1d
from geom import load_mask,Centerline,CrossSection,morphing_fiber,pca_frame
from harmonic import keep_largest,solve_laplace,grad_field,trace,surface
AT=json.load(open("/home/noura/Documents/Projects/PhD/simulation/emgforge/src/emgforge/mri/data/forearm_muscle_atlas.json"))

def _smooth(f):
    k=np.concatenate([[True],np.linalg.norm(np.diff(f,axis=0),axis=1)>1e-6]); f=f[k]
    return np.stack([gaussian_filter1d(f[:,d],1.5,mode="nearest") for d in range(3)],1) if len(f)>7 else f

def setup(L):
    mask,vs,_=load_mask(L); mask=keep_largest(mask)
    cl=Centerline(mask,vs); cs=CrossSection(mask,vs,cl); ctr,p1,p2,p3=pca_frame(mask,vs)
    key=AT["wr_label_to_muscle"][str(L)]; Lf=AT["muscles"][key]["fascicle_length_mm"]
    z0,z1=cl.z0,cl.z1
    vox=np.argwhere(mask); zf=(vox[:,2]*vs[2]-z0)/(z1-z0)
    d0=np.zeros(mask.shape,bool);d1=np.zeros(mask.shape,bool)
    for v,z in zip(vox,zf):
        if 0.05<=z<=0.10:d0[tuple(v)]=True
        elif 0.90<=z<=0.95:d1[tuple(v)]=True
    g=grad_field(solve_laplace(mask,vs,d0,d1),mask,vs)
    return dict(L=L,key=key,Lf=Lf,mask=mask,vs=vs,cl=cl,cs=cs,ctr=ctr,p1=p1,p2=p2,p3=p3,
                g=g,shape=np.array(mask.shape),z0=z0,z1=z1,zlo=z0+0.07*(z1-z0),zhi=z1-0.07*(z1-z0))

def dense_fill(D,grid_mm=2.0):
    mask,vs,g,shape=D["mask"],D["vs"],D["g"],D["shape"]; Lf=D["Lf"]; zlo,zhi=D["zlo"],D["zhi"]
    zc=0.5*(D["z0"]+D["z1"]); zi=int(round(zc/vs[2])); sm=np.argwhere(mask[:,:,zi])
    def stream(seed):
        up=trace(seed,g,mask,vs,sign=1,step=1.0,maxlen=320); dn=trace(seed,g,mask,vs,sign=-1,step=1.0,maxlen=320)
        f=_smooth(np.vstack([dn[::-1],up])); return f[(f[:,2]>=zlo)&(f[:,2]<=zhi)]
    segs=[];band=[];nmj=[]; step=grid_mm/vs[0]
    for gx in np.arange(sm[:,0].min(),sm[:,0].max(),step):
        for gy in np.arange(sm[:,1].min(),sm[:,1].max(),grid_mm/vs[1]):
            c=np.array([int(gx),int(gy),zi])
            if not(np.all(c>=0)and np.all(c<shape)and mask[tuple(c)]): continue
            f=stream(np.array([gx*vs[0],gy*vs[1],zc]))
            if len(f)<15: continue
            arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))]); tot=arc[-1]
            ns=max(1,int(round(tot/Lf))); Le=tot/ns
            for kk in range(ns):
                seg=f[(arc>=kk*Le)&(arc<=(kk+1)*Le)]
                if len(seg)>=4: segs.append(seg); band.append(kk); nmj.append(f[np.argmin(np.abs(arc-(kk+0.5)*Le))])
    return segs,np.array(band),np.array(nmj)

def _muscle_len_frac(D):
    coord=np.argwhere(D["mask"])*D["vs"]; s1=(coord-D["ctr"])@D["p1"]
    return float(s1.min()), float(s1.max()-s1.min())

def _cover_bands(lo,hi,atlas,step):
    """Return [(phi, is_atlas)] bands covering [lo,hi]: atlas IZ fractions pinned in place,
    plus geometric fill bands so no gap exceeds `step` (=Lf/Lm) — keeps fibres ~Lf long."""
    xs=sorted(f for f in atlas if lo+0.01<f<hi-0.01)
    res=[(f,True) for f in xs]
    pts=[lo]+xs+[hi]
    for a,b in zip(pts[:-1],pts[1:]):
        n=int(np.ceil((b-a)/step))-1
        for i in range(1,n+1): res.append((a+i*(b-a)/(n+1),False))
    return sorted(res)

def _cut_by_iz(f,D,s1min,Lm,atlas,step):
    """Cut a streamline into in-series segments whose NMJs sit on the covering bands.
    Returns [(segment, band_index, nmj_point, is_atlas)]."""
    phi=((f-D["ctr"])@D["p1"]-s1min)/Lm
    lo,hi=float(phi.min()),float(phi.max())
    bands=_cover_bands(lo,hi,atlas,step)
    if not bands: return [(f,0,f[len(f)//2],False)]
    fr=[b[0] for b in bands]
    bnds=[lo]+[(fr[k]+fr[k+1])/2 for k in range(len(fr)-1)]+[hi]
    out=[]
    for k,(phik,atl) in enumerate(bands):
        seg=f[(phi>=bnds[k])&(phi<=bnds[k+1])]
        if len(seg)>=4:
            j=int(np.argmin(np.abs(phi-phik))); out.append((seg,k,f[j],atl))
    return out

def _iz_params(D):
    atlas=sorted(AT["muscles"][D["key"]].get("IZ_fraction",[0.3]))
    s1min,Lm=_muscle_len_frac(D); step=float(np.clip(D["Lf"]/Lm,0.12,0.6))
    return atlas,s1min,Lm,step

def dense_fill_iz(D,grid_mm=2.0):
    """Like dense_fill but NMJs are PINNED to atlas IZ fractions (+ geometric fill).
    Returns segs, band(list), nmj(array), is_atlas(array-bool)."""
    mask,vs,g,shape=D["mask"],D["vs"],D["g"],D["shape"]; zlo,zhi=D["zlo"],D["zhi"]
    atlas,s1min,Lm,step=_iz_params(D)
    zc=0.5*(D["z0"]+D["z1"]); zi=int(round(zc/vs[2])); sm=np.argwhere(mask[:,:,zi])
    def stream(seed):
        up=trace(seed,g,mask,vs,sign=1,step=1.0,maxlen=320); dn=trace(seed,g,mask,vs,sign=-1,step=1.0,maxlen=320)
        f=_smooth(np.vstack([dn[::-1],up])); return f[(f[:,2]>=zlo)&(f[:,2]<=zhi)]
    segs=[];band=[];nmj=[];isatl=[]; st=grid_mm/vs[0]
    for gx in np.arange(sm[:,0].min(),sm[:,0].max(),st):
        for gy in np.arange(sm[:,1].min(),sm[:,1].max(),grid_mm/vs[1]):
            c=np.array([int(gx),int(gy),zi])
            if not(np.all(c>=0)and np.all(c<shape)and mask[tuple(c)]): continue
            f=stream(np.array([gx*vs[0],gy*vs[1],zc]))
            if len(f)<15: continue
            for seg,k,nm,atl in _cut_by_iz(f,D,s1min,Lm,atlas,step):
                segs.append(seg); band.append(k); nmj.append(nm); isatl.append(atl)
    return segs,np.array(band),np.array(nmj),np.array(isatl,bool)

def sparse_columns_iz(D,ncol=8):
    """A few seed columns, each cut by the atlas-pinned IZ bands.
    Returns list of stacks; each stack = [(segment, band_index, nmj_point, is_atlas)]."""
    mask,vs,p3,ctr=D["mask"],D["vs"],D["p3"],D["ctr"]
    atlas,s1min,Lm,step=_iz_params(D)
    zc=0.5*(D["z0"]+D["z1"]); zi=int(round(zc/vs[2])); sm=np.argwhere(mask[:,:,zi])
    P=np.column_stack([sm[:,0]*vs[0],sm[:,1]*vs[1],np.full(len(sm),zc)]); t3=(P-ctr)@p3
    order=np.argsort(t3); pick=order[np.linspace(0,len(order)-1,ncol).astype(int)]
    cols=[]
    for idx in pick:
        st=finite_from_seed_full(D,np.array([sm[idx,0]*vs[0],sm[idx,1]*vs[1],zc]))
        if st is None: continue
        cut=_cut_by_iz(st,D,s1min,Lm,atlas,step)
        if cut: cols.append(cut)
    return cols

def finite_from_seed_full(D,seed):
    """Full streamline (uncut) from a seed, for IZ-pinned cutting."""
    mask,vs,g=D["mask"],D["vs"],D["g"]; zlo,zhi=D["zlo"],D["zhi"]
    up=trace(seed,g,mask,vs,sign=1,step=1.0,maxlen=320); dn=trace(seed,g,mask,vs,sign=-1,step=1.0,maxlen=320)
    f=_smooth(np.vstack([dn[::-1],up])); f=f[(f[:,2]>=zlo)&(f[:,2]<=zhi)]
    return f if len(f)>=15 else None

def finite_from_seed(D,seed):
    """Trace ONE full streamline from seed and cut it into contiguous in-series Lf fibres.
    Returns [(segment, band_index, nmj_point), ...] — the in-series stack for that column."""
    mask,vs,g=D["mask"],D["vs"],D["g"]; Lf=D["Lf"]; zlo,zhi=D["zlo"],D["zhi"]
    up=trace(seed,g,mask,vs,sign=1,step=1.0,maxlen=320); dn=trace(seed,g,mask,vs,sign=-1,step=1.0,maxlen=320)
    f=_smooth(np.vstack([dn[::-1],up])); f=f[(f[:,2]>=zlo)&(f[:,2]<=zhi)]
    if len(f)<15: return []
    arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))]); tot=arc[-1]
    ns=max(1,int(round(tot/Lf))); Le=tot/ns; out=[]
    for kk in range(ns):
        seg=f[(arc>=kk*Le)&(arc<=(kk+1)*Le)]
        if len(seg)>=4: out.append((seg,kk,f[np.argmin(np.abs(arc-(kk+0.5)*Le))]))
    return out

def sparse_columns(D,ncol=7):
    """A handful of seed columns spread across the mid cross-section (thin axis), each
    returning its in-series stack — sparse enough to show begin/end/NMJ markers."""
    mask,vs,p3,ctr=D["mask"],D["vs"],D["p3"],D["ctr"]
    zc=0.5*(D["z0"]+D["z1"]); zi=int(round(zc/vs[2])); sm=np.argwhere(mask[:,:,zi])
    P=np.column_stack([sm[:,0]*vs[0],sm[:,1]*vs[1],np.full(len(sm),zc)]); t3=(P-ctr)@p3
    order=np.argsort(t3); pick=order[np.linspace(0,len(order)-1,ncol).astype(int)]
    cols=[]
    for idx in pick:
        seed=np.array([sm[idx,0]*vs[0],sm[idx,1]*vs[1],zc])
        st=finite_from_seed(D,seed)
        if st: cols.append(st)
    return cols

def min_gap(D,grid=4.0):
    """Non-crossing metric: trace streamlines from a grid, resample at common z-levels,
    return (min pairwise gap in mm, n streamlines)."""
    from scipy.spatial.distance import pdist
    mask,vs,g=D["mask"],D["vs"],D["g"]; zlo,zhi=D["zlo"],D["zhi"]
    zc=0.5*(D["z0"]+D["z1"]); zi=int(round(zc/vs[2])); sm=np.argwhere(mask[:,:,zi])*vs[:2]
    zg=np.linspace(zlo+5,zhi-5,30); R=[]
    for gx in np.arange(sm[:,0].min(),sm[:,0].max(),grid):
        for gy in np.arange(sm[:,1].min(),sm[:,1].max(),grid):
            c=np.array([int(round(gx/vs[0])),int(round(gy/vs[1])),zi])
            if not(0<=c[0]<mask.shape[0] and 0<=c[1]<mask.shape[1] and mask[tuple(c)]): continue
            up=trace(np.array([gx,gy,zc]),g,mask,vs,sign=1,step=1.0,maxlen=320)
            dn=trace(np.array([gx,gy,zc]),g,mask,vs,sign=-1,step=1.0,maxlen=320)
            sf=np.vstack([dn[::-1],up]); sf=sf[(sf[:,2]>=zlo)&(sf[:,2]<=zhi)]
            if len(sf)<15: continue
            r=np.stack([np.interp(zg,sf[:,2],sf[:,0]),np.interp(zg,sf[:,2],sf[:,1])],1)
            if all(np.min(np.linalg.norm(rk-r,axis=1))>1.5 for rk in R): R.append(r)  # dedup near-duplicate seeds
    R=np.array(R)
    if len(R)<2: return (0.0,len(R))
    return (min(pdist(R[:,z,:]).min() for z in range(len(zg))), len(R))

def morph(D,taper):
    cl,cs=D["cl"],D["cs"]; thp=np.degrees(np.arctan2(D["p3"][1],D["p3"][0]))
    zv=np.linspace(D["z0"],D["z1"],80)
    return [morphing_fiber(cl,cs,rn,th,zv,taper) for rn in (0.3,0.55,0.8) for th in (thp,thp+180)]

BANDCOL=["#1a9850","#66bd63","#006837","#a6d96a","#41ab5d"]
