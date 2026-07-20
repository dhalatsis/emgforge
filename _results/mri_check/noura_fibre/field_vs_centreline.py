"""Is the harmonic field NECESSARY, or does it just reproduce the RED/GREY centre-line direction?

Measures, per muscle:
  (A) angle between the field direction g and the GLOBAL long axis p1 (the naive 'straight fibre'):
      how much curvature/taper the field captures that a straight line misses.
  (B) angle between g (at RED fibre points) and the RED fibre's own tangent: how much the field
      redirects relative to what RED actually assumes.
  (C) CONTAINMENT: fraction of RED (centre-line-offset) fibre length that stays inside the muscle
      mask, vs the field streamlines (which are contained by construction). RED poking outside the
      muscle is a concrete failure a straight/offset model cannot avoid on curved/tapered muscles.

Note: densefib's validation field is LONGITUDINAL (z-cap BCs), so pennation is NOT in g here (it is
imposed separately). So this isolates the field's geometric value = containment + curvature, not the
bipennate convergence (that is the herringbone flagship, separate aponeurosis BCs).
"""
import sys, numpy as np
sys.path.insert(0,"/home/noura/Documents/Projects/PhD/simulation/emgforge/_results/mri_check/noura_fibre")
sys.path.insert(0,"/home/noura/Documents/Projects/PhD/simulation/emgforge/src")
import densefib as DF

def samp_g(g,mask,vs,p):
    c=p/vs; i=np.round(c).astype(int)
    if np.any(i<0) or np.any(i>=mask.shape) or not mask[i[0],i[1],i[2]]: return None
    v=g[i[0],i[1],i[2]]; n=np.linalg.norm(v)
    return v/n if n>1e-9 else None

def ang(u,v): return np.degrees(np.arccos(np.clip(abs(np.dot(u,v)),0,1)))   # unsigned line angle

LABELS=[11,8,13,21,23]     # BR(fusiform,2.4) FCU(12) FDS(6.8) pron.teres(9.6) pron.quad(10)
print(f"{'muscle':<20}{'penn':>5}{'Lf':>5} | {'g·p1 med/90':>14} | {'g·REDtan med/90':>16} | {'RED contain':>12} | field")
for L in LABELS:
    D=DF.setup(L); g,mask,vs,p1=D["g"],D["mask"],D["vs"],D["p1"]
    key=D["key"]; import json; penn=DF.AT["muscles"][key].get("pennation_deg","?")
    # (A) field vs global long axis
    gg=g[mask]; nn=np.linalg.norm(gg,axis=1); gg=gg[nn>1e-9]/nn[nn>1e-9,None]
    aA=np.degrees(np.arccos(np.clip(np.abs(gg@p1),0,1)))
    # (B,C) RED fibres: morphing centre-line offsets
    reds=DF.morph(D,None)
    aB=[]; contained=[]; total=0.0; inside=0.0
    for f in reds:
        f=np.asarray(f,float)
        seg=np.linalg.norm(np.diff(f,axis=0),axis=1)
        tang=np.diff(f,axis=0); tn=np.linalg.norm(tang,axis=1); tang=tang[tn>1e-9]/tn[tn>1e-9,None]
        mids=0.5*(f[:-1]+f[1:])[tn>1e-9]
        for pmid,t,dl in zip(mids,tang,seg[tn>1e-9]):
            total+=dl
            gv=samp_g(g,mask,vs,pmid)
            ci=np.round(pmid/vs).astype(int)
            ins = (np.all(ci>=0) and np.all(ci<mask.shape) and mask[ci[0],ci[1],ci[2]])
            inside+= dl if ins else 0.0
            if gv is not None: aB.append(ang(gv,t))
    aB=np.array(aB); cfrac=100*inside/max(total,1e-9)
    def med90(x): return (np.median(x),np.percentile(x,90)) if len(x) else (float('nan'),float('nan'))
    aAm,aA9=med90(aA); aBm,aB9=med90(aB)
    print(f"{key[:20]:<20}{str(penn):>5}{D['Lf']:>5.0f} | "
          f"{aAm:>5.1f}/{aA9:>5.1f}° | {aBm:>6.1f}/{aB9:>5.1f}° | "
          f"{cfrac:>10.1f}% | nvox={mask.sum()} ng={len(gg)} nred={len(aB)}")
