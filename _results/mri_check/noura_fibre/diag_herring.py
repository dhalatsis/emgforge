"""Investigate two reported defects on the FCU herringbone:
(1) fibre START not actually on the muscle edge;
(2) green/purple (top/bottom) sides mixing.
Measure, don't guess."""
import numpy as np, json
from scipy.ndimage import gaussian_filter1d, binary_erosion, distance_transform_edt
from geom import load_mask,Centerline,pca_frame
from harmonic import keep_largest,trace,surface
def _tangent(cl,z):
    z=float(np.clip(z,cl.z0,cl.z1)); dx=float(cl.sx(z,1)); dy=float(cl.sy(z,1))
    t=np.array([dx,dy,1.0]); return t/np.linalg.norm(t)
AT=json.load(open("/home/noura/Documents/Projects/PhD/simulation/emgforge/src/emgforge/mri/data/forearm_muscle_atlas.json"))
L=8; key=AT["wr_label_to_muscle"][str(L)]; th=np.radians(AT["muscles"][key]["pennation_deg"])
mask,vs,_=load_mask(L); mask=keep_largest(mask)
print(f"muscle={key}  voxel size vs={vs} mm  (note z spacing)")
cl=Centerline(mask,vs); ctr,p1,p2,p3=pca_frame(mask,vs)
z0,z1=cl.z0,cl.z1; zlo,zhi=z0+0.06*(z1-z0),z1-0.06*(z1-z0)

# ---- distance-to-boundary field: how far is a point from LEAVING the mask (in mm) ----
# EDT of the mask gives distance (in voxels*aniso) from each inside voxel to the nearest outside.
edt=distance_transform_edt(mask, sampling=vs)   # mm to the true boundary
def edge_dist_mm(P):
    c=np.round(P/vs).astype(int)
    if np.any(c<0) or np.any(c>=mask.shape): return np.nan
    return edt[tuple(c)]

# ---- analytic direction field ----
D=np.zeros(mask.shape+(3,))
for v in np.argwhere(mask):
    P=v*vs; C=cl.pos(np.array([P[2]]))[0]; t=_tangent(cl,P[2])
    r=C-P; r=r-(r@t)*t; n=np.linalg.norm(r)
    d=(np.cos(th)*t+np.sin(th)*(r/n)) if n>1e-6 else t
    D[tuple(v)]=d/np.linalg.norm(d)
def smooth(f):
    k=np.concatenate([[True],np.linalg.norm(np.diff(f,axis=0),axis=1)>1e-6]); f=f[k]
    return np.stack([gaussian_filter1d(f[:,d],1.2,mode="nearest") for d in range(3)],1) if len(f)>7 else f

# ---- (1) how far are surface() seeds from the TRUE edge? ----
surf=np.argwhere(surface(mask))
sd=[edge_dist_mm(v*vs) for v in surf]; sd=np.array([x for x in sd if not np.isnan(x)])
print(f"\n[1] surface() seed distance to TRUE mask boundary (mm): "
      f"median={np.median(sd):.2f}  p90={np.percentile(sd,90):.2f}  max={sd.max():.2f}")
print(f"    in-plane half-voxel={min(vs[0],vs[1])/2:.2f}mm ; z half-voxel={vs[2]/2:.2f}mm")

# ---- (2) do fibres OVERSHOOT the central tendon and cross sides? ----
SLAB=7.0; allv=np.argwhere(mask); coord=allv*vs
seeds=np.array([v for v,ok in zip(allv,(np.abs((coord-ctr)@p2)<SLAB)) if ok and tuple(v) in set(map(tuple,surf))])
def s3(P): return float(((np.asarray(P,float)-ctr)@p3))
def fib(seed):
    f=trace(seed,D,mask,vs,sign=1,step=1.0,maxlen=120)
    if len(f)<3: return None
    f=smooth(f); f=f[(f[:,2]>=zlo)&(f[:,2]<=zhi)]
    return np.vstack([seed,f]) if len(f)>=3 else None
rng=np.random.default_rng(0); cross=0;total=0;begin_off=[]
end_dist_cl=[];overshoot=[]
for v in seeds[rng.permutation(len(seeds))][:400]:
    P=v*vs
    if not(zlo<=P[2]<=zhi): continue
    f=fib(P)
    if f is None or len(f)<5: continue
    seg=np.linalg.norm(np.diff(f,axis=0),axis=1)
    if seg.sum()<9 or seg.max()>3: continue
    total+=1
    begin_off.append(edge_dist_mm(f[0]))
    a=s3(f[0]); b=s3(f[-1])
    if np.sign(a)!=np.sign(b) and abs(b)>1.0: cross+=1     # started one side, ended other => overshoot
    # end distance from central line (should be ~0 if it stops AT tendon)
    C=cl.pos(np.array([f[-1,2]]))[0]; end_dist_cl.append(np.linalg.norm(f[-1]-C))
    overshoot.append(b)   # signed side of the END
begin_off=np.array(begin_off); end_dist_cl=np.array(end_dist_cl)
print(f"\n[2] fibres analysed: {total}")
print(f"    begin distance to TRUE edge (mm): median={np.median(begin_off):.2f} p90={np.percentile(begin_off,90):.2f}")
print(f"    fibres that CROSS to the opposite side (overshoot the tendon): {cross}/{total} = {100*cross/total:.0f}%")
print(f"    END distance to central line (mm): median={np.median(end_dist_cl):.2f} p90={np.percentile(end_dist_cl,90):.2f}")
print(f"    (if end-dist-to-CL is large or crossings>0, fibres are NOT terminating at the tendon)")
