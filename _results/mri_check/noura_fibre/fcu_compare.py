"""FCU (real bipennate): RED / GREY / NEW on the SAME muscle.
NEW uses a SINGLE harmonic solve (outer walls -> central tendon) so it is a pure
gradient field => streamlines cannot cross (unlike the earlier two-field blend)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from geom import load_mask,Centerline,CrossSection,morphing_fiber,pca_frame
from harmonic import keep_largest,solve_laplace,grad_field,trace,surface
L=8; Lf=51.0; mask,vs,_=load_mask(L); mask=keep_largest(mask)
cl=Centerline(mask,vs); cs=CrossSection(mask,vs,cl); ctr,p1,p2,p3=pca_frame(mask,vs); shape=np.array(mask.shape)
def proj(P): d=np.atleast_2d(P)-ctr; return d@p1,d@p3
thp=np.degrees(np.arctan2(p3[1],p3[0]))
def smooth(f):
    k=np.concatenate([[True],np.linalg.norm(np.diff(f,axis=0),axis=1)>1e-6]); f=f[k]
    return np.stack([gaussian_filter1d(f[:,d],1.5,mode="nearest") for d in range(3)],1) if len(f)>7 else f

# RED / GREY on FCU
zv=np.linspace(cl.z0,cl.z1,80)
red=[morphing_fiber(cl,cs,rn,th,zv,None) for rn in (0.3,0.6,0.9) for th in (thp,thp+180)]
grey=[morphing_fiber(cl,cs,rn,th,zv,0.35) for rn in (0.3,0.6,0.9) for th in (thp,thp+180)]

# NEW bipennate: ONE Laplace solve, outer walls(0) -> central tendon(1), longitudinally offset for lean
vox=np.argwhere(mask); vp=vox*vs; s=(vp-ctr)@p1; s=(s-s.min())/(s.max()-s.min()); t3=(vp-ctr)@p3; t3max=np.abs(t3).max()
surf=surface(mask); sv=np.argwhere(surf); svp=sv*vs; ss=(svp-ctr)@p1; ss=(ss-ss.min())/(ss.max()-ss.min()); st3=(svp-ctr)@p3
d0=np.zeros(mask.shape,bool); d1=np.zeros(mask.shape,bool)
for v,x,t in zip(sv,ss,st3):
    if abs(t)>0.45*t3max and x<=0.72: d0[tuple(v)]=True      # outer walls, proximal-biased
for v,x,t in zip(vox,s,t3):
    if abs(t)<0.13*t3max and x>=0.28: d1[tuple(v)]=True       # central tendon, distal-biased
g=grad_field(solve_laplace(mask,vs,d0,d1),mask,vs)             # single gradient field -> non-crossing
zlo,zhi=cl.z0+0.10*(cl.z1-cl.z0),cl.z1-0.10*(cl.z1-cl.z0)
def stream(seed):
    up=trace(seed,g,mask,vs,sign=1,step=1.0,maxlen=Lf); dn=trace(seed,g,mask,vs,sign=-1,step=1.0,maxlen=Lf)
    f=smooth(np.vstack([dn[::-1],up])); return f[(f[:,2]>=zlo)&(f[:,2]<=zhi)]
top=[];bot=[]
for zc in np.linspace(zlo+8,zhi-8,12):
    pc=cl.pos(np.array([zc]))[0]
    for d3 in (5.0,3.0,-3.0,-5.0):            # symmetric absolute depths (muscle is ~+-6mm thick)
        for b in np.linspace(-9,9,3):
            seed=pc+d3*p3+b*p2; c=np.round(seed/vs).astype(int)
            if not(np.all(c>=0)and np.all(c<shape)and mask[tuple(c)]): continue
            f=stream(seed)
            if len(f)<6 or np.max(np.linalg.norm(np.diff(f,axis=0),axis=1))>3: continue
            (top if d3>0 else bot).append(f)
def dedup(fs,r=2.5):
    out=[];mids=[]
    for f in fs:
        m=f[len(f)//2]
        if all(np.linalg.norm(m-k)>r for k in mids): out.append(f); mids.append(m)
    return out
top=dedup(top); bot=dedup(bot); newf=top+bot
# non-crossing check: min distance between distinct fibre polylines (subsample)
def polymin(A,B): return cdist(A[::3],B[::3]).min()
mind=np.inf
for i in range(len(newf)):
    for j in range(i+1,len(newf)):
        mind=min(mind,polymin(newf[i],newf[j]))
print(f"NEW bipennate: {len(top)} top + {len(bot)} bot fibres; min gap between distinct fibres = {mind:.2f} mm (>0 => non-crossing)")

# figure: 3 rows on the SAME FCU
sv2=np.argwhere(surface(mask))*vs; U,W=proj(sv2)
fig,ax=plt.subplots(3,1,figsize=(11,7.2))
rows=[("#d62728","RED — morphing disk (full-length)",red,None),
      ("#7f7f7f","GREY — fusiform taper (full-length)",grey,None),
      (None,"NEW — bipennate (single harmonic field, non-crossing)",None,(top,bot))]
for a,(col,ttl,fs,tb) in zip(ax,rows):
    a.scatter(U,W,s=3,c="0.87",alpha=.5)
    if fs is not None:
        for f in fs: u,w=proj(f); a.plot(u,w,color=col,lw=1.1,alpha=.9)
    else:
        tt=np.linspace(zlo,zhi,20); cp=np.array([cl.pos(np.array([z]))[0] for z in tt]); cu,cw=proj(cp)
        a.plot(cu,cw,color="#d62728",lw=1.6,alpha=.7)
        for f in tb[0]: u,w=proj(f); a.plot(u,w,color="#1a9850",lw=1.1,alpha=.85)
        for f in tb[1]: u,w=proj(f); a.plot(u,w,color="#762a83",lw=1.1,alpha=.85)
    a.set_aspect("equal"); a.set_title(ttl,fontsize=9.5,loc="left"); a.set_ylabel("PC3 [mm]")
ax[-1].set_xlabel("along muscle (PC1) [mm]")
fig.suptitle(f"FCU (label 8, real bipennate) — the three methods on ONE muscle   (NEW min fibre gap {mind:.1f} mm)",fontsize=11)
plt.tight_layout(); plt.savefig("fcu_compare.png",dpi=108,bbox_inches="tight"); print("saved")
