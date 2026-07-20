"""Diagnostic: does the DIRECTION FIELD alone fill the muscle physiologically?
Uniform cross-section grid -> full streamlines. Two views + cross-section."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from geom import load_mask,Centerline,pca_frame
from harmonic import keep_largest,solve_laplace,grad_field,trace,surface
L=8; mask,vs,_=load_mask(L); mask=keep_largest(mask)
cl=Centerline(mask,vs); ctr,p1,p2,p3=pca_frame(mask,vs); shape=np.array(mask.shape)
z0,z1=cl.z0,cl.z1; span=z1-z0
# longitudinal field, internal caps
vox=np.argwhere(mask); zf=(vox[:,2]*vs[2]-z0)/span
d0=np.zeros(mask.shape,bool);d1=np.zeros(mask.shape,bool)
for v,z in zip(vox,zf):
    if 0.05<=z<=0.10:d0[tuple(v)]=True
    elif 0.90<=z<=0.95:d1[tuple(v)]=True
print("d0",d0.sum(),"d1",d1.sum())
phi=solve_laplace(mask,vs,d0,d1); g=grad_field(phi,mask,vs)
print("phi range",round(phi[mask].min(),2),round(phi[mask].max(),2))
zlo,zhi=z0+0.07*span,z1-0.07*span
def stream(seed):
    up=trace(seed,g,mask,vs,sign=1,step=1.0,maxlen=300); dn=trace(seed,g,mask,vs,sign=-1,step=1.0,maxlen=300)
    f=np.vstack([dn[::-1],up]); return f[(f[:,2]>=zlo)&(f[:,2]<=zhi)]
# UNIFORM grid across the cross-section at mid-length
zc=0.5*(z0+z1); zi=int(round(zc/vs[2])); sm=np.argwhere(mask[:,:,zi])*vs[:2]
fibs=[]
for gx in np.arange(sm[:,0].min(),sm[:,0].max(),3.0):
    for gy in np.arange(sm[:,1].min(),sm[:,1].max(),3.0):
        c=np.array([int(round(gx/vs[0])),int(round(gy/vs[1])),zi])
        if not(np.all(c>=0)and np.all(c<shape)and mask[tuple(c)]): continue
        f=stream(np.array([gx,gy,zc]))
        if len(f)>15: fibs.append(f)
print(f"{len(fibs)} full streamlines from uniform grid")
def proj(P,u,v): d=np.atleast_2d(P)-ctr; return d@u,d@v
sv=np.argwhere(surface(mask))*vs
fig,ax=plt.subplots(3,1,figsize=(11,8))
for a,(u,v,ttl) in zip(ax,[(p1,p3,"side view (PC1-PC3, thin axis)"),
                            (p1,p2,"side view (PC1-PC2, wide axis)"),
                            (p2,p3,"cross-section (PC2-PC3) at mid")]):
    if "cross" in ttl:
        s2=np.argwhere(mask[:,:,zi]); smid=np.column_stack([s2[:,0]*vs[0],s2[:,1]*vs[1],np.full(len(s2),zc)])
        U,W=proj(smid,u,v); a.scatter(U,W,s=4,c="0.85")
        pts=np.array([f[np.argmin(np.abs(f[:,2]-zc))] for f in fibs])
        U,W=proj(pts,u,v); a.scatter(U,W,s=18,c="#2ca02c",edgecolor="k",lw=.3)
    else:
        U,W=proj(sv,u,v); a.scatter(U,W,s=2,c="0.88",alpha=.5)
        for f in fibs: uu,ww=proj(f,u,v); a.plot(uu,ww,color="#2ca02c",lw=0.7,alpha=.7)
    a.set_aspect("equal"); a.set_title(ttl,fontsize=9,loc="left")
fig.suptitle(f"FCU DIRECTION-FIELD diagnostic — {len(fibs)} streamlines filling the muscle (longitudinal, k=0)",fontsize=11)
plt.tight_layout(); plt.savefig("diag_fill.png",dpi=105,bbox_inches="tight"); print("saved")
