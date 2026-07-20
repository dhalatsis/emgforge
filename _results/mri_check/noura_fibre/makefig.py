import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from geom import load_mask,Centerline,CrossSection,morphing_fiber,pca_frame
from harmonic import keep_largest,solve_laplace,grad_field,trace,surface

L=13; Lf=50.0                       # atlas fibre length (mm) for this muscle
mask,vs,_=load_mask(L); mask=keep_largest(mask)
cl=Centerline(mask,vs); cs=CrossSection(mask,vs,cl); ctr,p1,p2,p3=pca_frame(mask,vs)
def proj(P): d=np.atleast_2d(P)-ctr; return d@p1, d@p3
th_p3=np.degrees(np.arctan2(p3[1],p3[0]))

# ---- RED/GREY morphing ----
zv=np.linspace(cl.z0,cl.z1,80)
def morph_set(taper):
    return [morphing_fiber(cl,cs,rn,th,zv,taper=taper)
            for rn in (0.2,0.45,0.7,0.92) for th in (th_p3,th_p3+180)]
red=morph_set(None); grey=morph_set(0.35)

# ---- NEW: internal-aponeurosis harmonic field ----
# Dirichlet on two INTERNAL z-slabs (the origin/insertion aponeuroses), tilted by k
# for pennation. Because the slabs are internal (not the geometric end-caps), the
# muscle ends are free (Neumann) -> no end-of-muscle kink by construction.
def apo_caps(k):
    vp=np.argwhere(mask)*vs; zf=(vp[:,2]-cl.z0)/(cl.z1-cl.z0)
    t3=(vp-ctr)@p3; t3n=t3/(np.abs(t3).max()+1e-9)
    sig=zf+k*t3n
    d0=np.zeros(mask.shape,bool); d1=np.zeros(mask.shape,bool)
    for v,x in zip(np.argwhere(mask),sig):
        if 0.05<=x<=0.09: d0[tuple(v)]=True
        elif 0.91<=x<=0.95: d1[tuple(v)]=True
    return d0,d1
d0,d1=apo_caps(k=0.0); phi=solve_laplace(mask,vs,d0,d1); g=grad_field(phi,mask,vs)

from scipy.ndimage import gaussian_filter1d
def streamline(seed):
    up=trace(seed,g,mask,vs,sign=1,step=1.0,maxlen=260)
    dn=trace(seed,g,mask,vs,sign=-1,step=1.0,maxlen=260)
    f=np.vstack([dn[::-1],up])
    keep=np.concatenate([[True],(np.linalg.norm(np.diff(f,axis=0),axis=1)>1e-6)])
    f=f[keep]
    # light smoothing removes 6mm-voxel faceting/steps along the fibre
    if len(f)>7:
        f=np.stack([gaussian_filter1d(f[:,d],2.0,mode="nearest") for d in range(3)],1)
    return f
# belly = between the two aponeuroses; the gap beyond is tendon (fibres attach to
# the aponeuroses, not the tendon tips). Clip just inside the slabs to drop the
# streamline's bend INTO the Dirichlet slab.
zc=0.5*(cl.z0+cl.z1)
zlo=cl.z0+0.11*(cl.z1-cl.z0); zhi=cl.z1-0.11*(cl.z1-cl.z0)
streams=[]; kept_mid=[]
for zseed in np.linspace(zlo+12,zhi-12,5):
    pc=cl.pos(np.array([zseed]))[0]
    for a in np.linspace(-0.85,0.85,7):
        seed=pc+a*11*p3; c=np.round(seed/vs).astype(int)
        if not(np.all(c>=0) and np.all(c<np.array(mask.shape)) and mask[tuple(c)]): continue
        sfib=streamline(seed); sfib=sfib[(sfib[:,2]>=zlo)&(sfib[:,2]<=zhi)]
        if len(sfib)<25 or np.max(np.linalg.norm(np.diff(sfib,axis=0),axis=1))>3: continue
        mid=sfib[len(sfib)//2]
        if any(np.linalg.norm(mid-m)<3.0 for m in kept_mid): continue   # dedup shared paths
        streams.append(sfib); kept_mid.append(mid)
# TILE each streamline end-to-end into equal ~Lf fibres: the first fibre BEGINS at
# the proximal aponeurosis, the last ENDS at the distal one, IZ at each fibre centre.
new=[]; izs=[]; ends=[]
for sfib in streams:
    arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(sfib,axis=0),axis=1))])
    total=arc[-1]; nseg=max(1,int(round(total/Lf))); Le=total/nseg
    for k_ in range(nseg):
        m=(arc>=k_*Le)&(arc<=(k_+1)*Le); seg=sfib[m]
        if len(seg)<4: continue
        new.append(seg); ends.append(seg[0]); ends.append(seg[-1])
        izs.append(sfib[np.argmin(np.abs(arc-(k_+0.5)*Le))])
izs=np.array(izs); ends=np.array(ends)

# ---- metrics ----
def contained(fs):
    p=np.vstack(fs); ci=np.clip(np.round(p/vs).astype(int),0,np.array(mask.shape)-1)
    return mask[ci[:,0],ci[:,1],ci[:,2]].mean()*100
def penn(fs):
    ang=[]
    for f in fs:
        for k_ in range(1,len(f)):
            seg=f[k_]-f[k_-1]; nn=np.linalg.norm(seg)
            if nn<1e-6: continue
            t=cl.pos(np.array([f[k_,2]]))[0]-cl.pos(np.array([f[k_,2]-4]))[0]; t/=max(np.linalg.norm(t),1e-9)
            ang.append(np.degrees(np.arccos(np.clip(abs((seg/nn)@t),0,1))))
    return np.mean(ang)
def flen(fs): return np.mean([np.sum(np.linalg.norm(np.diff(f,axis=0),axis=1)) for f in fs if len(f)>1])
print(f"RED  contained {contained(red):.1f}%  len {flen(red):.0f}mm")
print(f"GREY contained {contained(grey):.1f}%  len {flen(grey):.0f}mm")
nbands=int(round(np.median([max(1,int(round((np.sum(np.linalg.norm(np.diff(s,axis=0),axis=1)))/Lf))) for s in streams])))
print(f"NEW  contained {contained(new):.1f}%  len {flen(new):.0f}mm  pennation {penn(new):.1f}deg  nfib {len(new)}  IZ bands {nbands}")

# ---- figure 3x2 ----
surfv=np.argwhere(surface(mask))*vs; U,W=proj(surfv); u0,u1=U.min()-5,U.max()+5
zoom=(-40,40)
fig,ax=plt.subplots(3,2,figsize=(13,9))
rows=[("#d62728","RED — morphing disk (constant offset, full-length)",red,False),
      ("#7f7f7f","GREY — fusiform taper (pinched ends, full-length)",grey,False),
      ("#2ca02c","NEW — harmonic streamlines (finite Lf, in-series IZ bands, follows the belly)",new,True)]
for r,(col,title,fs,isnew) in enumerate(rows):
    for c in range(2):
        a=ax[r,c]; a.scatter(U,W,s=2,c="0.87",alpha=.5,zorder=0)
        for f in fs:
            u,w=proj(f); a.plot(u,w,color=col,lw=1.1,alpha=.9)
            if isnew:  # mark where each fibre begins/ends (the in-series boundaries)
                eu,ew=proj(f[[0,-1]]); a.scatter(eu,ew,s=8,marker='|',c="#8c2d04",zorder=5)
        if isnew:
            iu,iw=proj(izs); a.scatter(iu,iw,s=11,c="k",zorder=6,label="IZ (fibre centre)")
        a.set_aspect("equal")
        if c==0: a.set_xlim(u0,u1); a.set_title(title,fontsize=10,loc="left"); a.set_ylabel("PC3 [mm]")
        else: a.set_xlim(*zoom); a.set_title("zoom (60 mm window)",fontsize=9)
        if r==2: a.set_xlabel("along muscle (PC1) [mm]")
        if isnew and c==1:
            a.scatter([],[],s=30,marker='|',c="#8c2d04",label="fibre begin/end")
            a.legend(fontsize=7,loc="upper right")
txt=(f"contained: RED {contained(red):.0f}%  GREY {contained(grey):.0f}%  NEW {contained(new):.0f}%\n"
     f"fibre length: RED/GREY {flen(red):.0f} mm (≈full muscle)   NEW {flen(new):.0f} mm (atlas Lf)\n"
     f"NEW pennation vs centreline: {penn(new):.1f}°   IZ bands: {nbands} (staggered in-series)")
fig.text(0.5,-0.02,txt,ha="center",fontsize=9,family="monospace",
         bbox=dict(boxstyle="round",fc="#f4f4f4",ec="0.7"))
fig.suptitle(f"WR forearm muscle L{L}: fibre-geometry methods — current (red/grey) vs proposed (new)",fontsize=12)
plt.tight_layout(); plt.savefig("compare_methods.png",dpi=100,bbox_inches="tight"); print("saved")
