import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter1d
from geom import load_mask,Centerline,CrossSection,morphing_fiber,pca_frame
from harmonic import keep_largest,solve_laplace,grad_field,trace,surface

L=13; Lf=50.0
mask,vs,_=load_mask(L); mask=keep_largest(mask)
cl=Centerline(mask,vs); cs=CrossSection(mask,vs,cl); ctr,p1,p2,p3=pca_frame(mask,vs)
shape=np.array(mask.shape); th_p3=np.degrees(np.arctan2(p3[1],p3[0]))
zlo=cl.z0+0.11*(cl.z1-cl.z0); zhi=cl.z1-0.11*(cl.z1-cl.z0)

def smooth(f):
    keep=np.concatenate([[True],(np.linalg.norm(np.diff(f,axis=0),axis=1)>1e-6)]); f=f[keep]
    if len(f)>7: f=np.stack([gaussian_filter1d(f[:,d],2.0,mode="nearest") for d in range(3)],1)
    return f
def streamf(seed,g,msk):
    up=trace(seed,g,msk,vs,sign=1,step=1.0,maxlen=260); dn=trace(seed,g,msk,vs,sign=-1,step=1.0,maxlen=260)
    return smooth(np.vstack([dn[::-1],up]))
def tile(sf):
    arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(sf,axis=0),axis=1))]); tot=arc[-1]
    nseg=max(1,int(round(tot/Lf))); Le=tot/nseg; out=[];iz=[]
    for k in range(nseg):
        m=(arc>=k*Le)&(arc<=(k+1)*Le); seg=sf[m]
        if len(seg)>=4: out.append(seg); iz.append(sf[np.argmin(np.abs(arc-(k+0.5)*Le))])
    return out,iz
def apo_caps(msk,k,b0=(0.05,0.09),b1=(0.91,0.95)):
    vp=np.argwhere(msk)*vs; zf=(vp[:,2]-cl.z0)/(cl.z1-cl.z0); t3=(vp-ctr)@p3; t3n=t3/(np.abs(t3).max()+1e-9)
    sig=zf+k*t3n; d0=np.zeros(msk.shape,bool); d1=np.zeros(msk.shape,bool)
    for v,x in zip(np.argwhere(msk),sig):
        if b0[0]<=x<=b0[1]: d0[tuple(v)]=True
        elif b1[0]<=x<=b1[1]: d1[tuple(v)]=True
    return grad_field(solve_laplace(msk,vs,d0,d1),msk,vs)
def build(g,msk,seeds,dedup=3.5):
    streams=[];kept=[]
    for seed in seeds:
        c=np.round(seed/vs).astype(int)
        if not(np.all(c>=0)and np.all(c<shape)and msk[tuple(c)]): continue
        sf=streamf(seed,g,msk); sf=sf[(sf[:,2]>=zlo)&(sf[:,2]<=zhi)]
        if len(sf)<20 or np.max(np.linalg.norm(np.diff(sf,axis=0),axis=1))>3: continue
        mid=sf[len(sf)//2]
        if any(np.linalg.norm(mid-m)<dedup for m in kept): continue
        streams.append(sf);kept.append(mid)
    fibs=[];izs=[]
    for sf in streams:
        o,iz=tile(sf); fibs+=o; izs+=iz
    return fibs,np.array(izs).reshape(-1,3),streams

# --- red / grey / new (unipennate, natural field) ---
zv=np.linspace(cl.z0,cl.z1,80)
red=[morphing_fiber(cl,cs,rn,th,zv,None) for rn in (0.35,0.7) for th in (th_p3,th_p3+180)]
grey=[morphing_fiber(cl,cs,rn,th,zv,0.35) for rn in (0.35,0.7) for th in (th_p3,th_p3+180)]
g_uni=apo_caps(mask,0.0); zc=0.5*(cl.z0+cl.z1); pc=cl.pos(np.array([zc]))[0]
seeds=[pc+a*11*p3+b*11*p2 for a in np.linspace(-0.8,0.8,7) for b in np.linspace(-0.6,0.6,3)]
new_fibs,new_iz,_=build(g_uni,mask,seeds)

# --- bipennate: two halves, opposite tilt ---
vox=np.argwhere(mask); t3all=(vox*vs-ctr)@p3; t3grid=np.zeros(mask.shape)
for v,t in zip(vox,t3all): t3grid[tuple(v)]=t
top=keep_largest(mask&(t3grid>1.0)); bot=keep_largest(mask&(t3grid<-1.0))
# opposite, CONVERGING tilt: top tilts down toward the central tendon distally, bottom up
g_top=apo_caps(top,-0.5); g_bot=apo_caps(bot,+0.5)
zlv=np.linspace(zlo+30,zhi-30,3)
st=[cl.pos(np.array([zs]))[0]+(3+i*3.5)*p3+b*7*p2 for zs in zlv for i in range(2) for b in np.linspace(-0.6,0.6,4)]
sb=[cl.pos(np.array([zs]))[0]-(3+i*3.5)*p3+b*7*p2 for zs in zlv for i in range(2) for b in np.linspace(-0.6,0.6,4)]
fb_top,iz_top,_=build(g_top,top,st,dedup=3.0); fb_bot,iz_bot,_=build(g_bot,bot,sb,dedup=3.0)

def proj(P): d=np.atleast_2d(P)-ctr; return d@p1,d@p3
sv=np.argwhere(surface(mask))*vs

# ================= FIGURE =================
fig=plt.figure(figsize=(15,5.4))

# Panel 1: 3D overlay of ALL THREE methods in their colours (Dimitrios style)
ax1=fig.add_subplot(131,projection="3d")
ax1.scatter(sv[::5,0],sv[::5,1],sv[::5,2],s=1,c="0.8",alpha=0.10)
for f in red:  ax1.plot(f[:,0],f[:,1],f[:,2],color="#d62728",lw=1.0,alpha=.7)
for f in grey: ax1.plot(f[:,0],f[:,1],f[:,2],color="#4a4a4a",lw=1.0,alpha=.7,ls=(0,(4,2)))
for f in new_fibs: ax1.plot(f[:,0],f[:,1],f[:,2],color="#2ca02c",lw=1.6,alpha=.95)
for lab,c in [("RED morphing","#d62728"),("GREY taper","#7f7f7f"),("NEW harmonic","#2ca02c")]:
    ax1.plot([],[],[],color=c,lw=2,label=lab)
ax1.set_box_aspect((np.ptp(sv[:,0]),np.ptp(sv[:,1]),np.ptp(sv[:,2])))
ax1.set_title("All three methods on one 3D muscle",fontsize=9)
ax1.set_xlabel("x",fontsize=7);ax1.set_ylabel("y",fontsize=7);ax1.set_zlabel("z (mm)",fontsize=7)
ax1.view_init(elev=14,azim=-72); ax1.legend(fontsize=6,loc="upper left")
for axis in (ax1.xaxis,ax1.yaxis,ax1.zaxis): axis.set_major_locator(plt.MaxNLocator(4))
ax1.tick_params(labelsize=6)

# Panel 2: bipennate herringbone (fixed)
ax2=fig.add_subplot(132)
U,W=proj(sv); ax2.scatter(U,W,s=2,c="0.87",alpha=.5)
for f in fb_top: u,w=proj(f); ax2.plot(u,w,color="#1a9850",lw=1.1)
for f in fb_bot: u,w=proj(f); ax2.plot(u,w,color="#762a83",lw=1.1)
tt=np.linspace(zlo,zhi,20); cp=np.array([cl.pos(np.array([z]))[0] for z in tt]); cu,cw=proj(cp)
ax2.plot(cu,cw,"k--",lw=1,alpha=.6,label="central tendon")
for iz in (iz_top,iz_bot):
    if len(iz): iu,iw=proj(iz); ax2.scatter(iu,iw,s=9,c="k",zorder=6)
ax2.scatter([],[],s=20,c="k",label="IZ (fibre centre)")
ax2.set_aspect("equal"); ax2.set_xlim(-55,55)
ax2.set_title("Bipennate variant (herringbone)\ntwo populations → central tendon",fontsize=9)
ax2.set_xlabel("along muscle (PC1) [mm]"); ax2.set_ylabel("PC3 [mm]"); ax2.legend(fontsize=7)

# Panel 3: non-crossing
ax3=fig.add_subplot(133)
zi=int(round(zc/vs[2])); sm=np.argwhere(mask[:,:,zi])*vs[:2]
xlo,xhi=sm[:,0].min(),sm[:,0].max(); ylo,yhi=sm[:,1].min(),sm[:,1].max()
gs=[np.array([gx,gy,zc]) for gx in np.arange(xlo,xhi,4.0) for gy in np.arange(ylo,yhi,4.0)
    if 0<=int(round(gx/vs[0]))<shape[0] and 0<=int(round(gy/vs[1]))<shape[1] and mask[int(round(gx/vs[0])),int(round(gy/vs[1])),zi]]
zg=np.linspace(zlo+5,zhi-5,40); Sg=[];R=[]
for s in gs:
    sf=streamf(s,g_uni,mask); sf=sf[(sf[:,2]>=zlo)&(sf[:,2]<=zhi)]
    if len(sf)<15: continue
    r=np.stack([np.interp(zg,sf[:,2],sf[:,0]),np.interp(zg,sf[:,2],sf[:,1])],1)
    if all(np.min(np.linalg.norm(rk-r,axis=1))>1.5 for rk in R): Sg.append(sf); R.append(r)
R=np.array(R); gmin=min(pdist(R[:,z,:]).min() for z in range(len(zg))) if len(R)>1 else 0
mids=R[:,len(zg)//2,:]
ax3.scatter(sm[:,0],sm[:,1],s=6,c="0.85")
ax3.scatter(mids[:,0],mids[:,1],s=34,c=plt.cm.coolwarm(np.linspace(.1,.9,len(mids))),edgecolor="k",lw=.4,zorder=5)
ax3.set_aspect("equal")
ax3.set_title(f"Non-crossing (1 direction per point)\n{len(Sg)} streamlines, min gap {gmin:.1f} mm",fontsize=9)
ax3.set_xlabel("x (mm)"); ax3.set_ylabel("y (mm)")

fig.suptitle("Harmonic-streamline method — 3-method 3D overlay, bipennate variant, non-crossing",fontsize=12)
plt.tight_layout(); plt.savefig("showcase.png",dpi=105,bbox_inches="tight")
print(f"new fibs {len(new_fibs)}, bip top {len(fb_top)} bot {len(fb_bot)}, xsec {len(Sg)} gap {gmin:.2f}mm")
