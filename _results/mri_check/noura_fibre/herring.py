"""Herringbone shown from THREE perspectives, each as a central SECTION so every fibre
begins on the muscle outline (no projecting a 3-D surface into a 2-D silhouette):
  (1) side  section — length x thickness (the textbook chevron) + zoom
  (2) face  section — length x width
  (3) cross section — width x thickness, transverse at mid-belly (radial convergence)
Analytic field: d = cos(th)*tendon_tangent + sin(th)*(unit vector toward the central tendon).
Each fibre: BEGIN on outline (orange ring), NMJ at mid-length (black), END on tendon (red square)."""
import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from geom import load_mask,Centerline,pca_frame
from harmonic import keep_largest,trace,surface
def _tangent(cl,z):
    z=float(np.clip(z,cl.z0,cl.z1)); dx=float(cl.sx(z,1)); dy=float(cl.sy(z,1))
    t=np.array([dx,dy,1.0]); return t/np.linalg.norm(t)
AT=json.load(open("/home/noura/Documents/Projects/PhD/simulation/emgforge/src/emgforge/mri/data/forearm_muscle_atlas.json"))
L=8; key=AT["wr_label_to_muscle"][str(L)]; th=np.radians(AT["muscles"][key]["pennation_deg"])
mask,vs,_=load_mask(L); mask=keep_largest(mask)
cl=Centerline(mask,vs); ctr,p1,p2,p3=pca_frame(mask,vs)
z0,z1=cl.z0,cl.z1; zlo,zhi=z0+0.06*(z1-z0),z1-0.06*(z1-z0)
_thick=np.ptp((np.argwhere(mask)*vs-ctr)@p3); REACH=max(1.5,0.10*_thick)
allv=np.argwhere(mask); coord=allv*vs; surfset=set(map(tuple,np.argwhere(surface(mask))))
s1all=(coord-ctr)@p1

# --- atlas innervation-zone bands: pin the herringbone NMJs into longitudinal compartments
#     (same atlas IZ the compare/dense figures use, now applied to the pennate case) ---
Lf=AT["muscles"][key]["fascicle_length_mm"]
atlasIZ=sorted(AT["muscles"][key].get("IZ_fraction",[0.3]))
s1min=float(s1all.min()); Lm=float(s1all.max()-s1min)
_step=float(np.clip(Lf/Lm,0.12,0.6))
def cover_bands(lo,hi):
    """atlas IZ fractions (measured) + geometric fill so no gap exceeds Lf/Lm (inferred)."""
    xs=[f for f in atlasIZ if lo+0.01<f<hi-0.01]
    res=[(f,True) for f in xs]; pts=[lo]+xs+[hi]
    for a,b in zip(pts[:-1],pts[1:]):
        n=int(np.ceil((b-a)/_step))-1
        for i in range(1,n+1): res.append((a+i*(b-a)/(n+1),False))
    return sorted(res)
_philo,_phihi=0.06,0.94
BANDS=cover_bands(_philo,_phihi)               # [(phi,is_atlas)] innervation compartments
_HW=0.04                                        # half-width (length fraction) of each compartment
def band_of(phi):
    """nearest cover band within _HW -> (phi_band,is_atlas); None in a tendinous gap."""
    best=None
    for phib,atl in BANDS:
        if abs(phi-phib)<=_HW and (best is None or abs(phi-phib)<abs(phi-best[0])):
            best=(phib,atl)
    return best
def phi_x(phib): return phib*Lm+s1min           # length fraction -> plot x (mm along p1)
print("atlas IZ",atlasIZ,"cover bands",[(round(b,2),a) for b,a in BANDS])

# analytic direction field
D=np.zeros(mask.shape+(3,))
for v in allv:
    P=v*vs; C=cl.pos(np.array([P[2]]))[0]; t=_tangent(cl,P[2])
    r=C-P; r=r-(r@t)*t; n=np.linalg.norm(r)
    d=(np.cos(th)*t+np.sin(th)*(r/n)) if n>1e-6 else t
    D[tuple(v)]=d/np.linalg.norm(d)
def smooth(f):
    k=np.concatenate([[True],np.linalg.norm(np.diff(f,axis=0),axis=1)>1e-6]); f=f[k]
    return np.stack([gaussian_filter1d(f[:,d],1.2,mode="nearest") for d in range(3)],1) if len(f)>7 else f
def fib(seed):
    f=trace(seed,D,mask,vs,sign=1,step=1.0,maxlen=120)
    if len(f)<3: return None
    f=smooth(f); f=f[(f[:,2]>=zlo)&(f[:,2]<=zhi)]
    if len(f)<3: return None
    dcl=np.array([np.linalg.norm(P-cl.pos(np.array([P[2]]))[0]) for P in f])   # cut the parallel tail at FIRST tendon reach
    below=np.where(dcl<=REACH)[0]; cut=int(below[0]) if len(below) else int(np.argmin(dcl))
    f=f[:cut+1]
    if len(f)<3: return None
    # EXTEND along the fibre's own pennation heading until it meets the tendon — NOT a perpendicular drop.
    k=min(5,len(f)-1); v=f[-1]-f[-1-k]; nv=np.linalg.norm(v)
    if nv>1e-6:
        v=v/nv; P=f[-1].astype(float); ext=[]; reached=False
        for _ in range(60):
            P=P+v
            zc=float(np.clip(P[2],cl.z0,cl.z1)); d=np.linalg.norm(P-cl.pos(np.array([zc]))[0])
            ext.append(P.copy())
            if d<=0.7: reached=True; break
        if reached:
            ext[-1]=cl.pos(np.array([float(np.clip(ext[-1][2],cl.z0,cl.z1))]))[0]   # land exactly on the aponeurosis
            f=np.vstack([f,np.array(ext)])
    return np.vstack([seed,f]) if len(f)>=3 else None
def thin(lst,keep=40):
    return lst if len(lst)<=keep else [lst[i] for i in np.linspace(0,len(lst)-1,keep).astype(int)]
rng=np.random.default_rng(0)

def longitudinal_section(slab_dir,side_dir,SLAB=7.0,band_filter=True,cap=40):
    """Fibres in a thin slab (|.,slab_dir|<SLAB); split by sign of side_dir. Returns top,bot,slab_all
    as lists of (fibre, is_atlas).
      band_filter=True  -> keep only fibres whose origin falls in an innervation compartment (banded view)
      band_filter=False -> keep EVERY surface fibre in the slab (density check, no gaps)
      cap               -> max fibres drawn per side (large => all)."""
    s=(coord-ctr)@slab_dir; inslab=np.abs(s)<SLAB
    seeds=np.array([v for v,ok in zip(allv,inslab) if ok and tuple(v) in surfset])
    top=[];bot=[]
    for v in seeds[rng.permutation(len(seeds))]:
        P=v*vs
        if not(zlo<=P[2]<=zhi): continue
        phi=((P-ctr)@p1-s1min)/Lm; bd=band_of(phi)     # innervation compartment (None in a tendinous gap)
        if band_filter and bd is None: continue
        C=cl.pos(np.array([P[2]]))[0]; side=np.sign((P-C)@side_dir)
        f=fib(P)
        if f is None or len(f)<5: continue
        seg=np.linalg.norm(np.diff(f,axis=0),axis=1)
        if seg.sum()<9 or seg.max()>3: continue
        if abs(((f[-1]-ctr)@slab_dir))>SLAB+3: continue
        (top if side>0 else bot).append((f,bd[1] if bd is not None else False))
    return thin(top,cap),thin(bot,cap),coord[inslab]

def transverse_dots(s1c,SLAB=3.0,binmm=3.0):
    """A transverse cut is PERPENDICULAR to the fibres, so each fibre pierces it as a DOT.
    Cut is taken THROUGH an innervation band (s1c). Return dot positions (u,w in the p2-p3 plane),
    the in-plane component of the fibre direction at each dot (tiny arrow => local pennation,
    converging on the tendon), and the outline."""
    inx=np.abs(s1all-s1c)<SLAB
    xvox=allv[inx]; xc=xvox*vs
    U=(xc-ctr)@p2; W=(xc-ctr)@p3
    seen=set(); du=[];dw=[];au=[];aw=[]
    order=rng.permutation(len(xvox))
    for i in order:
        b=(round(U[i]/binmm),round(W[i]/binmm))
        if b in seen: continue
        seen.add(b)
        dvec=D[tuple(xvox[i])]; ip=np.array([dvec@p2,dvec@p3]); n=np.linalg.norm(ip)
        du.append(U[i]); dw.append(W[i])
        au.append(ip[0]/n if n>1e-6 else 0.0); aw.append(ip[1]/n if n>1e-6 else 0.0)
    return (np.array(du),np.array(dw),np.array(au),np.array(aw),
            np.column_stack([U,W]))

def mkproj(u,w):
    def proj(P): d=np.atleast_2d(P)-ctr; return d@u,d@w
    return proj
def mark(ax,f,proj,is_atlas):
    u,w=proj(f)
    ax.scatter([u[0]],[w[0]],s=20,facecolors="none",edgecolors="#e08214",lw=1.0,zorder=6)
    ax.scatter([u[-1]],[w[-1]],s=15,marker="s",facecolors="none",edgecolors="#d62728",lw=1.0,zorder=6)
    arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))])
    im=np.argmin(np.abs(arc-arc[-1]/2))                                   # NMJ = mid-thickness endplate
    if is_atlas: ax.scatter([u[im]],[w[im]],s=13,c="k",zorder=7)          # pinned to a measured atlas IZ
    else: ax.scatter([u[im]],[w[im]],s=11,facecolors="none",edgecolors="0.45",lw=0.9,zorder=7)  # inferred fill band
def draw_long(ax,top,bot,slab_all,proj,markers=True,tendon=True,guides=True):
    U,W=proj(slab_all); ax.scatter(U,W,s=6,c="0.85",alpha=.5,zorder=0)
    if guides:                                                            # innervation-band guide lines
        for phib,atl in BANDS:
            x=phi_x(phib)
            if atl: ax.axvline(x,color="k",lw=0.8,ls="-",alpha=.30,zorder=1)
            else:   ax.axvline(x,color="0.5",lw=0.7,ls=":",alpha=.5,zorder=1)
    if tendon:
        tt=np.linspace(zlo,zhi,40); cp=np.array([cl.pos(np.array([z]))[0] for z in tt]); cu,cw=proj(cp)
        ax.plot(cu,cw,color="#d62728",lw=2.0,alpha=.9,zorder=4)
    for f,atl in top: u,w=proj(f); ax.plot(u,w,color="#1a9850",lw=0.8,alpha=.8,zorder=2); (mark(ax,f,proj,atl) if markers else None)
    for f,atl in bot: u,w=proj(f); ax.plot(u,w,color="#762a83",lw=0.8,alpha=.8,zorder=2); (mark(ax,f,proj,atl) if markers else None)
def draw_dense(ax,top,bot,slab_all,proj):
    """Every fibre, faint, no markers/compartments — purely to check the field fills the belly."""
    U,W=proj(slab_all); ax.scatter(U,W,s=6,c="0.9",alpha=.45,zorder=0)
    tt=np.linspace(zlo,zhi,40); cp=np.array([cl.pos(np.array([z]))[0] for z in tt]); cu,cw=proj(cp)
    ax.plot(cu,cw,color="#d62728",lw=1.6,alpha=.9,zorder=4)
    for f,_ in top: u,w=proj(f); ax.plot(u,w,color="#1a9850",lw=0.5,alpha=.6,zorder=2)
    for f,_ in bot: u,w=proj(f); ax.plot(u,w,color="#762a83",lw=0.5,alpha=.6,zorder=2)

# ---- build the sections ----
side_top,side_bot,side_bg=longitudinal_section(p2,p3)          # length x thickness (banded, capped)
face_top,face_bot,face_bg=longitudinal_section(p3,p2)          # length x width  (banded, capped)
dens_top,dens_bot,dens_bg=longitudinal_section(p2,p3,band_filter=False,cap=100000)  # EVERY fibre (density check)
_distIZ=max(atlasIZ)                                           # cut the transverse view through the distal IZ
du,dw,au,aw,xbg=transverse_dots(phi_x(_distIZ))               # width x thickness (transverse dots)
print(f"side {len(side_top)+len(side_bot)}  face {len(face_top)+len(face_bot)}  "
      f"dense {len(dens_top)+len(dens_bot)}  cross {len(du)} dots")

pj_side=mkproj(p1,p3); pj_face=mkproj(p1,p2)
fig=plt.figure(figsize=(12,15.2))
gs=fig.add_gridspec(5,2,height_ratios=[1.0,1.0,1.25,1.0,1.0],hspace=0.55,wspace=0.22)
axS=fig.add_subplot(gs[0,:]); axF=fig.add_subplot(gs[1,:])
axZ=fig.add_subplot(gs[2,0]); axX=fig.add_subplot(gs[2,1])
axD=fig.add_subplot(gs[3,:]); axDZ=fig.add_subplot(gs[4,:])

# (1) side section
draw_long(axS,side_top,side_bot,side_bg,pj_side)
axS.set_aspect("equal"); axS.set_title("(1) Side section — length × thickness  (the herringbone chevron)",fontsize=9.5,loc="left")
axS.set_xlabel("along muscle (length) [mm]",fontsize=8); axS.set_ylabel("thickness [mm]",fontsize=8)
axS.scatter([],[],s=6,c="0.85",label="outline"); axS.plot([],[],color="#d62728",lw=2,label="central tendon")
axS.scatter([],[],s=20,facecolors="none",edgecolors="#e08214",lw=1,label="fibre begin")
axS.scatter([],[],s=15,marker="s",facecolors="none",edgecolors="#d62728",lw=1,label="fibre end")
axS.scatter([],[],s=13,c="k",label="NMJ — atlas IZ")
axS.scatter([],[],s=11,facecolors="none",edgecolors="0.45",lw=0.9,label="NMJ — inferred band")
axS.legend(fontsize=6.2,loc="lower left",ncol=6,frameon=True)
# mark the zoom window on panel 1
from matplotlib.patches import Rectangle
zx0,zx1,zy0,zy1=-8,28,-11,11
axS.add_patch(Rectangle((zx0,zy0),zx1-zx0,zy1-zy0,fill=False,edgecolor="0.4",lw=0.8,ls="--",zorder=8))

# (2) face section
draw_long(axF,face_top,face_bot,face_bg,pj_face)
axF.set_aspect("equal"); axF.set_title("(2) Face section — length × width  (same muscle, broad face)",fontsize=9.5,loc="left")
axF.set_xlabel("along muscle (length) [mm]",fontsize=8); axF.set_ylabel("width [mm]",fontsize=8)
_ytop=axF.get_ylim()[1]
for f in atlasIZ:
    axF.text(phi_x(f),_ytop*0.98,f"IZ φ={f:g}",fontsize=7,color="k",ha="center",va="top")
for f,atl in BANDS:
    if not atl: axF.text(phi_x(f),_ytop*0.98,"fill",fontsize=6,color="0.5",ha="center",va="top")

# (3a) dedicated zoom of the chevron (its own panel — no overlap)
draw_long(axZ,side_top,side_bot,side_bg,pj_side,markers=True)
axZ.set_xlim(zx0,zx1); axZ.set_ylim(zy0,zy1); axZ.set_aspect("equal")
axZ.set_title("(3) Zoom of the 12° chevron  (side view, dashed box)",fontsize=9,loc="left")
axZ.set_xlabel("length [mm]",fontsize=8); axZ.set_ylabel("thickness [mm]",fontsize=8); axZ.tick_params(labelsize=7)

# (4) transverse cross-section: DOTS = fibres cut in cross-section; tiny ticks = in-plane direction
axX.scatter(xbg[:,0],xbg[:,1],s=8,c="0.86",alpha=.5,zorder=0)
axX.quiver(du,dw,au,aw,angles="xy",scale_units="xy",scale=0.32,width=0.005,color="0.55",zorder=2)
axX.scatter(du,dw,s=16,c="#1a9850",edgecolors="k",lw=.3,zorder=3,label="fibre (cut in cross-section)")
axX.scatter([0],[0],s=90,marker="*",c="#d62728",zorder=5,label="central tendon")
axX.set_aspect("equal"); axX.set_title(f"(4) Cross-section — width × thickness  (transverse, through the distal IZ φ={_distIZ:g})",fontsize=9,loc="left")
axX.set_xlabel("width [mm]",fontsize=8); axX.set_ylabel("thickness [mm]",fontsize=8); axX.tick_params(labelsize=7)
axX.legend(fontsize=6.5,loc="lower center",ncol=2,frameon=True)

# (5) ALL fibres — density check (no band filter, no cap): the belly fills, no gaps
draw_dense(axD,dens_top,dens_bot,dens_bg,pj_side)
axD.set_aspect("equal"); axD.set_title(f"(5) EVERY fibre rendered — density check (side view, {len(dens_top)+len(dens_bot)} fibres, no thinning, no compartments): the belly fills with no gaps",fontsize=9,loc="left")
axD.set_xlabel("along muscle (length) [mm]",fontsize=8); axD.set_ylabel("thickness [mm]",fontsize=8)
axD.add_patch(Rectangle((zx0,zy0),zx1-zx0,zy1-zy0,fill=False,edgecolor="0.4",lw=0.8,ls="--",zorder=8))

# (6) zoom of the dense fill — confirm no gaps up close
draw_dense(axDZ,dens_top,dens_bot,dens_bg,pj_side)
axDZ.set_xlim(zx0,zx1); axDZ.set_ylim(zy0,zy1); axDZ.set_aspect("equal")
axDZ.set_title("(6) Zoom of the full-density fill (dashed box) — fibres tile the belly, no gaps",fontsize=9,loc="left")
axDZ.set_xlabel("length [mm]",fontsize=8); axDZ.set_ylabel("thickness [mm]",fontsize=8); axDZ.tick_params(labelsize=7)

fig.suptitle(f"FCU herringbone — fibres lean {np.degrees(th):.0f}° to the central tendon; "
             f"NMJs pinned to atlas IZ ({', '.join(f'{f:g}' for f in atlasIZ)})",fontsize=12,y=0.992)
fig.text(0.5,0.004,
    "NEW vs earlier: NMJs no longer form one continuous mid-belly streak — fibres are grouped into in-series compartments\n"
    "PINNED to atlas IZ fractions (black), with inferred fill bands (grey) and thin tendinous gaps between.\n"
    "Panels 1–3 = central SECTIONS (thinned + band-filtered for marker legibility); panel 4 = a TRANSVERSE cut\n"
    "perpendicular to the fibres (each fibre a DOT, ticks converge on the tendon *).\n"
    "Panels 5–6 render EVERY fibre (no thinning) to confirm the field fills the belly with no gaps;\n"
    "the central tendon is modelled as a LINE, while a real bipennate tendon is a sheet.",
    ha="center",va="top",fontsize=8.5,style="italic",color="0.4")
plt.savefig("herring.png",dpi=115,bbox_inches="tight"); print("saved")
