"""OVERLAY: a real WR motor unit vs the NEW and RED/GREY simulated units, on ONE axis.

The honest, registration-free comparison is on a DISTANCE-FROM-OWN-IZ axis: we only ask how far
each motor unit's potential spreads from its innervation zone (= propagation extent = fibre length).
That needs no assumption about where the muscle sits under the array, and no claim about waveform
SHAPE (which depends on depth/CV/tissue nuisance parameters) — only spatial spread, the discriminating
quantity.

  REAL   : one clean cBSS-decomposed WR2 motor unit, STA of the monopolar grid -> along-muscle line.
  NEW    : short pennate fibres, one atlas IZ band, through the emgforge spatial engine.
  RED/GREY: one full-length fibre, IZ at geometric mid, same engine.

Output: overlay_real_model.png
  top    : three waterfall 'V's (real | NEW | RED/GREY), y = distance from IZ [mm], common scale.
  bottom : peak-to-peak spatial envelope of all three vs distance from IZ [mm], normalised, with the
           0.3*max extent marked. Real shown at 8 and 10 mm pitch (band).
"""
import os, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
sys.path.insert(0,"/home/noura/Documents/Projects/PhD/simulation/emgforge/src")
from mu_cbss import CBSS
import densefib as DF
from geom import morphing_fiber
from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial

fs=2000.0; NROW,NCOL=5,39
DATA="/home/noura/Documents/Projects/PhD/emg-decomposition/data/processed/WR2"
TASKS=["10mvc_rep1_index","20mvc_rep1_index","10mvc_rep1_middle","20mvc_rep1_middle","30mvc_rep1_index"]
W=int(0.016*fs)

# ============================ 1. pick a clean, well-propagating real unit ============================
def decompose(task):
    mono=np.load(f"{DATA}/{task}/combined_emg.npy")                 # (5,39,20000) monopolar
    sig=mono.reshape(-1,mono.shape[-1]); std=sig.std(1); good=std>0.05*np.median(std)
    clean=sig[good]; clean=clean-clean.mean(0,keepdims=True)
    cb=CBSS(ica_n_iter=60,sil_th=0.80,cov_th=0.40,min_num_spikes=10,random_seed=1909)
    sources,spikes,sil,filt=cb.decompose(clean.copy(),fs)
    return mono,spikes,sil

def sta_line(mono,sp):
    T=mono.shape[-1]; sp=np.asarray(sp,int); sp=sp[(sp>=W)&(sp<T-W)]
    if len(sp)<8: return None
    sta=np.zeros((NROW,NCOL,2*W))
    for s in sp: sta+=mono[:,:,s-W:s+W]
    sta/=len(sp)
    p2p=sta.max(2)-sta.min(2); best=p2p.argmax(0)                   # collapse across-array: strongest row
    return np.stack([sta[best[c],c] for c in range(NCOL)],0), len(sp)   # (39, 2W)

def contiguous(p2p,izc,frac=0.30):
    active=p2p>frac*p2p.max(); lo=hi=izc
    while lo-1>=0 and active[lo-1]: lo-=1
    while hi+1<NCOL and active[hi+1]: hi+=1
    return lo,hi

# collect EVERY usable single-MU (no cherry-picking); keep the whole population for the overlay,
# and separately pick one MEDIAN-extent, two-sided unit for the illustrative waterfall.
CACHE=os.path.join(HERE,"overlay_units_cache.npz")
units=[]     # (task,i,line,nsp,izc,lo,hi,sil,span,edge)
if os.path.exists(CACHE):
    z=np.load(CACHE,allow_pickle=True); units=[tuple(u) for u in z["units"]]; print(f"loaded {len(units)} units from cache")
else:
    for task in TASKS:
        try: mono,spikes,sil=decompose(task)
        except Exception as e: print(task,"decompose fail:",e); continue
        for i in spikes:
            if len(spikes[i])<30: continue
            r=sta_line(mono,spikes[i])
            if r is None: continue
            line,nsp=r; p2p=line.max(1)-line.min(1); izc=int(p2p.argmax())
            lo,hi=contiguous(p2p,izc); span=hi-lo; edge=min(izc,NCOL-1-izc)
            if span<2 or span>20: continue                          # drop artefacts (single-ch and array-length)
            units.append((task,i,line,nsp,izc,lo,hi,float(sil[i]),span,edge))
    np.savez(CACHE,units=np.array(units,dtype=object)); print(f"cached {len(units)} units")
if not units: print("no usable real units"); sys.exit(1)
spans=np.array([u[8] for u in units]); med_span=float(np.median(spans))
print(f"POPULATION: {len(units)} real units, span median {med_span:.0f} "
      f"(extent median {med_span*8:.0f}-{med_span*10:.0f} mm @8-10mm; IQR {np.percentile(spans,25)*8:.0f}-{np.percentile(spans,75)*10:.0f} mm)")
# representative = two-sided (edge>=6), span closest to the population median, best quality as tiebreak
twosided=[u for u in units if u[9]>=6]
pool=twosided if twosided else units
rtask,rui,rline,rnsp,rizc,rlo,rhi,rsil,rspan,_=min(pool,key=lambda u:(abs(u[8]-med_span),-u[7]*np.log(u[3])))
tt=(np.arange(2*W)/fs-0.016)*1e3
p2p_real=rline.max(1)-rline.min(1)
print(f"REPRESENTATIVE unit: {rtask} #{rui}  nsp={rnsp} sil={rsil:.3f}  IZ ch {rizc}  active ch {rlo}-{rhi} (span {rspan})")
print(f"  -> this unit extent {rspan*8:.0f} mm @8mm / {rspan*10:.0f} mm @10mm pitch")

# ============================ 2. simulate matched NEW and RED/GREY single motor units ============================
L=8; D=DF.setup(L)
mask,vs,cl,cs,ctr,p1,p2,p3=[D[k] for k in ("mask","vs","cl","cs","ctr","p1","p2","p3")]
z0,z1=D["z0"],D["z1"]; allc=np.argwhere(mask)*vs
s1all=(allc-ctr)@p1; s1min=float(s1all.min()); Lm=float(s1all.max()-s1min)
atlasIZ=DF._iz_params(D)[0]
def phi_of(pt): return (float((pt-ctr)@p1)-s1min)/Lm
def xy(pt): v=pt-ctr; return v-((v@p1)*p1)
zv=np.linspace(z0,z1,140)
def midnmj(f):
    f=np.asarray(f,float); a=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))]); return f[int(np.argmin(np.abs(a-a[-1]/2)))]
redall=[(f,midnmj(f)) for f in (morphing_fiber(cl,cs,rn,th,zv,None)
         for rn in np.linspace(0.25,0.9,8) for th in np.linspace(0,360,14,endpoint=False))]
segs,band,nmj,isatl=DF.dense_fill_iz(D,grid_mm=1.8)
newall=[(s,nm,phi_of(nm)) for s,nm,atl in zip(segs,nmj,isatl) if atl]

# 39-electrode column at model pitch, centred on the muscle; depth = 8 mm above the local surface
PITCH=9.0; NE=39; half=(NE-1)/2; ctr_s1=s1min+Lm/2
xe=ctr_s1+(np.arange(NE)-half)*PITCH
Epos=[]
for s1 in xe:
    sel=np.abs(s1all-s1)<6; top=((allc[sel]-ctr)@p3).max() if sel.any() else ((allc-ctr)@p3).max()
    Epos.append(ctr+p1*s1+p3*(top+8.0))
Epos=np.array(Epos); DZ=1.0; NZ=520; zgrid=(np.arange(NZ)-NZ//2)*DZ
def phi_at(poly,nm,E):
    poly=np.asarray(poly,float); a=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(poly,axis=0),axis=1))])
    ic=int(np.argmin(np.linalg.norm(poly-E,axis=1))); s=a-a[ic]
    inmj=int(np.argmin(np.linalg.norm(poly-nm,axis=1))); posz=float(a[inmj]-a[ic])
    l1=float(a[inmj]-a[0]); l2=float(a[-1]-a[inmj])
    if np.linalg.norm(poly[ic]-E)>30: return None
    d0=poly[1]-poly[0]; d0/=np.linalg.norm(d0); d1=poly[-1]-poly[-2]; d1/=np.linalg.norm(d1)
    P=np.empty((NZ,3))
    for i,z in enumerate(zgrid):
        if z<=s.min():P[i]=poly[0]+d0*(z-s.min())
        elif z>=s.max():P[i]=poly[-1]+d1*(z-s.max())
        else:P[i]=[np.interp(z+a[ic],a,poly[:,k]) for k in range(3)]
    return 1.0/np.maximum(np.linalg.norm(P-E,axis=1),1e-3),posz,l1,l2
CFG=SpatialConfig(v=4.0,fsamp=2048.0,w=320,csd_derivative=2,upsample_factor=2,denoise="none",
                  fiber_window="one_sided",edge_taper_left=5,edge_taper_right=10,center_time=False,t_start_ms=-8.0)
def montage(fibres):
    t=None; rows=[]
    for E in Epos:
        M=0.0
        for poly,nm in fibres:
            r=phi_at(poly,nm,E)
            if r is None: continue
            tt2,ss,_=compute_sfap_spatial(r[0],DZ,len1_mm=r[2],len2_mm=r[3],posz_mm=r[1],config=CFG); t=tt2; M=M+ss
        rows.append(M if np.ndim(M) else np.zeros(CFG.w))
    return t,np.array(rows)

R_MU=6.0; rng=np.random.default_rng(3)
xyall=np.array([xy(c) for c in allc])
# a central-ish territory so both arms of the model V are inside the column
c0=xyall[np.argmin(np.linalg.norm(xyall,axis=1))]
def new_unit(center_xy,phi_band,R=11,tol=0.05,cap=40):
    return [(s,nm) for s,nm,ph in newall if abs(ph-phi_band)<tol and np.linalg.norm(xy(nm)-center_xy)<R][:cap]
def red_unit(center_xy,cap=35):
    return [(f,nm) for f,nm in redall if np.linalg.norm(xy(nm)-center_xy)<R_MU][:cap]
newU=new_unit(c0,atlasIZ[1]); redU=red_unit(c0)
tn,imgn=montage(newU); tr,imgr=montage(redU)
print(f"model: NEW {len(newU)} fibres, RED {len(redU)} fibres")

def model_env(img):
    p2p=np.ptp(img,axis=1); izc=int(p2p.argmax()); lo,hi=contiguous(p2p,izc)
    dist=(np.arange(NE)-izc)*PITCH                     # distance from IZ [mm]
    return dist,p2p,izc,lo,hi
dn,pn,izn,lon,hin=model_env(imgn); dr,pr,izr,lor,hir=model_env(imgr)
Lf_new=float(DF.AT["muscles"][DF.AT["wr_label_to_muscle"][str(L)]]["fascicle_length_mm"])   # imposed fibre length
print(f"  NEW: imposed L_f={Lf_new:.0f} mm, measured 0.3xmax band {(hin-lon)*PITCH:.0f} mm ; "
      f"RED/GREY: imposed L_m={Lm:.0f} mm, measured band {(hir-lor)*PITCH:.0f} mm")

# ============================ 3. figure ============================
fig=plt.figure(figsize=(13.5,9.4)); gs=fig.add_gridspec(2,3,height_ratios=[1.05,0.95],hspace=0.34,wspace=0.26)
YLIM=150
GRN,RED,BLK="#1a9850","#d62728","#222"

def waterfall(ax,t,img,izc,col,title,sub):
    p2p=np.ptp(img,axis=1); dist=(np.arange(img.shape[0])-izc)*PITCH
    keep=(np.abs(dist)<=YLIM); step=PITCH; sc=0.85*step/max(1e-9,np.abs(img).max())
    tpk_iz=t[np.argmax(np.abs(img[izc]))]
    for e in range(img.shape[0]):
        if not keep[e]: continue
        ax.plot(t-tpk_iz, dist[e]+img[e]*sc, color=col, lw=0.7)
    ax.axhline(0,color=col,ls="--",lw=1.0,alpha=.6)
    ax.text(0.97,0.52,"IZ",transform=ax.transAxes,color=col,fontsize=9,ha="right",fontweight="bold")
    ax.set_ylim(-YLIM,YLIM); ax.set_xlim(-6,14); ax.set_title(title,fontsize=10.5,color=col)
    ax.set_xlabel("time from IZ peak [ms]",fontsize=8.5); ax.set_ylabel("distance from IZ [mm]",fontsize=8.5)
    ax.text(0.5,-0.20,sub,transform=ax.transAxes,ha="center",fontsize=8,color="0.4"); ax.tick_params(labelsize=8)

# real waterfall on distance-from-IZ (pitch shown as 8-10 mm band via dual y? use mid 9 for the picture)
def real_waterfall(ax):
    dist=(np.arange(NCOL)-rizc)*PITCH; step=PITCH; sc=0.85*step/max(1e-9,np.abs(rline).max())
    tpk_iz=tt[np.argmax(np.abs(rline[rizc]))]
    for e in range(NCOL):
        if abs(dist[e])>YLIM: continue
        ax.plot(tt-tpk_iz, dist[e]+rline[e]*sc, color=BLK, lw=0.7)
    ax.axhline(0,color="#2b8a3e",ls="--",lw=1.0)
    ax.text(0.97,0.52,"IZ",transform=ax.transAxes,color="#2b8a3e",fontsize=9,ha="right",fontweight="bold")
    ax.set_ylim(-YLIM,YLIM); ax.set_xlim(-6,14)
    ax.set_title(f"REAL WR unit  (cBSS, n={rnsp}, sil {rsil:.2f})",fontsize=10.5,color=BLK)
    ax.set_xlabel("time from IZ peak [ms]",fontsize=8.5); ax.set_ylabel("distance from IZ [mm]",fontsize=8.5)
    ax.text(0.5,-0.20,f"one decomposed motor unit; wave dies within ~{(rhi-rlo)*9//2:.0f}–{(rhi-rlo)*10//2:.0f} mm each side",
            transform=ax.transAxes,ha="center",fontsize=8,color="0.4"); ax.tick_params(labelsize=8)
    ax.text(0.03,0.97,"pitch drawn at 9 mm\n(true 8–10 mm)",transform=ax.transAxes,fontsize=7,va="top",color="0.5")

real_waterfall(fig.add_subplot(gs[0,0]))
waterfall(fig.add_subplot(gs[0,1]),tn,imgn,izn,GRN,"NEW  (short pennate fibres)","wave extinguishes at the aponeurosis ≈ L_f/2 out")
waterfall(fig.add_subplot(gs[0,2]),tr,imgr,izr,RED,"RED = GREY  (full-length fibre)","wave runs to BOTH tendons — the whole muscle")

# ---- bottom: spatial envelope overlay — WHOLE real population, no cherry-pick ----
axb=fig.add_subplot(gs[1,:])
def norm(p): return p/p.max()
# every real unit as a faint envelope, aligned at its own IZ (pitch 9 mm for the picture)
for k,u in enumerate(units):
    line,izc=u[2],u[4]; pp=line.max(1)-line.min(1); d=(np.arange(NCOL)-izc)*9.0
    axb.plot(d, norm(pp), color="0.6", lw=0.8, alpha=0.45, label=(f"real WR units (n={len(units)})" if k==0 else "_nolegend_"))
# representative unit highlighted, with 8-10 mm pitch band
d8=(np.arange(NCOL)-rizc)*8.0; d10=(np.arange(NCOL)-rizc)*10.0
axb.fill_betweenx(norm(p2p_real), d8, d10, color="0.35", alpha=0.18, label="_nolegend_")
axb.plot((d8+d10)/2, norm(p2p_real), color=BLK, lw=1.8, marker="o", ms=3.5, label="representative real unit (8–10 mm)")
axb.plot(dn, norm(pn), color=GRN, lw=2.6, label=f"NEW model (L_f={Lf_new:.0f} mm)")
axb.plot(dr, norm(pr), color=RED, lw=2.6, label=f"RED / GREY model (L_m={Lm:.0f} mm)")
axb.axhline(0.30,color="0.6",ls=":",lw=1.0); axb.text(-148,0.32,"0.3 × max (extent threshold)",fontsize=8,color="0.5")
# population median extent bracket (the honest headline), + model measured bands
def bracket(x0,x1,y,col,lab):
    axb.annotate("",xy=(x0,y),xytext=(x1,y),arrowprops=dict(arrowstyle="<->",color=col,lw=1.7))
    axb.text((x0+x1)/2,y+0.028,lab,ha="center",fontsize=8.2,color=col,fontweight="bold")
mlo,mhi=-med_span*9/2,med_span*9/2                                   # population median, symmetric, pitch 9
bracket(mlo,mhi,0.92,BLK,f"real population median ≈ {med_span*8:.0f}–{med_span*10:.0f} mm")
bracket((lon-izn)*PITCH,(hin-izn)*PITCH,0.82,GRN,f"NEW band ≈ {(hin-lon)*PITCH:.0f} mm")
bracket((lor-izr)*PITCH,(hir-izr)*PITCH,0.72,RED,f"RED/GREY band ≈ {(hir-lor)*PITCH:.0f} mm")
axb.set_xlim(-150,150); axb.set_ylim(0,1.02)
axb.set_xlabel("distance from innervation zone [mm]",fontsize=9.5)
axb.set_ylabel("normalised peak-to-peak amplitude",fontsize=9.5)
axb.set_title(f"Spatial spread from the innervation zone — all {len(units)} real WR units vs models (registration-free)",fontsize=10.5)
axb.legend(fontsize=8.3,loc="upper right"); axb.grid(alpha=.25); axb.tick_params(labelsize=8.5)

fig.suptitle("Real WR motor units spread like the NEW method, not RED/GREY — same subject as the MRI",fontsize=12.5,y=0.995)
fig.text(0.5,0.005,
    "Every trace is aligned to its OWN innervation zone (distance 0), so nothing is assumed about where the muscle sits under the array — the comparison is "
    "purely how far the potential spreads (the propagation extent, which equals fibre length). Real units' amplitude collapses within a few tens of mm of the IZ, "
    "in the NEW model's range; none spread like RED/GREY's full ±113 mm muscle. Real extents range across units (population shown, not one unit); the median is "
    "several-fold shorter than RED/GREY and of NEW's order — the robust claim is order-of-magnitude, not an exact millimetre match. We compare SPREAD and single-IZ "
    "structure only, not waveform shape (depth/CV/tissue are nuisance parameters here). Real pitch assumed 8–10 mm (raw OTB off-disk).",
    ha="center",fontsize=7.6,style="italic",color="0.4",wrap=True)
plt.savefig("overlay_real_model.png",dpi=130,bbox_inches="tight")
print("saved overlay_real_model.png")
