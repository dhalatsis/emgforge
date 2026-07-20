"""CORRECTED motor-unit model — fixes the 'zigzag' error.

The earlier HD-sEMG/example figures lumped fibres from BOTH atlas innervation bands (0.17 & 0.62)
into ONE simulated compound, giving a single MUAP with TWO innervation zones (a 'zigzag').
That conflates the whole muscle with one motor unit. A real motor unit's fibres share ONE
endplate band -> ONE innervation zone (confirmed on WR's real HD-sEMG, real_iz_check.py; and by
NeuroDec, which places one NMJ per unit).

Fix, demonstrated here:
  ROW 1  SINGLE MOTOR UNIT  — fibres from ONE band -> ONE IZ, for every method.
  ROW 2  WHOLE MUSCLE       — many single-IZ units summed. RED puts every unit's IZ at the
                              geometric mid (one tight band); NEW distributes units across the
                              muscle-specific atlas bands (scattered IZ). Each unit still has ONE IZ.

Only the fibre GEOMETRY differs (length + where the single IZ can sit); the motor-unit rule is now
identical and correct for all methods. Same electrode column / spatial engine as muap_hdsemg.py.
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/noura/Documents/Projects/PhD/simulation/emgforge/src")
import densefib as DF
from geom import morphing_fiber
from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial

L=8; D=DF.setup(L)
mask,vs,cl,cs,ctr,p1,p2,p3=D["mask"],D["vs"],D["cl"],D["cs"],D["ctr"],D["p1"],D["p2"],D["p3"]
z0,z1=D["z0"],D["z1"]; allc=np.argwhere(mask)*vs
s1all=(allc-ctr)@p1; s1min=float(s1all.min()); Lm=float(s1all.max()-s1min)
atlasIZ=DF._iz_params(D)[0]; rng=np.random.default_rng(3)
def phi_of(pt): return (float((pt-ctr)@p1)-s1min)/Lm       # muscle-length fraction of a point
def xy(pt):     v=pt-ctr; return v-((v@p1)*p1)             # in-plane (cross-section) position
zv=np.linspace(z0,z1,140)
def midnmj(f):
    f=np.asarray(f,float); a=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))]); return f[int(np.argmin(np.abs(a-a[-1]/2)))]

# ---- fibre pools ----
redall=[(f,midnmj(f)) for f in (morphing_fiber(cl,cs,rn,th,zv,None)
         for rn in np.linspace(0.25,0.9,8) for th in np.linspace(0,360,14,endpoint=False))]
segs,band,nmj,isatl=DF.dense_fill_iz(D,grid_mm=1.8)
newall=[(s,nm,phi_of(nm)) for s,nm,atl in zip(segs,nmj,isatl) if atl]   # short fibres, on atlas bands

# ---- motor-unit samplers: a unit = fibres in one xy territory, sharing ONE innervation band ----
R_MU=6.0
def red_unit(center_xy):                       # full-length, single IZ forced to geometric mid
    u=[(f,nm) for f,nm in redall if np.linalg.norm(xy(nm)-center_xy)<R_MU]
    return u[:35]
def new_unit(center_xy,phi_band,R=R_MU,tol=0.05,cap=35):   # short fibres whose single NMJ sits in ONE band
    u=[(s,nm) for s,nm,ph in newall if abs(ph-phi_band)<tol and np.linalg.norm(xy(nm)-center_xy)<R]
    return u[:cap]
# territory centres sampled inside the muscle cross-section
xyall=np.array([xy(c) for c in allc]); idx=rng.choice(len(xyall),30,replace=False); centres=xyall[idx]

# ---- electrode column + spatial engine (same as muap_hdsemg) ----
NE=25; xe=np.linspace(0.06,0.94,NE)*Lm+s1min; Epos=[]
for s1 in xe:
    sel=np.abs(s1all-s1)<6; top=((allc[sel]-ctr)@p3).max() if sel.any() else ((allc-ctr)@p3).max()
    Epos.append(ctr+p1*s1+p3*(top+8.0))
Epos=np.array(Epos)
DZ=1.0; NZ=520; zgrid=(np.arange(NZ)-NZ//2)*DZ
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
            tt,s,_=compute_sfap_spatial(r[0],DZ,len1_mm=r[2],len2_mm=r[3],posz_mm=r[1],config=CFG); t=tt; M=M+s
        rows.append(M if np.ndim(M) else np.zeros(CFG.w))
    return t,np.array(rows)
def peakarrival(img,t):
    amp=np.ptp(img,axis=1); active=amp>0.2*amp.max()
    tpk=np.array([t[np.argmax(np.abs(img[e]))] for e in range(NE)]); return tpk,active,amp

# ================= build =================
c0=centres[0]; mid=0.5*Lm+s1min
red1  = red_unit(c0)                                        # single unit, IZ forced to geometric mid (RED)
new1  = new_unit(c0,atlasIZ[1],R=11,cap=40)                # CORRECT single unit: fibres from ONE band (0.62)
lumped= new_unit(c0,atlasIZ[0],R=11,cap=40)+new_unit(c0,atlasIZ[1],R=11,cap=40)  # THE OLD BUG: one 'unit', BOTH bands
newM=[]; new_iz=[]; red_iz=[]
for k,c in enumerate(centres):
    bnd=atlasIZ[k%len(atlasIZ)]; u=new_unit(c,bnd)
    if u: newM+=u; new_iz.append(bnd*Lm+s1min)
    if red_unit(c): red_iz.append(mid)
tr,imgr=montage(red1); tpk_r,ac_r,_=peakarrival(imgr,tr)
tn,imgn=montage(new1); tpk_n,ac_n,_=peakarrival(imgn,tn)
tl,imgl=montage(lumped); tpk_l,ac_l,_=peakarrival(imgl,tl)
print(f"red1 {len(red1)} new1 {len(new1)} lumped {len(lumped)} | muscle new {len(new_iz)} red {len(red_iz)}")

def vband(tpk,active):     # contiguous propagating band around the IZ (earliest-arrival active channel)
    tp=tpk.copy(); tp[~active]=1e9; iz=int(np.argmin(tp)); lo=hi=iz
    while lo-1>=0 and active[lo-1]: lo-=1
    while hi+1<NE and active[hi+1]: hi+=1
    return iz,np.arange(lo,hi+1)

# ================= figure: the MUAP as a propagating WAVE (waterfall) — the 'V' =================
TMAX=30
def wave(ax,t,img,tpk,active,col,title,note):
    keep=t<=TMAX; tt=t[keep]; sub=img[:,keep]
    step=(xe[-1]-xe[0])/(NE-1); sc=0.9*step/max(1e-9,np.abs(sub).max())
    for e in range(NE): ax.plot(tt,xe[e]+sub[e]*sc,color=col,lw=0.8)
    iz,_=vband(tpk,active); ax.axhline(xe[iz],color="#2b8a3e" if col=="#1a9850" else "#111",ls="--",lw=1.0,alpha=.5)
    ax.text(TMAX-0.5,xe[iz]+3,"IZ",fontsize=9,ha="right",fontweight="bold",color="#2b8a3e" if col=="#1a9850" else "#111")
    # arrows showing the wave travels AWAY from the IZ in both directions (the two arms of the V)
    ax.annotate("",xy=(tt[0]+min(26,TMAX-2),xe[iz]+step*7),xytext=(tt[0]+2,xe[iz]+step*0.5),arrowprops=dict(arrowstyle="->",color="0.35",lw=1.1))
    ax.annotate("",xy=(tt[0]+min(26,TMAX-2),xe[iz]-step*7),xytext=(tt[0]+2,xe[iz]-step*0.5),arrowprops=dict(arrowstyle="->",color="0.35",lw=1.1))
    ax.set_title(title,fontsize=10.5); ax.text(0.5,-0.14,note,transform=ax.transAxes,ha="center",fontsize=8,color="0.4")
    ax.set_xlabel("time [ms]",fontsize=9); ax.set_ylabel("electrode along muscle [mm]",fontsize=9)
    ax.set_xlim(tt[0],TMAX); ax.set_ylim(s1min,s1min+Lm); ax.tick_params(labelsize=8)

fig,ax=plt.subplots(1,2,figsize=(13.5,7.4))
wave(ax[0],tr,imgr,tpk_r,ac_r,"#d62728","RED  =  GREY  (identical here) — one motor unit",
     "one IZ at the geometric mid → the wave spreads to BOTH tendon tips: a WIDE 'V' (full-length fibre)")
wave(ax[1],tn,imgn,tpk_n,ac_n,"#1a9850","NEW — one motor unit (fixed, no zigzag)",
     "one IZ at the anatomical band → the wave dies at the aponeurosis: a NARROW 'V' (short pennate fibre)")
fig.suptitle("The MUAP as a propagating wave along the electrode column — the classic 'V'. Now CORRECT: one motor unit = ONE innervation zone (one V) in every method.",fontsize=10.8)
fig.text(0.5,0.01,"Each faint line is one electrode's MUAP; stacked by position along the muscle. The potential is generated at the innervation zone (dashed) and "
         "PROPAGATES away in both directions (arrows) — the travelling peak traces the two arms of a 'V'. RED and GREY are identical: one full-length fibre, "
         "so the V is WIDE (the wave runs the whole muscle). NEW uses short pennate fibres, so the V is NARROW (the wave extinguishes at the aponeurosis ~Lₑ/2 out). "
         "Both show ONE V = ONE IZ — the earlier two-armed 'zigzag' (one 'unit' wrongly built from two bands) is gone. The whole muscle is many such single-V units, "
         "each at its own IZ (population picture in real_iz_check.py / §09).",
         ha="center",fontsize=7.4,style="italic",color="0.4",wrap=True)
plt.tight_layout(rect=[0,0.08,1,0.94]); plt.savefig("muap_fixed.png",dpi=130,bbox_inches="tight"); print("saved muap_fixed.png")
