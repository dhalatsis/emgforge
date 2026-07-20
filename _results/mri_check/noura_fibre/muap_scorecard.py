"""Scorecard: measured sEMG characteristics of each method vs the literature.

Computes EMERGENT quantities from the simulated linear-array signal (not the inputs):
  - conduction velocity  (slope of peak-arrival-time on the propagating arm)  [verifies the sim]
  - IZ count & location   (peak-arrival minima)
  - propagation semi-length (how far the wave travels from the IZ before it dies)
Then grades each vs literature values. Honest: CV is an input (4 m/s) so its recovery only
verifies the pipeline; the discriminating rows are IZ and propagation extent.
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/noura/Documents/Projects/PhD/simulation/emgforge/src")
import densefib as DF
from geom import morphing_fiber
from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial
from scipy.signal import find_peaks

L=8; D=DF.setup(L)
mask,vs,cl,cs,ctr,p1,p2,p3=D["mask"],D["vs"],D["cl"],D["cs"],D["ctr"],D["p1"],D["p2"],D["p3"]
z0,z1=D["z0"],D["z1"]; allc=np.argwhere(mask)*vs
s1all=(allc-ctr)@p1; s1min=float(s1all.min()); Lm=float(s1all.max()-s1min)
Lf=float(DF.AT["muscles"][DF.AT["wr_label_to_muscle"][str(L)]]["fascicle_length_mm"])
atlasIZ=DF._iz_params(D)[0]; rng=np.random.default_rng(1)
zv=np.linspace(z0,z1,140)
mid=lambda f:(f:=np.asarray(f,float))[int(np.argmin(np.abs(np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))])- \
             np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))])[-1]/2)))]
def midnmj(f):
    f=np.asarray(f,float); a=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))]); return f[int(np.argmin(np.abs(a-a[-1]/2)))]
rg=lambda taper:[morphing_fiber(cl,cs,rn,th,zv,taper) for rn in np.linspace(0.3,0.85,5) for th in np.linspace(0,360,8,endpoint=False)]
red=[(f,midnmj(f)) for f in rg(None)]; grey=[(f,midnmj(f)) for f in rg(0.35)]
segs,band,nmj,isatl=DF.dense_fill_iz(D,grid_mm=2.5); new=[(s,nm) for s,nm,atl in zip(segs,nmj,isatl) if atl]
take=lambda l,n: l if len(l)<=n else [l[i] for i in rng.choice(len(l),n,replace=False)]
red,grey,new=take(red,40),take(grey,40),take(new,80)

NE=27; xe=np.linspace(0.05,0.95,NE)*Lm+s1min; Epos=[]
for s1 in xe:
    sel=np.abs(s1all-s1)<6; top=((allc[sel]-ctr)@p3).max() if sel.any() else ((allc-ctr)@p3).max()
    Epos.append(ctr+p1*s1+p3*(top+8.0))
Epos=np.array(Epos); DZ=1.0; NZ=520; zgrid=(np.arange(NZ)-NZ//2)*DZ
def phi_at(poly,nm,E):
    poly=np.asarray(poly,float); a=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(poly,axis=0),axis=1))])
    ic=int(np.argmin(np.linalg.norm(poly-E,axis=1))); s=a-a[ic]
    inmj=int(np.argmin(np.linalg.norm(poly-nm,axis=1))); posz=float(a[inmj]-a[ic])
    if np.linalg.norm(poly[ic]-E)>30: return None
    d0=poly[1]-poly[0]; d0/=np.linalg.norm(d0); d1=poly[-1]-poly[-2]; d1/=np.linalg.norm(d1)
    P=np.empty((NZ,3))
    for i,z in enumerate(zgrid):
        if z<=s.min():P[i]=poly[0]+d0*(z-s.min())
        elif z>=s.max():P[i]=poly[-1]+d1*(z-s.max())
        else:P[i]=[np.interp(z+a[ic],a,poly[:,k]) for k in range(3)]
    return 1.0/np.maximum(np.linalg.norm(P-E,axis=1),1e-3),posz,float(a[inmj]-a[0]),float(a[-1]-a[inmj])
CFG=SpatialConfig(v=4.0,fsamp=2048.0,w=320,csd_derivative=2,upsample_factor=2,denoise="none",
                  fiber_window="one_sided",edge_taper_left=5,edge_taper_right=10,center_time=False,t_start_ms=-8.0)
def monopolar(fibres):
    t=None; rows=[]
    for E in Epos:
        M=0.0
        for poly,nm in fibres:
            r=phi_at(poly,nm,E)
            if r is None: continue
            tt,ss,_=compute_sfap_spatial(r[0],DZ,len1_mm=r[2],len2_mm=r[3],posz_mm=r[1],config=CFG); t=tt; M=M+ss
        rows.append(M if np.ndim(M) else np.zeros(CFG.w))
    return t,np.array(rows)
def measure(img,t):
    amp=np.ptp(img,axis=1); active=amp>0.2*amp.max()
    tpk=np.array([t[np.argmax(np.abs(img[e]))] for e in range(NE)])
    tp=tpk.copy(); tp[~active]=tpk[active].max()+5; tps=np.convolve(tp,np.ones(3)/3,mode="same")
    pk,_=find_peaks(-tps,prominence=1.2,distance=4); izs=[e for e in pk if active[e] and amp[e]>0.35*amp.max()]
    if not izs: izs=[int(np.argmin(tp))]
    # propagation semi-length: farthest active channel from the nearest IZ
    ext=max(min(abs(xe[e]-xe[i]) for i in izs) for e in range(NE) if active[e])
    # CV: fit |dx| vs |latency| on ONE IZ's own arm only (within its propagation extent)
    e0=izs[int(np.argmax(amp[izs]))]; d=[];lat=[]
    for e in range(NE):
        dd=abs(xe[e]-xe[e0])
        if active[e] and 8<dd<ext+12:                       # restrict to this IZ's propagating arm
            d.append(dd); lat.append(abs(tpk[e]-tpk[e0]))
    cv=abs(np.polyfit(lat,d,1)[0]) if len(d)>3 else np.nan   # mm per ms = m/s
    izfrac=sorted(round((xe[e]-s1min)/Lm,2) for e in izs)
    return dict(cv=cv,niz=len(izs),izfrac=izfrac,ext=ext,active_span=(xe[active].min(),xe[active].max()))

M={}
for name,fb in [("RED",red),("GREY",grey),("NEW",new)]:
    t,img=monopolar(fb); M[name]=measure(img,t); print(name,M[name])

# ---- scorecard table ----
def cell(v): return v
rows=[
 ("Conduction velocity (m/s)","3–5 (forearm 4.3–5)",
   f"{M['RED']['cv']:.1f}",f"{M['GREY']['cv']:.1f}",f"{M['NEW']['cv']:.1f}",
   "input=4; recovery just verifies the sim","=","=","="),
 ("Fibre semi-length / propagation (mm)","≈ Lf/2 ≈ 25 (pennate; Lieber Lf=51)",
   f"{M['RED']['ext']:.0f}",f"{M['GREY']['ext']:.0f}",f"{M['NEW']['ext']:.0f}",
   "RED/GREY span whole muscle; NEW ends at aponeurosis","x","x","ok"),
 ("IZ count","FCU ≥1, 'scattered' (Saito); ~2 split-bipennate (Safwat)",
   f"{M['RED']['niz']}",f"{M['GREY']['niz']}",f"{M['NEW']['niz']}",
   "NEW's 2 ~ 'distributed'; 'scattered' isn't exactly 2 bands","~","~","~"),
 ("IZ location (fraction)","FCU 'scattered around the belly' (Saito)",
   "0.50","0.50",f"{'/'.join(str(x) for x in M['NEW']['izfrac'])}",
   "neither sharp-mid nor 0.17/0.62 clearly right; real IZ broad","~","~","~"),
 ("MUAP duration (ms, with jitter)","5–15 (needle) / 8–14 (surface)",
   "~12–22","~17–18","~7","dispersion-driven; all ~in range (Fig 7)","=","=","~"),
]
fig,ax=plt.subplots(figsize=(15,4.6)); ax.axis("off")
cols=["sEMG characteristic","Literature (FCU / forearm)","RED","GREY","NEW","note"]
cmap={"ok":"#c8e6c9","x":"#ffcdd2","~":"#fff3c4","=":"#e3f2fd"}
tbl=[]; colors=[]
for r in rows:
    tbl.append([r[0],r[1],r[2],r[3],r[4],r[5]])
    colors.append(["white","white",cmap[r[6]],cmap[r[7]],cmap[r[8]],"white"])
T=ax.table(cellText=tbl,colLabels=cols,cellColours=colors,cellLoc="left",loc="center",
           colWidths=[0.20,0.26,0.07,0.07,0.09,0.31])
T.auto_set_font_size(False); T.set_fontsize(8.2); T.scale(1,2.1)
for (r,c),cell in T.get_celld().items():
    if r==0: cell.set_text_props(weight="bold",color="white"); cell.set_facecolor("#37474f")
    cell.set_edgecolor("#cfd8dc")
ax.set_title("sEMG characteristics vs literature — three fibre-geometry methods (FCU)\n"
             "green=matches · red=mismatch · yellow=uncertain/both approximate · blue=non-discriminating (input)",fontsize=11,pad=14)
fig.text(0.5,0.01,"Bottom line: the ROBUST, discriminating result is fibre semi-length / propagation extent — NEW (short, ends at aponeurosis) matches pennate "
         "architecture; RED/GREY (full-length) do not. On IZ location the literature says FCU is 'scattered around the belly', so NEITHER sharp-mid nor our "
         "sharp 0.17/0.62 is clearly correct — both are approximations of a broad zone. CV & duration don't discriminate (CV is an input; duration is dispersion-driven).",
         ha="center",fontsize=7.4,style="italic",color="0.35",wrap=True)
plt.tight_layout(rect=[0,0.05,1,1]); plt.savefig("muap_scorecard.png",dpi=130,bbox_inches="tight"); print("saved muap_scorecard.png")
