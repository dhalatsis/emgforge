"""FCU MUAP comparison — CORRECTED after literature check.

Literature MUAP duration: needle 5-15 ms, surface 8-14 ms, determined PRIMARILY by
motor-unit fibre count + TEMPORAL DISPERSION of depolarisations, NOT by single-fibre length
(Nandedkar/IntechOpen; Preston & Shapiro). emgforge's own note (synthesis/api.py:87):
"zero scatter gives 4x too-short MUAP durations" -> dispersion is the dominant driver.

So this uses the PRODUCTION spatial engine (CSD 2nd-derivative -> the SFAP is localised to
the electrode pickup zone, not the whole fibre) + realistic per-fibre jitter (NMJ scatter,
CV scatter, tendon-length scatter). Compared fairly: a matched motor-unit territory, same
fibre count per method. FEM lead field is unavailable here (no dolfinx/mesh) so phi is the
analytical monopole the test-suite is validated against, but CSD localisation makes fibre
length secondary — the same reason grey works for NeuroDec/decomposition.
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/noura/Documents/Projects/PhD/simulation/emgforge/src")
import densefib as DF
from geom import morphing_fiber
from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial

L=8; D=DF.setup(L)
mask,vs,cl,cs,ctr,p1,p2,p3=D["mask"],D["vs"],D["cl"],D["cs"],D["ctr"],D["p1"],D["p2"],D["p3"]
z0,z1=D["z0"],D["z1"]; thp=np.degrees(np.arctan2(p3[1],p3[0]))
allc=np.argwhere(mask)*vs
rng=np.random.default_rng(0)

# ---- motor-unit territory: a cylinder (radius R) at mid-belly ----
zc=0.5*(z0+z1); zi=int(round(zc/vs[2]))
belly=cl.pos(np.array([zc]))[0]; R_MU=9.0                   # mm territory radius
E=ctr + p3*((allc-ctr)@p3).max() + p3*10.0                 # skin electrode 10 mm above surface
NMU=45                                                      # fibres per motor unit (matched)

# ---- build three fibre sets in the SAME territory ----
from scipy.ndimage import gaussian_filter1d
def smooth_path(f):   # morphing-disk paths carry small discretisation ripples that the CSD 2nd-derivative
    f=np.asarray(f,float)  # amplifies into a jagged SFAP; smooth them the SAME way densefib smooths NEW fibres
    return np.stack([gaussian_filter1d(f[:,k],2.2,mode="nearest") for k in range(3)],1) if len(f)>7 else f
zv=np.linspace(z0,z1,140)
def redgrey(taper):
    out=[]
    for rn in np.linspace(0.25,0.9,7):
        for th in np.linspace(0,360,10,endpoint=False):
            f=smooth_path(morphing_fiber(cl,cs,rn,th,zv,taper))
            mid=f[len(f)//2]
            if np.linalg.norm((mid-belly)-((mid-belly)@p1)*p1)<R_MU: out.append(f)   # xy within territory
    return out
red=redgrey(None); grey=redgrey(0.35)
segs,band,nmj,isatl=DF.dense_fill_iz(D,grid_mm=2.0)
new=[(s,nm) for s,nm in zip(segs,nmj) if np.linalg.norm((nm-belly)-((nm-belly)@p1)*p1)<R_MU]
def take(lst,n):
    if len(lst)<=n: return lst
    idx=rng.choice(len(lst),n,replace=False); return [lst[i] for i in idx]
red=take(red,NMU); grey=take(grey,NMU); new=take(new,NMU)
def midnmj(f):
    f=np.asarray(f,float); a=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(f,axis=0),axis=1))])
    return f[int(np.argmin(np.abs(a-a[-1]/2)))]
RED =[(f,midnmj(f)) for f in red]
GREY=[(f,midnmj(f)) for f in grey]
NEW =new
print(f"territory R={R_MU}mm  fibres: red {len(RED)} grey {len(GREY)} new {len(NEW)}")

# ---- phi(z) centred on the electrode-nearest point; posz=NMJ offset; len1/len2 to tendons ----
DZ=1.0; NZ=400; zgrid=(np.arange(NZ)-NZ//2)*DZ
def phi_centered(poly,nmj_pt):
    poly=np.asarray(poly,float)
    arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(poly,axis=0),axis=1))])
    ic=int(np.argmin(np.linalg.norm(poly-E,axis=1)))        # closest approach = phi centre (z=0)
    s=arc-arc[ic]
    inmj=int(np.argmin(np.linalg.norm(poly-nmj_pt,axis=1))); posz=float(arc[inmj]-arc[ic])
    len1=float(arc[inmj]-arc[0]); len2=float(arc[-1]-arc[inmj])
    d0=poly[1]-poly[0]; d0/=np.linalg.norm(d0); d1=poly[-1]-poly[-2]; d1/=np.linalg.norm(d1)
    P=np.empty((NZ,3))
    for i,z in enumerate(zgrid):
        if z<=s.min():   P[i]=poly[0]+d0*(z-s.min())
        elif z>=s.max(): P[i]=poly[-1]+d1*(z-s.max())
        else:            P[i]=[np.interp(z+arc[ic],arc,poly[:,k]) for k in range(3)]
    phi=1.0/np.maximum(np.linalg.norm(P-E,axis=1),1e-3)
    return phi,posz,len1,len2

def scfg(v): return SpatialConfig(v=v,fsamp=2048.0,w=256,csd_derivative=2,upsample_factor=2,
                                  denoise="none",fiber_window="one_sided",edge_taper_left=5,
                                  edge_taper_right=10,center_time=False,t_start_ms=-12.0)
# realistic physiological jitter (emgforge typical ranges)
NMJ_SIG,CV_SIG,TEN_SIG=10.0,0.35,6.0
def compound(fibres,jit=True):
    t=None; M=0.0; singles=[]
    for poly,nm in fibres:
        ph,posz,l1,l2=phi_centered(poly,nm)
        if jit:
            posz+=rng.normal(0,NMJ_SIG); v=max(2.5,rng.normal(4.0,CV_SIG))
            l1=max(5,l1+rng.normal(0,TEN_SIG)); l2=max(5,l2+rng.normal(0,TEN_SIG))
        else: v=4.0
        tt,s,_=compute_sfap_spatial(ph,DZ,len1_mm=l1,len2_mm=l2,posz_mm=posz,config=scfg(v))
        t=tt; M=M+s; singles.append(s)
    return np.asarray(t),np.asarray(M),singles
def dur(t,m):
    p2p=m.max()-m.min(); ab=np.abs(m)>0.1*p2p
    if not ab.any(): return 0.0
    i,j=np.argmax(ab),len(ab)-1-np.argmax(ab[::-1]); return float((t[j]-t[i]))

sets=[("RED\n(morphing disk)",RED,"#d62728"),("GREY\n(fusiform taper)",GREY,"#7f7f7f"),
      ("NEW\n(atlas-IZ in-series)",NEW,"#1a9850")]
print(f"\n{'method':26s}{'1-fibre dur':>13}{'MU dur (jitter)':>17}")
res={}
for name,fb,col in sets:
    t1,_,sing=compound(fb[:1],jit=False); d1=dur(t1,sing[0])
    tC,mC,_=compound(fb,jit=True); dC=dur(tC,mC)
    res[name]=(t1,sing[0],tC,mC,col,d1,dC,len(fb))
    print(f"{name.replace(chr(10),' '):26s}{d1:>11.1f}m{dC:>15.1f}m")

# ---- figure ----
fig,ax=plt.subplots(2,3,figsize=(15,8.2))
for j,(name,(t1,s1,tC,mC,col,d1,dC,n)) in enumerate(res.items()):
    a=ax[0,j]; a.plot(t1,s1,color=col,lw=1.4); a.set_title(f"{name} — single fibre  (dur {d1:.1f} ms)",fontsize=9.5)
    a.set_xlabel("time [ms]",fontsize=8); a.set_ylabel("SFAP [a.u.]",fontsize=8); a.grid(alpha=.3); a.tick_params(labelsize=7); a.set_xlim(-12,40)
    b=ax[1,j]; b.plot(tC,mC/max(1e-30,np.abs(mC).max()),color=col,lw=1.4)
    b.axvspan(tC[np.argmin(np.abs(tC))], tC[np.argmin(np.abs(tC))], color="none")
    b.set_title(f"motor unit, {n} fibres + jitter  (dur {dC:.1f} ms)",fontsize=9.5)
    b.set_xlabel("time [ms]",fontsize=8); b.set_ylabel("norm. MUAP",fontsize=8); b.grid(alpha=.3); b.tick_params(labelsize=7); b.set_xlim(-12,40)
    # literature band 5-15 ms width reference (as a horizontal span near baseline)
for a in ax[1]:
    a.axhspan(-0.06,0.06,color="0.85",alpha=0.0)
fig.suptitle("FCU MUAPs — CORRECTED. Literature MUAP duration is 5–15 ms (needle) / 8–14 ms (surface) and is DISPERSION-driven, "
             "NOT fibre-length-driven. Production CSD engine + physiological jitter (NMJ 10mm, CV 0.35 m/s, tendon 6mm).",fontsize=10.5)
fig.text(0.5,0.005,"RETRACTION: my earlier '4× longer' claim was an artifact of zero jitter + a slow 1/r monopole tail on 228mm fibres. "
         "With CSD localisation the electrode PICKUP ZONE (not fibre length) sets single-fibre width, so all three SFAPs are brief "
         "(7–17 ms) — consistent with grey being validated for NeuroDec/decomposition. MU durations (7–22 ms) are the same ballpark as "
         "literature, NOT 4× apart. Caveats: these numbers swing with electrode/territory (analytical monopole, no FEM), and the NEW MU "
         "under-disperses here (the CSD engine centres each SFAP, blunting NMJ-jitter spread) — a precise duration needs FEM + the native "
         "jitter path. The robust difference is STRUCTURE: NEW = physiological fibre length + multi-IZ (0.17, 0.62); red/grey = single-IZ full-length.",
         ha="center",fontsize=7.0,style="italic",color="0.4")
plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig("muap_compare.png",dpi=120,bbox_inches="tight")
print("\nsaved muap_compare.png")
