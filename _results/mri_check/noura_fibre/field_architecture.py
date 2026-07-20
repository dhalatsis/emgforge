"""Where does the fibre field DEPART from the RED/GREY centre-line — by architecture class?

RED/GREY: fibres parallel to the centre-line => pennation 0deg, no transverse lean, for EVERY muscle.
NEW: fibres lean toward the aponeurosis/central tendon at the muscle's pennation angle (analytic
     field d = cos(th)*tangent + sin(th)*toward-tendon, the same one the herringbone flagship uses).

This isolates the field's geometric contribution:
  (A) angle(NEW field, centre-line) per muscle  -> equals the imposed pennation (2.4deg fusiform .. 12deg bipennate)
  (B) transverse cross-section quiver: the in-plane convergence on the tendon (chevron), strength ~ sin(pennation)
      -> RED/GREY have ZERO in-plane component in all of these (categorical difference, not a matter of degree).
Honest: pennation is IMPOSED from the atlas (can't be read off a belly-only T2); the central tendon is modelled
as a LINE (a real bipennate tendon is a sheet), so 'bipennate' here = larger pennation + atlas IZ bands + radial
convergence, an approximation of the true two-sided herringbone.
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"."); sys.path.insert(0,"/home/noura/Documents/Projects/PhD/simulation/emgforge/src")
import json
from geom import load_mask, Centerline, pca_frame
from harmonic import keep_largest
AT=json.load(open("/home/noura/Documents/Projects/PhD/simulation/emgforge/src/emgforge/mri/data/forearm_muscle_atlas.json"))

def _tangent(cl,z):
    z=float(np.clip(z,cl.z0,cl.z1)); return (lambda t:t/np.linalg.norm(t))(np.array([float(cl.sx(z,1)),float(cl.sy(z,1)),1.0]))

def build(L):
    key=AT["wr_label_to_muscle"][str(L)]; th=np.radians(AT["muscles"][key]["pennation_deg"])
    mask,vs,_=load_mask(L); mask=keep_largest(mask); cl=Centerline(mask,vs); ctr,p1,p2,p3=pca_frame(mask,vs)
    allv=np.argwhere(mask); coord=allv*vs
    D=np.zeros(mask.shape+(3,)); ang=[]
    for v in allv:
        P=v*vs; C=cl.pos(np.array([P[2]]))[0]; t=_tangent(cl,P[2])
        r=C-P; r=r-(r@t)*t; n=np.linalg.norm(r)
        d=(np.cos(th)*t+np.sin(th)*(r/n)) if n>1e-6 else t; d/=np.linalg.norm(d)
        D[tuple(v)]=d
        ang.append(np.degrees(np.arccos(np.clip(abs(d@t),0,1))))       # angle vs centre-line tangent (=RED direction)
    return dict(key=key,penn=AT["muscles"][key]["pennation_deg"],mask=mask,vs=vs,cl=cl,ctr=ctr,
                p1=p1,p2=p2,p3=p3,D=D,ang=np.array(ang),coord=coord,allv=allv)

def transverse(M,frac=0.5,binmm=3.0):
    mask,vs,ctr,p1,p2,p3,D=M["mask"],M["vs"],M["ctr"],M["p1"],M["p2"],M["p3"],M["D"]
    s1=(M["coord"]-ctr)@p1; s1c=s1.min()+frac*(s1.max()-s1.min())
    inx=np.abs(s1-s1c)<3.0; xv=M["allv"][inx]; xc=xv*vs
    U=(xc-ctr)@p2; W=(xc-ctr)@p3; seen={}; du=[];dw=[];au=[];aw=[]
    for i in range(len(xv)):
        b=(round(U[i]/binmm),round(W[i]/binmm))
        if b in seen: continue
        seen[b]=1; dv=D[tuple(xv[i])]; ip=np.array([dv@p2,dv@p3])   # in-plane magnitude ~ sin(pennation)
        du.append(U[i]);dw.append(W[i]);au.append(ip[0]);aw.append(ip[1])   # UNnormalised: arrow length shows convergence strength
    return np.array(du),np.array(dw),np.array(au),np.array(aw),np.column_stack([U,W])

LABELS=[11,13,18,21,8]     # sorted by pennation: BR 2.4, FDS 6.8, APL 7.9, pron.teres 9.6, FCU 12 (bipennate)
Ms=[build(L) for L in LABELS]
for M in Ms: print(f"{M['key'][:22]:<22} penn={M['penn']:>4}  angle(field,centreline) med={np.median(M['ang']):.1f} p90={np.percentile(M['ang'],90):.1f}")

fig=plt.figure(figsize=(13,7.6)); gs=fig.add_gridspec(2,3,height_ratios=[1,1.05],hspace=0.42,wspace=0.28)
# (A) imposed pennation vs architecture
axA=fig.add_subplot(gs[0,:])
penns=[M["penn"] for M in Ms]; meds=[np.median(M["ang"]) for M in Ms]; names=[M["key"].split("_")[0][:10] for M in Ms]
x=np.arange(len(Ms))
axA.bar(x,meds,width=0.6,color=["#4d9221" if p<5 else "#f46d43" if p<10 else "#d73027" for p in penns],alpha=.85)
axA.axhline(0,color="#7a7f83",lw=2.5); axA.text(len(Ms)-0.5,0.5,"RED / GREY = 0° (all muscles)",ha="right",fontsize=9,color="#555",fontweight="bold")
for i,(m,p) in enumerate(zip(meds,penns)): axA.text(i,m+0.3,f"{m:.1f}°",ha="center",fontsize=9,fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels([f"{n}\n({p}°)" for n,p in zip(names,penns)],fontsize=8.5)
axA.set_ylabel("field angle from centre-line [°]\n(= imposed pennation)",fontsize=9)
axA.set_title("How far NEW's fibre direction departs from RED/GREY — scales with architecture (fusiform → bipennate)",fontsize=10.5)
axA.set_ylim(0,14)

# (B) transverse cross-sections: convergence structure
for j,idx in enumerate([0,3,4]):      # BR (fusiform), pron.teres (mid), FCU (bipennate)
    M=Ms[idx]; ax=fig.add_subplot(gs[1,j]); du,dw,au,aw,bg=transverse(M)
    ax.scatter(bg[:,0],bg[:,1],s=7,c="0.87",alpha=.5,zorder=0)
    ax.quiver(du,dw,au,aw,angles="xy",scale_units="xy",scale=0.055,width=0.007,color="#1a9850",zorder=2)
    ax.scatter([0],[0],s=90,marker="*",c="#d62728",zorder=4)
    ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    ax.set_title(f"{M['key'].split('_')[0][:12]} — pennation {M['penn']}° (arrow ∝ lean)",fontsize=9)
    ax.set_xlabel("width [mm]",fontsize=8); ax.set_ylabel("thickness [mm]",fontsize=8)
fig.text(0.5,0.5,"transverse cross-sections — green ticks = in-plane fibre direction (converging on the central tendon ★); RED/GREY have ZERO in-plane component everywhere",
         ha="center",fontsize=8,style="italic",color="0.45")
fig.suptitle("Where the field's gain over RED/GREY comes from: pennation & convergence, scaling with architecture",fontsize=12,y=0.98)
fig.text(0.5,0.005,"For a fusiform muscle (brachioradialis, 2.4°) the field is within ~2–3° of the centre-line — barely different from RED/GREY. "
    "For a bipennate muscle (FCU, 12°) the fibres lean 12° and converge on the central tendon (right) — a chevron RED/GREY cannot represent at any angle. "
    "The magnitude of the direction gain ≈ the pennation angle; the convergence STRUCTURE is categorical (RED/GREY = 0). Pennation is imposed from the atlas; "
    "the tendon is modelled as a line, so this approximates (not fully resolves) the true two-sided herringbone.",
    ha="center",fontsize=7.4,style="italic",color="0.4",wrap=True)
plt.savefig("field_architecture.png",dpi=130,bbox_inches="tight"); print("saved field_architecture.png")
