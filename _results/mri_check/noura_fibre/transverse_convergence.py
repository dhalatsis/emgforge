"""Can the TRANSVERSE (across-muscle) dimension of the WR grid see bipennate convergence?

A linear (along-muscle) array is ~blind to pennation. But the WR grid is 5 (across) x 39 (along).
If FCU fibres converge on a central tendon, motor units on the two sides of the tendon lean in
OPPOSITE transverse directions -> their action potentials propagate with opposite transverse
velocity components. So the SIGNATURE of convergence is a BIMODAL +/- distribution of transverse
tilt across units. A uniform tilt (all same sign) would instead just mean the array is rotated vs
the muscle (misalignment), NOT convergence. A tilt ~0 means no detectable transverse structure.

For each clean cBSS unit we STA the full 5x39 grid, take the peak-arrival time t(row,col), and on the
dominant propagating arm fit t ~ a + b*(col-col_IZ) + c*row. Transverse tilt angle = atan((c/pitch)/(b/pitch))
= atan(c/b) in electrode units -> apparent fibre inclination in the grid plane. Honest: 5 rows is very
few, registration to the tendon is unknown, so a null/weak result is a real and expected outcome.
"""
import os, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from mu_cbss import CBSS
fs=2000.0; NROW,NCOL=5,39; W=int(0.016*fs)
DATA="/home/noura/Documents/Projects/PhD/emg-decomposition/data/processed/WR2"
TASKS=["10mvc_rep1_index","20mvc_rep1_index","30mvc_rep1_index"]   # subset for speed

def decompose(task):
    mono=np.load(f"{DATA}/{task}/combined_emg.npy")
    sig=mono.reshape(-1,mono.shape[-1]); std=sig.std(1); good=std>0.05*np.median(std)
    clean=sig[good]; clean=clean-clean.mean(0,keepdims=True)
    cb=CBSS(ica_n_iter=60,sil_th=0.80,cov_th=0.40,min_num_spikes=10,random_seed=1909)
    sources,spikes,sil,filt=cb.decompose(clean.copy(),fs)
    return mono,spikes,sil

def grid_sta(mono,sp):
    T=mono.shape[-1]; sp=np.asarray(sp,int); sp=sp[(sp>=W)&(sp<T-W)]
    if len(sp)<30: return None
    sta=np.zeros((NROW,NCOL,2*W))
    for s in sp: sta+=mono[:,:,s-W:s+W]
    return sta/len(sp), len(sp)

def analyse(sta):
    """Return dict: IZ col, along-slope b, transverse-slope c, tilt angle, transverse IZ shift, quality."""
    p2p=sta.max(2)-sta.min(2)                          # (5,39)
    col_amp=p2p.mean(0)                                # along-muscle amplitude profile
    izc=int(col_amp.argmax())
    active=col_amp>0.30*col_amp.max()
    # peak-arrival time per (row,col), in ms
    tpk=np.argmax(np.abs(sta),axis=2).astype(float)/fs*1e3   # (5,39)
    # per-row IZ column (transverse shift of the innervation zone) — only rows with real signal
    strong=[r for r in range(NROW) if p2p[r].max()>0.5*p2p.max()]
    izcol=np.array([int(p2p[r].argmax()) for r in strong]) if strong else np.array([izc])
    # pick the LONGER propagating arm around izc
    lo=hi=izc
    while lo-1>=0 and active[lo-1]: lo-=1
    while hi+1<NCOL and active[hi+1]: hi+=1
    arm = np.arange(izc,hi+1) if (hi-izc)>=(izc-lo) else np.arange(lo,izc+1)
    if len(arm)<3: return None
    # design matrix over (row,col) on the arm
    R=[];C=[];Tt=[]
    for c in arm:
        for r in range(NROW):
            if p2p[r,c]>0.30*p2p.max():
                R.append(r); C.append(c-izc); Tt.append(tpk[r,c])
    if len(Tt)<8: return None
    A=np.column_stack([np.ones(len(Tt)),C,R]); coef,*_=np.linalg.lstsq(A,np.array(Tt),rcond=None)
    b,c=coef[1],coef[2]                               # ms per along-col, ms per across-row
    # transverse tilt: lean of the wavefront ACROSS vs ALONG, independent of propagation DIRECTION
    # (use |b| so a reverse-propagating unit doesn't read as 180deg). +/- sign of tilt = lean direction.
    tilt=np.degrees(np.arctan2(c, abs(b) if abs(b)>1e-6 else 1e-6))
    resid=np.array(Tt)-A@coef; r2=1-np.var(resid)/max(np.var(Tt),1e-9)
    return dict(izc=izc,b=b,c=c,tilt=tilt,izshift=izcol.max()-izcol.min(),r2=r2,sta=sta,arm=arm,tpk=tpk,p2p=p2p)

units=[]
for task in TASKS:
    try: mono,spikes,sil=decompose(task)
    except Exception as e: print(task,"fail",e); continue
    for i in spikes:
        if len(spikes[i])<40 or sil[i]<0.88: continue
        r=grid_sta(mono,spikes[i])
        if r is None: continue
        sta,nsp=r; a=analyse(sta)
        if a is None or a["r2"]<0.3: continue
        a["nsp"]=nsp; a["sil"]=float(sil[i]); a["task"]=task; units.append(a)
    print(f"{task}: kept {sum(u['task']==task for u in units)} units so far")

print(f"\nTOTAL clean units analysed: {len(units)}")
if len(units)<3: print("too few units for a distribution — inconclusive");
tilts=np.array([u["tilt"] for u in units]); izsh=np.array([u["izshift"] for u in units])
# along-velocity for context (sign): b in ms/electrode; |CV| = pitch/|b|
print(f"transverse tilt (deg): {np.round(tilts,1).tolist()}")
print(f"  median |tilt| = {np.median(np.abs(tilts)):.1f} deg ; signed mean = {tilts.mean():.1f} ; both-signs? "
      f"{'YES (bimodal -> convergence-like)' if (tilts.min()<-5 and tilts.max()>5) else 'no (unimodal)'}")
print(f"transverse IZ shift across the 5 rows (electrodes): {izsh.tolist()} (median {np.median(izsh):.0f})")

# ---- figure ----
fig=plt.figure(figsize=(13,7)); gs=fig.add_gridspec(2,3,height_ratios=[1,1.1],hspace=0.5,wspace=0.35)
# (a) tilt distribution
axT=fig.add_subplot(gs[0,0])
axT.hist(tilts,bins=np.arange(-45,50,7.5),color="#2166ac",alpha=.8,edgecolor="w")
axT.axvline(0,color="k",lw=1.2); axT.set_xlabel("transverse tilt of propagation [°]",fontsize=9)
axT.set_ylabel("units",fontsize=9); axT.set_title(f"Transverse tilt per unit (n={len(units)})\nbimodal ± ⇒ convergence; ~0 ⇒ none",fontsize=9.5)
# (b) IZ transverse shift
axI=fig.add_subplot(gs[0,1])
axI.hist(izsh,bins=np.arange(0,12,1),color="#1a9850",alpha=.8,edgecolor="w")
axI.set_xlabel("IZ column shift across 5 rows [electrodes]",fontsize=9); axI.set_ylabel("units",fontsize=9)
axI.set_title("Does the IZ move across the muscle?\n(large ⇒ oblique innervation band)",fontsize=9.5)
# (c) along vs across velocity scatter
axV=fig.add_subplot(gs[0,2])
axV.scatter([u["b"] for u in units],[u["c"] for u in units],c="#762a83",s=30)
axV.axhline(0,color="k",lw=.8); axV.axvline(0,color="k",lw=.8)
axV.set_xlabel("along-slope b [ms/elec]",fontsize=9); axV.set_ylabel("across-slope c [ms/elec]",fontsize=9)
axV.set_title("Wavefront slope: along vs across\n(c≈0 ⇒ propagation is purely longitudinal)",fontsize=9.5)
# (d,e,f) peak-arrival maps of up to 3 example units
ex=sorted(units,key=lambda u:-abs(u["tilt"]))[:3]
for j,u in enumerate(ex):
    ax=fig.add_subplot(gs[1,j]); im=ax.imshow(u["tpk"],aspect="auto",origin="lower",cmap="viridis")
    ax.axvline(u["izc"],color="r",lw=1.2); ax.set_title(f"peak-arrival t(row,col)\ntilt {u['tilt']:.0f}°, sil {u['sil']:.2f}",fontsize=8.5)
    ax.set_xlabel("along muscle [electrode]",fontsize=8); ax.set_ylabel("across [row]",fontsize=8); ax.tick_params(labelsize=7)
    plt.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
verdict=("BIMODAL ± tilt — a convergence-like signature (fibres leaning both ways)" if (len(tilts) and tilts.min()<-5 and tilts.max()>5)
         else "tilt clusters near 0 / one sign — NO clear transverse convergence detectable")
fig.suptitle(f"Transverse (5-row) analysis of WR motor units — looking for bipennate convergence.  Result: {verdict}",fontsize=10.5,y=0.99)
fig.text(0.5,0.005,"Honest read: 5 across-electrodes, unknown registration to the central tendon, ~12° geometric effect. A bimodal ± tilt would be positive "
         "evidence of convergence; a near-zero / unimodal tilt means the surface grid cannot resolve it here (consistent with the ~2% projection effect).",
         ha="center",fontsize=7.4,style="italic",color="0.4",wrap=True)
plt.savefig("transverse_convergence.png",dpi=130,bbox_inches="tight"); print("saved transverse_convergence.png")
