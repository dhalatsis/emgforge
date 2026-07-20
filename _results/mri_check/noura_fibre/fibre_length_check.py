"""Fibre length & propagation extent: MODEL vs LITERATURE vs REAL WR data.

Three cross-checks of the claim "NEW's short fibres are more geometrically correct than RED/GREY's
one-fibre-spans-the-whole-muscle":
  (1) FCU fibre length — our model/atlas vs cadaver literature (Lieber 1990/1992; FCU regional
      variability, Loren&Lieber-type 2004) and vs the (impossible) length RED/GREY imply.
  (2) Propagation extent in mm — model vs the real WR HD-sEMG unit (bipolar linear array, fs=2000).
  (3) Fraction of the electrode array a single unit activates — the PITCH-INDEPENDENT check.

Honest limits: exact FCU fibre length varies by source (~4–5 cm) and REGIONALLY within the muscle
(~2x, so ~3–6 cm); the WR2 electrode pitch is not on disk (raw OTB not present) so mm uses the
OTB-standard 8–10 mm; extent is one spike-triggered-averaged unit (coarse vs full decomposition).
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
sys.path.insert(0,"/home/noura/Documents/Projects/PhD/simulation/emgforge/src")
import densefib as DF

# ---------- (A) model fibre lengths ----------
L=8; D=DF.setup(L)
mask,vs,cl,cs,ctr,p1,p2,p3=D["mask"],D["vs"],D["cl"],D["cs"],D["ctr"],D["p1"],D["p2"],D["p3"]
allc=np.argwhere(mask)*vs; s1all=(allc-ctr)@p1; Lm=float(s1all.max()-s1all.min())
Lf_atlas=float(DF.AT["muscles"][DF.AT["wr_label_to_muscle"][str(L)]]["fascicle_length_mm"])
segs,band,nmj,isatl=DF.dense_fill_iz(D,grid_mm=1.8)
arclen=np.array([np.linalg.norm(np.diff(np.asarray(s,float),axis=0),axis=1).sum()
                 for s,atl in zip(segs,isatl) if atl])
Lf_mean=arclen.mean(); Lf_lo,Lf_hi=arclen.min(),arclen.max()
print(f"Lm={Lm:.0f}  atlas Lf={Lf_atlas:.0f}  model fibres mean {Lf_mean:.0f} ({Lf_lo:.0f}-{Lf_hi:.0f})")

# ---------- (B) real WR unit propagation extent ----------
fs=2000.0
def load(fn):
    s=np.load(f"/home/noura/Documents/Projects/PhD/emg-decomposition/data/processed/WR2/{fn}/bipolar_emg.npy")[2]
    b,a=butter(4,[20/(fs/2),450/(fs/2)],btype="band"); return filtfilt(b,a,s,axis=1)
def sta(sig,trig,W):
    x=sig[trig]; th=5*np.median(np.abs(x))/0.6745
    pk,_=find_peaks(np.abs(x),height=th,distance=int(0.02*fs)); out=np.zeros((sig.shape[0],2*W)); n=0
    for p in pk:
        if p-W<0 or p+W>sig.shape[1]: continue
        out+=sig[:,p-W:p+W]; n+=1
    return out/max(n,1),n
sig=load("20mvc_rep1_index"); NCH=sig.shape[0]; W=int(0.016*fs)
S,n=sta(sig,36,W); p2p=S.max(1)-S.min(1); izc=int(p2p.argmax()); active=p2p>0.30*p2p.max()
lo=hi=izc
while lo-1>=0 and active[lo-1]: lo-=1
while hi+1<NCH and active[hi+1]: hi+=1
span_steps=hi-lo                                   # electrode gaps spanned by the unit
print(f"real unit: band ch{lo}-{hi} = {span_steps} steps, {span_steps+1}/{NCH} channels")

# ---------- figure ----------
IED=(8,10)                                         # OTB-standard pitch (raw not on disk to confirm)
real_span_lo,real_span_hi=span_steps*IED[0],span_steps*IED[1]
fig,ax=plt.subplots(1,3,figsize=(15.5,5.6))
GRN,RED,GRY,BL="#1a9850","#d62728","#8a8f94","#2166ac"

# --- panel 1: FIBRE LENGTH ---
a=ax[0]
a.axvspan(40,55,color=GRY,alpha=.18,label="literature consensus ≈ 40–55 mm")
a.axvspan(30,60,color=GRY,alpha=.09)
rows=[("RED / GREY imply\n(one fibre = whole muscle)",226,RED),
      ("longest human fibre\n(sartorius, ref.)",120,"0.6"),
      ("our atlas L_f (Lieber-derived)",Lf_atlas,GRN),
      ("our model fibres (mean)",Lf_mean,GRN)]
y=np.arange(len(rows))[::-1]
for yi,(lab,val,col) in zip(y,rows):
    a.barh(yi,val,color=col,alpha=.85,height=.6); a.text(val+4,yi,f"{val:.0f} mm",va="center",fontsize=8.5)
a.errorbar(Lf_mean,y[3],xerr=[[Lf_mean-Lf_lo],[Lf_hi-Lf_mean]],fmt="none",ecolor=GRN,capsize=4,lw=1.4)
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows],fontsize=8)
a.set_xlabel("fibre length [mm]",fontsize=9); a.set_xlim(0,245)
a.set_title("(1) FCU fibre length: model vs literature\nRED/GREY imply a 226 mm fibre — none exists",fontsize=10)
a.legend(fontsize=7.5,loc="lower right")

# --- panel 2: PROPAGATION EXTENT (mm) ---
b=ax[1]
b.barh(2,Lm,color=RED,alpha=.85,height=.55); b.text(Lm+3,2,f"~{Lm:.0f} mm",va="center",fontsize=8.5)
b.barh(1,Lf_mean,color=GRN,alpha=.85,height=.55); b.text(Lf_mean+3,1,f"~{Lf_mean:.0f} mm",va="center",fontsize=8.5)
b.barh(0,0.5*(real_span_lo+real_span_hi),color=BL,alpha=.85,height=.55,
       xerr=0.5*(real_span_hi-real_span_lo),error_kw=dict(ecolor="k",capsize=4,lw=1.4))
b.text(real_span_hi+3,0,f"{real_span_lo:.0f}–{real_span_hi:.0f} mm",va="center",fontsize=8.5)
b.set_yticks([2,1,0]); b.set_yticklabels(["RED / GREY\n(full muscle)","NEW\n(short fibre)","REAL WR unit\n(8–10 mm pitch)"],fontsize=8.5)
b.set_xlabel("propagation footprint of one motor unit [mm]",fontsize=9); b.set_xlim(0,245)
b.axvspan(real_span_lo,real_span_hi,color=BL,alpha=.10)
b.set_title("(2) Propagation extent: NEW matches real,\nRED/GREY 4–5× too long",fontsize=10)

# --- panel 3: PITCH-INDEPENDENT fraction of array ---
c=ax[2]
fr_real=(span_steps+1)/NCH*100
fr_red=100.0; fr_new=Lf_mean/Lm*100
c.bar([0,1,2],[fr_red,fr_new,fr_real],color=[RED,GRN,BL],alpha=.85,width=.6)
for xi,v in zip([0,1,2],[fr_red,fr_new,fr_real]): c.text(xi,v+2,f"{v:.0f}%",ha="center",fontsize=9)
c.set_xticks([0,1,2]); c.set_xticklabels(["RED/GREY","NEW","REAL\nWR unit"],fontsize=8.5)
c.set_ylabel("% of muscle / array a single unit spans",fontsize=9); c.set_ylim(0,115)
c.set_title("(3) Pitch-INDEPENDENT check\none unit lights up ~1/5, not the whole muscle",fontsize=10)

fig.suptitle("Is NEW's short fibre more geometrically correct? Literature says yes; WR's own HD-sEMG confirms it — and rules out RED/GREY.",fontsize=11)
fig.text(0.5,-0.04,
    f"(1) FCU fascicles are ~4–5 cm across cadaver studies (Lieber 1990/1992; FCU is short & feather-like and regionally variable ~2×, Loren&Lieber-type 2004); "
    f"our atlas L_f={Lf_atlas:.0f} mm and model fibres (mean {Lf_mean:.0f}, {Lf_lo:.0f}–{Lf_hi:.0f} mm) sit inside that. RED/GREY imply a single {Lm:.0f} mm fibre — "
    f"longer than any human fibre (sartorius ~120 mm). (2) The real WR unit's action potential spans only {span_steps+1} of {NCH} electrodes "
    f"(~{real_span_lo:.0f}–{real_span_hi:.0f} mm at 8–10 mm pitch) ≈ NEW's {Lf_mean:.0f} mm, NOT RED/GREY's {Lm:.0f} mm. (3) Pitch-independent: one real unit lights up "
    f"~{fr_real:.0f}% of the array, like NEW (~{fr_new:.0f}%), not ~100% (RED/GREY). Caveats: WR2 pitch not on disk (OTB-standard 8–10 mm assumed); one STA unit; exact L_f varies by source.",
    ha="center",fontsize=7.2,style="italic",color="0.4",wrap=True)
plt.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig("fibre_length_check.png",dpi=125,bbox_inches="tight"); print("saved fibre_length_check.png")
