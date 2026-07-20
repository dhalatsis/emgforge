"""Literature-grounded check: real MUAP characteristics ACROSS muscles vs what each method predicts.

Non-circular structural argument:
  - RED/GREY assume ONE innervation zone at the geometric MID (0.5) for EVERY muscle.
  - Literature (Safwat & Abdel-Meguid 2007 forearm motor-points; Saito 2000 HD-sEMG; Lateva 2010
    brachioradialis; Barbero/Merletti/Rainoldi 2012 Atlas of Muscle Innervation Zones) shows real IZ
    locations are SPREAD across the muscle and several muscles have MULTIPLE IZ bands.
  - NOTE: Barbero's exact forearm fractions are paywalled, so our atlas IZ values are grounded in
    Safwat 2007 + Saito 2000 + Lateva 2010 at thirds-level resolution (~0.17/0.33/0.5/0.62/0.8);
    Barbero is cited as corroboration, not read off directly.
  - MUAP DURATION, by contrast, is ~constant across muscles (8-14 ms; FDI 8.9, BB 10.1, ES 10.4, LV 11.1)
    -> duration does NOT encode architecture; IZ location/count is the characteristic that does.
So: the one MUAP characteristic that varies muscle-to-muscle is exactly the one our method encodes and
red/grey cannot. (Caveat: our atlas IZ VALUES come from this literature, so exact values aren't an
independent validation; the STRUCTURAL point — variation exists, one-size-fits-all is wrong — is.)
"""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
AT=json.load(open("/home/noura/Documents/Projects/PhD/simulation/emgforge/src/emgforge/mri/data/forearm_muscle_atlas.json"))
rows=[]
for key,m in AT["muscles"].items():
    iz=m.get("IZ_fraction")
    if not iz: continue
    rows.append((key.replace("_"," "),sorted(iz),int(m.get("n_iz_bands",len(iz))),m.get("pennation_type","?")))
rows.sort(key=lambda r:r[1][0])

# quantify red/grey's fixed-0.5 error and the multi-IZ gap
nearest_err=[min(abs(f-0.5) for f in iz) for _,iz,_,_ in rows]           # dist of NEAREST IZ from 0.5
multi=[r for r in rows if r[2]>1]
far=[e for e in nearest_err if e>0.15]
allfr=[f for _,iz,_,_ in rows for f in iz]
print(f"{len(rows)} muscles with IZ data; IZ fractions span {min(allfr):.2f}..{max(allfr):.2f}")
print(f"median nearest-IZ distance from 0.5: {np.median(nearest_err):.2f}  (red/grey error)")
print(f"muscles whose nearest IZ is >0.15 from mid: {len(far)}/{len(rows)}")
print(f"muscles with MULTIPLE IZ bands (red/grey cannot represent): {len(multi)}/{len(rows)}")

fig,ax=plt.subplots(figsize=(12.5,9))
y=np.arange(len(rows))
ax.axvspan(0.5-0.05,0.5+0.05,color="#d62728",alpha=0.12,zorder=0)
ax.axvline(0.5,color="#d62728",lw=2,label="RED/GREY: assume IZ at mid (0.5) for ALL muscles")
for i,(name,iz,nb,pt) in enumerate(rows):
    ax.plot([min(iz),max(iz)] if len(iz)>1 else [iz[0],iz[0]],[i,i],color="0.8",lw=1,zorder=1)
    col="#1a9850" if nb>1 else "#238b45"
    ax.scatter(iz,[i]*len(iz),s=70 if nb>1 else 45,color=col,edgecolor="k",lw=.4,zorder=3)
    ax.text(1.02,i,f"{pt}"+("  ⟵ MULTIPLE IZ" if nb>1 else ""),fontsize=6.5,va="center",
            color="#1a9850" if nb>1 else "0.5")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=7)
ax.set_xlim(0,1.28); ax.set_xlabel("innervation-zone location  (fraction of muscle length, proximal=0 → distal=1)",fontsize=9)
ax.scatter([],[],s=45,color="#238b45",edgecolor="k",lw=.4,label="literature IZ (our method places fibres here)")
ax.scatter([],[],s=70,color="#1a9850",edgecolor="k",lw=.4,label="muscle with MULTIPLE IZ bands (red/grey can't represent)")
ax.legend(fontsize=8,loc="center right",bbox_to_anchor=(1.0,0.62),framealpha=.95)
ax.set_title("Real innervation-zone locations vary widely across muscles — the one MUAP characteristic that encodes architecture\n"
             f"(our forearm atlas, {len(rows)} muscles; IZ span {min(allfr):.2f}–{max(allfr):.2f}; {len(multi)} muscles have multiple IZ bands)",fontsize=10.5)
fig.text(0.5,-0.02,
        f"RED/GREY put the IZ at 0.5 for every muscle → median error {np.median(nearest_err):.2f} of muscle length, >0.15 off for {len(far)}/{len(rows)} muscles, and they CANNOT represent the {len(multi)} multi-IZ muscles at all.\n"
        f"MUAP DURATION, by contrast, is ~constant across muscles (8–14 ms) so it does NOT discriminate. IZ sources: Safwat & Abdel-Meguid 2007 (forearm motor-points), Saito 2000 (HD-sEMG), Lateva 2010 (BR); Barbero/Merletti/Rainoldi 2012 corroborates\n"
        f"(exact forearm table paywalled → our values are thirds-level). Honest caveat: these IZ values are sourced FROM this literature, so they validate the STRUCTURE (variation exists, one-size-fits-all is wrong), not the exact fractions.",
        ha="center",fontsize=7.6,style="italic",color="0.35")
plt.tight_layout(); plt.savefig("muap_literature.png",dpi=130,bbox_inches="tight"); print("saved muap_literature.png")
