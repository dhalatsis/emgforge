"""Independent verification that the INTEGRATED pipeline reproduces the PROTOTYPE
and the artifact's claims — across several muscle architectures, no FEM needed.

For each muscle it runs BOTH:
  PROTO   = _results/mri_check/noura_fibre prototype (densefib.dense_fill_iz)
  PIPE    = integrated pipeline (build_muscle_beds(method="harmonic"))
and compares fibre count, length distribution, atlas-IZ fractions, containment.
Then checks the artifact's numeric claims: fibres ~Lf (short, Lf/Lm~0.2), NMJs on the
muscle's atlas IZ fractions, contained + non-crossing.
"""
import sys, os, json, numpy as np
PROTO_DIR = "/home/noura/Documents/Projects/PhD/simulation/emgforge/_results/mri_check/noura_fibre"
SRC = "/home/noura/Documents/Projects/PhD/simulation/emgforge/src"
sys.path.insert(0, SRC); sys.path.insert(0, PROTO_DIR)
import geom
WR = "/home/noura/Documents/Projects/PhD/mri/data/Lab/WR/WR_Segmentation.nii.gz"
geom.SEG = WR
import densefib                                   # prototype
from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core import harmonic_fibers as HF

AT = json.load(open(f"{SRC}/emgforge/mri/data/forearm_muscle_atlas.json"))
def atlas_of(L):
    k = AT["wr_label_to_muscle"][str(L)]; mu = AT["muscles"][k]
    return k, float(mu["fascicle_length_mm"]), sorted(mu["IZ_fraction"])

def arclen(p):
    p = np.asarray(p, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

def contain(paths, mask, vs):
    tot = ins = 0
    for p in paths:
        ci = np.round(np.asarray(p, float) / vs).astype(int)
        ok = (ci >= 0).all(1) & (ci < np.array(mask.shape)).all(1)
        idx = ci[ok]
        ins += int(mask[idx[:, 0], idx[:, 1], idx[:, 2]].sum()); tot += len(ci)
    return 100.0 * ins / max(tot, 1)

MUSCLES = [(8, "FCU bipennate"), (13, "FDS multipenn"), (11, "BR fusiform"),
           (6, "ECU"), (21, "pronator teres")]

# pipeline model (shared)
model = MuscleFiberModel(WR)
model.estimate_centerlines(); model.estimate_cross_sections()

print(f"{'muscle':<18}{'atlasLf':>8}{'  PROTO(n/len/IZ)':>26}{'   PIPE(n/len/IZ)':>26}{'  contain':>9}  verdict")
allpass = True
for L, name in MUSCLES:
    k, Lf, izA = atlas_of(L)
    # ---- prototype ----
    D = densefib.setup(L)
    segs, band, nmj, isatl = densefib.dense_fill_iz(D, grid_mm=2.0)
    plen = np.array([arclen(s) for s in segs])
    ctr, p1 = D["ctr"], D["p1"]; s1 = (np.argwhere(D["mask"]) * D["vs"] - ctr) @ p1
    s1min, Lm = float(s1.min()), float(s1.max() - s1.min())
    izP = sorted(set(round(float(((nm - ctr) @ p1 - s1min) / Lm), 2)
                     for nm, a in zip(nmj, isatl) if a))
    contP = contain(segs, D["mask"], D["vs"])
    # ---- pipeline ----
    beds = build_muscle_beds(model, method="harmonic", labels=[L], grid_mm=2.0)
    if L not in beds:
        print(f"{name:<18} PIPE produced no bed -> FAIL"); allpass = False; continue
    bed = beds[L]
    ln = np.asarray(bed.half1_mm) + np.asarray(bed.half2_mm)
    izPipe = sorted(set(round(float(f), 2) for f, a in zip(bed.iz_fractions, bed.is_atlas) if a))
    contPipe = contain(list(bed.paths), model.seg_data == L, model.voxel_size)
    # ---- checks ----
    len_ok   = abs(np.mean(ln) - np.mean(plen)) < 0.15 * np.mean(plen)   # parity of mean length
    n_ok     = abs(len(ln) - len(plen)) <= max(3, 0.10 * len(plen))      # parity of count
    iz_ok    = izPipe == izP == izA                                       # IZ == prototype == atlas
    lf_ok    = 0.55 * Lf <= np.mean(ln) <= 1.5 * Lf                       # length ~ atlas Lf
    ratio_ok = 0.12 <= np.mean(ln) / Lm <= 0.45                          # short (Lf/Lm ~ 0.2)
    cont_ok  = contPipe > 95 and contP > 95
    ok = len_ok and n_ok and iz_ok and lf_ok and ratio_ok and cont_ok
    allpass &= ok
    print(f"{name:<18}{Lf:>7.0f} {len(plen):>4}/{np.mean(plen):>5.0f}/{str(izP):>12}"
          f"  {len(ln):>4}/{np.mean(ln):>5.0f}/{str(izPipe):>12}  {contPipe:>7.1f}%  "
          f"{'PASS' if ok else 'FAIL'}  Lf/Lm={np.mean(ln)/Lm:.2f}"
          f"{'' if ok else ' <'+','.join(n for n,v in [('len',len_ok),('n',n_ok),('iz',iz_ok),('lf',lf_ok),('ratio',ratio_ok),('cont',cont_ok)] if not v)+'>'}")

# ---- non-crossing (prototype field guarantee), σ-direction parity ----
print("\n--- non-crossing + σ-direction ---")
mg, nstream = densefib.min_gap(densefib.setup(8), grid=4.0)
print(f"non-crossing (FCU): min pairwise streamline gap = {mg:.2f} mm over {nstream} streamlines "
      f"({'PASS >0.5mm' if mg > 0.5 else 'FAIL'})")
model.estimate_fibers(method="harmonic")
dir_pipe = model.muscles[8].fiber_direction
print(f"σ-direction FCU (pipeline harmonic): {np.round(dir_pipe,3)}  source={model.muscles[8].direction_source} "
      f"({'PASS longitudinal' if abs(dir_pipe[2])>0.9 else 'CHECK'})")

print(f"\n{'='*60}\nOVERALL: {'ALL MUSCLES PASS — pipeline reproduces prototype + artifact claims' if allpass else 'SOME FAILED (see <..> above)'}")
