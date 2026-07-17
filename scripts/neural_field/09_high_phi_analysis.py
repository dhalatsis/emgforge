"""Does accuracy on HIGH potentials predict MUAP quality?

Hypothesis (user): being accurate on high phi matters more than on low phi. The physics
agrees — deep/low-phi fibres attenuate to 1-8% of a superficial one, and SFAP ~ phi'' so
curvature (largest where phi is sharp) drives the MUAP. If true, a model's error in the
TOP |phi| decile should predict its MUAP score far better than its overall rel-L2 does.

Measures, per trained target: rel-L2 stratified by |phi| decile on held-out electrodes,
then correlates each stratum against the frozen-benchmark MUAP score.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
S = import_module("04_score_bench")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/neural_field"
dev = "cuda" if torch.cuda.is_available() else "cpu"

d = np.load(OUT / "cyl_elec_64.npz")
P, E, PHI = d["points"], d["electrodes"], d["phi"]

TARGETS = [("asinh", "cyl_mlp.pt", "score_cyl_mlp.npz"),
           ("phir", "cyl_mlp_phir.pt", "score_cyl_mlp_phir.npz"),
           ("phir0", "cyl_mlp_phir0.pt", "score_cyl_mlp_phir0.npz"),
           ("raw", "cyl_mlp_raw.pt", "score_cyl_mlp_raw.npz")]

rows = {}
for name, ck, sc in TARGETS:
    if not (OUT / ck).exists():
        print(f"  (skip {name} — no checkpoint)"); continue
    net, c = S.load(str(OUT / ck), dev)
    va = c["val_elec"]
    pred = np.concatenate([S.predict_phi(net, c, dev, P, E[i]) for i in va])
    true = PHI[va].reshape(-1)
    a = np.abs(true)
    # stratify by |phi| decile
    qs = np.percentile(a, np.arange(0, 101, 10))
    strat = []
    for i in range(10):
        m = (a >= qs[i]) & (a < qs[i + 1] if i < 9 else a <= qs[10])
        if m.sum() < 10:
            strat.append(np.nan); continue
        strat.append(np.linalg.norm(pred[m] - true[m]) / (np.linalg.norm(true[m]) + 1e-30))
    overall = np.linalg.norm(pred - true) / np.linalg.norm(true)
    # top-decile |phi| error and bottom-half error
    hi = a >= qs[9]; lo = a < qs[5]
    e_hi = np.linalg.norm(pred[hi] - true[hi]) / np.linalg.norm(true[hi])
    e_lo = np.linalg.norm(pred[lo] - true[lo]) / np.linalg.norm(true[lo])
    Sc = np.load(OUT / sc)
    rows[name] = dict(strat=np.array(strat), overall=overall, e_hi=e_hi, e_lo=e_lo,
                      muap_med=float(np.median(Sc["r"])), muap_min=float(Sc["r"].min()),
                      muap_mean=float(Sc["r"].mean()),
                      p2p=float(np.median(Sc["p2p_ratio"])),
                      n94=int((Sc["r"] >= 0.94).sum()))
    print(f"  {name:6s} done")

print()
print("=" * 96)
print("ACCURACY BY |phi| MAGNITUDE  vs  MUAP QUALITY")
print("=" * 96)
print(f"{'target':>7} | {'relL2 all':>9} {'relL2 TOP10%':>13} {'relL2 bot50%':>13} | "
      f"{'MUAP med':>9} {'MUAP min':>9} {'>=0.94':>7} {'p2p':>6}")
print("-" * 96)
for n, r in rows.items():
    print(f"{n:>7} | {r['overall']:9.4f} {r['e_hi']:13.4f} {r['e_lo']:13.4f} | "
          f"{r['muap_med']:+9.3f} {r['muap_min']:+9.3f} {r['n94']:5d}/27 {r['p2p']:6.2f}")

# which predicts MUAP quality better: overall error or TOP-decile error?
if len(rows) >= 3:
    nm = list(rows)
    ov = np.array([rows[n]["overall"] for n in nm])
    hi = np.array([rows[n]["e_hi"] for n in nm])
    mm = np.array([rows[n]["muap_min"] for n in nm])
    md = np.array([rows[n]["muap_med"] for n in nm])
    def cc(x, y):
        x, y = x - x.mean(), y - y.mean()
        dn = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / dn) if dn else np.nan
    print()
    print(f"  corr(overall relL2, MUAP min) = {cc(ov, mm):+.3f}")
    print(f"  corr(TOP-10% relL2, MUAP min) = {cc(hi, mm):+.3f}   <- if stronger, the "
          f"hypothesis holds")
    print(f"  corr(overall relL2, MUAP med) = {cc(ov, md):+.3f}")
    print(f"  corr(TOP-10% relL2, MUAP med) = {cc(hi, md):+.3f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for n, r in rows.items():
    ax[0].plot(np.arange(1, 11), r["strat"], "o-", label=n)
ax[0].set_xlabel("|φ| decile (10 = the highest potentials)"); ax[0].set_ylabel("rel-L2 within decile")
ax[0].set_yscale("log"); ax[0].set_title("Where each target is accurate")
ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
for n, r in rows.items():
    ax[1].scatter(r["e_hi"], r["muap_min"], s=90, label=f"{n}")
    ax[1].annotate(n, (r["e_hi"], r["muap_min"]), fontsize=8,
                   textcoords="offset points", xytext=(6, 4))
ax[1].set_xlabel("rel-L2 on the TOP 10% |φ|"); ax[1].set_ylabel("worst-case MUAP corr")
ax[1].set_title("High-φ accuracy vs worst-case MUAP"); ax[1].grid(alpha=0.3)
fig.suptitle("Is accuracy on HIGH potentials what matters?", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "high_phi_analysis.png", dpi=130, bbox_inches="tight")
print(f"\nwrote {OUT/'high_phi_analysis.png'}")
