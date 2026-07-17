"""Plots for the target/loss investigation: who is accurate on HIGH potentials?"""
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
from emgforge.fem.geometry import ParametricGeometry
from emgforge.synthesis import FibreBed, field_to_muap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/neural_field"
dev = "cuda" if torch.cuda.is_available() else "cpu"

b = np.load(OUT / "muap_bench_cyl.npz")
t, Wt, cfg, p2p_t = b["t_ms"], b["muap"], b["cfg"], b["p2p"] * 1e6

V = [("asinh", "cyl_mlp.pt", "score_cyl_mlp.npz", "tab:gray"),
     ("φ·r", "cyl_mlp_phir.pt", "score_cyl_mlp_phir.npz", "tab:orange"),
     ("φ·(r+r₀)", "cyl_mlp_phir0.pt", "score_cyl_mlp_phir0.npz", "tab:green"),
     ("raw", "cyl_mlp_raw.pt", "score_cyl_mlp_raw.npz", "tab:blue"),
     ("raw+|φ|¹", "cyl_mlp_raw_w1.pt", "score_cyl_mlp_raw_w1.npz", "tab:purple"),
     ("raw+|φ|²", "cyl_mlp_raw_w2.pt", "score_cyl_mlp_raw_w2.npz", "tab:red")]
V = [(n, c, s, col) for n, c, s, col in V if (OUT / s).exists()]

fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25)

# (A) p2p ratio vs truth strength — THE trade-off
axA = fig.add_subplot(gs[0, :2])
o = np.argsort(p2p_t)
for n, ck, sc, col in V:
    D = np.load(OUT / sc)
    axA.plot(p2p_t[o], D["p2p_ratio"][o], "o-", color=col, label=n, ms=4, lw=1.2, alpha=0.85)
axA.axhline(1.0, color="k", ls="--", lw=1.2, label="perfect amplitude")
axA.axvspan(1.0, p2p_t.max() * 1.3, color="gold", alpha=0.15)
axA.text(2.5, 0.08, "HIGH POTENTIALS\n(what matters)", fontsize=9, ha="center")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("truth MUAP amplitude (µV)"); axA.set_ylabel("p2p ratio (pred/truth)")
axA.set_title("Amplitude accuracy vs signal strength — only φ·(r+r₀) holds at the top")
axA.legend(fontsize=8, ncol=2); axA.grid(alpha=0.3, which="both")

# (B) MUAP corr vs truth strength
axB = fig.add_subplot(gs[0, 2])
for n, ck, sc, col in V:
    D = np.load(OUT / sc)
    axB.plot(p2p_t[o], D["r"][o], "o-", color=col, ms=3, lw=1, alpha=0.85, label=n)
axB.axhline(0.94, color="k", ls="--", lw=1)
axB.set_xscale("log"); axB.set_xlabel("truth amplitude (µV)"); axB.set_ylabel("MUAP corr")
axB.set_ylim(0.3, 1.03); axB.set_title("shape accuracy"); axB.grid(alpha=0.3, which="both")

# (C) the tallest peak, recreated by each variant
axC = fig.add_subplot(gs[1, :2])
k = int(np.argmax(p2p_t))
th, z, dep = cfg[k]
g = ParametricGeometry(**S.CYL)
paths = S.mu_fibre_paths(g, dep)
elec = g.electrode_on_skin(th, z)
sb = FibreBed.from_arrays(np.full(S.N_FIB, S.DZ), np.full(S.N_FIB, S.LP), np.full(S.N_FIB, S.LD),
                          np.full(S.N_FIB, S.POSZ), np.full(S.N_FIB, S.V))
axC.plot(t, Wt[k] * 1e6, "k", lw=3, label=f"FEM truth ({p2p_t[k]:.2f}µV)", zorder=5)
for n, ck, sc, col in V:
    net, c = S.load(str(OUT / ck), dev)
    phi = np.array([S.predict_phi(net, c, dev, p, elec) for p in paths])
    m = field_to_muap(phi, sb, S.SPCFG).muap
    axC.plot(t, m * 1e6, color=col, lw=1.4, alpha=0.9, label=f"{n} ({np.ptp(m)*1e6:.2f}µV)")
axC.set_xlim(-10, 55); axC.axhline(0, color="k", lw=0.3)
axC.set_xlabel("t (ms)"); axC.set_ylabel("µV")
axC.set_title(f"The tallest benchmark MUAP recreated by each variant "
              f"(θ={th:.0f}° z={z:.0f} depth {dep:.0f}mm)")
axC.legend(fontsize=8); axC.grid(alpha=0.3)

# (D) summary: strong-config accuracy
axD = fig.add_subplot(gs[1, 2])
names = [n for n, _, _, _ in V]
strong = p2p_t > 1.0
sp = [np.median(np.load(OUT / s)["p2p_ratio"][strong]) for _, _, s, _ in V]
cols = [c for _, _, _, c in V]
axD.barh(names, sp, color=cols, alpha=0.85)
axD.axvline(1.0, color="k", ls="--", lw=1.5)
axD.set_xlabel("p2p ratio on STRONG (>1µV) configs"); axD.set_xscale("log")
axD.set_title("who gets the high potentials right"); axD.grid(alpha=0.3, axis="x", which="both")

fig.suptitle("Target/loss investigation — accuracy on HIGH potentials  ·  "
             "55 train VC solutions × 20k pts = 1.1M samples", fontsize=13)
fig.savefig(OUT / "variant_comparison.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "variant_comparison.png")
for n, _, s, _ in V:
    D = np.load(OUT / s)
    print(f"  {n:10s} strong p2p {np.median(D['p2p_ratio'][strong]):5.2f} · "
          f"strong r {np.median(D['r'][strong]):+.3f} · all-med r {np.median(D['r']):+.3f}")
