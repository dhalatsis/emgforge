"""Final Phase-2 figure: FEM truth vs the learned VC (MLP+Fourier, SIREN) on the frozen
MUAP benchmark, plus the metric summary. Everything here is held-out electrodes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/neural_field"

b = np.load(OUT / "muap_bench_cyl.npz")
t, Wt, cfg = b["t_ms"], b["muap"], b["cfg"]
S = {a: np.load(OUT / f"score_cyl_{a}.npz") for a in ("mlp", "siren")}

# rebuild the model MUAPs for the figure by re-scoring is expensive; instead show the
# score distributions + the truth waveforms, which is what the benchmark actually asserts.
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)

# (A) MUAP corr per config, both archs
axA = fig.add_subplot(gs[0, :2])
x = np.arange(len(cfg))
axA.plot(x, S["mlp"]["r"], "o-", color="tab:blue", label="MLP+Fourier", ms=4)
axA.plot(x, S["siren"]["r"], "s-", color="tab:red", label="SIREN", ms=4, alpha=0.8)
axA.axhline(0.94, color="k", ls="--", lw=1, label="Gate 2 (0.94)")
axA.set_xticks(x)
axA.set_xticklabels([f"{c[0]:.0f}/{c[1]:.0f}/{c[2]:.0f}" for c in cfg], rotation=90, fontsize=6)
axA.set_ylabel("MUAP corr vs FEM truth"); axA.set_ylim(-0.1, 1.05)
axA.set_xlabel("benchmark config  (elec θ / elec z / MU depth)")
axA.set_title("Per-config MUAP score on the frozen benchmark (held-out electrodes)")
axA.legend(fontsize=8); axA.grid(alpha=0.3)

# (B) the phi-vs-MUAP dissociation — the whole point of Phase 1
axB = fig.add_subplot(gs[0, 2])
for a, c in (("mlp", "tab:blue"), ("siren", "tab:red")):
    axB.scatter(np.abs(S[a]["p2p_ratio"]), S[a]["r"], c=c, s=28, label=a, alpha=0.75)
axB.axhline(0.94, color="k", ls="--", lw=1); axB.axvline(1.0, color="k", ls=":", lw=1)
axB.set_xscale("log"); axB.set_xlabel("p2p ratio (1 = correct amplitude)")
axB.set_ylabel("MUAP corr"); axB.set_title("amplitude vs shape")
axB.legend(fontsize=8); axB.grid(alpha=0.3, which="both")

# (C) where MLP fails: corr vs MU depth / electrode angle
axC = fig.add_subplot(gs[1, 0])
for dep, mk in zip((8., 15., 25.), ("o", "s", "^")):
    m = cfg[:, 2] == dep
    axC.scatter(cfg[m, 0], S["mlp"]["r"][m], marker=mk, s=45, label=f"depth {dep:.0f}mm")
axC.axhline(0.94, color="k", ls="--", lw=1)
axC.set_xlabel("electrode θ (deg)"); axC.set_ylabel("MUAP corr (MLP)")
axC.set_title("MLP failures localise: superficial MU under the electrode")
axC.legend(fontsize=8); axC.grid(alpha=0.3)

# (D) truth MUAPs across the difficulty gradient
axD = fig.add_subplot(gs[1, 1:])
for k, lab in ((0, "θ0 z90 d8 — 8.7µV (strong, superficial)"),
               (4, "θ0 z120 d15 — 0.67µV"),
               (14, "θ20 z120 d25 — 0.14µV (deep, weak)")):
    axD.plot(t, Wt[k] / (np.abs(Wt[k]).max() + 1e-30), lw=1.5, label=lab)
axD.set_xlim(-10, 60); axD.axhline(0, color="k", lw=0.3)
axD.set_xlabel("t (ms)"); axD.set_ylabel("normalised")
axD.set_title("The frozen truth spans a real difficulty gradient (8.7µV → 0.14µV)")
axD.legend(fontsize=8); axD.grid(alpha=0.3)

fig.suptitle("Learned VC on the cylinder — Gate 2: MLP+Fourier PASS (median r=0.991), "
             "SIREN FAIL (0.883, 128× overfit)", fontsize=13)
fig.savefig(OUT / "phase2_final.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "phase2_final.png")
for a in ("mlp", "siren"):
    r = S[a]["r"]
    print(f"  {a:6s} median r {np.median(r):+.3f} · >=0.94 {int((r>=0.94).sum())}/{len(r)} "
          f"· p2p med {np.median(S[a]['p2p_ratio']):.2f} · Djag med {np.median(S[a]['d_jag']):+.5f}")
