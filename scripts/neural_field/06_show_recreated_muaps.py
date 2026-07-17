"""Show the RECREATED MUAPs: FEM truth vs the learned VC, on held-out benchmark configs.

Reuses 04_score_bench's machinery, but plots the waveforms instead of only scoring them.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
S = import_module("04_score_bench")

from emgforge.fem.geometry import ParametricGeometry
from emgforge.synthesis import FibreBed, field_to_muap
from emgforge.synthesis.metrics import align_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/neural_field"
dev = "cuda" if torch.cuda.is_available() else "cpu"

b = np.load(OUT / "muap_bench_cyl.npz")
t, Wt, cfg, p2p_t = b["t_ms"], b["muap"], b["cfg"], b["p2p"]
g = ParametricGeometry(**S.CYL)
beds = {d: S.mu_fibre_paths(g, d) for d in np.unique(cfg[:, 2])}
sb = FibreBed.from_arrays(np.full(S.N_FIB, S.DZ), np.full(S.N_FIB, S.LP),
                          np.full(S.N_FIB, S.LD), np.full(S.N_FIB, S.POSZ), np.full(S.N_FIB, S.V))

nets = {}
for arch in ("mlp", "siren"):
    nets[arch] = S.load(str(OUT / f"cyl_{arch}.pt"), dev)


def recreate(arch, k):
    net, c = nets[arch]
    th, z, dep = cfg[k]
    elec = g.electrode_on_skin(th, z)
    phi = np.array([S.predict_phi(net, c, dev, p, elec) for p in beds[dep]])
    return field_to_muap(phi, sb, S.SPCFG).muap


# pick a spread: 3 best-case, 3 mid, 3 of the known near-field failures
sc = np.load(OUT / "score_cyl_mlp.npz")
order = np.argsort(sc["r"])
picks = list(order[:3]) + list(order[len(order)//2 - 1:len(order)//2 + 2]) + list(order[-3:])

fig, axes = plt.subplots(3, 3, figsize=(16, 10))
for ax, k in zip(axes.ravel(), picks):
    th, z, dep = cfg[k]
    m_mlp, m_sir = recreate("mlp", k), recreate("siren", k)
    r_m = align_score(t, Wt[k], t, m_mlp)[1]
    r_s = align_score(t, Wt[k], t, m_sir)[1]
    ax.plot(t, Wt[k] * 1e6, "k", lw=2.4, label=f"FEM truth ({p2p_t[k]*1e6:.2f}µV)")
    ax.plot(t, m_mlp * 1e6, "tab:blue", lw=1.5, label=f"MLP+FF (r={r_m:+.2f})")
    ax.plot(t, m_sir * 1e6, "tab:red", lw=1.2, ls="--", alpha=0.8, label=f"SIREN (r={r_s:+.2f})")
    ax.set_xlim(-10, 55); ax.axhline(0, color="k", lw=0.3); ax.grid(alpha=0.25)
    ax.set_title(f"elec θ={th:.0f}° z={z:.0f} · MU depth {dep:.0f}mm", fontsize=10)
    ax.legend(fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel("t (ms)")
for ax in axes[:, 0]:
    ax.set_ylabel("µV")
fig.suptitle("Recreated MUAPs — learned VC vs FEM truth (held-out electrodes)\n"
             "top row = worst (near-field amplitude bug) · bottom row = best",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "recreated_muaps.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "recreated_muaps.png")
for k in picks:
    th, z, dep = cfg[k]
    print(f"  θ{th:3.0f} z{z:3.0f} d{dep:2.0f}: truth {p2p_t[k]*1e6:6.2f}µV · "
          f"mlp r={sc['r'][k]:+.3f} p2p×{sc['p2p_ratio'][k]:.2f}")
