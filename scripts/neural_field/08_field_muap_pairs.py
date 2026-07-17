"""Pair the recreated MUAPs with the recreated FIELDS.

For selected benchmark configs: FEM-truth phi(z) along the MU's fibres vs the learned VC's
phi, and directly beneath it the MUAP each one produces. Shows the causal chain
field-error -> MUAP-error, and should expose WHY the near-field configs fail.

Needs an FEM solve per shown config (the benchmark stored MUAPs, not phi).
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

from emgforge.fem import FEMModel
from emgforge.fem.conductivity import TissueTable
from emgforge.fem.geometry import ParametricGeometry
from emgforge.synthesis import FibreBed, field_to_muap
from emgforge.synthesis.metrics import align_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/neural_field"
MESH = ROOT / "_results/sanity/fem_cache/cyl_10_35_38_40.msh"
dev = "cuda" if torch.cuda.is_available() else "cpu"

b = np.load(OUT / "muap_bench_cyl.npz")
t, Wt, cfg, p2p_t = b["t_ms"], b["muap"], b["cfg"], b["p2p"]
sc = np.load(OUT / "score_cyl_mlp.npz")
g = ParametricGeometry(**S.CYL)
net, ck = S.load(str(OUT / "cyl_mlp.pt"), dev)
sb = FibreBed.from_arrays(np.full(S.N_FIB, S.DZ), np.full(S.N_FIB, S.LP),
                          np.full(S.N_FIB, S.LD), np.full(S.N_FIB, S.POSZ), np.full(S.N_FIB, S.V))

order = np.argsort(sc["r"])
picks = [int(order[0]), int(order[1]), int(order[-2]), int(order[-1])]   # 2 worst, 2 best
print("solving FEM for the shown configs...")
fem = FEMModel(str(MESH), conductivity=TissueTable.analytical(), source_sigma=3.0)

fig, axes = plt.subplots(2, len(picks), figsize=(5.0 * len(picks), 8))
zc = (np.arange(S.NZ) - S.NZ // 2) * S.DZ
for col, k in enumerate(picks):
    th, z, dep = cfg[k]
    paths = S.mu_fibre_paths(g, dep)
    uh = fem.solve_for_point(g.electrode_on_skin(th, z))
    phi_true = np.array([fem.evaluate_solution_at_points(p, uh) for p in paths])
    elec = g.electrode_on_skin(th, z)
    phi_pred = np.array([S.predict_phi(net, ck, dev, p, elec) for p in paths])
    m_pred = field_to_muap(phi_pred, sb, S.SPCFG).muap
    r = align_score(t, Wt[k], t, m_pred)[1]

    # --- top: the FIELD (mean over the MU's fibres) ---
    a = axes[0, col]
    a.plot(zc, phi_true.mean(0) * 1e3, "k", lw=2.4, label="FEM truth φ")
    a.plot(zc, phi_pred.mean(0) * 1e3, "tab:blue", lw=1.6, label="learned φ")
    err = np.abs(phi_pred.mean(0) - phi_true.mean(0)).max() / (np.abs(phi_true.mean(0)).max() + 1e-30)
    a.set_title(f"θ={th:.0f}° z={z:.0f} · depth {dep:.0f}mm\nφ peak {phi_true.mean(0).max()*1e3:.1f}"
                f" vs {phi_pred.mean(0).max()*1e3:.1f} mV · max err {err*100:.0f}%", fontsize=9)
    a.set_xlabel("z along fibre (mm)"); a.set_ylabel("mean φ (mV)")
    a.legend(fontsize=7); a.grid(alpha=0.3)

    # --- bottom: the MUAP it produces ---
    a = axes[1, col]
    a.plot(t, Wt[k] * 1e6, "k", lw=2.4, label=f"FEM MUAP ({p2p_t[k]*1e6:.2f}µV)")
    a.plot(t, m_pred * 1e6, "tab:blue", lw=1.6, label=f"learned ({m_pred.ptp()*1e6:.2f}µV, r={r:+.2f})")
    a.set_xlim(-10, 55); a.axhline(0, color="k", lw=0.3)
    a.set_xlabel("t (ms)"); a.set_ylabel("µV"); a.legend(fontsize=7); a.grid(alpha=0.3)
    print(f"  θ{th:3.0f} z{z:3.0f} d{dep:2.0f}: φ peak {phi_true.mean(0).max()*1e3:6.2f} → "
          f"{phi_pred.mean(0).max()*1e3:6.2f} mV ({err*100:5.1f}% err) | "
          f"MUAP {p2p_t[k]*1e6:5.2f} → {m_pred.ptp()*1e6:5.2f} µV, r={r:+.3f}")

fig.suptitle("Recreated FIELD (top) → the MUAP it produces (bottom)  ·  "
             "left 2 = worst configs, right 2 = best", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "field_muap_pairs.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "field_muap_pairs.png")
