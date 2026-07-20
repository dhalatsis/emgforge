"""Phase 4.3 — the MUAP gallery: learned VC (best model) vs FROZEN FEM truth, on real MRI.

Reuses the scorer's exact setup (13_score_bench_mri) so the waveforms are byte-identical to
what Gate 4 scored. Lays the 27 benchmark configs out as 3 MUs (rows: strong/mid/weak) x 9
skin electrodes (cols), each panel overlaying the learned MUAP (blue) on the FEM MUAP (black)
with r and amplitude annotated.

Run: PYTHONPATH=src python scripts/neural_field/15_muap_gallery_mri.py \
        --ckpt _results/neural_field/cyl_mlp_raw_mriN1024.pt
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
S = import_module("04_score_bench")            # load(), predict_phi()
S13 = import_module("13_score_bench_mri")       # mu_synth_bed(), SPCFG, FCU, DENSITY, ...

from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core.motor_unit_pool import sample_henneman_pool
from emgforge.mri.core.fem_solver import MRIFEMModel
from emgforge.synthesis import field_to_muap
from emgforge.synthesis.metrics import align_score

OUT = S13.OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "cyl_mlp_raw_mriN1024.pt"))
    ap.add_argument("--bench", default=str(OUT / "muap_bench_mri.npz"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    b = np.load(a.bench)
    t_ms, Wt, cfg = b["t_ms"], b["muap"], b["cfg"]
    A = b["p2p"] * 1e6                                    # truth amplitude, µV

    # exact scorer setup — deterministic bed + pool so fibre paths match the frozen truth
    fm = MuscleFiberModel(str(S13.SEG)); fm.estimate_centerlines(); fm.estimate_cross_sections()
    bed = build_muscle_beds(fm, density=S13.DENSITY, method="poisson", labels=[S13.FCU],
                            min_fibers=30)[S13.FCU]
    pool = sample_henneman_pool(bed, n_mu=100, size_min=5,
                                size_max=min(400, len(bed.r_norms)), seed=S13.POOL_SEED)
    seg = np.linalg.norm(np.diff(bed.paths, axis=1), axis=2)
    arc_dz, L_fib = seg.mean(1), seg.sum(1)
    fem = MRIFEMModel(str(S13.MESH), fiber_config=str(S13.CFG), nifti_path=str(S13.SEG),
                      skin_shell_mm=1.5, sigma_mode="centerline")
    net, c = S.load(a.ckpt, dev)

    # 27 configs are grouped: 9 electrodes, 3 MUs each (build order). row = MU, col = electrode.
    MUS = [int(m) for m in dict.fromkeys(cfg[:, 2])]      # preserves first-seen order
    n_el = len(cfg) // len(MUS)
    fig, ax = plt.subplots(len(MUS), n_el, figsize=(2.05 * n_el, 2.5 * len(MUS)),
                           sharex=True, squeeze=False)

    rows = []
    for k in range(len(cfg)):
        th, zf, mi, _ = cfg[k]
        mu = pool[int(mi)]
        elec = fem.get_skin_surface_point(th, zf)
        P = bed.paths[mu.fiber_idxs]
        phi = np.array([S.predict_phi(net, c, dev, p, elec) for p in P])
        m = field_to_muap(phi, S13.mu_synth_bed(mu, arc_dz, L_fib), S13.SPCFG).muap
        r = align_score(t_ms, Wt[k], t_ms, m)[1]
        p2p_r = float(np.ptp(m) / (np.ptp(Wt[k]) + 1e-30))
        rows.append((int(mi), A[k], r, p2p_r))

        row, col = MUS.index(int(mi)), k // len(MUS)
        p = ax[row][col]
        p.plot(t_ms, Wt[k] * 1e6, "k", lw=1.9)
        p.plot(t_ms, m * 1e6, "tab:blue", lw=1.3)
        p.axhline(0, color="k", lw=0.3); p.set_xlim(-10, 55)
        det = A[k] >= 1.0
        p.text(0.03, 0.94, f"r={r:+.3f}", transform=p.transAxes, fontsize=7.5,
               va="top", color=("tab:green" if r >= 0.94 else "tab:red"))
        p.text(0.97, 0.94, f"{A[k]:.1f}µV", transform=p.transAxes, fontsize=7.5,
               va="top", ha="right", color=("k" if det else "0.6"))
        p.tick_params(labelsize=6)
        if row == 0:
            p.set_title(f"θ={th:.0f}° z={zf:.2f}", fontsize=7.5)
        if col == 0:
            p.set_ylabel(f"MU {int(mi)} ({mu.size} fib)\nµV", fontsize=8)
        if row == len(MUS) - 1:
            p.set_xlabel("t (ms)", fontsize=7)

    R = np.array([x[2] for x in rows]); Aarr = np.array([x[1] for x in rows])
    w = Aarr / Aarr.sum(); det = Aarr >= 1.0
    fig.suptitle(
        f"Learned VC (black=FEM truth, blue=learned)  ·  {Path(a.ckpt).stem}  ·  "
        f"amp-wtd r {float((w*R).sum()):+.3f} · {(R>=0.94).sum()}/{len(R)} ≥0.94 · "
        f"worst detectable r {R[det].min():+.3f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = OUT / "muap_gallery_mri.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")

    print(f"\n{'MU':>5}{'truth µV':>10}{'r':>9}{'p2p×':>8}")
    print("-" * 32)
    for mi, amp, r, p2p_r in sorted(rows, key=lambda x: -x[1]):
        flag = "" if amp >= 1.0 else "  (sub-µV)"
        print(f"{mi:5d}{amp:10.2f}{r:+9.3f}{p2p_r:8.2f}{flag}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
