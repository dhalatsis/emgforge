"""HD-EMG: an M×M grid of skin electrodes over FCU, each cell showing the MUAP that
electrode records for a motor unit. Rows run along the arm (fibre direction → you see
the AP propagate / the IZ), columns run around the arm (transverse amplitude footprint).

The 25 FEM reciprocity solves are done ONCE and φ is sampled along every bed fibre, so
several MUs (``--mus 85,46,26``) reuse the same lead fields. One figure per MU.

Run: python scripts/mri/mu_electrode_grid.py [--m 5] [--mus 99] [--dtheta 15]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core.motor_unit_pool import sample_henneman_pool
from emgforge.mri.core.fem_solver import MRIFEMModel
from emgforge.synthesis import FibreBed, field_to_muap, production_config

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
MESH = ROOT / "_results/sanity/fem_cache/forearm_WR.msh"
CFG = ROOT / "_results/sanity/fem_cache/forearm_WR_fibers.json"
OUT = ROOT / "_results/mu_pool/electrode_grid"
LABELS = json.load(open(ROOT / "src/emgforge/mri/data/pd_lab_labels.json"))["common_labels"]
IZ_FRAC = 0.305
# the production recipe (MRI regime fs=2048, v=4, w=256) — defined once in emgforge.synthesis
SPCFG = production_config(fs=2048.0, v=4.0, w=256)


def mu_bed(bed, mu, arc_dz, L_fib):
    """The synthesis FibreBed for one MU: IZ at 0.305 → asymmetric Lp/Ld, posz=(Lp-Ld)/2."""
    idx = mu.fiber_idxs
    rng = np.random.default_rng(int(mu.idx))
    izf = np.clip(IZ_FRAC + rng.normal(0, 0.02, mu.size), 0.1, 0.9)
    Lp, Ld = izf * L_fib[idx], (1 - izf) * L_fib[idx]
    return FibreBed.from_arrays(arc_dz[idx], Lp, Ld, (Lp - Ld) / 2, 4.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--mus", default="99", help="comma-separated pool MU indices")
    ap.add_argument("--muscle", type=int, default=8, help="segmentation label (8=FCU, 11=Brachioradialis, …)")
    ap.add_argument("--dtheta", type=float, default=15.0)
    ap.add_argument("--zlo", type=float, default=0.30)
    ap.add_argument("--zhi", type=float, default=0.70)
    args = ap.parse_args()
    M = args.m
    lbl = args.muscle
    mname = LABELS.get(str(lbl), {}).get("name", f"L{lbl}")
    mus_req = [int(x) for x in args.mus.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    fm = MuscleFiberModel(str(SEG)); fm.estimate_centerlines(); fm.estimate_cross_sections()
    bed = build_muscle_beds(fm, density=4.0, method="poisson", labels=[lbl], min_fibers=30)[lbl]
    print(f"muscle {lbl} = {mname}: {len(bed.r_norms)} fibres in the bed")
    N, Nz = bed.paths.shape[:2]
    pool = sample_henneman_pool(bed, n_mu=100, size_min=5, size_max=min(400, N), seed=0)
    seg = np.linalg.norm(np.diff(bed.paths, axis=1), axis=2)
    arc_dz, L_fib = seg.mean(1), seg.sum(1)

    # ---- lead fields for this grid geometry: solve once, then cache on disk ----
    # (the bed is deterministic, so the 25 φ-grids only depend on M/θ/z → reusable
    # across MUs and across runs; a new MU then costs only its own MUAP sum.)
    cache = OUT / f"_phigrid_L{lbl}_M{M}_dt{args.dtheta:.0f}_z{args.zlo:.2f}-{args.zhi:.2f}_N{N}.npz"
    if cache.exists():
        d = np.load(cache)
        phi_grid, elec_xyz, fcu_ang = d["phi_grid"], d["elec_xyz"], float(d["fcu_ang"])
        print(f"loaded cached lead fields: {cache.name}")
    else:
        fem = MRIFEMModel(str(MESH), fiber_config=str(CFG), nifti_path=str(SEG),
                          skin_shell_mm=1.5, sigma_mode="centerline")
        limb = np.array([fem.mesh.geometry.x[:, 0].mean(), fem.mesh.geometry.x[:, 1].mean()])
        fcu_ang = float(np.degrees(np.arctan2(bed.centroid_xy[1] - limb[1], bed.centroid_xy[0] - limb[0])))
        thetas = fcu_ang + np.linspace(-args.dtheta, args.dtheta, M)
        zfracs = np.linspace(args.zlo, args.zhi, M)
        print(f"solving {M*M} electrodes over {mname} (θ={fcu_ang:.0f}°±{args.dtheta:.0f}, "
              f"z {args.zlo:.2f}–{args.zhi:.2f}), sampling φ for {N} fibres each...")
        phi_grid = np.zeros((M, M, N, Nz)); elec_xyz = np.zeros((M, M, 3))
        for i, zf in enumerate(zfracs):
            for j, th in enumerate(thetas):
                elec = fem.get_skin_surface_point(th, zf); elec_xyz[i, j] = elec
                # source_sigma=5.0 mm is the legacy FEM default kept so this cache stays
                # comparable with the released lead fields; the paper finds 5 mm narrows
                # the lateral footprint (FWHM 25 vs 40 mm analytical) and 1 mm matches.
                fem.solve_for_point(elec, source_sigma=5.0)
                phi_grid[i, j] = np.array([fem.evaluate_solution_at_points(p) for p in bed.paths])
            print(f"  row {i+1}/{M} solved ({time.time()-t0:.0f}s)")
        np.savez_compressed(cache, phi_grid=phi_grid, elec_xyz=elec_xyz, fcu_ang=fcu_ang)
    ied_z = np.median(np.linalg.norm(np.diff(elec_xyz, axis=0), axis=2))
    ied_t = np.median(np.linalg.norm(np.diff(elec_xyz, axis=1), axis=2))

    # ---- per MU: MUAP grid + figure (reusing the solves) ----
    t_ms = None
    for mu_idx in mus_req:
        mu = pool[mu_idx]; idx = mu.fiber_idxs
        sb = mu_bed(bed, mu, arc_dz, L_fib)
        W = np.zeros((M, M, SPCFG.w))
        for i in range(M):
            for j in range(M):
                res = field_to_muap(phi_grid[i, j][idx], sb, SPCFG)
                W[i, j] = res.muap
                t_ms = res.t_ms if t_ms is None else t_ms
        p2p = W.ptp(axis=2)
        print(f"  MU{mu_idx}: {mu.size} fibres · p2p {p2p.min()*1e6:.1f}–{p2p.max()*1e6:.1f} µV")
        _plot(lbl, mname, mu_idx, mu.size, M, t_ms, W, p2p, ied_z, ied_t)
    print(f"done in {time.time()-t0:.0f}s → {OUT}")


def _plot(lbl, mname, mu_idx, mu_size, M, t_ms, W, p2p, ied_z, ied_t):
    ymax = np.abs(W).max() * 1e6
    peak = np.unravel_index(np.argmax(p2p), p2p.shape)
    fig, axes = plt.subplots(M, M, figsize=(2.3 * M, 2.0 * M), sharex=True, sharey=True)
    for i in range(M):
        for j in range(M):
            ax = axes[i, j]
            ax.plot(t_ms, W[i, j] * 1e6, lw=1.1,
                    color="tab:red" if (i, j) == peak else "tab:blue")
            ax.set_ylim(-ymax, ymax); ax.axhline(0, color="k", lw=0.3)
            ax.set_xlim(-10, 60); ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.03, 0.9, f"{p2p[i, j]*1e6:.1f}µV", transform=ax.transAxes, fontsize=7)
    axes[-1, M // 2].set_xlabel("→ around arm →", fontsize=9)
    axes[M // 2, 0].set_ylabel("← along arm ←", fontsize=9)
    fig.suptitle(f"HD-EMG grid ({M}×{M}, IED ≈ {ied_z:.0f}×{ied_t:.0f} mm) — {mname} · MU{mu_idx} "
                 f"({mu_size} fibres), monopolar MUAP per electrode · shared ±{ymax:.1f}µV",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"grid_L{lbl}_mu{mu_idx}_{M}x{M}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
