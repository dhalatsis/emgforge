"""Phase 3.2 — MRI electrode-sweep dataset on the real WR forearm.

Same hybrid recipe the cylinder work landed on: a shared/cached coverage set (cell_ids
computed once — Gate 0b's 18x) PLUS a per-electrode near-field set (uniform in r, so
density ~1/r^2), because uniform-in-volume sampling starves the high-phi regime that
drives MUAPs.

The WR mesh is 10.5k nodes (solve 0.39s) vs the cylinder's 194k (7.7s), so MRI datasets
are ~20x cheaper — a real data-scaling study is affordable locally.

Run: PYTHONPATH=src python scripts/neural_field/12_gen_mri_dataset.py [--n 256]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from dolfinx import geometry

from emgforge.mri.core.fem_solver import MRIFEMModel

MUSCLE_TAG = 4      # mesh cell marker for muscle

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
MESH = ROOT / "_results/sanity/fem_cache/forearm_WR.msh"
CFG = ROOT / "_results/sanity/fem_cache/forearm_WR_fibers.json"
OUT = ROOT / "_results/neural_field"


def _in_muscle(fem, pts):
    """True where a point falls in a MUSCLE cell. The MUAP only queries phi along fibres,
    which live in muscle — phi in skin/fat/bone is NEVER used. Sampling it wastes capacity
    and inflates the dynamic range 762x -> 51x (the near-electrode skin peak, sigma=4.55e-4)."""
    lf = fem._leadfield
    c = geometry.compute_closest_entity(lf.tree, lf.midpoints, lf.mesh, pts).squeeze()
    return fem.cell_markers.values[c] == MUSCLE_TAG


def sample_muscle_points(fem, X, n, rng):
    """Uniform coverage of the MUSCLE only."""
    lo, hi = X.min(0), X.max(0)
    out = []
    while sum(len(o) for o in out) < n:
        p = rng.uniform(lo, hi, size=(int(6 * n), 3))
        out.append(p[_in_muscle(fem, p)])
    return np.concatenate(out)[:n]


def sample_near_muscle(elec, n, rng, fem, r_max=35.0):
    """Near the electrode AND in muscle. The electrode is on the SKIN, so the nearest
    muscle is ~7mm down — these are the highest-phi points a MUAP can actually see.
    (A naive near-electrode sampler puts 86% of its points in fat/skin.)"""
    out = []
    while sum(len(o) for o in out) < n:
        m = int(8 * n)
        r = rng.uniform(0.5, r_max, m)                  # uniform in r => density ~1/r^2
        u, v = rng.uniform(-1, 1, m), rng.uniform(0, 2 * np.pi, m)
        s = np.sqrt(1 - u ** 2)
        p = elec + np.column_stack([r * s * np.cos(v), r * s * np.sin(v), r * u])
        out.append(p[_in_muscle(fem, p)])
    return np.concatenate(out)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="electrodes (WR solves are ~0.39s)")
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--near", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    fem = MRIFEMModel(str(MESH), fiber_config=str(CFG), nifti_path=str(SEG),
                      skin_shell_mm=1.5, sigma_mode="centerline")
    X = fem.mesh.geometry.x
    print(f"model built {time.time()-t0:.1f}s · mesh {X.shape[0]} nodes")
    rng = np.random.default_rng(a.seed)
    pts = sample_muscle_points(fem, X, a.points, rng)
    lf = fem._leadfield
    tc = time.time()
    cids = geometry.compute_closest_entity(lf.tree, lf.midpoints, lf.mesh, pts).squeeze()
    print(f"cell_ids cached for {a.points} shared pts in {time.time()-tc:.1f}s")

    # electrodes: full coverage around the arm, over the muscle-bearing z-band
    rng2 = np.random.default_rng(a.seed + 1)
    th = rng2.uniform(0, 360, a.n)
    zf = rng2.uniform(0.25, 0.75, a.n)
    elecs = np.array([fem.get_skin_surface_point(t_, z_) for t_, z_ in zip(th, zf)])

    PHI = np.zeros((a.n, a.points), np.float32)
    NP = np.zeros((a.n, a.near, 3), np.float32) if a.near else None
    NPHI = np.zeros((a.n, a.near), np.float32) if a.near else None
    rng3 = np.random.default_rng(a.seed + 2)
    for i, e in enumerate(elecs):
        uh = fem.solve_for_point(e, source_sigma=5.0)
        PHI[i] = np.asarray(uh.eval(pts, cids)).reshape(-1)
        if a.near:
            npi = sample_near_muscle(e, a.near, rng3, fem)
            NP[i] = npi
            NPHI[i] = fem.evaluate_solution_at_points(npi, uh)
        if (i + 1) % 32 == 0:
            print(f"  {i+1}/{a.n} electrodes ({time.time()-t0:.0f}s)")

    out = OUT / f"mri_elec_{a.n}{'_near' if a.near else ''}_muscle.npz"
    d = dict(points=pts.astype(np.float32), electrodes=elecs.astype(np.float32), phi=PHI,
             elec_theta=th, elec_zfrac=zf)
    if a.near:
        d.update(near_points=NP, near_phi=NPHI)
    np.savez_compressed(out, **d)
    print(f"\n{a.n} electrodes × {a.points} shared + {a.near} near")
    print(f"phi shared [{PHI.min():.3e}, {PHI.max():.3e}]")
    if a.near:
        print(f"phi near   [{NPHI.min():.3e}, {NPHI.max():.3e}]  <- the high-phi regime")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
