"""Phase 2.2 — cylinder electrode-sweep dataset for the learned VC.

u(x,y,z | electrode_xyz) -> phi. One FEM solve per electrode; phi sampled at a FIXED
volume point set so `cell_ids` is computed once and reused (Gate 0b: that is 99.5% of
the sampling cost). The benchmark's 9 electrodes are NOT in the training set — they are
the held-out generalisation test.

Run: PYTHONPATH=src python scripts/neural_field/02_gen_cyl_dataset.py [--n 64]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from dolfinx import geometry

from emgforge.fem import FEMModel
from emgforge.fem.conductivity import TissueTable
from emgforge.fem.geometry import ParametricGeometry

ROOT = Path(__file__).resolve().parents[2]
MESH = ROOT / "_results/sanity/fem_cache/cyl_10_35_38_40.msh"
OUT = ROOT / "_results/neural_field"
CYL = dict(r_bone=10.0, r_muscle=35.0, r_fat=38.0, r_skin=40.0, length=240.0)
# training electrodes live in this band; the benchmark's (θ∈{0,20,40}, z∈{90,120,150}) are held out
Z_LO, Z_HI = 60.0, 180.0


def sample_volume_points(n, seed=0):
    """Uniform in the cylinder out to the fat boundary, spanning the fibre z-range."""
    rng = np.random.default_rng(seed)
    r = 38.0 * np.sqrt(rng.uniform(0, 1, n))          # area-uniform
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(10.0, 230.0, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th), z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="training electrodes")
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    g = ParametricGeometry(**CYL)
    fem = FEMModel(str(MESH), conductivity=TissueTable.analytical(), source_sigma=3.0)
    print(f"model built {time.time()-t0:.1f}s")

    pts = sample_volume_points(a.points, a.seed)
    lf = fem._leadfield
    tc = time.time()
    cids = geometry.compute_closest_entity(lf.tree, lf.midpoints, lf.mesh, pts).squeeze()
    print(f"cell_ids cached for {a.points} fixed points in {time.time()-tc:.1f}s "
          f"(reused for all {a.n} electrodes)")

    rng = np.random.default_rng(a.seed + 1)
    th = rng.uniform(0, 360, a.n)
    z = rng.uniform(Z_LO, Z_HI, a.n)
    elecs = np.array([g.electrode_on_skin(t_, z_) for t_, z_ in zip(th, z)])

    PHI = np.zeros((a.n, a.points), dtype=np.float32)
    for i, e in enumerate(elecs):
        uh = fem.solve_for_point(e)
        PHI[i] = np.asarray(uh.eval(pts, cids)).reshape(-1)
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{a.n} electrodes ({time.time()-t0:.0f}s)")

    out = OUT / f"cyl_elec_{a.n}.npz"
    np.savez_compressed(out, points=pts.astype(np.float32), electrodes=elecs.astype(np.float32),
                        phi=PHI, elec_theta=th, elec_z=z)
    print(f"\n{a.n} electrodes × {a.points} pts · phi range [{PHI.min():.3e}, {PHI.max():.3e}]")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
