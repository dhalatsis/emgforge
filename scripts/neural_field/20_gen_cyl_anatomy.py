"""Phase 5.1 — parametrized cylinder dataset for ANATOMY conditioning.

One (r_fat, pennation) anatomy per call, N electrodes. The learned VC will condition on
(query_xyz, electrode_xyz, r_fat, pennation) and be tested on fat/pennation held out from
training — the 594x "varying anatomy" regime (Gate 0b), not the cached electrode sweep.

Two anatomy axes:
  * fat      — r_fat changes the MESH (new gmsh build per value; the dominant cost).
  * pennation— rotates the muscle sigma tensor (same mesh, new sigma-assembly + solve). The
               fibre query paths must rotate by the same angle at scoring (pennation_rotation
               _matrix is the single source of truth for the convention).

Muscle-only + hybrid near-field sampling, as the MRI work landed on. Muscle is analytic here
(r_bone < r < r_muscle), so no cell-marker lookup — cheaper than the MRI sampler.

Run: PYTHONPATH=src python scripts/neural_field/20_gen_cyl_anatomy.py \
        --r_fat 38 --pennation 10 --n_elec 64 --out _results/neural_field/anat
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

# fixed cylinder tiers; fat and pennation are the swept knobs
R_BONE, R_MUSCLE, SKIN_MM, LENGTH = 10.0, 35.0, 2.0, 240.0
Z_LO, Z_HI = 60.0, 180.0            # electrode + fibre z-band (mm)
CHAR_LENGTH = 0.3                    # gmsh; MUST match across anatomies or resolution confounds


def geom(r_fat: float) -> ParametricGeometry:
    # r_fat is the OUTER fat boundary; it must sit outside the muscle or the fat shell is
    # degenerate/inverted and gmsh fails with a cryptic "could not assign volume to layer".
    if r_fat <= R_MUSCLE:
        raise ValueError(f"r_fat={r_fat} must be > r_muscle={R_MUSCLE} (fat thickness "
                         f"{r_fat - R_MUSCLE:g}mm <= 0)")
    return ParametricGeometry(r_bone=R_BONE, r_muscle=R_MUSCLE, r_fat=r_fat,
                              r_skin=r_fat + SKIN_MM, length=LENGTH)


def in_muscle(p: np.ndarray) -> np.ndarray:
    r = np.hypot(p[:, 0], p[:, 1])
    return (r > R_BONE) & (r < R_MUSCLE) & (p[:, 2] > 5.0) & (p[:, 2] < LENGTH - 5.0)


def sample_muscle(n, rng):
    out = []
    while sum(len(o) for o in out) < n:
        r = R_MUSCLE * np.sqrt(rng.uniform(0, 1, 6 * n))          # area-uniform in the disc
        th = rng.uniform(0, 2 * np.pi, 6 * n)
        z = rng.uniform(Z_LO - 20, Z_HI + 20, 6 * n)
        p = np.column_stack([r * np.cos(th), r * np.sin(th), z])
        out.append(p[in_muscle(p)])
    return np.concatenate(out)[:n]


def sample_near_muscle(elec, n, rng, r_max=25.0):
    """Near the electrode AND in muscle (density ~1/r^2): the high-phi regime the MUAP needs."""
    out = []
    while sum(len(o) for o in out) < n:
        m = int(8 * n)
        r = rng.uniform(0.5, r_max, m)
        u, v = rng.uniform(-1, 1, m), rng.uniform(0, 2 * np.pi, m)
        s = np.sqrt(1 - u ** 2)
        p = elec + np.column_stack([r * s * np.cos(v), r * s * np.sin(v), r * u])
        out.append(p[in_muscle(p)])
    return np.concatenate(out)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r_fat", type=float, required=True, help="fat boundary radius (mm)")
    ap.add_argument("--pennation", type=float, required=True, help="fibre pennation (deg)")
    ap.add_argument("--n_elec", type=int, default=64)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--near", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--char_length", type=float, default=CHAR_LENGTH)
    ap.add_argument("--mesh_dir", default="_results/sanity/fem_cache/anat")
    ap.add_argument("--out", default="_results/neural_field/anat")
    a = ap.parse_args()
    t0 = time.time()
    mesh_dir, out_dir = Path(a.mesh_dir), Path(a.out)
    mesh_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)

    g = geom(a.r_fat)
    # mesh is keyed by r_fat ONLY (pennation doesn't change geometry) -> shared across pennations
    msh = mesh_dir / f"cyl_fat{a.r_fat:g}_cl{a.char_length:g}.msh"
    g.build(msh, msh.with_suffix(".json"), char_length=a.char_length)
    print(f"mesh {msh.name} ({time.time()-t0:.0f}s)")

    fem = FEMModel(str(msh), conductivity=TissueTable.analytical_pennated(a.pennation),
                   source_sigma=3.0)
    lf = fem._leadfield
    rng = np.random.default_rng(a.seed)
    pts = sample_muscle(a.points, rng)
    cids = geometry.compute_closest_entity(lf.tree, lf.midpoints, lf.mesh, pts).squeeze()

    rng2 = np.random.default_rng(a.seed + 1)
    th = rng2.uniform(0, 360, a.n_elec)
    z = rng2.uniform(Z_LO, Z_HI, a.n_elec)
    elecs = np.array([g.electrode_on_skin(t_, z_) for t_, z_ in zip(th, z)])

    PHI = np.zeros((a.n_elec, a.points), np.float32)
    NP = np.zeros((a.n_elec, a.near, 3), np.float32)
    NPHI = np.zeros((a.n_elec, a.near), np.float32)
    rng3 = np.random.default_rng(a.seed + 2)
    for i, e in enumerate(elecs):
        fem.solve_for_point(e)
        PHI[i] = np.asarray(fem.uh.eval(pts, cids)).reshape(-1)
        npi = sample_near_muscle(e, a.near, rng3)
        nci = geometry.compute_closest_entity(lf.tree, lf.midpoints, lf.mesh, npi).squeeze()
        NP[i] = npi
        NPHI[i] = np.asarray(fem.uh.eval(npi, nci)).reshape(-1)
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{a.n_elec} ({time.time()-t0:.0f}s)")

    out = out_dir / f"cyl_fat{a.r_fat:g}_pen{a.pennation:g}_n{a.n_elec}.npz"
    np.savez_compressed(
        out, points=pts.astype(np.float32), electrodes=elecs.astype(np.float32), phi=PHI,
        near_points=NP, near_phi=NPHI, elec_theta=th, elec_z=z,
        r_fat=np.float32(a.r_fat), pennation=np.float32(a.pennation),
        r_bone=np.float32(R_BONE), r_muscle=np.float32(R_MUSCLE))
    print(f"\nanatomy r_fat={a.r_fat} pennation={a.pennation}° · {a.n_elec} electrodes")
    print(f"phi shared [{PHI.min():.3e},{PHI.max():.3e}] near [{NPHI.min():.3e},{NPHI.max():.3e}]")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
