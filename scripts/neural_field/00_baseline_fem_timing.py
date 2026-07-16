"""Phase 0.3 — re-baseline the FEM cost per electrode on the WR anatomy.

The closed cylinder work claimed ~75x (150 ms/electrode net vs ~13 s/electrode FEM).
The FEM has since got much faster, so the speed justification for a learned surrogate
needs re-measuring BEFORE we invest in one. Measures the three costs separately:
model build (one-time), solve per electrode, and phi sampling per query set.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core.fem_solver import MRIFEMModel

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
MESH = ROOT / "_results/sanity/fem_cache/forearm_WR.msh"
CFG = ROOT / "_results/sanity/fem_cache/forearm_WR_fibers.json"
N_ELEC = 8


def main():
    print("building FCU bed (for a realistic fibre query set)...")
    fm = MuscleFiberModel(str(SEG)); fm.estimate_centerlines(); fm.estimate_cross_sections()
    bed = build_muscle_beds(fm, density=4.0, method="poisson", labels=[8], min_fibers=30)[8]
    fibre_pts = bed.paths.reshape(-1, 3)                     # 637*200 = 127k points
    rng = np.random.default_rng(0)
    X = None

    t0 = time.time()
    fem = MRIFEMModel(str(MESH), fiber_config=str(CFG), nifti_path=str(SEG),
                      skin_shell_mm=1.5, sigma_mode="centerline")
    t_build = time.time() - t0
    X = fem.mesh.geometry.x
    limb = np.array([X[:, 0].mean(), X[:, 1].mean()])
    th = float(np.degrees(np.arctan2(bed.centroid_xy[1] - limb[1], bed.centroid_xy[0] - limb[0])))

    # volume query set of the size a neural-field dataset would use
    lo, hi = X.min(0), X.max(0)
    vol_pts = rng.uniform(lo, hi, size=(50_000, 3))

    t_solve, t_fib, t_vol = [], [], []
    for k in range(N_ELEC):
        elec = fem.get_skin_surface_point(th + (k - N_ELEC / 2) * 4.0, 0.5)
        t = time.time(); fem.solve_for_point(elec, source_sigma=5.0); t_solve.append(time.time() - t)
        t = time.time(); fem.evaluate_solution_at_points(fibre_pts); t_fib.append(time.time() - t)
        t = time.time(); fem.evaluate_solution_at_points(vol_pts); t_vol.append(time.time() - t)

    s, f, v = np.median(t_solve), np.median(t_fib), np.median(t_vol)
    print("\n================ FEM cost on WR (median of %d electrodes) ================" % N_ELEC)
    print(f"  model build (ONE-TIME)          : {t_build:6.2f} s")
    print(f"  solve_for_point   / electrode   : {s:6.3f} s")
    print(f"  phi @ 127k fibre pts / electrode: {f:6.3f} s")
    print(f"  phi @  50k volume pts/ electrode: {v:6.3f} s")
    print(f"  --> MUAP-ready per electrode    : {s+f:6.3f} s   (solve + fibre sampling)")
    print(f"  --> dataset-gen per electrode   : {s+v:6.3f} s   (solve + volume sampling)")
    print("\n---------------- vs the closed cylinder claim ----------------")
    print(f"  prior FEM baseline              : ~13.0 s/electrode")
    print(f"  prior net inference             : ~0.15 s/electrode  (claimed 75x)")
    print(f"  ACTUAL FEM now (MUAP-ready)     : {s+f:6.3f} s/electrode")
    print(f"  => a 0.15 s net would now be    : {(s+f)/0.15:6.1f}x  (not 75x)")
    print(f"  => dataset of 2048 electrodes   : {(s+v)*2048/60:6.1f} min")
    print("\nRead: if the multiple is small, the surrogate must be justified by")
    print("differentiability / mesh-free inference / amortisation, NOT raw speed.")


if __name__ == "__main__":
    main()
