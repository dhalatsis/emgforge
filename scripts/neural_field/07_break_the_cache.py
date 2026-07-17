"""Gate 0b, part 2 — WHEN does cell_id caching break, and what does the FEM then cost?

Gate 0b measured the FIXED-anatomy electrode sweep, where caching applies and the FEM is
0.43 s/electrode (surrogate only ~3x). But caching only holds while (mesh, query points)
are fixed. Vary the ANATOMY and the cache is void AND you additionally pay mesh generation
+ sigma assembly — costs the electrode sweep amortises to zero.

That is the regime the science actually lives in (pennation / fat sweeps = the prior work's
Exp 2-3; cross-subject; inverse problems where geometry is the variable). Measure it.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from dolfinx import geometry

from emgforge.fem import FEMModel
from emgforge.fem.conductivity import TissueTable
from emgforge.fem.geometry import ParametricGeometry

ROOT = Path(__file__).resolve().parents[2]
CACHED = ROOT / "_results/sanity/fem_cache/cyl_10_35_38_40.msh"
N_PTS = 20_000


def timed(f):
    t = time.time(); r = f(); return r, time.time() - t


def main():
    rng = np.random.default_rng(0)
    r = 38.0 * np.sqrt(rng.uniform(0, 1, N_PTS)); th = rng.uniform(0, 2 * np.pi, N_PTS)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th), rng.uniform(10, 230, N_PTS)])

    print("=" * 74)
    print("REGIME A — FIXED anatomy, sweep electrodes  (what Gate 0b measured)")
    print("=" * 74)
    fem, t_build = timed(lambda: FEMModel(str(CACHED), conductivity=TissueTable.analytical(),
                                          source_sigma=3.0))
    lf = fem._leadfield
    cids, t_loc = timed(lambda: geometry.compute_closest_entity(lf.tree, lf.midpoints,
                                                                lf.mesh, pts).squeeze())
    g = ParametricGeometry(r_bone=10., r_muscle=35., r_fat=38., r_skin=40., length=240.)
    ts = []
    for k in range(3):
        _, dt = timed(lambda: fem.solve_for_point(g.electrode_on_skin(k * 30.0, 120.0)))
        ts.append(dt)
    t_solve = float(np.median(ts))
    _, t_eval = timed(lambda: np.asarray(fem.uh.eval(pts, cids)).reshape(-1))
    print(f"  mesh generation      : 0 s        (reused from cache)")
    print(f"  FEMModel build       : {t_build:6.1f} s   (once, amortised)")
    print(f"  cell_ids location    : {t_loc:6.1f} s   (once, amortised — the cache)")
    print(f"  --- per electrode ---")
    print(f"  solve                : {t_solve:6.2f} s")
    print(f"  eval (cached cids)   : {t_eval:6.3f} s")
    amort_A = t_solve + t_eval
    print(f"  => marginal cost     : {amort_A:6.2f} s / electrode")

    print()
    print("=" * 74)
    print("REGIME B — VARYING anatomy (pennation / fat / subject)  → THE CACHE IS VOID")
    print("=" * 74)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # a different anatomy: thicker fat layer (the prior work's Exp 3 axis)
        g2 = ParametricGeometry(r_bone=10., r_muscle=35., r_fat=39.5, r_skin=42.,
                                length=240.)
        (msh, _), t_mesh = timed(lambda: g2.build(d / "v.msh", d / "v.json", char_length=0.3))
        fem2, t_build2 = timed(lambda: FEMModel(str(msh), conductivity=TissueTable.analytical(),
                                                source_sigma=3.0))
        lf2 = fem2._leadfield
        _, t_loc2 = timed(lambda: geometry.compute_closest_entity(lf2.tree, lf2.midpoints,
                                                                  lf2.mesh, pts).squeeze())
        _, t_solve2 = timed(lambda: fem2.solve_for_point(g2.electrode_on_skin(0.0, 120.0)))
        print(f"  mesh generation      : {t_mesh:6.1f} s   <- NEW: cannot be cached")
        print(f"  FEMModel build       : {t_build2:6.1f} s   <- NEW: sigma re-assembled")
        print(f"  cell_ids location    : {t_loc2:6.1f} s   <- NEW: new mesh => cache VOID")
        print(f"  solve                : {t_solve2:6.2f} s")
        total_B = t_mesh + t_build2 + t_loc2 + t_solve2
        print(f"  => cost for ONE sample of a new anatomy : {total_B:6.1f} s")

    print()
    print("=" * 74)
    print("THE SPEED ARGUMENT, HONESTLY")
    print("=" * 74)
    net = 0.15
    print(f"  a conditioned net costs ~{net:.2f} s/sample either way (it just evaluates)")
    print()
    print(f"  Regime A (fixed anatomy, sweep electrodes):")
    print(f"     FEM {amort_A:6.2f} s  vs net {net:.2f} s  ->  {amort_A/net:5.1f}x   ← weak; caching wins")
    print(f"  Regime B (vary anatomy — pennation/fat/subject/inverse):")
    print(f"     FEM {total_B:6.1f} s  vs net {net:.2f} s  ->  {total_B/net:5.0f}x   ← the real case")
    print()
    print(f"  ratio between regimes: {total_B/amort_A:.0f}x  — the cache is doing ALL the work in A,")
    print(f"  and NONE of it in B. The surrogate's speed case was never about A.")


if __name__ == "__main__":
    main()
