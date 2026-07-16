"""Phase 1 — build the FROZEN MUAP benchmark on the cylinder.

The scoreboard every learned-VC model is judged on. Models are NOT ranked by phi error
(R^2=0.999 already coexisted with bad MUAPs, because SFAP ∝ phi''); they are ranked by
whether they reproduce these MUAPs and their physiological properties.

Fixed configs = {MU depth} x {electrode (theta, z)}. For each: FEM phi along the MU's
fibres -> field_to_muap -> waveform + properties. Everything seeded; rerunning must
reproduce it byte-for-byte (that is Gate 1).

Run: PYTHONPATH=src python scripts/neural_field/01_build_muap_bench.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from emgforge.fem import FEMModel
from emgforge.fem.conductivity import TissueTable
from emgforge.fem.geometry import ParametricGeometry
from emgforge.synthesis import FibreBed, SpatialConfig, field_to_muap
from emgforge.synthesis.metrics import jaggedness, lobe_metrics

ROOT = Path(__file__).resolve().parents[2]
MESH = ROOT / "_results/sanity/fem_cache/cyl_10_35_38_40.msh"
OUT = ROOT / "_results/neural_field"

# ---- the FROZEN spec (changing any of this invalidates the benchmark) ----
CYL = dict(r_bone=10.0, r_muscle=35.0, r_fat=38.0, r_skin=40.0, length=240.0)
MU_DEPTHS = (8.0, 15.0, 25.0)                 # mm below skin (r=32,25,15 → inside muscle)
ELECTRODES = tuple((th, z) for th in (0.0, 20.0, 40.0) for z in (90.0, 120.0, 150.0))
N_FIB, MU_RADIUS_MM, MU_SEED = 20, 2.0, 7
NZ, DZ, Z_CENTROID = 200, 1.07, 120.0          # fibre: 200 pts × 1.07mm = 213mm
LP, LD, V = 65.0, 148.0, 4.0                   # MU-113-like asymmetric geometry
POSZ = (LP - LD) / 2.0                         # = -41.5mm: IZ offset from the fibre centre
SPCFG = SpatialConfig(denoise="monopole", denoise_n_poles=3, fiber_window="one_sided",
                      tukey_alpha=0.25, csd_derivative=2, upsample_factor=2, fsamp=2048.0,
                      w=256, edge_taper_left=5, edge_taper_right=10, t_start_ms=-10.0, v=V)


def mu_fibre_paths(g, depth):
    """N_FIB fibres in a disk of MU_RADIUS_MM around the point `depth` below the skin at θ=0."""
    x0, y0 = g.fibre_xy_below_skin(depth, 0.0)
    rng = np.random.default_rng(MU_SEED)
    r = MU_RADIUS_MM * np.sqrt(rng.uniform(0, 1, N_FIB))
    a = rng.uniform(0, 2 * np.pi, N_FIB)
    paths = [g.fibre_points(x0 + ri * np.cos(ai), y0 + ri * np.sin(ai), Z_CENTROID, NZ, DZ)[0]
             for ri, ai in zip(r, a)]
    return np.asarray(paths)                    # (N_FIB, NZ, 3)


def muap_properties(t, m):
    trough, before, after = lobe_metrics(t, m, win_ms=12.0)
    return dict(p2p=float(m.ptp()), duration_ms=0.0, jaggedness=float(jaggedness(m)),
                eof=float(after), latency=float(t[np.argmax(np.abs(m))]), trough=float(trough))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    g = ParametricGeometry(**CYL)
    print(f"cylinder FEM ({MESH.name})...")
    fem = FEMModel(str(MESH), conductivity=TissueTable.analytical(), source_sigma=3.0)
    print(f"  model built in {time.time()-t0:.1f}s")

    beds = {d: mu_fibre_paths(g, d) for d in MU_DEPTHS}
    sb = FibreBed.from_arrays(DZ, LP, LD, POSZ, V)           # scalars broadcast to 1 fibre...
    sb = FibreBed.from_arrays(np.full(N_FIB, DZ), np.full(N_FIB, LP),
                              np.full(N_FIB, LD), np.full(N_FIB, POSZ), np.full(N_FIB, V))

    cfgs, waves, props, t_ms = [], [], [], None
    for ei, (th, z) in enumerate(ELECTRODES):
        elec = g.electrode_on_skin(th, z)
        ts = time.time(); uh = fem.solve_for_point(elec); t_solve = time.time() - ts
        for d in MU_DEPTHS:
            phi = np.array([fem.evaluate_solution_at_points(p, uh) for p in beds[d]])
            res = field_to_muap(phi, sb, SPCFG)
            if t_ms is None:
                t_ms = res.t_ms
            pr = muap_properties(res.t_ms, res.muap)
            pr["duration_ms"] = float(res.metrics.get("duration_ms", 0.0))
            cfgs.append((th, z, d)); waves.append(res.muap); props.append(pr)
        print(f"  elec {ei+1}/{len(ELECTRODES)} (θ={th:.0f}°,z={z:.0f}) solved {t_solve:.1f}s "
              f"· {len(MU_DEPTHS)} MUs · {time.time()-t0:.0f}s total")

    cfgs = np.array(cfgs, dtype=float)          # (K, 3): theta, z, depth
    waves = np.array(waves)                     # (K, w)
    keys = ("p2p", "duration_ms", "jaggedness", "eof", "latency", "trough")
    P = {k: np.array([p[k] for p in props]) for k in keys}
    np.savez_compressed(OUT / "muap_bench_cyl.npz", t_ms=t_ms, muap=waves, cfg=cfgs,
                        cfg_cols=np.array(["elec_theta_deg", "elec_z_mm", "mu_depth_mm"]), **P)

    print(f"\n=== FROZEN BENCHMARK: {len(cfgs)} configs ({len(ELECTRODES)} electrodes × "
          f"{len(MU_DEPTHS)} MU depths) ===")
    print(f"{'θ':>4} {'z':>5} {'depth':>6} | {'p2p µV':>8} {'dur ms':>7} {'lat ms':>7} "
          f"{'jagged':>7} {'EOF':>6}")
    print("-" * 64)
    for c, k in zip(cfgs, range(len(cfgs))):
        print(f"{c[0]:4.0f} {c[1]:5.0f} {c[2]:6.0f} | {P['p2p'][k]*1e6:8.2f} "
              f"{P['duration_ms'][k]:7.1f} {P['latency'][k]:7.1f} {P['jaggedness'][k]:7.4f} "
              f"{P['eof'][k]:6.2f}")
    print(f"\nwrote {OUT/'muap_bench_cyl.npz'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
