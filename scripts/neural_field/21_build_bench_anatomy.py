"""Phase 5.2 — FROZEN MUAP benchmark over HELD-OUT anatomies (fat x pennation).

The referee for the anatomy-conditioning experiment. Same synthesis as the cylinder benchmark
(01_build_muap_bench), but the truth MUAPs are computed at (r_fat, pennation) anatomies that
are deliberately OFF the training grid — so a passing model has interpolated the VC field to
unseen anatomy, not memorised it.

MODELLING CHOICE (documented, revisit if needed): pennation enters via the muscle sigma-tensor
rotation (TissueTable.analytical_pennated) — that is what changes the VC field phi the network
must learn. The benchmark fibres stay z-aligned, so a config's MUAP change is attributable to
the FIELD, not to also moving the sampling path. Coupling fibre geometry to pennation is a
separate extension: a full-length (~213mm) fibre tilted 25deg would exit a 35mm-radius muscle;
real pennate muscles use short fibres between aponeuroses, which this straight-cylinder model
does not represent.

IMPORTANT: the manifest's TRAINING grid MUST exclude these anatomies (they are the held-out
test). HELDOUT_ANATOMY sits at half-steps interior to the intended train grid
(fat {34..44 step2}, pennation {0..25 step5}) -> interpolation, not extrapolation.

Run: PYTHONPATH=src python scripts/neural_field/21_build_bench_anatomy.py
"""
from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
B = import_module("01_build_muap_bench")        # mu_fibre_paths, SPCFG, fibre consts, props
G = import_module("20_gen_cyl_anatomy")          # geom(r_fat)

from emgforge.fem import FEMModel
from emgforge.fem.conductivity import TissueTable
from emgforge.synthesis import FibreBed, field_to_muap

OUT = Path(__file__).resolve().parents[2] / "_results/neural_field"
MESH_DIR = Path(__file__).resolve().parents[2] / "_results/sanity/fem_cache/anat"

# ---- FROZEN spec ----
HELDOUT_ANATOMY = [(37.0, 12.5), (39.0, 7.5), (41.0, 2.5), (43.0, 17.5)]   # (r_fat, pennation)
ELECTRODES = tuple((th, z) for th in (0.0, 20.0, 40.0) for z in (110.0, 140.0))
MU_DEPTHS = (8.0, 15.0, 25.0)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True); MESH_DIR.mkdir(parents=True, exist_ok=True)
    sb = FibreBed.from_arrays(np.full(B.N_FIB, B.DZ), np.full(B.N_FIB, B.LP),
                              np.full(B.N_FIB, B.LD), np.full(B.N_FIB, B.POSZ),
                              np.full(B.N_FIB, B.V))

    cfgs, waves, props, t_ms = [], [], [], None
    for r_fat, penn in HELDOUT_ANATOMY:
        g = G.geom(r_fat)
        msh = MESH_DIR / f"cyl_fat{r_fat:g}_cl{G.CHAR_LENGTH:g}.msh"
        g.build(msh, msh.with_suffix(".json"), char_length=G.CHAR_LENGTH)
        fem = FEMModel(str(msh), conductivity=TissueTable.analytical_pennated(penn),
                       source_sigma=3.0)
        beds = {d: B.mu_fibre_paths(g, d) for d in MU_DEPTHS}
        print(f"anatomy r_fat={r_fat} penn={penn}° ({time.time()-t0:.0f}s)")
        for th, z in ELECTRODES:
            uh = fem.solve_for_point(g.electrode_on_skin(th, z))
            for d in MU_DEPTHS:
                phi = np.array([fem.evaluate_solution_at_points(p, uh) for p in beds[d]])
                res = field_to_muap(phi, sb, B.SPCFG)
                if t_ms is None:
                    t_ms = res.t_ms
                pr = B.muap_properties(res.t_ms, res.muap)
                pr["duration_ms"] = float(res.metrics.get("duration_ms", 0.0))
                cfgs.append((r_fat, penn, th, z, d)); waves.append(res.muap); props.append(pr)

    cfgs = np.array(cfgs, float); waves = np.array(waves)
    keys = ("p2p", "duration_ms", "jaggedness", "eof", "latency", "trough")
    P = {k: np.array([p[k] for p in props]) for k in keys}
    np.savez_compressed(OUT / "muap_bench_anatomy.npz", t_ms=t_ms, muap=waves, cfg=cfgs,
                        cfg_cols=np.array(["r_fat", "pennation_deg", "elec_theta_deg",
                                           "elec_z_mm", "mu_depth_mm"]), **P)
    A = P["p2p"] * 1e6
    print(f"\n=== FROZEN ANATOMY BENCHMARK: {len(cfgs)} configs "
          f"({len(HELDOUT_ANATOMY)} anatomies × {len(ELECTRODES)} elec × {len(MU_DEPTHS)} MU) ===")
    print(f"{'fat':>5}{'pen':>5}{'θ':>5}{'z':>5}{'dep':>5} | {'p2p µV':>8}{'lat':>7}{'jag':>8}")
    print("-" * 52)
    for c, k in zip(cfgs, range(len(cfgs))):
        print(f"{c[0]:5.0f}{c[1]:5.1f}{c[2]:5.0f}{c[3]:5.0f}{c[4]:5.0f} | {A[k]:8.2f}"
              f"{P['latency'][k]:7.1f}{P['jaggedness'][k]:8.4f}")
    print(f"\ndetectable (>1µV): {(A>=1.0).sum()}/{len(cfgs)} · {A.min():.2f}–{A.max():.2f} µV")
    print(f"wrote {OUT/'muap_bench_anatomy.npz'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
