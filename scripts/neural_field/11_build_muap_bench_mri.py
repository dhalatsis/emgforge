"""Phase 3.1 — the FROZEN MUAP benchmark on the REAL MRI anatomy (WR forearm, FCU).

Mirrors 01_build_muap_bench.py (cylinder), but on real anatomy with a real Henneman MU
pool. Uses the corrected fibre geometry from the MU-pool work: innervation zone at
fibre-fraction 0.305 -> asymmetric Lp/Ld, posz=(Lp-Ld)/2, per-fibre ARC-LENGTH dz.

Configs = {3 MUs spanning the detectable amplitude range} x {9 skin electrodes over FCU}.
Seeded; a rerun must reproduce it byte-for-byte (Gate 3).

Run: PYTHONPATH=src python scripts/neural_field/11_build_muap_bench_mri.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from dolfinx import geometry

from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core.motor_unit_pool import sample_henneman_pool
from emgforge.mri.core.fem_solver import MRIFEMModel
from emgforge.synthesis import FibreBed, SpatialConfig, field_to_muap
from emgforge.synthesis.metrics import jaggedness, lobe_metrics

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
MESH = ROOT / "_results/sanity/fem_cache/forearm_WR.msh"
CFG = ROOT / "_results/sanity/fem_cache/forearm_WR_fibers.json"
OUT = ROOT / "_results/neural_field"

# ---- FROZEN spec ----
FCU, DENSITY, POOL_SEED, IZ_FRAC = 8, 4.0, 0, 0.305
MUS = (99, 59, 26)                     # strong / mid / weak — spans the detectable range
DTHETA = (-20.0, 0.0, 20.0)            # relative to the FCU centroid angle
ZFRACS = (0.40, 0.50, 0.60)
SPCFG = SpatialConfig(denoise="monopole", denoise_n_poles=3, fiber_window="one_sided",
                      tukey_alpha=0.25, csd_derivative=2, upsample_factor=2, fsamp=2048.0,
                      w=256, edge_taper_left=5, edge_taper_right=10, t_start_ms=-10.0, v=4.0)


def mu_synth_bed(mu, arc_dz, L_fib):
    """IZ at 0.305 -> asymmetric Lp/Ld, posz=(Lp-Ld)/2 (the validated pm_lib convention)."""
    idx = mu.fiber_idxs
    rng = np.random.default_rng(int(mu.idx))
    izf = np.clip(IZ_FRAC + rng.normal(0, 0.02, mu.size), 0.1, 0.9)
    Lp, Ld = izf * L_fib[idx], (1 - izf) * L_fib[idx]
    return FibreBed.from_arrays(arc_dz[idx], Lp, Ld, (Lp - Ld) / 2, 4.0)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fm = MuscleFiberModel(str(SEG)); fm.estimate_centerlines(); fm.estimate_cross_sections()
    bed = build_muscle_beds(fm, density=DENSITY, method="poisson", labels=[FCU],
                            min_fibers=30)[FCU]
    N = len(bed.r_norms)
    pool = sample_henneman_pool(bed, n_mu=100, size_min=5, size_max=min(400, N), seed=POOL_SEED)
    seg = np.linalg.norm(np.diff(bed.paths, axis=1), axis=2)
    arc_dz, L_fib = seg.mean(1), seg.sum(1)
    print(f"FCU bed {N} fibres · pool 100 MUs · using MUs {MUS} "
          f"({[pool[m].size for m in MUS]} fibres)")

    fem = MRIFEMModel(str(MESH), fiber_config=str(CFG), nifti_path=str(SEG),
                      skin_shell_mm=1.5, sigma_mode="centerline")
    limb = np.array([fem.mesh.geometry.x[:, 0].mean(), fem.mesh.geometry.x[:, 1].mean()])
    fcu_ang = float(np.degrees(np.arctan2(bed.centroid_xy[1] - limb[1],
                                          bed.centroid_xy[0] - limb[0])))
    lf = fem._leadfield
    # cache cell_ids per MU fibre-set (fixed across electrodes — Gate 0b's 18x)
    cids = {m: geometry.compute_closest_entity(
        lf.tree, lf.midpoints, lf.mesh,
        bed.paths[pool[m].fiber_idxs].reshape(-1, 3)).squeeze() for m in MUS}

    cfgs, waves, props, t_ms = [], [], [], None
    for th_off in DTHETA:
        for zf in ZFRACS:
            elec = fem.get_skin_surface_point(fcu_ang + th_off, zf)
            uh = fem.solve_for_point(elec, source_sigma=5.0)
            for m in MUS:
                mu = pool[m]
                P = bed.paths[mu.fiber_idxs]
                phi = np.asarray(uh.eval(P.reshape(-1, 3), cids[m])).reshape(len(P), -1)
                res = field_to_muap(phi, mu_synth_bed(mu, arc_dz, L_fib), SPCFG)
                if t_ms is None:
                    t_ms = res.t_ms
                tr, _, after = lobe_metrics(res.t_ms, res.muap, 12.0)
                cfgs.append((fcu_ang + th_off, zf, m, mu.size))
                waves.append(res.muap)
                props.append(dict(p2p=float(res.muap.ptp()),
                                  duration_ms=float(res.metrics.get("duration_ms", 0.0)),
                                  jaggedness=float(jaggedness(res.muap)), eof=float(after),
                                  latency=float(res.t_ms[np.argmax(np.abs(res.muap))]),
                                  trough=float(tr)))
            print(f"  θ{fcu_ang+th_off:5.0f}° z{zf:.2f} · {len(MUS)} MUs ({time.time()-t0:.0f}s)")

    cfgs = np.array(cfgs, float); waves = np.array(waves)
    keys = ("p2p", "duration_ms", "jaggedness", "eof", "latency", "trough")
    P = {k: np.array([p[k] for p in props]) for k in keys}
    np.savez_compressed(OUT / "muap_bench_mri.npz", t_ms=t_ms, muap=waves, cfg=cfgs,
                        cfg_cols=np.array(["elec_theta_deg", "elec_z_frac", "mu_idx",
                                           "mu_size"]),
                        fcu_ang=fcu_ang, **P)
    print(f"\n=== FROZEN MRI BENCHMARK: {len(cfgs)} configs ===")
    print(f"{'θ':>6}{'zfrac':>7}{'MU':>5}{'nfib':>6} | {'p2p µV':>8}{'dur ms':>8}{'lat':>7}"
          f"{'jag':>8}{'EOF':>6}")
    print("-" * 62)
    for c, k in zip(cfgs, range(len(cfgs))):
        print(f"{c[0]:6.0f}{c[1]:7.2f}{int(c[2]):5d}{int(c[3]):6d} | {P['p2p'][k]*1e6:8.2f}"
              f"{P['duration_ms'][k]:8.1f}{P['latency'][k]:7.1f}{P['jaggedness'][k]:8.4f}"
              f"{P['eof'][k]:6.2f}")
    det = P["p2p"] * 1e6 >= 1.0
    print(f"\ndetectable (>1µV): {det.sum()}/{len(cfgs)} · amplitude "
          f"{P['p2p'].min()*1e6:.2f}–{P['p2p'].max()*1e6:.2f} µV")
    print(f"wrote {OUT/'muap_bench_mri.npz'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
