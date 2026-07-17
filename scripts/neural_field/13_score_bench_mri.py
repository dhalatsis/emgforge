"""Phase 3.3 / 4 — score a learned VC against the FROZEN MRI MUAP benchmark.

Same referee as the cylinder (04_score_bench.py) but on real WR/FCU anatomy. Rebuilds the
pool deterministically to recover each config's fibre paths, queries the model there, runs
field_to_muap, and diffs against the frozen FEM truth.

Amplitude-aware: reports the plain median AND the amplitude-weighted mean + the detectable
(>1µV) subset — on MRI 21/27 configs are detectable, so the honest gate has real teeth.

Needs dolfinx only to rebuild the fibre bed (geometry), NOT to solve. Training + this
scoring remain the cluster-portable half.

Run: PYTHONPATH=src python scripts/neural_field/13_score_bench_mri.py --ckpt <ckpt>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
S = import_module("04_score_bench")          # reuse load() / predict_phi()

from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core.motor_unit_pool import sample_henneman_pool
from emgforge.mri.core.fem_solver import MRIFEMModel
from emgforge.synthesis import FibreBed, SpatialConfig, field_to_muap
from emgforge.synthesis.metrics import align_score, jaggedness, lobe_metrics

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
MESH = ROOT / "_results/sanity/fem_cache/forearm_WR.msh"
CFG = ROOT / "_results/sanity/fem_cache/forearm_WR_fibers.json"
OUT = ROOT / "_results/neural_field"
FCU, DENSITY, POOL_SEED, IZ_FRAC = 8, 4.0, 0, 0.305
SPCFG = SpatialConfig(denoise="monopole", denoise_n_poles=3, fiber_window="one_sided",
                      tukey_alpha=0.25, csd_derivative=2, upsample_factor=2, fsamp=2048.0,
                      w=256, edge_taper_left=5, edge_taper_right=10, t_start_ms=-10.0, v=4.0)


def mu_synth_bed(mu, arc_dz, L_fib):
    idx = mu.fiber_idxs
    rng = np.random.default_rng(int(mu.idx))
    izf = np.clip(IZ_FRAC + rng.normal(0, 0.02, mu.size), 0.1, 0.9)
    Lp, Ld = izf * L_fib[idx], (1 - izf) * L_fib[idx]
    return FibreBed.from_arrays(arc_dz[idx], Lp, Ld, (Lp - Ld) / 2, 4.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", default=str(OUT / "muap_bench_mri.npz"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = np.load(a.bench)
    t_ms, Wt, cfg = b["t_ms"], b["muap"], b["cfg"]
    A = b["p2p"] * 1e6

    fm = MuscleFiberModel(str(SEG)); fm.estimate_centerlines(); fm.estimate_cross_sections()
    bed = build_muscle_beds(fm, density=DENSITY, method="poisson", labels=[FCU],
                            min_fibers=30)[FCU]
    pool = sample_henneman_pool(bed, n_mu=100, size_min=5,
                                size_max=min(400, len(bed.r_norms)), seed=POOL_SEED)
    seg = np.linalg.norm(np.diff(bed.paths, axis=1), axis=2)
    arc_dz, L_fib = seg.mean(1), seg.sum(1)
    # the electrode xyz must match the benchmark exactly -> rebuild via the same call
    fem = MRIFEMModel(str(MESH), fiber_config=str(CFG), nifti_path=str(SEG),
                      skin_shell_mm=1.5, sigma_mode="centerline")
    net, c = S.load(a.ckpt, dev)

    rows = []
    for k in range(len(cfg)):
        th, zf, mi, _ = cfg[k]
        mu = pool[int(mi)]
        elec = fem.get_skin_surface_point(th, zf)
        P = bed.paths[mu.fiber_idxs]
        phi = np.array([S.predict_phi(net, c, dev, p, elec) for p in P])
        m = field_to_muap(phi, mu_synth_bed(mu, arc_dz, L_fib), SPCFG).muap
        _, r = align_score(t_ms, Wt[k], t_ms, m)
        _, _, eof_p = lobe_metrics(t_ms, m, 12.0)
        rows.append(dict(th=th, zf=zf, mu=int(mi), r=r,
                         p2p_ratio=float(m.ptp() / (Wt[k].ptp() + 1e-30)),
                         d_jag=float(jaggedness(m) - b["jaggedness"][k]),
                         d_eof=float(eof_p - b["eof"][k]),
                         d_lat=float(t_ms[np.argmax(np.abs(m))] - b["latency"][k])))

    R = np.array([x["r"] for x in rows]); Pr = np.array([x["p2p_ratio"] for x in rows])
    print(f"\n====== MRI MUAP BENCHMARK — {Path(a.ckpt).name} ======")
    print(f"{'θ':>5}{'zf':>6}{'MU':>4}{'truth µV':>10} | {'r':>7}{'p2p×':>7}{'Δjag':>9}{'Δlat':>7}")
    print("-" * 58)
    for x, amp in zip(rows, A):
        print(f"{x['th']:5.0f}{x['zf']:6.2f}{x['mu']:4d}{amp:10.2f} | {x['r']:+7.3f}"
              f"{x['p2p_ratio']:7.2f}{x['d_jag']:+9.4f}{x['d_lat']:+7.1f}")
    w = A / A.sum(); det = A >= 1.0
    print("-" * 58)
    print(f"  plain median r      : {np.median(R):+.3f} · ≥0.94 {int((R>=0.94).sum())}/{len(R)}")
    print(f"  amp-weighted mean r : {float((w*R).sum()):+.3f}")
    print(f"  DETECTABLE >1µV (n={det.sum()}) : median r {np.median(R[det]):+.3f} · "
          f"min {R[det].min():+.3f} · p2p med {np.median(Pr[det]):.2f}")
    print(f"  GATE 4 (amp-weighted mean ≥0.94): "
          f"{'PASS ✅' if float((w*R).sum())>=0.94 else 'FAIL ❌'}")
    np.savez(OUT / f"scoremri_{Path(a.ckpt).stem}.npz",
             **{k: np.array([x[k] for x in rows]) for k in rows[0]}, amp=A)


if __name__ == "__main__":
    main()
