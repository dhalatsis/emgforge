"""Precompute the (n_mu, n_elec, w) MUAP tensor for the FCU pool over the 5×5 grid,
reusing the cached lead fields (no FEM re-solve). Saved once; the HD-EMG demo reuses it.

Run: python scripts/activation/build_grid_tensor.py [n_mu]
"""
import sys, time
from pathlib import Path
import numpy as np

from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core.motor_unit_pool import sample_henneman_pool
from emgforge.synthesis import FibreBed, SpatialConfig, field_to_muap

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
CACHE = ROOT / "_results/mu_pool/electrode_grid/_phigrid_M5_dt15_z0.30-0.70_N637.npz"
OUT = ROOT / "_results/mu_pool/electrode_grid/muap_tensor_L8_M5.npz"
IZ_FRAC = 0.305
NMU = int(sys.argv[1]) if len(sys.argv) > 1 else 65

fm = MuscleFiberModel(str(SEG)); fm.estimate_centerlines(); fm.estimate_cross_sections()
bed = build_muscle_beds(fm, density=4.0, method="poisson", labels=[8], min_fibers=30)[8]
N = len(bed.r_norms)
pool = sample_henneman_pool(bed, n_mu=100, size_min=5, size_max=min(400, N), seed=0)
seg = np.linalg.norm(np.diff(bed.paths, axis=1), axis=2); arc_dz, L_fib = seg.mean(1), seg.sum(1)

phi_grid = np.load(CACHE)["phi_grid"]                  # (M, M, 637, 200)
M = phi_grid.shape[0]
cfg = SpatialConfig(denoise="monopole", denoise_n_poles=3, fiber_window="one_sided",
                    tukey_alpha=0.25, csd_derivative=2, upsample_factor=2, fsamp=2048.0,
                    w=256, edge_taper_left=5, edge_taper_right=10, t_start_ms=-10.0, v=4.0)

sizes = np.array([pool[k].size for k in range(NMU)])
W = np.zeros((NMU, M * M, cfg.w)); t_ms = None; done = 0
# resume from a checkpoint of the same shape (large MUs are slow — don't recompute)
if OUT.exists():
    ck = np.load(OUT)
    if ck["W"].shape == W.shape:
        W, t_ms, done = ck["W"], ck["t_ms"], int(ck["done"])
        print(f"resuming from checkpoint: {done}/{NMU} MUs already done")

def save(k):
    np.savez_compressed(OUT, W=W, t_ms=t_ms, M=M, sizes=sizes, done=k)

t0 = time.time()
print(f"computing MUAP tensor: {NMU} MUs × {M*M} electrodes (from {done})...")
for k in range(done, NMU):
    mu = pool[k]; idx = mu.fiber_idxs
    r = np.random.default_rng(int(mu.idx)); izf = np.clip(IZ_FRAC + r.normal(0, 0.02, mu.size), 0.1, 0.9)
    Lp, Ld = izf * L_fib[idx], (1 - izf) * L_fib[idx]
    sb = FibreBed.from_arrays(arc_dz[idx], Lp, Ld, (Lp - Ld) / 2, 4.0)
    for e in range(M * M):
        res = field_to_muap(phi_grid[e // M, e % M][idx], sb, cfg)
        W[k, e] = res.muap
        if t_ms is None:
            t_ms = res.t_ms
    if k % 4 == 0 or k == NMU - 1:
        save(k + 1); print(f"  MU {k:3d}/{NMU} ({time.time()-t0:.0f}s) [checkpoint]")
save(NMU)
print(f"done {time.time()-t0:.0f}s → {OUT}  (W {W.shape})")
