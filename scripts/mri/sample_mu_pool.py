"""Sample a 100-MU motor-unit pool from the WR-FCU and compute each MU's MUAP.

Pipeline
--------
1. WR segmentation → MuscleFiberModel → centerlines + cross-sections.
2. build_muscle_beds → a whole-muscle FiberBed for FCU (label 8): N fibre paths.
3. sample_henneman_pool → 100 MUs (exponential sizes, growing territories,
   Henneman recruitment order). Each MU is a set of fibre indices into the bed.
4. ONE FEM reciprocity solve for a skin electrode over FCU → φ(z) sampled along
   every bed fibre (one solve serves all MUs).
5. Per MU: field_to_muap(φ[mu fibres], jittered bed, MRI config) → its MUAP.
6. Save the pool + MUAPs; plot territories, size distribution, recruitment
   (amplitude vs size), sample waveforms, and a compound EMG interference pattern.

Run:  python scripts/mri/sample_mu_pool.py
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core.motor_unit_pool import sample_henneman_pool, simulate_compound_emg
from emgforge.mri.core.fem_solver import MRIFEMModel
from emgforge.synthesis import FibreBed, field_to_muap, get_mri_config

# --- paths (WR subject) ---
ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
MESH = ROOT / "_results/sanity/fem_cache/forearm_WR.msh"
FIBER_CFG = ROOT / "_results/sanity/fem_cache/forearm_WR_fibers.json"
OUT = ROOT / "_results/mu_pool"
OUT.mkdir(parents=True, exist_ok=True)

FCU = 8
N_MU = 100
DENSITY = 4.0        # fibres / mm² (→ ~640 fibres in FCU; oversamples the pool)
SEED = 0


def main():
    t_start = time.time()

    # 1-2. fibre model + FCU bed --------------------------------------------
    print("building fibre model + FCU bed...")
    fm = MuscleFiberModel(str(SEG))
    fm.estimate_centerlines()
    fm.estimate_cross_sections()
    bed = build_muscle_beds(fm, density=DENSITY, method="poisson",
                            labels=[FCU], min_fibers=30)[FCU]
    cl, cs = fm.muscles[FCU].centerline, fm.muscles[FCU].cross_section
    N = len(bed.r_norms)
    dz = float(bed.z_vals[1] - bed.z_vals[0])
    half = float(bed.half_mm)
    print(f"  FCU bed: {N} fibres, area {bed.cross_section_area_mm2:.0f} mm², "
          f"fibre half-length {half:.0f} mm, dz {dz:.2f} mm")

    # 3. Henneman pool -------------------------------------------------------
    pool = sample_henneman_pool(bed, n_mu=N_MU, size_min=5,
                                size_max=min(400, N), seed=SEED)
    sizes = np.array([mu.size for mu in pool])
    print(f"  pool: {len(pool)} MUs | sizes {sizes.min()}–{sizes.max()} "
          f"(median {int(np.median(sizes))})")

    # 4. one FEM solve for a skin electrode over FCU -------------------------
    print("FEM: solving lead field for a skin electrode over FCU...")
    fem = MRIFEMModel(str(MESH), fiber_config=str(FIBER_CFG), nifti_path=str(SEG),
                      skin_shell_mm=1.5, sigma_mode="centerline")
    # electrode angle: FCU centroid direction from the limb centre
    limb_c = np.array([fem.mesh.geometry.x[:, 0].mean(), fem.mesh.geometry.x[:, 1].mean()])
    fcu_c = bed.centroid_xy
    theta = float(np.degrees(np.arctan2(fcu_c[1] - limb_c[1], fcu_c[0] - limb_c[0])))
    elec = fem.get_skin_surface_point(theta, 0.5)
    t0 = time.time()
    fem.solve_for_point(elec, source_sigma=5.0)
    print(f"  electrode θ={theta:.0f}° · solved in {time.time()-t0:.1f}s")

    # φ(z) along every bed fibre (one solve → all fibres)
    t0 = time.time()
    phi_all = np.array([fem.evaluate_solution_at_points(p) for p in bed.paths])
    print(f"  sampled φ along {N} fibres in {time.time()-t0:.1f}s · φ shape {phi_all.shape}")

    # 5. per-MU MUAP ---------------------------------------------------------
    print("computing per-MU MUAPs...")
    cfg = replace(get_mri_config(), len1_mm=half, len2_mm=half, v=4.0)
    muaps = {}       # mu.idx -> (t_ms, muap)
    p2p = np.zeros(len(pool))
    dur = np.zeros(len(pool))
    for k, mu in enumerate(pool):
        sb = FibreBed.jittered(mu.size, dz_mm=dz, len1_mm=half, len2_mm=half, v=4.0,
                               nmj_sigma_mm=8.0, tendon_sigma_mm=4.0,
                               fibre_length_sigma_mm=6.0, seed=mu.idx)
        res = field_to_muap(phi_all[mu.fiber_idxs], sb, cfg)
        muaps[mu.idx] = (res.t_ms, res.muap)
        p2p[k] = res.metrics.get("peak_to_peak", 0.0)
        dur[k] = res.metrics.get("duration_ms", 0.0)
    print(f"  MUAP p2p: {p2p.min():.2e} … {p2p.max():.2e} V  ({p2p.max()/max(p2p.min(),1e-30):.0f}× range)")

    # compound EMG at 30% activation
    t_s, emg, fired = simulate_compound_emg(pool, muaps, duration_s=1.0,
                                            fsamp=2048.0, activation_level=0.3, seed=SEED)
    print(f"  compound EMG: {len(fired)}/{len(pool)} MUs active @30% · "
          f"p2p {emg.max()-emg.min():.2e} V")

    # 6. save + plot ---------------------------------------------------------
    np.savez_compressed(
        OUT / "mu_pool.npz",
        sizes=sizes, territory_r=np.array([m.territory_radius_mm for m in pool]),
        centre_xy=np.array([m.centre_xy for m in pool]),
        recruit_thr=np.array([m.recruitment_threshold for m in pool]),
        firing_hz=np.array([m.firing_rate_hz for m in pool]),
        p2p=p2p, duration_ms=dur, electrode_theta_deg=theta,
        bed_N=N, density=DENSITY, half_mm=half,
    )
    _plot(pool, bed, cl, cs, sizes, p2p, muaps, t_s, emg, fired, theta)
    print(f"done in {time.time()-t_start:.0f}s → {OUT}")


def _plot(pool, bed, cl, cs, sizes, p2p, muaps, t_s, emg, fired, theta):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.26)
    z_mid = 0.5 * (cl.z_min + cl.z_max)
    th = np.linspace(0, 360, 361)
    Rb = cs.boundary_radius(np.full_like(th, z_mid), th)
    cent = cl.position(z_mid)
    bx = cent[0] + Rb * np.cos(np.radians(th))
    by = cent[1] + Rb * np.sin(np.radians(th))

    # (A) territory map — a few representative MUs
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(bx, by, "k-", lw=1.3)
    axA.scatter(bed.xy_mid[:, 0], bed.xy_mid[:, 1], s=2, c="0.75", label=f"{len(bed.r_norms)} fibres")
    show = [pool[i] for i in np.linspace(0, len(pool) - 1, 6).astype(int)]
    colors = plt.cm.viridis(np.linspace(0, 1, len(show)))
    for mu, c in zip(show, colors):
        fx = bed.xy_mid[mu.fiber_idxs]
        axA.scatter(fx[:, 0], fx[:, 1], s=8, color=c, label=f"MU{mu.idx} (n={mu.size})")
    axA.set_aspect("equal"); axA.set_title("FCU cross-section — MU territories")
    axA.set_xlabel("x (mm)"); axA.set_ylabel("y (mm)"); axA.legend(fontsize=6, loc="upper right")

    # (B) size distribution
    axB = fig.add_subplot(gs[0, 1])
    axB.hist(sizes, bins=25, color="tab:blue", alpha=0.8)
    axB.set_title("MU size distribution (Henneman, exponential)")
    axB.set_xlabel("fibres per MU"); axB.set_ylabel("count")

    # (C) recruitment — MUAP amplitude vs size
    axC = fig.add_subplot(gs[0, 2])
    axC.scatter(sizes, p2p * 1e3, s=14, c=np.arange(len(pool)), cmap="viridis")
    axC.set_title("MUAP amplitude vs MU size"); axC.set_xlabel("fibres per MU")
    axC.set_ylabel("MUAP peak-to-peak (mV)"); axC.set_xscale("log"); axC.set_yscale("log")
    axC.grid(alpha=0.3, which="both")

    # (D) sample MUAP waveforms across the size range
    axD = fig.add_subplot(gs[1, 0])
    for mu, c in zip(show, colors):
        t, m = muaps[mu.idx]
        axD.plot(t, m * 1e3, color=c, lw=1.2, label=f"MU{mu.idx} n={mu.size}")
    axD.set_title("Sample MUAP waveforms"); axD.set_xlabel("t (ms)")
    axD.set_ylabel("mV"); axD.legend(fontsize=6); axD.grid(alpha=0.3)

    # (E) compound EMG interference pattern
    axE = fig.add_subplot(gs[1, 1:])
    axE.plot(t_s * 1e3, emg * 1e3, "k-", lw=0.5)
    axE.set_title(f"Compound EMG — {len(fired)} MUs active @30% activation")
    axE.set_xlabel("t (ms)"); axE.set_ylabel("mV"); axE.set_xlim(0, 1000); axE.grid(alpha=0.3)

    fig.suptitle(f"WR-FCU · 100-MU pool · electrode θ={theta:.0f}° · one FEM solve", fontsize=13)
    fig.savefig(OUT / "mu_pool.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
