"""Sample a 100-MU motor-unit pool from the WR-FCU and compute each MU's MUAP.

Pipeline
--------
1. WR segmentation → MuscleFiberModel → centerlines + cross-sections.
2. build_muscle_beds → a whole-muscle FiberBed for FCU (label 8): N fibre paths.
3. sample_henneman_pool → 100 MUs (exponential sizes, growing territories,
   Henneman recruitment order). Each MU is a set of fibre indices into the bed.
4. ONE FEM reciprocity solve for a skin electrode over FCU → φ(z) sampled along
   every bed fibre (one solve serves all MUs).
5. Per MU: field_to_muap(φ[mu fibres], jittered bed, config) → its MUAP.
6. Save the pool + MUAPs; plot territories, size distribution, recruitment
   (amplitude vs size), sample waveforms, and a compound EMG interference pattern.

Engine (``--engine``): the **spatial** line-source recipe (default) is the production
choice for FEM lead fields — monopole denoise + one-sided fibre-end window, so the
CSD's 2nd derivative doesn't amplify mesh ripple into a noisy MUAP. The **fourier**
engine is kept for comparison but butterworth-smoothed FEM φ still rings. Outputs go
to ``_results/mu_pool/<engine>/`` so both survive side by side.

Run:  python scripts/mri/sample_mu_pool.py [--engine spatial|fourier]
"""
from __future__ import annotations

import argparse
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
from emgforge.synthesis import FibreBed, SpatialConfig, field_to_muap, get_mri_config
from emgforge.synthesis.preprocessing import denoise_field_n

# --- paths (WR subject) ---
ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
MESH = ROOT / "_results/sanity/fem_cache/forearm_WR.msh"
FIBER_CFG = ROOT / "_results/sanity/fem_cache/forearm_WR_fibers.json"

FCU = 8
N_MU = 100
DENSITY = 4.0        # fibres / mm² (→ ~640 fibres in FCU; oversamples the pool)
IZ_FRAC = 0.305      # innervation-zone position along the fibre (FCU, from MU-113 iz_norm)
SEED = 0


def build_config(engine: str, half: float):
    """The per-fibre synthesis config for the chosen engine."""
    if engine == "spatial":
        # Production FEM recipe: monopole denoise (removes mesh ripple BEFORE the
        # CSD 2nd-derivative amplifies it) + one-sided tendon window. Physical-time,
        # so a small pre-roll gives lead-in when the NMJ sits under the electrode.
        return SpatialConfig(
            denoise="monopole", denoise_n_poles=3,
            fiber_window="one_sided", tukey_alpha=0.25,
            csd_derivative=2, upsample_factor=2,
            fsamp=2048.0, w=256, edge_taper_left=5, edge_taper_right=10,
            t_start_ms=-10.0, v=4.0,
        )
    if engine == "fourier":
        return replace(get_mri_config(), len1_mm=half, len2_mm=half, v=4.0)
    raise ValueError(f"unknown engine {engine!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["spatial", "fourier"], default="spatial")
    args = ap.parse_args()
    engine = args.engine
    out = ROOT / "_results/mu_pool" / engine
    out.mkdir(parents=True, exist_ok=True)
    print(f"engine: {engine} → {out}")
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

    # Per-fibre ARC-LENGTH geometry (not the z-step): the AP propagates ALONG the
    # curved fibre, so dz must be the arc-length sample spacing and L the arc length.
    seg = np.linalg.norm(np.diff(bed.paths, axis=1), axis=2)     # (N, Nz-1) segment lengths
    arc_dz = seg.mean(axis=1)                                    # (N,) mean arc-length step
    L_fib = seg.sum(axis=1)                                      # (N,) total fibre length
    # Innervation zone at fibre-fraction IZ_FRAC (FCU ≈ 0.305, from MU-113 iz_norm).
    # The electrode sits at muscle-mid (φ peaks near the fibre centre), so an IZ at
    # 0.305 puts the detection point ~40 mm off the endplate → a propagating MUAP,
    # not the on-endplate spike. posz = (Lp-Ld)/2 = the IZ offset from the array centre.
    print(f"  fibre arc-length {L_fib.mean():.0f} mm (dz≈{arc_dz.mean():.3f} mm); IZ frac {IZ_FRAC}")

    # field visualization (raw φ vs monopole fit + field slices)
    _plot_fields(out, engine, fem, phi_all, bed, arc_dz, elec)

    # 5. per-MU MUAP ---------------------------------------------------------
    print(f"computing per-MU MUAPs ({engine} engine)...")
    cfg = build_config(engine, half)
    muaps = {}       # mu.idx -> (t_ms, muap)
    muap_wave = np.zeros((len(pool), int(cfg.w)))   # (100, w) waveform stack
    t_ms = None
    p2p = np.zeros(len(pool))
    dur = np.zeros(len(pool))
    for k, mu in enumerate(pool):
        idx = mu.fiber_idxs
        rng = np.random.default_rng(int(mu.idx))
        izf = np.clip(IZ_FRAC + rng.normal(0.0, 0.02, mu.size), 0.1, 0.9)  # IZ spread
        Lp = izf * L_fib[idx]                       # NMJ → proximal tendon
        Ld = (1.0 - izf) * L_fib[idx]               # NMJ → distal tendon
        posz = (Lp - Ld) / 2.0                      # IZ offset from the array centre
        sb = FibreBed.from_arrays(dz_mm=arc_dz[idx], len1_mm=Lp, len2_mm=Ld,
                                  posz_mm=posz, v=4.0)
        res = field_to_muap(phi_all[idx], sb, cfg)
        muaps[mu.idx] = (res.t_ms, res.muap)
        if t_ms is None:
            t_ms = res.t_ms
        muap_wave[k] = res.muap
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
        out / "mu_pool.npz",
        engine=engine,
        sizes=sizes, territory_r=np.array([m.territory_radius_mm for m in pool]),
        centre_xy=np.array([m.centre_xy for m in pool]),
        recruit_thr=np.array([m.recruitment_threshold for m in pool]),
        firing_hz=np.array([m.firing_rate_hz for m in pool]),
        p2p=p2p, duration_ms=dur, electrode_theta_deg=theta,
        bed_N=N, density=DENSITY, half_mm=half,
        t_ms=t_ms, muap_wave=muap_wave,     # the per-MU MUAP waveforms
    )
    _plot(out, engine, pool, bed, cl, cs, sizes, p2p, muaps, t_s, emg, fired, theta)
    _plot_muap_samples(out, engine, pool, muaps, sizes, p2p)
    _plot_muap_overlay(out, engine, pool, muaps, sizes)
    print(f"done in {time.time()-t_start:.0f}s → {out}")


def _plot_muap_overlay(out, engine, pool, muaps, sizes, n=16):
    """Peak-aligned, peak-normalised MUAPs coloured by size — shape vs MU size."""
    pick = np.linspace(0, len(pool) - 1, n).astype(int)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(pick)))
    for k, c in zip(pick, colors):
        t, m = muaps[pool[k].idx]
        pkt = t[np.argmax(np.abs(m))]
        ax.plot(t - pkt, m / (np.abs(m).max() + 1e-30), color=c, lw=1.3,
                label=f"{pool[k].size} fib")
    ax.set_xlim(-25, 25); ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("t − t_peak (ms)"); ax.set_ylabel("normalised")
    ax.set_title(f"MUAP shape vs MU size ({engine} engine, peak-aligned)")
    ax.legend(fontsize=7, ncol=2, title="MU size"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "mu_pool_muaps_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_fields(out, engine, fem, phi_all, bed, arc_dz, elec, n_show=6):
    """Raw φ(z) vs the 3-monopole interpolation, + longitudinal & cross-section slices."""
    N, Nz = phi_all.shape
    d = np.linalg.norm(bed.xy_mid - elec[:2], axis=1)          # fibre→electrode distance
    pick = np.argsort(d)[np.linspace(0, N - 1, n_show).astype(int)]
    zc = np.arange(Nz) - Nz // 2

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.27, wspace=0.2)

    # (A) raw φ(z) vs 3-monopole fit — the interpolated lead field the engine integrates
    axA = fig.add_subplot(gs[0, 0])
    for j, i in enumerate(pick):
        raw = phi_all[i]
        mono = denoise_field_n(raw, float(arc_dz[i]), n=3)
        z_mm = zc * arc_dz[i]
        axA.plot(z_mm, raw * 1e3, color=f"C{j}", lw=0.8, alpha=0.45)
        axA.plot(z_mm, mono * 1e3, color=f"C{j}", lw=1.9, label=f"d={d[i]:.0f} mm")
    axA.axvline(0, color="k", ls=":", lw=0.8)
    axA.set_title("φ(z) along fibres — raw (thin) vs 3-monopole fit (thick)")
    axA.set_xlabel("z along fibre (mm; 0 = array centre)"); axA.set_ylabel("φ (mV)")
    axA.legend(fontsize=7, title="fibre→electrode"); axA.grid(alpha=0.3)

    # (B) all fibres' φ(z) — the peak marks the detection point (electrode)
    axB = fig.add_subplot(gs[0, 1])
    for i in range(0, N, max(1, N // 120)):
        axB.plot(np.arange(Nz), phi_all[i] * 1e3, color="0.75", lw=0.4)
    axB.plot(np.arange(Nz), phi_all.mean(0) * 1e3, "k", lw=2, label="mean φ")
    axB.axvline(Nz // 2, color="tab:blue", ls=":", label="array centre")
    axB.axvline(np.median(np.argmax(phi_all, 1)), color="tab:red", ls="--", label="φ-peak (electrode)")
    axB.set_title("φ(z), all fibres — peak = detection point")
    axB.set_xlabel("z index"); axB.set_ylabel("φ (mV)"); axB.legend(fontsize=8); axB.grid(alpha=0.3)

    # (C/D) φ on a longitudinal slice + the electrode cross-section
    X = fem.mesh.geometry.x
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 80)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 80)
    zs = np.linspace(X[:, 2].min(), X[:, 2].max(), 80)

    XX, ZZ = np.meshgrid(xs, zs)
    phiL = fem.evaluate_solution_at_points(
        np.c_[XX.ravel(), np.full(XX.size, elec[1]), ZZ.ravel()]).reshape(ZZ.shape)
    axC = fig.add_subplot(gs[1, 0])
    im = axC.pcolormesh(xs, zs, np.log10(np.abs(phiL) + 1e-12), shading="auto", cmap="magma")
    axC.plot(elec[0], elec[2], "c*", ms=15)
    axC.set_title("longitudinal φ  (x–z at electrode y, log₁₀|φ|)")
    axC.set_xlabel("x (mm)"); axC.set_ylabel("z (mm)"); fig.colorbar(im, ax=axC, shrink=0.85)

    XX2, YY2 = np.meshgrid(xs, ys)
    phiX = fem.evaluate_solution_at_points(
        np.c_[XX2.ravel(), YY2.ravel(), np.full(XX2.size, elec[2])]).reshape(YY2.shape)
    axD = fig.add_subplot(gs[1, 1])
    im2 = axD.pcolormesh(xs, ys, np.log10(np.abs(phiX) + 1e-12), shading="auto", cmap="magma")
    axD.scatter(bed.xy_mid[:, 0], bed.xy_mid[:, 1], s=3, c="cyan", alpha=0.5, label="FCU fibres")
    axD.plot(elec[0], elec[1], "w*", ms=15, label="electrode")
    axD.set_aspect("equal"); axD.set_title("cross-section φ at electrode z (log₁₀|φ|) + FCU")
    axD.set_xlabel("x (mm)"); axD.set_ylabel("y (mm)"); axD.legend(fontsize=7)
    fig.colorbar(im2, ax=axD, shrink=0.85)

    fig.suptitle(f"WR-FCU FEM lead field φ — {engine} run, electrode over FCU", fontsize=13)
    fig.savefig(out / "mu_pool_fields.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_muap_samples(out, engine, pool, muaps, sizes, p2p, n=16):
    """A grid of individual MUAP waveforms spanning small → large MUs."""
    pick = np.linspace(0, len(pool) - 1, n).astype(int)   # pool is size-sorted
    nc = 4
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(15, 3.0 * nr))
    for ax, k in zip(axes.ravel(), pick):
        mu = pool[k]
        t, m = muaps[mu.idx]
        ax.plot(t, m * 1e6, "tab:blue", lw=1.4)
        ax.axhline(0, color="k", lw=0.3)
        pk = t[np.argmax(np.abs(m))]
        ax.set_xlim(pk - 30, pk + 30)
        ax.set_title(f"MU{mu.idx}: {mu.size} fibres · {p2p[k]*1e6:.2f} µV", fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("t (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("µV")
    fig.suptitle(f"WR-FCU MUAP samples — 16 MUs across the size range "
                 f"({engine} engine → field_to_muap)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out / "mu_pool_muaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot(out, engine, pool, bed, cl, cs, sizes, p2p, muaps, t_s, emg, fired, theta):
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
    axC.scatter(sizes, p2p * 1e6, s=14, c=np.arange(len(pool)), cmap="viridis")
    axC.set_title("MUAP amplitude vs MU size"); axC.set_xlabel("fibres per MU")
    axC.set_ylabel("MUAP peak-to-peak (µV)"); axC.set_xscale("log"); axC.set_yscale("log")
    axC.grid(alpha=0.3, which="both")

    # (D) sample MUAP waveforms across the size range
    axD = fig.add_subplot(gs[1, 0])
    for mu, c in zip(show, colors):
        t, m = muaps[mu.idx]
        axD.plot(t, m * 1e6, color=c, lw=1.2, label=f"MU{mu.idx} n={mu.size}")
    axD.set_title("Sample MUAP waveforms"); axD.set_xlabel("t (ms)")
    axD.set_ylabel("µV"); axD.legend(fontsize=6); axD.grid(alpha=0.3)

    # (E) compound EMG interference pattern
    axE = fig.add_subplot(gs[1, 1:])
    axE.plot(t_s * 1e3, emg * 1e6, "k-", lw=0.5)
    axE.set_title(f"Compound EMG — {len(fired)} MUs active @30% activation")
    axE.set_xlabel("t (ms)"); axE.set_ylabel("µV"); axE.set_xlim(0, 1000); axE.grid(alpha=0.3)

    fig.suptitle(f"WR-FCU · 100-MU pool · {engine} engine · electrode θ={theta:.0f}° · one FEM solve",
                 fontsize=13)
    fig.savefig(out / "mu_pool.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
