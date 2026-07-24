#!/usr/bin/env python3
"""Build a more realistic MUAP from a Gaussian-density fiber bundle
with NMJ-position jitter and CV jitter.

A realistic motor unit:
  - has 10-200 muscle fibers (depending on muscle and MU type)
  - in a compact territory ~1-5mm radius
  - fibers are Gaussian-density (not uniform ring) around the centre
  - each fiber's NMJ sits at a random z within the innervation zone
    (σ ≈ 5-15mm)
  - each fiber's conduction velocity is drawn from N(μ, σ) with
    σ ≈ 0.3 m/s

This script:
  1. Picks a MU centre (r_norm_c, θ_c) and territory radius.
  2. Samples N fibers with 2D Gaussian density in (r_norm, θ) around
     the centre, in the muscle's cross-section.
  3. Solves FEM at one electrode (or loads a cached one).
  4. Evaluates φ(z) along each fiber path.
  5. Generates 4 MUAPs:
       'ideal'   : posz=0, v=4.0 — synchronous (sharp, like current default)
       'NMJ'     : posz ~ N(0, 10mm), v=4.0
       'CV'      : posz=0, v ~ N(4.0, 0.3)
       'real'    : both jitters
  6. Saves a figure overlaying the four MUAPs.

Output:
    mri/figures/realistic_muap/L{label}.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from emgforge.synthesis.api import MUAPConfig, generate_muap_from_phi
from emgforge.mri.core._pipeline_config import build_muap_config

# NOTE: ``fem_solver`` (FEniCS/ufl) and ``MuscleFiberModel`` are imported lazily
# inside ``main()`` so this module's reusable helpers (``compute_per_fiber_muap``,
# ``compute_harmonic_muap``, the sampling utilities) import cleanly in
# environments without the heavy FEM stack.

NIFTI = "mri/data/PD_PROPELLER_5MM_FATS_FLX_0012/full.nii.gz"
MESH = "mri/mesh/forearm.msh"
FIBER_CFG = "mri/mesh/muscle_fibers.json"


def _xy_to_rnorm_theta(x, y, mx, my, cs, z_mid):
    """Map physical (x, y) to (r_norm, θ_deg) in the muscle cross-section
    at z=z_mid. Returns (r_norm, θ_deg) or None if outside the boundary.
    """
    rdx = x - mx
    rdy = y - my
    r_rel = (rdx * rdx + rdy * rdy) ** 0.5
    theta_deg = float(np.degrees(np.arctan2(rdy, rdx)) % 360.0)
    R_b = float(cs.boundary_radius(
        np.array([z_mid]), np.array([theta_deg]),
    )[0])
    if R_b < 1e-3:
        return None
    r_norm = r_rel / max(cs.r_inset * R_b, 1e-6)
    if r_norm >= 1.0:
        return None
    return float(r_norm), theta_deg


def sample_mu_fibers(centerline, cross_section, mu_r_norm, mu_theta_deg,
                      radius_mm, n_fibers, seed=0, method="uniform",
                      min_spacing_mm=0.0):
    """Sample N fibers around a MU centre at z=midpoint of the centerline.

    Sampling happens in PHYSICAL (x, y) at z_mid, then each sample's
    (r_norm, θ) inside the muscle's cross-section is computed (so fibers
    map to the morphing-disk system used by ``fiber_path_morphing``).

    Parameters
    ----------
    method : {"uniform", "gaussian", "poisson"}
        - "uniform":   uniform density on the disk (polar inverse-CDF).
                       **Default since 2026-05-29** — best match to the
                       Neurodec MRI-FEM reference across all 6 muscles
                       (mean Wasserstein-2D rank 1.25 vs 3.21 Gaussian,
                       2.88 Poisson; KS p > 0.05 for PL).
        - "gaussian":  σ = radius/2. Was the pre-2026-05-29 default.
                       Use this only when you specifically want a
                       centre-dense Gaussian bundle (legacy behaviour).
        - "poisson":   Bridson Poisson-disk sampling with minimum spacing
                       ``min_spacing_mm``. Hard minimum spacing — useful
                       when you need strict spatial regularity, but
                       over-packs by ~3x vs Neurodec's natural near-
                       neighbour shoulder.
    min_spacing_mm : float
        Minimum allowed distance between any two sampled fibres (mm).
        Only used when ``method == "poisson"``. If 0, defaults to
        ``radius * sqrt(π / (N · 4))`` ≈ half the spacing you'd expect
        if the disk were tiled with N hexagons.

    Returns
    -------
    list of (r_norm, θ_deg) tuples for ``fiber_path_morphing``.
    """
    cs = cross_section
    cl = centerline
    z_mid = 0.5 * (cl.z_min + cl.z_max)
    cx_c, cy_c = cl.fiber_path_morphing(
        mu_r_norm, mu_theta_deg, np.array([z_mid]), cs,
    )[0, :2]
    cl_pos = cl.position(z_mid)
    mx, my = cl_pos[0], cl_pos[1]

    rng = np.random.default_rng(seed)

    if method == "gaussian":
        sigma = radius_mm / 2.0
        out, tries = [], 0
        while len(out) < n_fibers and tries < n_fibers * 20:
            tries += 1
            dx, dy = rng.normal(0, sigma), rng.normal(0, sigma)
            if dx * dx + dy * dy > radius_mm * radius_mm * 1.5:
                continue
            res = _xy_to_rnorm_theta(cx_c + dx, cy_c + dy, mx, my, cs, z_mid)
            if res is not None:
                out.append(res)
        return out

    if method == "uniform":
        # r ~ R*sqrt(U), θ ~ 2π U — exact uniform on the disk
        out, tries = [], 0
        while len(out) < n_fibers and tries < n_fibers * 20:
            tries += 1
            r = radius_mm * np.sqrt(rng.uniform())
            th = 2 * np.pi * rng.uniform()
            dx, dy = r * np.cos(th), r * np.sin(th)
            res = _xy_to_rnorm_theta(cx_c + dx, cy_c + dy, mx, my, cs, z_mid)
            if res is not None:
                out.append(res)
        return out

    if method == "poisson":
        # Bridson Poisson-disk on the disk of radius radius_mm.
        # Bug-fix (2026-05-29): let Bridson saturate (max_pts=0 triggers
        # the auto theoretical-saturation bound) and randomly subsample
        # the result to n_fibers. The previous insertion-order trim
        # biased toward the centre and under-filled disk RMS by ~8-30%
        # depending on R/r_min ratio.
        if min_spacing_mm <= 0:
            min_spacing_mm = float(radius_mm * np.sqrt(np.pi / max(n_fibers, 1)) / 2)
        pts = _poisson_disk_on_disk(
            radius_mm, min_spacing_mm, rng, max_pts=0,
        )
        # Random subsample (avoids insertion-order centre bias)
        if len(pts) > n_fibers:
            idx = rng.choice(len(pts), size=n_fibers, replace=False)
            pts = [pts[i] for i in idx]
        out = []
        for dx, dy in pts:
            res = _xy_to_rnorm_theta(cx_c + dx, cy_c + dy, mx, my, cs, z_mid)
            if res is not None:
                out.append(res)
        return out

    raise ValueError(f"unknown method={method!r}")


def _poisson_disk_on_disk(R, r_min, rng, max_pts=0, k=30):
    """Bridson Poisson-disk sampling on a 2D disk of radius R, with
    minimum point spacing r_min. Returns list of (x, y).

    Parameters
    ----------
    R, r_min : float
        Disk radius and minimum point spacing (same units).
    rng : np.random.Generator
    max_pts : int
        Hard cap on number of returned points. **Default 0 = auto** —
        sets max_pts to a theoretical-saturation bound of
        ``4 · π · (R/r_min)²`` so Bridson saturates the disk naturally.
        Pre-2026-05-29 default was 500; this caused the disk to
        under-fill for any R/r_min > ~12. Pass an explicit max_pts > 0
        to restore the legacy cap behaviour.
    k : int
        Bridson "k samples per active cell" parameter.

    Bug-fix (2026-05-29): the seed point was previously near the disk
    centre (``uniform(-r_min/2, r_min/2)``), which combined with the
    growth-from-active-list traversal order produced a centre bias when
    callers used insertion-order trimming. Seed is now drawn uniformly
    from the entire disk so the radial CDF is unbiased.
    """
    # Cell side for uniform grid lookup
    cell = r_min / np.sqrt(2)
    n_cells = int(np.ceil(2 * R / cell)) + 1
    grid = {}  # (i, j) → (x, y)

    if max_pts <= 0:
        # Bridson naturally saturates at ~hex-packing density.
        # Hex cell area = r_min² · √3/2 ⇒ ~2π/√3 · (R/r_min)² ≈ 3.63 N.
        # Use 4·π·(R/r_min)² as a safe upper bound (~12% headroom).
        max_pts = int(4 * np.pi * (R / max(r_min, 1e-9))**2) + 50

    def cell_idx(p):
        return (int((p[0] + R) / cell), int((p[1] + R) / cell))

    def in_disk(p):
        return p[0] * p[0] + p[1] * p[1] <= R * R

    def far_from_neighbours(p):
        ci, cj = cell_idx(p)
        for i in range(max(0, ci - 2), min(n_cells, ci + 3)):
            for j in range(max(0, cj - 2), min(n_cells, cj + 3)):
                q = grid.get((i, j))
                if q is None:
                    continue
                if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < r_min * r_min:
                    return False
        return True

    # Seed uniformly inside the disk (was: near centre — biased)
    r0 = R * np.sqrt(rng.uniform())
    th0 = 2 * np.pi * rng.uniform()
    p0 = (r0 * np.cos(th0), r0 * np.sin(th0))
    grid[cell_idx(p0)] = p0
    active = [p0]
    pts = [p0]

    while active and len(pts) < max_pts:
        idx = rng.integers(0, len(active))
        base = active[idx]
        found = False
        for _ in range(k):
            theta = 2 * np.pi * rng.uniform()
            r = r_min * (1 + rng.uniform())
            cand = (base[0] + r * np.cos(theta), base[1] + r * np.sin(theta))
            if not in_disk(cand):
                continue
            if not far_from_neighbours(cand):
                continue
            grid[cell_idx(cand)] = cand
            active.append(cand)
            pts.append(cand)
            found = True
            break
        if not found:
            active.pop(idx)
    return pts


def compute_per_fiber_muap(phi_mat, dz, half_mm, vs_per_fiber, posz_per_fiber,
                           auto=False):
    """Sum N per-fiber MUAPs each with its own v and NMJ offset."""
    n_fib = phi_mat.shape[0]
    t_common = None
    muap_sum = None
    for fi in range(n_fib):
        cfg = build_muap_config(half_mm=half_mm, auto=auto,
                                v=float(vs_per_fiber[fi]))
        posz = np.array([float(posz_per_fiber[fi])])
        res_i = generate_muap_from_phi(
            phi_mat[fi:fi + 1], dz, config=cfg, posz_mm_arr=posz,
        )
        if t_common is None:
            t_common = res_i.t_ms
            muap_sum = res_i.muap.copy()
        else:
            muap_sum += res_i.muap
    return t_common, muap_sum


def apply_finite_electrode(phi, dz_mm, electrode_diameter_mm):
    """Finite-electrode spatial low-pass of a lead field ``phi(z)``.

    A real recording electrode integrates the potential over its surface, which
    spatially averages (low-passes) the field along the fibre. Point-electrode
    reciprocity omits this, leaving high-frequency content the electrode would
    smooth away — which shows up as ringing when the SFAP is assembled (most
    visibly in the Fourier/`pare` engine) and decorrelates the MUAP from a
    full-FEM reference (NeuroDec), whose electrodes have area.

    Modelled as a Gaussian along-fibre kernel with ``σ = diameter / 4`` mm (a
    smooth stand-in for averaging over a disk of that diameter). ``0`` disables
    it (point electrode) and is byte-identical to no filtering. Validated on the
    NeuroDec MU-113 twin (denoise="none" baseline, per-fibre v/polarity held
    fixed): sweeping the diameter gives a clean unimodal shape-NCC curve —
    median 0.70 at D=0, rising to a peak ~0.87 near D≈13 mm, then declining
    gently (0.85 by D=20 mm). The optimum diameter is physically sensible for a
    surface electrode and leaves the end-of-fibre amplitude (which the Fourier
    engine already gets right) unchanged. Note this gain is SEPARATE from — and
    additive with — using denoise="none" instead of the default butterworth
    (which contributes a further ~+0.15 on its own); don't conflate the two.
    """
    if not electrode_diameter_mm or electrode_diameter_mm <= 0:
        return np.asarray(phi, dtype=float)
    from scipy.ndimage import gaussian_filter1d
    sigma_samples = (electrode_diameter_mm / 4.0) / float(dz_mm)
    return gaussian_filter1d(np.asarray(phi, dtype=float), sigma_samples)


def _spatial_to_muap_config(config):
    """Coerce ``None`` / a ``SpatialConfig`` / a ``MUAPConfig`` → ``MUAPConfig``.

    Lets ``compute_harmonic_muap(engine="fourier")`` accept a spatial-engine
    config (or none) without the caller juggling two config types.

    The constructed configs use ``denoise="none"``: the finite-electrode model
    (:func:`apply_finite_electrode`) is the physically-motivated low-pass on this
    path, and stacking the default φ butterworth on top over-smooths (drops the
    all-electrode NCC vs NeuroDec by ~0.15). An explicitly-passed ``MUAPConfig``
    is honoured verbatim, so callers who want the butterworth can still set it.
    """
    from emgforge.synthesis.api import MUAPConfig
    if config is None:
        return MUAPConfig(denoise="none")
    if isinstance(config, MUAPConfig):
        return config
    from emgforge.synthesis.conventions import Conventions
    return MUAPConfig(
        v=getattr(config, "v", 4.0),
        fsamp=getattr(config, "fsamp", 4096.0),
        w=getattr(config, "w", 256),
        upsample_factor=getattr(config, "upsample_factor", 1),
        apply_z_window=True,
        denoise="none",
        conventions=Conventions(
            polarity=getattr(config, "polarity", 1),
            center_time=getattr(config, "center_time", False),
        ),
    )


def compute_harmonic_muap(phi_per_fiber, dz_mm, half1_mm, half2_mm, posz_mm,
                          vs_per_fiber=None, config=None,
                          engine="fourier", electrode_diameter_mm=0.0):
    """Sum SHORT-fibre SFAPs for a harmonic motor unit.

    Unlike :func:`compute_per_fiber_muap` (one global ``half_mm``, mid-belly
    NMJ — the RED/GREY assumption), this consumes the PER-FIBRE semi-lengths
    and placed NMJ produced by
    :mod:`emgforge.mri.core.harmonic_fibers` / a ``method="harmonic"``
    ``FiberBed``. Each fibre's lead field ``phi_per_fiber[i]`` is sampled at
    uniform ``dz_mm`` along its short path and combined with that fibre's
    ``len1_mm=half1``, ``len2_mm=half2`` and ``posz_mm`` — so the active
    propagation band spans only ~one fascicle length, not the whole muscle.

    Parameters
    ----------
    phi_per_fiber : sequence of (Nz_i,) arrays
        Lead field along each short fibre (fibres may differ in length).
    dz_mm : float
        Uniform arc-length step of every fibre's ``phi`` (== the bed ``dz_mm``).
    half1_mm, half2_mm, posz_mm : (N,) arrays
        Per-fibre semi-lengths and NMJ position (bed ``half1_mm`` etc.).
    vs_per_fiber : (N,) array or None
        Optional per-fibre conduction velocity (m/s). If None, the config ``v``
        is used for all fibres.
    config : SpatialConfig | MUAPConfig | None
        Engine config. ``None`` → the engine's default. A ``SpatialConfig`` is
        accepted for either engine (translated for the Fourier path).
    engine : {"fourier", "spatial"}
        Forward operator (default "fourier"). The **Fourier/`pare`** engine
        (Farina 2004) models tendon generation/extinction explicitly and
        reproduces the end-of-fibre potential — on the NeuroDec twin its
        far-field EOF/wave matches (7.9 vs 7.6) and the amplitude scale is
        correct, where the **spatial** engine (no `pare`, end term only a
        window-derivative byproduct) under-produces the EOF ~15× in the far
        field and under-scales amplitude ~20×. Use "spatial" only for
        near, wave-dominated pickups where its smoothness is an asset.
    electrode_diameter_mm : float
        Finite-electrode spatial low-pass (default 0 = point electrode). See
        :func:`apply_finite_electrode`. Recommended ~10 mm for surface-EMG /
        full-FEM comparisons; needed for the Fourier engine to match a
        finite-electrode reference without high-frequency ringing.

    Returns
    -------
    (t_ms, muap) : the common time axis and the summed MUAP.
    """
    half1_mm = np.asarray(half1_mm, dtype=float)
    half2_mm = np.asarray(half2_mm, dtype=float)
    posz_mm = np.asarray(posz_mm, dtype=float)
    n_fib = len(phi_per_fiber)
    phi_e = [apply_finite_electrode(phi_per_fiber[fi], dz_mm, electrode_diameter_mm)
             for fi in range(n_fib)]

    if engine == "fourier":
        from emgforge.synthesis.api import generate_muap_from_phi
        # Fibres may differ in length; pad to a common grid for the (Nfib, Nz)
        # matrix the Fourier engine consumes. Pad SYMMETRICALLY so each fibre's
        # electrode-centred midpoint stays at the array centre (the engine
        # treats the array midpoint as the electrode-nearest z=0).
        nz = max(len(p) for p in phi_e)
        phi_mat = np.zeros((n_fib, nz), dtype=float)
        for fi, p in enumerate(phi_e):
            off = (nz - len(p)) // 2
            phi_mat[fi, off:off + len(p)] = p
        mcfg = _spatial_to_muap_config(config)
        res = generate_muap_from_phi(
            phi_mat, dz_mm, config=mcfg,
            posz_mm_arr=posz_mm,
            vs_per_fiber=(None if vs_per_fiber is None
                          else np.asarray(vs_per_fiber, dtype=float)),
            len1_per_fiber=half1_mm, len2_per_fiber=half2_mm,
        )
        return res.t_ms, res.muap

    if engine == "spatial":
        from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial
        cfg = config if isinstance(config, SpatialConfig) else SpatialConfig()
        t_common = None
        muap_sum = None
        for fi in range(n_fib):
            c = cfg
            if vs_per_fiber is not None and float(vs_per_fiber[fi]) != cfg.v:
                c = SpatialConfig(**{**cfg.__dict__, "v": float(vs_per_fiber[fi])})
            t_ms, sfap, _ = compute_sfap_spatial(
                phi_e[fi], dz_mm,
                len1_mm=float(half1_mm[fi]), len2_mm=float(half2_mm[fi]),
                posz_mm=float(posz_mm[fi]), config=c,
            )
            if t_common is None:
                t_common = t_ms
                muap_sum = sfap.copy()
            else:
                muap_sum += sfap
        return t_common, muap_sum

    raise ValueError(f"unknown engine {engine!r}; expected 'fourier' or 'spatial'")


def compute_muap_from_bed(bed, phi_per_fiber, fiber_idxs=None,
                          vs_per_fiber=None, config=None,
                          half1_mm=None, half2_mm=None, posz_mm=None,
                          engine="fourier", electrode_diameter_mm=0.0):
    """Convenience wrapper: MUAP from a ``method="harmonic"`` ``FiberBed``.

    Pulls each selected fibre's ``half1_mm`` / ``half2_mm`` / ``posz_mm`` from
    the bed and forwards to :func:`compute_harmonic_muap`. ``phi_per_fiber``
    must be aligned with ``fiber_idxs`` (or with all bed fibres if
    ``fiber_idxs`` is None).

    To render a motor unit with literature-aligned single-IZ + endplate-jitter
    geometry, pass the ``half1_mm``/``half2_mm``/``posz_mm`` overrides from
    :func:`motor_unit_pool.mu_fiber_geometry` — they take precedence over the
    bed's stored per-fibre semi-lengths. ``engine`` and ``electrode_diameter_mm``
    are forwarded to :func:`compute_harmonic_muap` (default: Fourier/`pare`
    engine, point electrode).
    """
    if not getattr(bed, "is_harmonic", False):
        raise ValueError(
            "compute_muap_from_bed requires a method='harmonic' FiberBed; "
            "use compute_per_fiber_muap for uniform/poisson/hex beds."
        )
    idx = np.arange(len(bed.half1_mm)) if fiber_idxs is None else np.asarray(fiber_idxs)
    h1 = bed.half1_mm[idx] if half1_mm is None else half1_mm
    h2 = bed.half2_mm[idx] if half2_mm is None else half2_mm
    pz = bed.posz_mm[idx] if posz_mm is None else posz_mm
    return compute_harmonic_muap(
        phi_per_fiber, bed.dz_mm, h1, h2, pz,
        vs_per_fiber=vs_per_fiber, config=config,
        engine=engine, electrode_diameter_mm=electrode_diameter_mm,
    )


def _align(t, x):
    return t - t[np.argmax(np.abs(x))], x


def _norm(x):
    return x / max(np.abs(x).max(), 1e-30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", type=int, default=7)
    p.add_argument("--electrode_theta", type=float, default=90.0)
    p.add_argument("--electrode_z_frac", type=float, default=0.5)
    p.add_argument("--mu_r_norm", type=float, default=0.5)
    p.add_argument("--mu_theta_deg", type=float, default=0.0)
    p.add_argument("--mu_radius_mm", type=float, default=3.0)
    p.add_argument("--n_fibers", type=int, default=60)
    p.add_argument("--nmj_sigma_mm", type=float, default=10.0)
    p.add_argument("--cv_sigma", type=float, default=0.3)
    p.add_argument("--out_dir", default="mri/figures/realistic_muap")
    p.add_argument("--auto", action="store_true",
                   help="Use the new auto pipeline config "
                        "(adaptive w + auto-smoothing + auto-edge_taper).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from emgforge.mri.core.fem_solver import MRIFEMModel
    from emgforge.mri.core.fiber_directions import MuscleFiberModel

    fm = MuscleFiberModel(NIFTI)
    fm.estimate_centerlines()
    fm.estimate_cross_sections()
    muscle = fm.muscles[args.label]
    cl, cs = muscle.centerline, muscle.cross_section

    print(f"Sampling {args.n_fibers} fibers in MU @ "
          f"(r_norm={args.mu_r_norm}, θ={args.mu_theta_deg}°), "
          f"radius {args.mu_radius_mm}mm...")
    # This script's module docstring specifies a 2D Gaussian-density
    # bundle, so we keep "gaussian" here even though the function's
    # default switched to "uniform" on 2026-05-29.
    fiber_keys = sample_mu_fibers(
        cl, cs, args.mu_r_norm, args.mu_theta_deg,
        args.mu_radius_mm, args.n_fibers, seed=0, method="gaussian",
    )
    print(f"  got {len(fiber_keys)} fibers")

    z_margin = 0.05 * (cl.z_max - cl.z_min)
    dz = 1.0
    z_vals = np.arange(cl.z_min + z_margin, cl.z_max - z_margin, dz)
    half = 0.5 * float(z_vals[-1] - z_vals[0])
    paths = [cl.fiber_path_morphing(rn, th, z_vals, cs) for rn, th in fiber_keys]

    print("Loading FEM, solving...")
    fem = MRIFEMModel(MESH, fiber_config=FIBER_CFG, nifti_path=NIFTI)
    elec = fem.get_skin_surface_point(args.electrode_theta, args.electrode_z_frac)
    t0 = time.time()
    fem.solve_for_point(elec, source_sigma=5.0)
    print(f"  solved in {time.time()-t0:.1f}s")
    phi = np.array([fem.evaluate_solution_at_points(p) for p in paths])
    print(f"  phi shape: {phi.shape}")

    rng = np.random.default_rng(0)
    n = len(fiber_keys)

    # Generate four MUAP variants
    print(f"Computing 4 MUAP variants (auto={args.auto})...")
    cfg_ideal = build_muap_config(half_mm=half, auto=args.auto)
    res_ideal = generate_muap_from_phi(phi, dz, config=cfg_ideal)

    posz_nmj = rng.normal(0, args.nmj_sigma_mm, size=n)
    res_nmj = generate_muap_from_phi(
        phi, dz, config=cfg_ideal, posz_mm_arr=posz_nmj)

    vs = rng.normal(4.0, args.cv_sigma, size=n)
    t_cv, m_cv = compute_per_fiber_muap(
        phi, dz, half, vs_per_fiber=vs, posz_per_fiber=np.zeros(n),
        auto=args.auto)

    t_real, m_real = compute_per_fiber_muap(
        phi, dz, half, vs_per_fiber=vs, posz_per_fiber=posz_nmj,
        auto=args.auto)

    # Plot
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.27)

    # (A) Cross-section showing MU territory and sampled fibers
    axA = fig.add_subplot(gs[0, 0])
    z_mid = 0.5 * (cl.z_min + cl.z_max)
    theta_arr = np.linspace(0, 360, 361)
    R_b = cs.boundary_radius(np.full_like(theta_arr, z_mid), theta_arr)
    cent = cl.position(z_mid)
    axA.plot(
        cent[0] + R_b * np.cos(np.radians(theta_arr)),
        cent[1] + R_b * np.sin(np.radians(theta_arr)),
        "k-", lw=1.2, label="muscle boundary",
    )
    axA.plot(cent[0], cent[1], "k+", ms=10, mew=1.5)
    # MU centre and territory
    cx_c, cy_c = cl.fiber_path_morphing(
        args.mu_r_norm, args.mu_theta_deg, np.array([z_mid]), cs,
    )[0, :2]
    axA.add_patch(plt.Circle(
        (cx_c, cy_c), args.mu_radius_mm, fill=False, color="red", lw=1.5))
    axA.plot(cx_c, cy_c, "r*", ms=14, mec="black", mew=0.4)
    # Sampled fibers
    fiber_xy = np.array([
        cl.fiber_path_morphing(rn, th, np.array([z_mid]), cs)[0, :2]
        for rn, th in fiber_keys
    ])
    axA.plot(fiber_xy[:, 0], fiber_xy[:, 1], "b.", ms=4, alpha=0.6)
    axA.set_aspect("equal")
    # Frame around MU
    pad = max(args.mu_radius_mm * 2, 8)
    axA.set_xlim(cx_c - pad, cx_c + pad)
    axA.set_ylim(cy_c - pad, cy_c + pad)
    axA.set_xlabel("x [mm]"); axA.set_ylabel("y [mm]")
    axA.set_title(f"(A) {len(fiber_keys)} fibers sampled in "
                   f"{args.mu_radius_mm}mm-radius MU @ z={z_mid:.0f}mm",
                   fontsize=10)
    axA.grid(alpha=0.3)
    axA.legend(fontsize=8)

    # (B) NMJ position histogram (jitter)
    axB = fig.add_subplot(gs[0, 1])
    axB.hist(posz_nmj, bins=20, color="#1f77b4", edgecolor="white")
    axB.axvline(0, color="k", lw=0.5)
    axB.set_xlabel("per-fiber NMJ offset [mm]")
    axB.set_ylabel("count")
    axB.set_title(f"(B) NMJ jitter σ = {args.nmj_sigma_mm:.1f} mm "
                   f"(innervation zone width)", fontsize=10)
    axB.grid(alpha=0.3)

    # (C) CV histogram (jitter)
    axC = fig.add_subplot(gs[0, 2])
    axC.hist(vs, bins=20, color="#d62728", edgecolor="white")
    axC.axvline(4.0, color="k", lw=0.5)
    axC.set_xlabel("per-fiber CV [m/s]")
    axC.set_ylabel("count")
    axC.set_title(f"(C) CV jitter σ = {args.cv_sigma:.2f} m/s", fontsize=10)
    axC.grid(alpha=0.3)

    # (D) 4 MUAP variants overlaid (peak-aligned, peak-normalized)
    axD = fig.add_subplot(gs[1, :])
    variants = [
        ("ideal — synchronous", res_ideal.t_ms, res_ideal.muap),
        (f"NMJ jitter only (σ={args.nmj_sigma_mm:.0f}mm)",
         res_nmj.t_ms, res_nmj.muap),
        (f"CV jitter only (σ={args.cv_sigma:.2f}m/s)",
         t_cv, m_cv),
        (f"REAL — both jitters",
         t_real, m_real),
    ]
    colors = ["#888888", "#1f77b4", "#d62728", "#2ca02c"]
    for (name, t, m), col in zip(variants, colors):
        ta, ma = _align(t, m)
        lw = 1.0 if "ideal" in name else (2.2 if "REAL" in name else 1.4)
        axD.plot(ta, _norm(ma), lw=lw, color=col, label=name)
    axD.set_xlim(-10, 30)
    axD.set_xlabel("t - t_peak [ms]")
    axD.set_ylabel("MUAP (peak-normalized)")
    axD.set_title(
        f"(D) Synchronous MUAP (grey) vs realistic with physical heterogeneity (green)\n"
        f"L{args.label}  electrode θ={args.electrode_theta:.0f}° z={args.electrode_z_frac:.2f}  "
        f"MU @ (r={args.mu_r_norm}, θ={args.mu_theta_deg:.0f}°), "
        f"n={len(fiber_keys)} fibers",
        fontsize=11,
    )
    axD.grid(alpha=0.3)
    axD.legend(fontsize=9, loc="upper right")
    axD.axhline(0, color="k", lw=0.3)

    suffix = "_auto" if args.auto else "_legacy"
    out_path = out_dir / (
        f"L{args.label}_mu{int(args.mu_r_norm*100):03d}_"
        f"{int(args.mu_theta_deg):03d}{suffix}.png"
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
