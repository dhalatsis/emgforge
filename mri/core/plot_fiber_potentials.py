#!/usr/bin/env python3
"""Visualize fiber groups within muscles and their lead-field potentials / MUAPs.

For a selected muscle, generates:
  1. 3D fiber group within muscle volume (curved centerline fibers)
  2. Lead field φ(z) along each fiber (FEM solution evaluated along curved paths)
  3. MUAP waveforms: individual SFAPs and summed MUAP
  4. Comparison: curved-fiber vs straight-fiber lead fields and MUAPs

Usage:
    conda activate fenicsx-env
    export PYTHONPATH=src
    python mri/plot_fiber_potentials.py [--label LABEL] [--theta THETA] [--z_frac Z_FRAC]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mri.core.fiber_directions import MuscleFiberModel, MuscleCenterline, MuscleCrossSection

NIFTI_PATH = "mri/data/PD_PROPELLER_5MM_FATS_FLX_0012/full.nii.gz"
MESH_PATH = "mri/mesh/forearm.msh"
FIBER_CONFIG = "mri/mesh/muscle_fibers.json"
OUT_DIR = Path("mri/figures/fiber_potentials")  # updated per-label in main()

CMAP_FIBERS = plt.cm.coolwarm


def pick_muscle(fiber_model: MuscleFiberModel, label: int | None = None) -> int:
    """Pick a muscle label — user-specified or the most curved z-aligned one."""
    if label is not None:
        if label not in fiber_model.muscles:
            raise ValueError(f"Label {label} not found")
        return label

    best_label, best_dev = None, 0.0
    for lbl, m in fiber_model.muscles.items():
        if m.tissue_type != "muscle" or m.centerline is None:
            continue
        if m.fiber_angle_from_z_deg > 45:
            continue
        cl = m.centerline
        if not cl._use_spline:
            continue
        z = np.linspace(cl.z_min, cl.z_max, 100)
        t = (z - cl.z_min) / max(cl.z_max - cl.z_min, 1e-6)
        x_s = cl.cx_mm[0] + t * (cl.cx_mm[-1] - cl.cx_mm[0])
        y_s = cl.cy_mm[0] + t * (cl.cy_mm[-1] - cl.cy_mm[0])
        dev = np.sqrt((cl._spline_x(z) - x_s)**2 + (cl._spline_y(z) - y_s)**2).max()
        if dev > best_dev:
            best_dev = dev
            best_label = lbl

    if best_label is None:
        # Fallback: largest muscle
        best_label = max(fiber_model.muscles, key=lambda l: fiber_model.muscles[l].n_voxels)

    return best_label


def generate_fiber_group(
    cl: MuscleCenterline,
    n_radial: int = 8,
    radii_mm: list[float] = [0, 1, 2, 3, 4],
    dz: float = 1.0,
    z_margin_frac: float = 0.05,
    cross_section: MuscleCrossSection | None = None,
    r_norms: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Generate a group of fibers at radial offsets from centerline.

    When cross_section is provided and r_norms is set, uses morphing-disk
    coordinates so all fibers stay inside the muscle by construction.
    Otherwise falls back to constant lateral mm offsets.

    Parameters
    ----------
    cl : MuscleCenterline
    n_radial : int
        Number of angular samples per ring.
    radii_mm : list[float]
        Ring radii in mm (used when cross_section is None).
    dz : float
        z-spacing in mm.
    z_margin_frac : float
        Fraction of z-span to skip at each end.
    cross_section : MuscleCrossSection or None
        If provided, uses morphing-disk mapping.
    r_norms : list[float] or None
        Normalized radii [0-1] for morphing-disk mode. Default: [0, 0.25, 0.5, 0.75, 0.95].

    Returns
    -------
    z_values : (Nz,) array of z coordinates
    fiber_paths : list of (Nz, 3) arrays — one per fiber
    offsets : list of (r_norm, theta_deg) or (dx_mm, dy_mm) tuples for each fiber
    """
    z_margin = z_margin_frac * (cl.z_max - cl.z_min)
    z_values = np.arange(cl.z_min + z_margin, cl.z_max - z_margin, dz)

    fiber_paths = []
    offsets = []
    angles_rad = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)
    angles_deg = np.degrees(angles_rad)

    if cross_section is not None:
        # Morphing-disk mode
        if r_norms is None:
            r_norms = [0.0, 0.25, 0.5, 0.75, 0.95]

        for r_n in r_norms:
            if r_n == 0:
                path = cl.fiber_path_morphing(0.0, 0.0, z_values, cross_section)
                fiber_paths.append(path)
                offsets.append((0.0, 0.0))
            else:
                for theta_d in angles_deg:
                    path = cl.fiber_path_morphing(r_n, theta_d, z_values, cross_section)
                    fiber_paths.append(path)
                    offsets.append((r_n, theta_d))
    else:
        # Legacy constant-offset mode
        for r in radii_mm:
            if r == 0:
                path = cl.fiber_path(0.0, 0.0, z_values)
                fiber_paths.append(path)
                offsets.append((0.0, 0.0))
            else:
                for theta in angles_rad:
                    dx = r * np.cos(theta)
                    dy = r * np.sin(theta)
                    path = cl.fiber_path(dx, dy, z_values)
                    fiber_paths.append(path)
                    offsets.append((dx, dy))

    return z_values, fiber_paths, offsets


def generate_straight_fibers(
    cl: MuscleCenterline,
    offsets: list[tuple[float, float]],
    z_values: np.ndarray,
    cross_section: MuscleCrossSection | None = None,
) -> list[np.ndarray]:
    """Generate straight-line fibers (start-to-end interpolation) at same offsets.

    In morphing mode (cross_section provided), offsets are (r_norm, theta_deg).
    The straight fiber uses the same morphing-disk coords but with a straight
    centerline (linear interpolation of start/end).
    """
    straight_paths = []
    x0, y0 = cl.cx_mm[0], cl.cy_mm[0]
    x1, y1 = cl.cx_mm[-1], cl.cy_mm[-1]
    t = (z_values - cl.z_min) / max(cl.z_max - cl.z_min, 1e-6)

    if cross_section is not None:
        # Morphing mode: offsets are (r_norm, theta_deg)
        for r_norm, theta_deg in offsets:
            cx_straight = x0 + t * (x1 - x0)
            cy_straight = y0 + t * (y1 - y0)
            if r_norm == 0:
                path = np.column_stack([cx_straight, cy_straight, z_values])
            else:
                theta_arr = np.full(len(z_values), theta_deg % 360.0)
                R = cross_section.boundary_radius(z_values, theta_arr)
                theta_rad = np.radians(theta_deg)
                scale = r_norm * cross_section.r_inset * R
                xs = cx_straight + scale * np.cos(theta_rad)
                ys = cy_straight + scale * np.sin(theta_rad)
                path = np.column_stack([xs, ys, z_values])
            straight_paths.append(path)
    else:
        # Legacy mode: offsets are (dx_mm, dy_mm)
        for dx, dy in offsets:
            xs = x0 + t * (x1 - x0) + dx
            ys = y0 + t * (y1 - y0) + dy
            path = np.column_stack([xs, ys, z_values])
            straight_paths.append(path)

    return straight_paths


def solve_and_evaluate(
    fiber_paths: list[np.ndarray],
    electrode_point: np.ndarray,
) -> list[np.ndarray]:
    """Solve FEM and evaluate along each fiber path.

    Imports FEM solver here to keep it optional (requires fenicsx-env).
    """
    from mri.core.fem_solver import MRIFEMModel

    print("  Initializing FEM solver (v3 with centerlines)...")
    t0 = time.time()
    model = MRIFEMModel(
        MESH_PATH,
        fiber_config=FIBER_CONFIG,
        nifti_path=NIFTI_PATH,
    )
    print(f"  FEM model: {model.n_cells} cells, {model.n_dofs} DOFs ({time.time()-t0:.1f}s)")

    print(f"  Solving for electrode at {electrode_point}...")
    t0 = time.time()
    uh = model.solve_for_point(electrode_point, source_sigma=5.0)
    iters = model._last_solver.getIterationNumber()
    print(f"  Solved in {iters} iterations ({time.time()-t0:.1f}s)")

    print(f"  Evaluating along {len(fiber_paths)} fibers...")
    phi_list = []
    for path in fiber_paths:
        phi = model.evaluate_solution_at_points(path)
        phi_list.append(phi)

    return phi_list


def compute_muaps(
    phi_list: list[np.ndarray],
    dz_mm: float,
    fiber_half_len_mm: float = 60.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute SFAP for each fiber."""
    from emgforge.synthesis.api import MUAPConfig, generate_muap_from_phi

    config = MUAPConfig(
        denoise="butterworth",
        butterworth_cutoff=0.03,
        butterworth_order=2,
        v=4.0,
        fsamp=4096.0,
        w=256,
        len1_mm=fiber_half_len_mm,
        len2_mm=fiber_half_len_mm,
        apply_z_window=False,
    )

    sfaps = []
    for phi in phi_list:
        phi_2d = phi.reshape(1, -1)
        result = generate_muap_from_phi(phi_2d, dz_mm, config)
        sfaps.append((result.t_ms, result.muap))

    return sfaps


def sum_muap(sfaps: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Sum individual SFAPs into a MUAP."""
    t_ms = sfaps[0][0]
    muap = np.sum([s[1] for s in sfaps], axis=0)
    return t_ms, muap


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_fiber_group_3d(
    fiber_paths: list[np.ndarray],
    straight_paths: list[np.ndarray],
    offsets: list[tuple[float, float]],
    electrode_point: np.ndarray,
    label: int,
):
    """Fig 1: 3D view of curved fiber group + electrode."""
    fig = plt.figure(figsize=(14, 6))

    # Panel A: 3D curved fibers
    ax1 = fig.add_subplot(121, projection="3d")
    n_fibers = len(fiber_paths)
    colors = CMAP_FIBERS(np.linspace(0.1, 0.9, n_fibers))

    for i, path in enumerate(fiber_paths):
        lw = 2.0 if offsets[i] == (0.0, 0.0) else 0.6
        ax1.plot(path[:, 0], path[:, 1], path[:, 2],
                 color=colors[i], linewidth=lw, alpha=0.8)

    ax1.scatter(*electrode_point, color="red", s=80, marker="*",
                zorder=10, label="Electrode")
    ax1.set_xlabel("x (mm)", fontsize=8)
    ax1.set_ylabel("y (mm)", fontsize=8)
    ax1.set_zlabel("z (mm)", fontsize=8)
    ax1.set_title(f"Curved Fibers (Label {label})", fontsize=10)
    ax1.legend(fontsize=8)

    # Panel B: axial cross-section at 3 z-levels
    ax2 = fig.add_subplot(122)
    z_all = fiber_paths[0][:, 2]
    z_levels = [z_all[len(z_all)//4], z_all[len(z_all)//2], z_all[3*len(z_all)//4]]
    markers = ["o", "s", "D"]
    zcolors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for zi, (z_lev, mk, zc) in enumerate(zip(z_levels, markers, zcolors)):
        for i, path in enumerate(fiber_paths):
            idx = np.argmin(np.abs(path[:, 2] - z_lev))
            ms = 8 if offsets[i] == (0.0, 0.0) else 4
            ax2.plot(path[idx, 0], path[idx, 1], mk, color=zc,
                     markersize=ms, alpha=0.7)
        # Also plot straight reference
        for i, spath in enumerate(straight_paths):
            idx = np.argmin(np.abs(spath[:, 2] - z_lev))
            ax2.plot(spath[idx, 0], spath[idx, 1], mk, color=zc,
                     markersize=3, alpha=0.3, markerfacecolor="none")
        ax2.plot([], [], mk, color=zc, markersize=6,
                 label=f"z={z_lev:.0f} mm")

    ax2.plot(electrode_point[0], electrode_point[1], "r*",
             markersize=14, label="Electrode")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.set_title("Axial Cross-Section (filled=curved, open=straight)")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fiber_group_3d.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fiber_group_3d.png")
    plt.close(fig)


def plot_lead_fields(
    z_values: np.ndarray,
    phi_curved: list[np.ndarray],
    phi_straight: list[np.ndarray],
    offsets: list[tuple[float, float]],
    label: int,
):
    """Fig 2: Lead field φ(z) for curved vs straight fibers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    n_fibers = len(phi_curved)
    colors = CMAP_FIBERS(np.linspace(0.1, 0.9, n_fibers))

    # (a) All curved fiber lead fields
    ax = axes[0, 0]
    for i in range(n_fibers):
        lw = 2.0 if offsets[i] == (0.0, 0.0) else 0.5
        alpha = 1.0 if offsets[i] == (0.0, 0.0) else 0.5
        ax.plot(z_values, phi_curved[i], color=colors[i], linewidth=lw, alpha=alpha)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Potential (V)")
    ax.set_title(f"(a) Lead Field — Curved Fibers (Label {label})")
    ax.grid(True, alpha=0.3)

    # (b) All straight fiber lead fields
    ax = axes[0, 1]
    for i in range(n_fibers):
        lw = 2.0 if offsets[i] == (0.0, 0.0) else 0.5
        alpha = 1.0 if offsets[i] == (0.0, 0.0) else 0.5
        ax.plot(z_values, phi_straight[i], color=colors[i], linewidth=lw, alpha=alpha)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Potential (V)")
    ax.set_title("(b) Lead Field — Straight Fibers")
    ax.grid(True, alpha=0.3)

    # (c) Centerline fiber: curved vs straight overlay
    ax = axes[1, 0]
    ax.plot(z_values, phi_curved[0], "b-", linewidth=2, label="Curved")
    ax.plot(z_values, phi_straight[0], "r--", linewidth=2, label="Straight")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Potential (V)")
    ax.set_title("(c) Centerline Fiber: Curved vs Straight")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Difference (curved - straight) for all fibers
    ax = axes[1, 1]
    for i in range(n_fibers):
        diff = phi_curved[i] - phi_straight[i]
        lw = 2.0 if offsets[i] == (0.0, 0.0) else 0.5
        alpha = 1.0 if offsets[i] == (0.0, 0.0) else 0.5
        ax.plot(z_values, diff, color=colors[i], linewidth=lw, alpha=alpha)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Potential Difference (V)")
    ax.set_title("(d) Difference: Curved - Straight")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Lead Fields Along Muscle Fibers — Label {label}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lead_fields.png", dpi=150, bbox_inches="tight")
    print(f"  Saved lead_fields.png")
    plt.close(fig)


def plot_muaps(
    sfaps_curved: list[tuple[np.ndarray, np.ndarray]],
    sfaps_straight: list[tuple[np.ndarray, np.ndarray]],
    offsets: list[tuple[float, float]],
    label: int,
):
    """Fig 3: MUAP waveforms — individual SFAPs and summed MUAP."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    n_fibers = len(sfaps_curved)
    colors = CMAP_FIBERS(np.linspace(0.1, 0.9, n_fibers))

    # (a) Individual SFAPs (curved)
    ax = axes[0, 0]
    for i, (t, s) in enumerate(sfaps_curved):
        lw = 1.5 if offsets[i] == (0.0, 0.0) else 0.4
        alpha = 1.0 if offsets[i] == (0.0, 0.0) else 0.4
        ax.plot(t, s, color=colors[i], linewidth=lw, alpha=alpha)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title(f"(a) Individual SFAPs — Curved Fibers")
    ax.grid(True, alpha=0.3)

    # (b) Individual SFAPs (straight)
    ax = axes[0, 1]
    for i, (t, s) in enumerate(sfaps_straight):
        lw = 1.5 if offsets[i] == (0.0, 0.0) else 0.4
        alpha = 1.0 if offsets[i] == (0.0, 0.0) else 0.4
        ax.plot(t, s, color=colors[i], linewidth=lw, alpha=alpha)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("(b) Individual SFAPs — Straight Fibers")
    ax.grid(True, alpha=0.3)

    # (c) Summed MUAP: curved vs straight
    ax = axes[1, 0]
    t_c, muap_c = sum_muap(sfaps_curved)
    t_s, muap_s = sum_muap(sfaps_straight)
    ax.plot(t_c, muap_c, "b-", linewidth=2, label="Curved fibers")
    ax.plot(t_s, muap_s, "r--", linewidth=2, label="Straight fibers")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title(f"(c) Summed MUAP ({n_fibers} fibers)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) MUAP difference and metrics
    ax = axes[1, 1]
    diff = muap_c - muap_s
    ax.plot(t_c, diff, "k-", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude Difference (V)")
    ax.set_title("(d) MUAP Difference: Curved - Straight")
    ax.grid(True, alpha=0.3)

    # Compute correlation
    if np.std(muap_c) > 0 and np.std(muap_s) > 0:
        r = np.corrcoef(muap_c, muap_s)[0, 1]
        nrmse = np.sqrt(np.mean(diff**2)) / (muap_c.max() - muap_c.min()) * 100
        ptp_c = muap_c.max() - muap_c.min()
        ptp_s = muap_s.max() - muap_s.min()
        ax.text(0.02, 0.98,
                f"r = {r:.4f}\nNRMSE = {nrmse:.2f}%\n"
                f"PtP curved: {ptp_c:.3e}\nPtP straight: {ptp_s:.3e}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"MUAP Waveforms — Label {label}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "muap_waveforms.png", dpi=150, bbox_inches="tight")
    print(f"  Saved muap_waveforms.png")
    plt.close(fig)


def plot_depth_comparison(
    z_values: np.ndarray,
    phi_curved: list[np.ndarray],
    sfaps_curved: list[tuple[np.ndarray, np.ndarray]],
    offsets: list[tuple[float, float]],
    label: int,
):
    """Fig 4: Lead field and SFAP by fiber distance from centerline."""
    # Group fibers by radius
    radii_map: dict[float, list[int]] = {}
    for i, (dx, dy) in enumerate(offsets):
        r = round(np.sqrt(dx**2 + dy**2), 1)
        radii_map.setdefault(r, []).append(i)

    unique_radii = sorted(radii_map.keys())
    n_radii = len(unique_radii)
    colors_r = plt.cm.viridis(np.linspace(0.1, 0.9, n_radii))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ri, r in enumerate(unique_radii):
        indices = radii_map[r]
        # Average lead field and SFAP across fibers at this radius
        phi_avg = np.mean([phi_curved[i] for i in indices], axis=0)
        sfap_avg = np.mean([sfaps_curved[i][1] for i in indices], axis=0)
        t_ms = sfaps_curved[indices[0]][0]

        ax1.plot(z_values, phi_avg, color=colors_r[ri], linewidth=1.5,
                 label=f"r={r:.0f}mm (n={len(indices)})")
        ax2.plot(t_ms, sfap_avg, color=colors_r[ri], linewidth=1.5,
                 label=f"r={r:.0f}mm")

    ax1.set_xlabel("z (mm)")
    ax1.set_ylabel("Potential (V)")
    ax1.set_title("(a) Mean Lead Field by Fiber Radius")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude (V)")
    ax2.set_title("(b) Mean SFAP by Fiber Radius")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Fiber Depth Analysis — Label {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "depth_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved depth_comparison.png")
    plt.close(fig)


def plot_fibers_in_volume(
    fiber_paths: list[np.ndarray],
    offsets: list[tuple[float, float]],
    electrode_point: np.ndarray,
    label: int,
    fiber_model: MuscleFiberModel,
):
    """Fig 5: Fiber sample points within the muscle volume on axial slices."""
    seg = fiber_model.seg_data
    voxel_size = fiber_model.voxel_size

    # Pick 5 z-levels evenly spaced within the fiber z-range
    z_all = fiber_paths[0][:, 2]
    z_levels = np.linspace(z_all[0], z_all[-1], 7)[1:-1]  # skip endpoints

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Fiber Sample Points Within Muscle Volume — Label {label}",
                 fontsize=13, y=1.02)

    # Color fibers by radial offset
    radii = np.array([np.sqrt(dx**2 + dy**2) for dx, dy in offsets])
    r_max = max(radii.max(), 1.0)

    for ax, z_target in zip(axes, z_levels):
        z_idx = int(round(z_target / voxel_size[2]))
        z_idx = np.clip(z_idx, 0, seg.shape[2] - 1)
        z_actual = z_idx * voxel_size[2]

        # Segmentation slice
        slice_data = seg[:, :, z_idx].T
        extent = [0, seg.shape[0] * voxel_size[0],
                  0, seg.shape[1] * voxel_size[1]]

        # Background: all tissue faint
        ax.imshow(slice_data > 0, origin="lower", extent=extent,
                  cmap="Greys", alpha=0.15, interpolation="nearest")

        # Highlight target muscle
        muscle_mask = np.ma.masked_where(slice_data != label, slice_data)
        ax.imshow(muscle_mask, origin="lower", extent=extent,
                  cmap="Blues", alpha=0.4, interpolation="nearest",
                  vmin=label - 1, vmax=label + 1)

        # Show other muscles as outlines
        for other_label in fiber_model.muscles:
            if other_label == label:
                continue
            m = fiber_model.muscles[other_label]
            if m.tissue_type != "muscle":
                continue
            other_mask = slice_data == other_label
            if not np.any(other_mask):
                continue
            # Find boundary pixels
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(other_mask, iterations=1)
            boundary = dilated & ~other_mask
            boundary_ma = np.ma.masked_where(~boundary, boundary.astype(float))
            ax.imshow(boundary_ma, origin="lower", extent=extent,
                      cmap="Greys", alpha=0.3, interpolation="nearest",
                      vmin=0, vmax=1)

        # Plot fiber sample points at this z-level
        for i, path in enumerate(fiber_paths):
            # Find closest z
            idx = np.argmin(np.abs(path[:, 2] - z_actual))
            if abs(path[idx, 2] - z_actual) > 1.5:
                continue
            r = radii[i]
            color = plt.cm.hot(0.2 + 0.6 * r / r_max)
            ms = 7 if r == 0 else 4
            marker = "o" if r == 0 else "."
            ax.plot(path[idx, 0], path[idx, 1], marker, color=color,
                    markersize=ms, markeredgecolor="white" if r == 0 else "none",
                    markeredgewidth=0.5)

        # Electrode
        if abs(electrode_point[2] - z_actual) < 20:
            ax.plot(electrode_point[0], electrode_point[1], "r*",
                    markersize=10, zorder=10)

        # Zoom to muscle region with padding
        muscle_voxels = np.argwhere(seg[:, :, z_idx] == label)
        if len(muscle_voxels) > 0:
            phys = muscle_voxels * voxel_size[:2]
            pad = 10  # mm
            x_lo, x_hi = phys[:, 0].min() - pad, phys[:, 0].max() + pad
            y_lo, y_hi = phys[:, 1].min() - pad, phys[:, 1].max() + pad
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

        ax.set_title(f"z = {z_actual:.0f} mm", fontsize=10)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("y (mm)", fontsize=8)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fibers_in_volume.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fibers_in_volume.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fiber group & potential visualization")
    parser.add_argument("--label", type=int, default=None,
                        help="Muscle label (default: most curved z-aligned muscle)")
    parser.add_argument("--theta", type=float, default=0.0,
                        help="Electrode angle (degrees from +x)")
    parser.add_argument("--z_frac", type=float, default=0.5,
                        help="Electrode z-position (0-1 fraction)")
    parser.add_argument("--n_radial", type=int, default=8,
                        help="Number of fibers per ring")
    parser.add_argument("--radii", type=float, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Fiber ring radii in mm (legacy mode)")
    parser.add_argument("--r_norms", type=float, nargs="+", default=None,
                        help="Normalized radii [0-1] for morphing-disk mode "
                             "(default: 0 0.25 0.5 0.75 0.95)")
    parser.add_argument("--dz", type=float, default=1.0,
                        help="Fiber z-spacing in mm")
    parser.add_argument("--no_morphing", action="store_true",
                        help="Disable morphing-disk mapping (use legacy constant offsets)")
    args = parser.parse_args()

    # Step 1: Load fiber model with centerlines + cross-sections
    print("Loading fiber model with centerlines...")
    fiber_model = MuscleFiberModel(NIFTI_PATH)
    fiber_model.estimate_fibers(method="pca")
    fiber_model.estimate_centerlines(min_slices=3)
    if not args.no_morphing:
        fiber_model.estimate_cross_sections()
    fiber_model.save_config(FIBER_CONFIG)

    # Step 2: Pick muscle and set per-label output directory
    label = pick_muscle(fiber_model, args.label)
    global OUT_DIR
    OUT_DIR = Path(f"mri/figures/fiber_potentials/L{label}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = fiber_model.muscles[label]
    cl = m.centerline
    cs = m.cross_section if not args.no_morphing else None
    print(f"\nSelected muscle: Label {label}")
    print(f"  Tissue: {m.tissue_type}, Voxels: {m.n_voxels}, Slices: {m.n_slices}")
    print(f"  z-range: {m.z_min_mm:.0f} — {m.z_max_mm:.0f} mm (span {m.z_span_mm:.0f} mm)")
    print(f"  PCA fiber angle: {m.fiber_angle_from_z_deg:.1f} deg")
    if cl:
        print(f"  Centerline: {len(cl.z_mm)} waypoints, spline={cl._use_spline}")
    if cs:
        print(f"  Cross-section: {len(cs.z_mm)} slices, {len(cs.theta_deg)} angles, "
              f"r_inset={cs.r_inset}")

    # Step 3: Generate fiber group
    if cs is not None:
        r_norms = args.r_norms or [0, 0.25, 0.5, 0.75, 0.95]
        print(f"\nGenerating fiber group (morphing-disk): {args.n_radial} per ring, "
              f"r_norms={r_norms}...")
    else:
        print(f"\nGenerating fiber group (legacy): {args.n_radial} per ring, "
              f"radii={args.radii} mm...")
    z_values, fiber_paths_curved, offsets = generate_fiber_group(
        cl, n_radial=args.n_radial, radii_mm=args.radii, dz=args.dz,
        cross_section=cs, r_norms=args.r_norms,
    )
    fiber_paths_straight = generate_straight_fibers(cl, offsets, z_values, cross_section=cs)
    print(f"  {len(fiber_paths_curved)} fibers, {len(z_values)} z-samples "
          f"(dz={args.dz}mm, z=[{z_values[0]:.0f}, {z_values[-1]:.0f}])")

    # Step 4: Get electrode position
    # Import FEM solver to get skin surface point
    from mri.core.fem_solver import MRIFEMModel
    # We just need get_skin_surface_point, use a lightweight init
    print(f"\nLocating electrode at theta={args.theta}°, z_frac={args.z_frac}...")
    model_tmp = MRIFEMModel.__new__(MRIFEMModel)
    from dolfinx import io
    from mpi4py import MPI
    model_tmp.mesh, model_tmp.cell_markers, model_tmp.facet_markers = io.gmshio.read_from_msh(
        MESH_PATH, MPI.COMM_WORLD, gdim=3
    )
    model_tmp.mesh_geometry = model_tmp.mesh.geometry
    electrode_point = model_tmp.get_skin_surface_point(args.theta, args.z_frac)
    del model_tmp
    print(f"  Electrode: [{electrode_point[0]:.1f}, {electrode_point[1]:.1f}, {electrode_point[2]:.1f}] mm")

    # Step 5: Solve FEM and evaluate lead fields
    print("\n--- FEM Solve ---")
    all_paths = fiber_paths_curved + fiber_paths_straight
    phi_all = solve_and_evaluate(all_paths, electrode_point)

    n = len(fiber_paths_curved)
    phi_curved = phi_all[:n]
    phi_straight = phi_all[n:]

    print(f"  Lead field range (curved): [{min(p.min() for p in phi_curved):.3e}, "
          f"{max(p.max() for p in phi_curved):.3e}]")

    # Step 6: Compute MUAPs
    print("\n--- MUAP Generation ---")
    fiber_z_extent = z_values[-1] - z_values[0]
    half_len = fiber_z_extent / 2.0
    print(f"  Fiber half-length: {half_len:.1f} mm, dz: {args.dz} mm")

    print("  Computing SFAPs (curved)...")
    sfaps_curved = compute_muaps(phi_curved, args.dz, fiber_half_len_mm=half_len)
    print("  Computing SFAPs (straight)...")
    sfaps_straight = compute_muaps(phi_straight, args.dz, fiber_half_len_mm=half_len)

    t_c, muap_c = sum_muap(sfaps_curved)
    t_s, muap_s = sum_muap(sfaps_straight)
    if np.std(muap_c) > 0 and np.std(muap_s) > 0:
        r = np.corrcoef(muap_c, muap_s)[0, 1]
        print(f"  MUAP correlation (curved vs straight): r = {r:.4f}")
        ptp_c = muap_c.max() - muap_c.min()
        ptp_s = muap_s.max() - muap_s.min()
        print(f"  Peak-to-peak: curved={ptp_c:.3e}, straight={ptp_s:.3e}")

    # Step 7: Generate plots
    print("\n--- Generating Plots ---")
    plot_fiber_group_3d(fiber_paths_curved, fiber_paths_straight, offsets,
                        electrode_point, label)
    plot_lead_fields(z_values, phi_curved, phi_straight, offsets, label)
    plot_muaps(sfaps_curved, sfaps_straight, offsets, label)
    plot_depth_comparison(z_values, phi_curved, sfaps_curved, offsets, label)
    plot_fibers_in_volume(fiber_paths_curved, offsets, electrode_point,
                          label, fiber_model)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
