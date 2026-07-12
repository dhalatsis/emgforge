"""Muscle-wide fiber bed: sample ALL fibers in each muscle at fixed density.

Real muscle isn't innervated in isolated 'territories' from scratch each
time you want a MUAP. The fibers form a continuous bed; any motor unit is
defined as a SUBSET of those fibers within some territory radius.

This module builds that bed for each muscle once at a chosen fiber
density (fibers / mm² of cross-section at z-mid). MU sampling then
becomes a cheap distance lookup over the precomputed positions.

Three sampling methods supported (see ``FiberBed.build``):
  - "uniform"   uniform density on the muscle cross-section
  - "poisson"   Bridson Poisson-disk → minimum-spacing biology-like packing
  - "hex"       hex lattice + small jitter (most anatomical, simplest math)

The bed records each fiber as a dict:
  - r_norm, theta_deg     morphing-disk coordinates
  - x_mid, y_mid          physical xy at z-mid
  - path (Nz, 3)          full 3D path along z (built once)
  - tangent (Nz, 3)       unit tangents (for σ rotation if wanted)

Typical use:

    from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
    beds = build_muscle_beds(fiber_model, density=2.0, method="poisson")
    bed = beds[7]  # FiberBed for muscle L7
    mu_fibers = bed.fibers_in_mu(cx, cy, radius_mm=3.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from emgforge.mri.core.realistic_muap import _xy_to_rnorm_theta, _poisson_disk_on_disk


@dataclass
class FiberBed:
    """All fibers in one muscle's cross-section, plus their 3D paths."""

    label: int
    density: float                       # fibers / mm² target
    method: str                          # "uniform" | "poisson" | "hex"
    r_norms: np.ndarray                  # (N,)
    theta_degs: np.ndarray               # (N,)
    xy_mid: np.ndarray                   # (N, 2) physical (x, y) at z_mid
    z_vals: np.ndarray                   # (Nz,) common z-grid
    paths: np.ndarray                    # (N, Nz, 3)
    tangents: np.ndarray                 # (N, Nz, 3)
    half_mm: float                       # fiber half-length used downstream
    centroid_xy: np.ndarray              # (2,) muscle centroid at z_mid
    cross_section_area_mm2: float

    def fibers_in_mu(self, mu_cx, mu_cy, radius_mm):
        """Indices of fibers whose (x_mid, y_mid) is within radius_mm
        of (mu_cx, mu_cy)."""
        dx = self.xy_mid[:, 0] - mu_cx
        dy = self.xy_mid[:, 1] - mu_cy
        return np.where(dx * dx + dy * dy < radius_mm * radius_mm)[0]


def _build_uniform(R_max, mx, my, cs, z_mid, n_target, rng):
    """Rejection-sample uniform on the muscle's enclosed disk-like region."""
    keys = []
    tries = 0
    max_tries = n_target * 60
    while len(keys) < n_target and tries < max_tries:
        tries += 1
        # uniform in bounding disk of radius R_max
        r = R_max * np.sqrt(rng.uniform())
        th = 2 * np.pi * rng.uniform()
        x = mx + r * np.cos(th)
        y = my + r * np.sin(th)
        res = _xy_to_rnorm_theta(x, y, mx, my, cs, z_mid)
        if res is not None:
            keys.append(res)
    return keys


def _build_poisson(R_max, mx, my, cs, z_mid, n_target, rng, area_mm2):
    """Poisson-disk over the bounding disk; filter to muscle shape.

    The bounding disk (πR_max²) is usually larger than the actual muscle
    area, so Bridson on the disk + boundary filter under-shoots the
    target. We scale r_min by the disk/muscle area ratio and let
    Bridson saturate, then accept only points inside the muscle.

    Bridson achieves ~0.7 of hex max packing density at the same r_min.
    """
    disk_area = np.pi * R_max ** 2
    # Bridson saturation: density_sat ≈ 0.7 / r_min² · (2/√3) ≈ 0.81/r_min²
    # Choose r_min so that on the disk we get n_disk = n_target * disk/muscle
    n_disk_target = n_target * (disk_area / max(area_mm2, 1e-3))
    target_disk_density = n_disk_target / disk_area
    r_min = float(np.sqrt(0.81 / target_disk_density))
    pts = _poisson_disk_on_disk(
        R_max, r_min, rng, max_pts=int(n_disk_target * 1.4) + 200,
    )
    keys = []
    for dx, dy in pts:
        x, y = mx + dx, my + dy
        res = _xy_to_rnorm_theta(x, y, mx, my, cs, z_mid)
        if res is not None:
            keys.append(res)
    return keys


def _build_hex(R_max, mx, my, cs, z_mid, density, rng, jitter_frac=0.15):
    """Hexagonal lattice + small jitter, clipped to muscle boundary."""
    # spacing s so that hex density = 2/(s² √3) → s = sqrt(2 / (density √3))
    s = float(np.sqrt(2 / (density * np.sqrt(3))))
    keys = []
    # generate hex lattice over bounding box
    n_layers = int(np.ceil(R_max / s)) + 1
    for i in range(-n_layers, n_layers + 1):
        for j in range(-n_layers, n_layers + 1):
            dx = (j + 0.5 * (i % 2)) * s
            dy = i * s * np.sqrt(3) / 2
            if dx * dx + dy * dy > R_max * R_max * 1.05:
                continue
            # add small jitter
            dx += rng.uniform(-jitter_frac, jitter_frac) * s
            dy += rng.uniform(-jitter_frac, jitter_frac) * s
            x, y = mx + dx, my + dy
            res = _xy_to_rnorm_theta(x, y, mx, my, cs, z_mid)
            if res is not None:
                keys.append(res)
    return keys


def build_muscle_beds(
    fiber_model,
    density=2.0,
    method="poisson",
    dz=1.0,
    seed=0,
    labels=None,
    min_fibers=30,
):
    """Build a FiberBed for each muscle in ``fiber_model``.

    Parameters
    ----------
    fiber_model : MuscleFiberModel
        With centerlines + cross-sections already estimated.
    density : float
        Target fiber density (fibers / mm² at z-mid). Real muscle has
        ~50-500 fibers/mm² at typical light-microscopy scale; for our
        coarse MUAP grid 1-5 fibers/mm² is enough to oversample MUs.
    method : {"uniform", "poisson", "hex"}
    dz : float
        z-spacing for fiber paths (mm).
    seed : int
        RNG seed.
    labels : list[int] or None
        If None, build for all viable muscles.
    min_fibers : int
        Skip muscles that yield fewer than this many fibers.

    Returns
    -------
    dict[label] = FiberBed
    """
    rng = np.random.default_rng(seed)
    beds = {}

    if labels is None:
        labels = sorted(
            l for l, m in fiber_model.muscles.items()
            if m.tissue_type == "muscle"
            and m.centerline is not None
            and m.cross_section is not None
        )

    for label in labels:
        m = fiber_model.muscles[label]
        cl, cs = m.centerline, m.cross_section
        z_margin = 0.05 * (cl.z_max - cl.z_min)
        z_vals = np.arange(cl.z_min + z_margin, cl.z_max - z_margin, dz)
        if len(z_vals) < 5:
            continue
        half = 0.5 * float(z_vals[-1] - z_vals[0])

        z_mid = 0.5 * (cl.z_min + cl.z_max)
        cent = cl.position(z_mid)
        mx, my = cent[0], cent[1]
        R_max = float(cs.R_boundary.max()) * cs.r_inset

        # Approx cross-section area at z_mid via ray-cast
        theta_arr = np.linspace(0, 360, 360)
        R_b = cs.boundary_radius(
            np.full_like(theta_arr, z_mid), theta_arr,
        ) * cs.r_inset
        # polygon area (shoelace) of (R, θ) → (x, y)
        xs = R_b * np.cos(np.radians(theta_arr))
        ys = R_b * np.sin(np.radians(theta_arr))
        area = 0.5 * float(np.abs(
            np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)
        ))
        n_target = max(int(round(density * area)), 1)

        if method == "uniform":
            keys = _build_uniform(R_max, mx, my, cs, z_mid, n_target, rng)
        elif method == "poisson":
            keys = _build_poisson(R_max, mx, my, cs, z_mid, n_target, rng, area)
        elif method == "hex":
            keys = _build_hex(R_max, mx, my, cs, z_mid, density, rng)
        else:
            raise ValueError(method)

        if len(keys) < min_fibers:
            continue

        # Build paths + tangents (vectorized loop)
        n = len(keys)
        paths = np.zeros((n, len(z_vals), 3))
        tangents = np.zeros((n, len(z_vals), 3))
        xy_mid = np.zeros((n, 2))
        rnorms = np.zeros(n)
        thetas = np.zeros(n)
        for fi, (rn, th) in enumerate(keys):
            paths[fi] = cl.fiber_path_morphing(rn, th, z_vals, cs)
            tangents[fi] = cl.fiber_tangent_morphing(rn, th, z_vals, cs)
            xy_mid[fi] = paths[fi][len(z_vals) // 2, :2]
            rnorms[fi] = rn
            thetas[fi] = th

        beds[label] = FiberBed(
            label=label, density=density, method=method,
            r_norms=rnorms, theta_degs=thetas, xy_mid=xy_mid,
            z_vals=z_vals, paths=paths, tangents=tangents,
            half_mm=half, centroid_xy=np.array([mx, my]),
            cross_section_area_mm2=area,
        )

    return beds
