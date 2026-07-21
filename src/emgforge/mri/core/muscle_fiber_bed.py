"""Muscle-wide fiber bed: sample ALL fibers in each muscle at fixed density.

Real muscle isn't innervated in isolated 'territories' from scratch each
time you want a MUAP. The fibers form a continuous bed; any motor unit is
defined as a SUBSET of those fibers within some territory radius.

This module builds that bed for each muscle once at a chosen fiber
density (fibers / mm² of cross-section at z-mid). MU sampling then
becomes a cheap distance lookup over the precomputed positions.

Four sampling methods supported (see ``build_muscle_beds``):
  - "uniform"   uniform density on the muscle cross-section
  - "poisson"   Bridson Poisson-disk → minimum-spacing biology-like packing
  - "hex"       hex lattice + small jitter (most anatomical, simplest math)
  - "harmonic"  masked-Laplace streamlines (Noura's fibre-geometry method,
                :mod:`emgforge.mri.core.harmonic_fibers`). Default = single-NMJ:
                one full-length fibre per streamline, mid-belly NMJ. With
                ``series=True`` (EXPERIMENTAL) each streamline is cut into SHORT
                in-series fibres, each with an atlas-placed NMJ.

The uniform/poisson/hex methods record each fibre as a FULL-length morphing-disk
path with a single global ``half_mm`` (the RED/GREY assumption). The "harmonic"
method uses curved, non-crossing, depth-following streamlines instead, and records
each fibre's own semi-lengths + NMJ in the per-fibre arrays ``half1_mm`` /
``half2_mm`` / ``posz_mm`` (single-NMJ full-length by default; short in-series when
``series=True``).

The bed records each fiber as a dict:
  - r_norm, theta_deg     morphing-disk coordinates (harmonic: geometric proxy)
  - x_mid, y_mid          physical xy at z-mid
  - path (Nz, 3)          3D path along z (full-length; harmonic: short segment)
  - tangent (Nz, 3)       unit tangents (for σ rotation if wanted)

Typical use:

    from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
    beds = build_muscle_beds(fiber_model, density=2.0, method="poisson")
    bed = beds[7]  # FiberBed for muscle L7
    mu_fibers = bed.fibers_in_mu(cx, cy, radius_mm=3.0)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from emgforge.mri.core.realistic_muap import _xy_to_rnorm_theta, _poisson_disk_on_disk


@dataclass
class FiberBed:
    """All fibers in one muscle's cross-section, plus their 3D paths."""

    label: int
    density: float                       # fibers / mm² target
    method: str                          # "uniform" | "poisson" | "hex" | "harmonic"
    r_norms: np.ndarray                  # (N,)
    theta_degs: np.ndarray               # (N,)
    xy_mid: np.ndarray                   # (N, 2) physical (x, y) at z_mid
    z_vals: np.ndarray                   # (Nz,) common z-grid
    paths: np.ndarray                    # (N, Nz, 3)
    tangents: np.ndarray                 # (N, Nz, 3)
    half_mm: float                       # fiber half-length used downstream
    centroid_xy: np.ndarray              # (2,) muscle centroid at z_mid
    cross_section_area_mm2: float

    # --- SHORT-fibre / placed-IZ arrays (method="harmonic" only) ---------
    # None for uniform/poisson/hex (which use the single global ``half_mm``).
    # When set, each fibre is a short in-series segment with its OWN semi-
    # lengths and NMJ, and ``paths``/``tangents`` are object arrays of
    # variable-length (Nz_i, 3) polylines (uniform ``dz_mm`` arc spacing).
    half1_mm: Optional[np.ndarray] = None      # (N,) NMJ->proximal semi-length
    half2_mm: Optional[np.ndarray] = None      # (N,) NMJ->distal   semi-length
    posz_mm: Optional[np.ndarray] = None       # (N,) NMJ pos in centred-array convention
    iz_fractions: Optional[np.ndarray] = None  # (N,) longitudinal fraction of each NMJ
    is_atlas: Optional[np.ndarray] = None      # (N,) bool: atlas IZ vs geometric fill
    dz_mm: Optional[float] = None              # arc-length step of harmonic paths

    @property
    def is_harmonic(self) -> bool:
        """True when this bed holds short, per-fibre-length harmonic fibres."""
        return self.half1_mm is not None

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


def _build_harmonic_bed(fiber_model, label, dz, min_fibers, grid_mm, atlas, series=False):
    """Build a harmonic-streamline FiberBed for one muscle
    (``emgforge.mri.core.harmonic_fibers``).

    ``series=False`` (default) → single-NMJ: one full-length fibre per streamline,
    mid-belly NMJ. ``series=True`` → EXPERIMENTAL series-fibering: short in-series
    fibres with atlas-placed IZs. Either way each fibre carries its own semi-lengths
    + NMJ, so the bed populates ``half1_mm`` / ``half2_mm`` / ``posz_mm`` and stores
    the variable-length paths/tangents as object arrays.
    """
    from emgforge.mri.core.harmonic_fibers import (
        HarmonicFibreField,
        atlas_fibre_params,
    )

    m = fiber_model.muscles[label]
    mask = fiber_model.seg_data == label
    vs = np.asarray(fiber_model.voxel_size, dtype=float)

    field = HarmonicFibreField(mask, vs, solve=True)
    if series:
        Lf, iz, _ = atlas_fibre_params(label, atlas)
        fibres = field.short_fibers(Lf, iz, grid_mm=grid_mm, dz_mm=dz)
    else:
        fibres = field.long_fibers(grid_mm=grid_mm, dz_mm=dz)
    if len(fibres) < min_fibers:
        return None

    n = len(fibres)
    xy_mid = np.zeros((n, 2))
    half1 = np.zeros(n)
    half2 = np.zeros(n)
    posz = np.zeros(n)
    iz_fr = np.zeros(n)
    is_atl = np.zeros(n, dtype=bool)
    paths = np.empty(n, dtype=object)
    tangents = np.empty(n, dtype=object)
    for i, fb in enumerate(fibres):
        paths[i] = fb.path
        tangents[i] = fb.tangents
        xy_mid[i] = fb.path[len(fb.path) // 2, :2]
        half1[i] = fb.half1_mm
        half2[i] = fb.half2_mm
        posz[i] = fb.posz_mm
        iz_fr[i] = fb.iz_fraction
        is_atl[i] = fb.is_atlas

    cent = field.ctr[:2]
    d = np.linalg.norm(xy_mid - cent, axis=1)
    R_max = float(d.max()) if n else 1.0
    # Geometric r_norm / theta proxies (length-correct; used only for MU
    # sampling bookkeeping, not for the morphing-disk path harmonic bypasses).
    r_norms = d / max(R_max, 1e-6)
    thetas = np.degrees(np.arctan2(xy_mid[:, 1] - cent[1], xy_mid[:, 0] - cent[0])) % 360.0

    return FiberBed(
        label=label, density=float("nan"), method="harmonic",
        r_norms=r_norms, theta_degs=thetas, xy_mid=xy_mid,
        z_vals=np.array([field.z0, field.z1]),
        paths=paths, tangents=tangents,
        half_mm=float(np.mean(half1 + half2) / 2.0),
        centroid_xy=cent.copy(),
        cross_section_area_mm2=float("nan"),
        half1_mm=half1, half2_mm=half2, posz_mm=posz,
        iz_fractions=iz_fr, is_atlas=is_atl, dz_mm=float(dz),
    )


def build_muscle_beds(
    fiber_model,
    density=2.0,
    method="poisson",
    dz=1.0,
    seed=0,
    labels=None,
    min_fibers=30,
    grid_mm=2.0,
    atlas=None,
    series=False,
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
    method : {"uniform", "poisson", "hex", "harmonic"}
        The morphing-disk methods ("uniform"/"poisson"/"hex") build
        full-length fibres with a single global ``half_mm``. "harmonic"
        builds masked-Laplace-streamline fibres (needs only ``seg_data``, not
        centerlines / cross-sections) — single-NMJ full-length by default.
    series : bool
        (harmonic only) EXPERIMENTAL. When True, cut each streamline into short
        in-series fibres with atlas-placed multiple IZs instead of one mid-belly
        NMJ. Emits a warning; the single-NMJ model is the production default.
    dz : float
        z-spacing for fiber paths (mm). For "harmonic" this is the uniform
        arc-length spacing of the short-fibre polylines.
    seed : int
        RNG seed.
    labels : list[int] or None
        If None, build for all viable muscles.
    min_fibers : int
        Skip muscles that yield fewer than this many fibers.
    grid_mm : float
        (harmonic only) cross-section seeding grid spacing (mm).
    atlas : dict or None
        (harmonic only) pre-loaded forearm atlas; loaded from disk if None.

    Returns
    -------
    dict[label] = FiberBed
    """
    rng = np.random.default_rng(seed)
    beds = {}

    if method == "harmonic":
        if series:
            warnings.warn(
                "harmonic series-fibering (multi-NMJ) is EXPERIMENTAL; the single-NMJ "
                "model (series=False) is the production default.",
                stacklevel=2,
            )
        if labels is None:
            labels = sorted(
                l for l, mm in fiber_model.muscles.items()
                if mm.tissue_type == "muscle"
            )
        for label in labels:
            bed = _build_harmonic_bed(
                fiber_model, label, dz, min_fibers, grid_mm, atlas, series=series,
            )
            if bed is not None:
                beds[label] = bed
        return beds

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
