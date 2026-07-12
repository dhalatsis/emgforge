"""
Analytical conductivity computation for FEM-free PINN data generation.

Computes conductivity tensors at arbitrary points using tissue geometry
and known material properties — no FEM solve, mesh, or DOLFINx required.

This enables PINN training data generation using only:
  - Tissue radii from metadata (geometry_params)
  - Conductivity values from emgforge.fem.constants
  - Analytical sampling (emgforge.pointcloud.sampling)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

# Muscle anisotropy from the single source of truth (emgforge.tissue is dolfinx-free,
# so this keeps the analytical path free of the FEM stack). The isotropic layer
# values are the fixed 5-layer physical constants.
from emgforge.tissue import ANISOTROPY_RATIO, SIGMA_MUSCLE_CROSS

_CONDUCTIVITY_ISO = {
    0: 0.075,       # Cancellous Bone (S/m)
    1: 0.02,        # Cortical Bone
    3: 0.0379,      # Fat
    4: 4.55e-4,     # Skin
}
_MUSCLE_TRANS = SIGMA_MUSCLE_CROSS                  # Muscle transverse (S/m) = 0.2455
_MUSCLE_FIBER = ANISOTROPY_RATIO * _MUSCLE_TRANS    # Muscle fiber = 1.2275


def compute_conductivity_analytical(
    points: np.ndarray,
    meta: Dict,
    pennation_angle: float = 0.0,
) -> np.ndarray:
    """
    Compute conductivity tensor at points using geometry only.

    Uses radial distance to determine tissue layer, then assigns
    known conductivity values from emgforge.fem.constants.CONDUCTIVITY.

    Parameters
    ----------
    points : np.ndarray
        Query points of shape (N, 3).
    meta : dict
        Mesh metadata with geometry_params containing tissue radii.
    pennation_angle : float
        Muscle fiber pennation angle in degrees. If non-zero, rotates
        the muscle conductivity tensor via Rodrigues' formula.

    Returns
    -------
    np.ndarray
        Conductivity tensors at points, shape (N, 6).
        Channels: [xx, yy, zz, xy, xz, yz] (symmetric tensor).
    """
    from .evaluate import compute_tissue_labels

    tissue_labels = compute_tissue_labels(meta, points)
    N = len(points)
    sigma = np.zeros((N, 6), dtype=np.float32)

    # Isotropic tissues: [xx, yy, zz, xy, xz, yz] = [s, s, s, 0, 0, 0]
    # Tissue label mapping: 0=cancellous, 1=cortical, 2=muscle, 3=fat, 4=skin
    for label, s in _CONDUCTIVITY_ISO.items():
        mask = tissue_labels == label
        sigma[mask, 0] = s   # xx
        sigma[mask, 1] = s   # yy
        sigma[mask, 2] = s   # zz
        # xy, xz, yz remain 0

    # Muscle: anisotropic (σ_zz = ANISOTROPY_RATIO × σ_xx)
    muscle_mask = tissue_labels == 2
    s_trans = _MUSCLE_TRANS   # 0.2455 (transverse)
    s_fiber = _MUSCLE_FIBER   # 1.2275 (fiber direction = z)

    if abs(pennation_angle) < 1e-6:
        # No pennation — fiber direction is z-axis
        sigma[muscle_mask, 0] = s_trans  # xx
        sigma[muscle_mask, 1] = s_trans  # yy
        sigma[muscle_mask, 2] = s_fiber  # zz
        # Off-diagonal remain 0
    else:
        # Rotate conductivity tensor by pennation angle
        # Pennation rotates fiber direction from z toward x in the xz-plane
        sigma_rot = _rotate_muscle_conductivity(
            s_trans, s_fiber, pennation_angle
        )
        sigma[muscle_mask, 0] = sigma_rot[0]  # xx
        sigma[muscle_mask, 1] = sigma_rot[1]  # yy
        sigma[muscle_mask, 2] = sigma_rot[2]  # zz
        sigma[muscle_mask, 3] = sigma_rot[3]  # xy
        sigma[muscle_mask, 4] = sigma_rot[4]  # xz
        sigma[muscle_mask, 5] = sigma_rot[5]  # yz

    return sigma


def _rotate_muscle_conductivity(
    s_trans: float,
    s_fiber: float,
    pennation_angle_deg: float,
) -> np.ndarray:
    """
    Rotate muscle conductivity tensor by pennation angle.

    Pennation rotates the fiber direction from z toward x in the xz-plane.
    Uses Rodrigues' rotation about the y-axis.

    Parameters
    ----------
    s_trans : float
        Transverse conductivity (xx, yy).
    s_fiber : float
        Fiber direction conductivity (zz).
    pennation_angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        6-component symmetric tensor [xx, yy, zz, xy, xz, yz].
    """
    theta = np.deg2rad(pennation_angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    # Rotation matrix around y-axis
    R = np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c],
    ])

    # Original diagonal conductivity
    S = np.diag([s_trans, s_trans, s_fiber])

    # Rotated: S' = R @ S @ R^T
    S_rot = R @ S @ R.T

    return np.array([
        S_rot[0, 0],  # xx
        S_rot[1, 1],  # yy
        S_rot[2, 2],  # zz
        S_rot[0, 1],  # xy
        S_rot[0, 2],  # xz
        S_rot[1, 2],  # yz
    ], dtype=np.float32)


__all__ = [
    "compute_conductivity_analytical",
]
