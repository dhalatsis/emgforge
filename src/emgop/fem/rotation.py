"""
Conductivity tensor rotation for pennation angle support.

Pennation angle represents the angle of muscle fibers relative to the
longitudinal axis of the cylinder. This module provides functions to
rotate the conductivity tensor to represent angled muscle fibers.
"""

import numpy as np


def rotate_conductivity_tensor(
    centroid: np.ndarray,
    tensor: np.ndarray,
    theta_deg: float,
) -> np.ndarray:
    """
    Rotate a conductivity tensor by theta degrees around an axis perpendicular
    to the radial direction at the given centroid.

    For a cylindrical muscle model, this simulates pennation where muscle fibers
    are oriented at an angle to the longitudinal (Z) axis. The rotation axis
    is tangent to the cylinder surface (perpendicular to the radial direction
    in the XY plane).

    Parameters
    ----------
    centroid : np.ndarray
        Cell centroid (x, y, z) in Cartesian coordinates.
    tensor : np.ndarray
        3x3 conductivity tensor to rotate.
    theta_deg : float
        Rotation angle in degrees. Positive angles tilt the fiber direction
        away from Z toward the radial direction.

    Returns
    -------
    np.ndarray
        Rotated 3x3 conductivity tensor.

    Notes
    -----
    Uses Rodrigues' rotation formula: R = I + sin(θ)K + (1-cos(θ))K²
    where K is the skew-symmetric matrix from the rotation axis.

    The tensor is rotated as: σ' = R σ Rᵀ
    """
    theta_rad = np.radians(theta_deg)

    x, y, z = centroid

    # Radial vector in XY plane
    v_xy = np.array([x, y, 0.0])
    r_xy = np.linalg.norm(v_xy)

    # At or near center, rotation is undefined - return unchanged
    if r_xy < 1e-10:
        return tensor.copy()

    # Rotation axis: perpendicular to radial direction in XY plane
    # This is the tangent direction to the cylinder surface
    k = np.array([0.0, 0.0, 1.0])
    v_perp = -np.cross(v_xy, k)
    v_perp = v_perp / np.linalg.norm(v_perp)

    # Rodrigues' rotation formula components
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Skew-symmetric matrix K from rotation axis v_perp
    K = np.array([
        [0.0, -v_perp[2], v_perp[1]],
        [v_perp[2], 0.0, -v_perp[0]],
        [-v_perp[1], v_perp[0], 0.0],
    ])

    # Rotation matrix: R = I + sin(θ)K + (1-cos(θ))K²
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)

    # Rotate tensor: σ' = R σ Rᵀ
    rotated = R @ tensor @ R.T

    return rotated


def rotate_point_in_cylinder(
    point: np.ndarray,
    matrix: np.ndarray,
    theta: float,
) -> np.ndarray:
    """
    Rotate a conductivity tensor at a point in a cylinder.

    This is an alias for rotate_conductivity_tensor() for backwards
    compatibility with existing code.

    Parameters
    ----------
    point : np.ndarray
        Point coordinates (x, y, z).
    matrix : np.ndarray
        3x3 conductivity tensor.
    theta : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        Rotated 3x3 conductivity tensor.
    """
    return rotate_conductivity_tensor(point, matrix, theta)


__all__ = ["rotate_conductivity_tensor", "rotate_point_in_cylinder"]
