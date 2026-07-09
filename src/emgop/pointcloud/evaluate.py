"""
Evaluate FEM solutions and conductivity tensors at arbitrary points.

Provides functions to query the FEM solution and material properties
at point cloud locations for neural network training data generation.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path


def evaluate_fem_solution(
    model,
    uh: fem.Function,
    points: np.ndarray,
) -> np.ndarray:
    """
    Evaluate FEM solution at given points.

    Parameters
    ----------
    model : FEMModel
        FEM model with mesh and bounding box tree.
    uh : fem.Function
        Solution function to evaluate.
    points : np.ndarray
        Query points of shape (N, 3).

    Returns
    -------
    np.ndarray
        Solution values at points, shape (N,).
    """
    from dolfinx import fem, geometry  # lazy import: FEM-only dependency
    return model.evaluate_solution_at_points(points, uh=uh)


def evaluate_conductivity_at_points(
    model,
    points: np.ndarray,
) -> np.ndarray:
    """
    Evaluate conductivity tensor at given points.

    Parameters
    ----------
    model : FEMModel
        FEM model with sigma_anisotropic function.
    points : np.ndarray
        Query points of shape (N, 3).

    Returns
    -------
    np.ndarray
        Conductivity tensors at points, shape (N, 6).
        Channels: [xx, yy, zz, xy, xz, yz]
    """
    from dolfinx import geometry  # lazy import: FEM-only dependency

    if not hasattr(model, "sigma_anisotropic"):
        raise ValueError("Model does not have sigma_anisotropic (conductivity map not built)")

    # Find cells containing each point
    cell_ids = geometry.compute_closest_entity(
        model.tree, model.midpoints, model.mesh, points
    ).squeeze()

    # Get sigma values - sigma_anisotropic is DG0 tensor, constant per cell
    sigma_vec = np.asarray(model.sigma_anisotropic.vector.array)

    # Each cell has 9 values (3x3 tensor flattened)
    n_components = 9
    sigma_per_cell = sigma_vec.reshape(-1, n_components)

    # Extract sigma for each point's cell
    sigma_full = sigma_per_cell[cell_ids]  # (N, 9)

    # Convert from full 3x3 to symmetric 6-component format
    # Full tensor layout (row-major): [00, 01, 02, 10, 11, 12, 20, 21, 22]
    # Symmetric format: [xx, yy, zz, xy, xz, yz]
    sigma_6 = np.zeros((len(points), 6), dtype=np.float32)
    sigma_6[:, 0] = sigma_full[:, 0]  # xx = [0,0]
    sigma_6[:, 1] = sigma_full[:, 4]  # yy = [1,1]
    sigma_6[:, 2] = sigma_full[:, 8]  # zz = [2,2]
    sigma_6[:, 3] = sigma_full[:, 1]  # xy = [0,1]
    sigma_6[:, 4] = sigma_full[:, 2]  # xz = [0,2]
    sigma_6[:, 5] = sigma_full[:, 5]  # yz = [1,2]

    return sigma_6


def compute_tissue_labels(
    meta: Dict,
    points: np.ndarray,
) -> np.ndarray:
    """
    Compute tissue label for each point based on geometry.

    Supports circular, elliptical, off-center bone, two bones, and tapering.

    Parameters
    ----------
    meta : dict
        Mesh metadata with geometry_params.
    points : np.ndarray
        Query points of shape (N, 3).

    Returns
    -------
    np.ndarray
        Tissue labels (N,). Values: 0=cancellous, 1=cortical, 2=muscle, 3=fat, 4=skin, -1=outside.
    """
    from .geometry import compute_tissue_labels_general
    return compute_tissue_labels_general(meta, points)


def compute_source_field(
    points: np.ndarray,
    source_position: np.ndarray,
    source_sigma: float = 0.1,
    volume: Optional[float] = None,
    meta: Optional[Dict] = None,
) -> np.ndarray:
    """
    Compute source field f(x) at points (for PINN training).

    Uses Gaussian source with mean-zero constraint:
    f(x) = G(x; x_s, sigma) - 1/V

    Parameters
    ----------
    points : np.ndarray
        Query points of shape (N, 3).
    source_position : np.ndarray
        Source electrode position (3,).
    source_sigma : float
        Gaussian standard deviation.
    volume : float, optional
        Domain volume. Estimated from meta if not provided.
    meta : dict, optional
        Mesh metadata (for volume estimation).

    Returns
    -------
    np.ndarray
        Source field values at points, shape (N,).
    """
    # Gaussian blob
    diff = points - source_position.reshape(1, 3)
    dist_sq = np.sum(diff**2, axis=1)
    sigma_sq = source_sigma**2
    gauss = np.exp(-dist_sq / (2 * sigma_sq)) / ((2 * np.pi * sigma_sq) ** 1.5)

    # Estimate volume if not provided
    if volume is None:
        if meta is not None:
            gp = meta["geometry_params"]
            r_skin = float(gp["radius_skin"])
            length = float(gp["length"])
            volume = np.pi * r_skin**2 * length
        else:
            raise ValueError("Either volume or meta must be provided")

    # Mean-zero source
    gauss_integral = np.mean(gauss) * volume  # Approximate integral
    f = gauss - gauss_integral / volume

    return f.astype(np.float32)


def evaluate_all_fields(
    model,
    uh: fem.Function,
    points: np.ndarray,
    meta: Dict,
    source_position: np.ndarray,
    source_sigma: float = 0.1,
    compute_source: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Evaluate all fields at points for training data.

    Parameters
    ----------
    model : FEMModel
        FEM model.
    uh : fem.Function
        FEM solution.
    points : np.ndarray
        Query points (N, 3).
    meta : dict
        Mesh metadata.
    source_position : np.ndarray
        Electrode position.
    source_sigma : float
        Gaussian source sigma.
    compute_source : bool
        Whether to compute source field (for PINN).

    Returns
    -------
    dict
        Dictionary with:
        - "u": Solution values (N,)
        - "sigma": Conductivity tensors (N, 6)
        - "tissue_labels": Tissue labels (N,)
        - "source_field": Source field values (N,) [if compute_source]
    """
    result = {}

    # Solution
    result["u"] = evaluate_fem_solution(model, uh, points).astype(np.float32)

    # Conductivity
    result["sigma"] = evaluate_conductivity_at_points(model, points)

    # Tissue labels
    result["tissue_labels"] = compute_tissue_labels(meta, points)

    # Source field (for PINN)
    if compute_source:
        result["source_field"] = compute_source_field(
            points, source_position, source_sigma, meta=meta
        )

    return result


__all__ = [
    "evaluate_fem_solution",
    "evaluate_conductivity_at_points",
    "compute_tissue_labels",
    "compute_source_field",
    "evaluate_all_fields",
]
