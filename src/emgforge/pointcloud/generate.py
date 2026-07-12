"""
Point cloud dataset generation from FEM solutions.

Provides high-level functions to generate training data in point cloud format,
suitable for neural field training and PINN training.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from .sampling import sample_points, sample_boundary_points
from .evaluate import evaluate_all_fields, compute_tissue_labels


def generate_pointcloud_sample(
    model,
    uh,
    meta: Dict,
    source_position: np.ndarray,
    n_interior_points: int = 10000,
    n_boundary_points: int = 2000,
    sampling_strategy: str = "uniform",
    seed: int = 42,
    ground_position: Optional[np.ndarray] = None,
    pennation_angle: float = 0.0,
    source_sigma: float = 0.1,
    include_pinn_data: bool = True,
    **sampling_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Generate a complete point cloud training sample from an FEM solution.

    Parameters
    ----------
    model : FEMModel
        FEM model with solved mesh.
    uh : fem.Function
        FEM solution function.
    meta : dict
        Mesh metadata with geometry_params.
    source_position : np.ndarray
        Source electrode position (3,).
    n_interior_points : int
        Number of interior domain points to sample.
    n_boundary_points : int
        Number of boundary points (for PINN). Set to 0 to skip.
    sampling_strategy : str
        Interior point sampling strategy.
    seed : int
        Random seed for reproducibility.
    ground_position : np.ndarray, optional
        Ground electrode position (for bipolar config).
    pennation_angle : float
        Pennation angle in degrees (0 if not used).
    source_sigma : float
        Gaussian source sigma for source field computation.
    include_pinn_data : bool
        Include data needed for PINN (source field, boundary data).
    **sampling_kwargs : dict
        Additional arguments for sampling strategy.

    Returns
    -------
    dict
        Training sample dictionary with:

        Interior data:
        - "points": (N, 3) - Query point coordinates
        - "u": (N,) - FEM solution at points
        - "sigma": (N, 6) - Conductivity tensor at points [xx,yy,zz,xy,xz,yz]
        - "tissue_labels": (N,) - Tissue label per point (0-4)

        Electrode data:
        - "electrode_position": (3,) - Source electrode coords
        - "ground_position": (3,) - Ground electrode coords (if bipolar)

        PINN data (if include_pinn_data):
        - "source_field": (N,) - Source term f(x) at interior points
        - "boundary_points": (M, 3) - Boundary point coordinates
        - "boundary_normals": (M, 3) - Outward unit normals

        Metadata:
        - "pennation_angle": (1,) - Pennation angle in degrees
        - "r_skin": (1,) - Cylinder radius
        - "length": (1,) - Cylinder length
    """
    rng = np.random.default_rng(seed)

    gp = meta["geometry_params"]
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])

    # Sample interior points
    sampling_result = sample_points(
        meta=meta,
        n_points=n_interior_points,
        strategy=sampling_strategy,
        rng=rng,
        electrode_position=source_position,
        ground_position=ground_position,
        **sampling_kwargs,
    )
    interior_points = sampling_result["points"]

    # Evaluate fields at interior points
    fields = evaluate_all_fields(
        model=model,
        uh=uh,
        points=interior_points,
        meta=meta,
        source_position=source_position,
        source_sigma=source_sigma,
        compute_source=include_pinn_data,
    )

    # Build result dictionary
    result = {
        # Interior data
        "points": interior_points.astype(np.float32),
        "u": fields["u"],
        "sigma": fields["sigma"],
        "tissue_labels": fields["tissue_labels"],

        # Electrode data
        "electrode_position": source_position.astype(np.float32),

        # Metadata
        "pennation_angle": np.array([pennation_angle], dtype=np.float32),
        "r_skin": np.array([r_skin], dtype=np.float32),
        "length": np.array([length], dtype=np.float32),
    }

    if ground_position is not None:
        result["ground_position"] = ground_position.astype(np.float32)

    # Add tissue labels from sampling if stratified
    if "tissue_labels" in sampling_result:
        result["tissue_labels"] = sampling_result["tissue_labels"]

    # PINN data
    if include_pinn_data:
        result["source_field"] = fields["source_field"]

        if n_boundary_points > 0:
            boundary_pts, boundary_normals = sample_boundary_points(
                meta=meta,
                n_points=n_boundary_points,
                rng=rng,
                include_caps=True,
            )
            result["boundary_points"] = boundary_pts.astype(np.float32)
            result["boundary_normals"] = boundary_normals.astype(np.float32)

    return result


def save_pointcloud_sample(
    sample: Dict[str, np.ndarray],
    output_path: Path,
    compressed: bool = True,
):
    """
    Save a point cloud sample to disk.

    Parameters
    ----------
    sample : dict
        Point cloud sample from generate_pointcloud_sample().
    output_path : Path
        Output file path (.npz).
    compressed : bool
        Use compression (smaller files, slightly slower).
    """
    if compressed:
        np.savez_compressed(output_path, **sample)
    else:
        np.savez(output_path, **sample)


def load_pointcloud_sample(path: Path) -> Dict[str, np.ndarray]:
    """
    Load a point cloud sample from disk.

    Parameters
    ----------
    path : Path
        Path to .npz file.

    Returns
    -------
    dict
        Point cloud sample dictionary.
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}


__all__ = [
    "generate_pointcloud_sample",
    "save_pointcloud_sample",
    "load_pointcloud_sample",
]
