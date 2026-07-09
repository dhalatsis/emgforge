"""
Point cloud sampling strategies for neural network training.

Supports various sampling strategies for interior points and boundary points,
suitable for both neural field training and PINN training.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, List
from enum import Enum


class SamplingStrategy(Enum):
    """Available point sampling strategies."""
    UNIFORM = "uniform"
    NEAR_ELECTRODE = "near_electrode"
    STRATIFIED_TISSUE = "stratified_tissue"
    SURFACE_BIASED = "surface_biased"
    MIXED = "mixed"


def sample_points_uniform_cylinder(
    r_skin: float,
    length: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample points uniformly inside a cylinder.

    Parameters
    ----------
    r_skin : float
        Cylinder radius.
    length : float
        Cylinder length (z from 0 to length).
    n_points : int
        Number of points to sample.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Points array of shape (n_points, 3).
    """
    points = []
    while len(points) < n_points:
        # Oversample to account for rejection
        n_needed = (n_points - len(points)) * 2

        # Sample in bounding box
        x = rng.uniform(-r_skin, r_skin, n_needed)
        y = rng.uniform(-r_skin, r_skin, n_needed)
        z = rng.uniform(0, length, n_needed)

        # Keep points inside cylinder
        r = np.sqrt(x**2 + y**2)
        mask = r <= r_skin

        valid_points = np.stack([x[mask], y[mask], z[mask]], axis=1)
        points.extend(valid_points.tolist())

    return np.array(points[:n_points], dtype=np.float64)


def sample_points_near_electrode(
    r_skin: float,
    length: float,
    n_points: int,
    electrode_position: np.ndarray,
    concentration_radius: float,
    concentration_fraction: float,
    rng: np.random.Generator,
    ground_position: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample points with higher density near electrode(s).

    Parameters
    ----------
    r_skin : float
        Cylinder radius.
    length : float
        Cylinder length.
    n_points : int
        Total number of points.
    electrode_position : np.ndarray
        Source electrode position (3,).
    concentration_radius : float
        Radius around electrode for concentrated sampling.
    concentration_fraction : float
        Fraction of points to place near electrode(s) (0-1).
    rng : np.random.Generator
        Random number generator.
    ground_position : np.ndarray, optional
        Ground electrode position for bipolar config.

    Returns
    -------
    np.ndarray
        Points array of shape (n_points, 3).
    """
    n_concentrated = int(n_points * concentration_fraction)
    n_uniform = n_points - n_concentrated

    # Uniform samples
    uniform_pts = sample_points_uniform_cylinder(r_skin, length, n_uniform, rng)

    # Concentrated samples near electrode(s)
    electrodes = [electrode_position]
    if ground_position is not None:
        electrodes.append(ground_position)

    n_per_electrode = n_concentrated // len(electrodes)
    concentrated_pts = []

    for elec_pos in electrodes:
        pts = _sample_sphere_clipped(
            center=elec_pos,
            radius=concentration_radius,
            n_points=n_per_electrode,
            r_skin=r_skin,
            length=length,
            rng=rng,
        )
        concentrated_pts.append(pts)

    concentrated_pts = np.vstack(concentrated_pts)

    return np.vstack([uniform_pts, concentrated_pts])


def _sample_sphere_clipped(
    center: np.ndarray,
    radius: float,
    n_points: int,
    r_skin: float,
    length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points in a sphere, clipped to cylinder bounds."""
    points = []
    max_attempts = n_points * 20
    attempts = 0

    while len(points) < n_points and attempts < max_attempts:
        # Sample in sphere using rejection
        n_needed = (n_points - len(points)) * 4

        # Uniform in cube, reject outside sphere
        offset = rng.uniform(-radius, radius, (n_needed, 3))
        dist = np.linalg.norm(offset, axis=1)
        mask_sphere = dist <= radius

        pts = center + offset[mask_sphere]

        # Clip to cylinder
        r_pts = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        mask_cyl = (r_pts <= r_skin) & (pts[:, 2] >= 0) & (pts[:, 2] <= length)

        valid_pts = pts[mask_cyl]
        points.extend(valid_pts.tolist())
        attempts += n_needed

    if len(points) < n_points:
        # Fill remainder with uniform if sphere is mostly outside domain
        extra = sample_points_uniform_cylinder(
            r_skin, length, n_points - len(points), rng
        )
        points.extend(extra.tolist())

    return np.array(points[:n_points], dtype=np.float64)


def sample_points_stratified_tissue(
    meta: Dict,
    n_points: int,
    rng: np.random.Generator,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points stratified by tissue layer.

    Parameters
    ----------
    meta : dict
        Mesh metadata with geometry_params.
    n_points : int
        Total number of points.
    rng : np.random.Generator
        Random number generator.
    weights : dict, optional
        Sampling weights per tissue. Default: equal weight.
        Keys: "cancellous", "cortical", "muscle", "fat", "skin"

    Returns
    -------
    points : np.ndarray
        Points array of shape (n_points, 3).
    tissue_labels : np.ndarray
        Tissue label for each point (0-4).
    """
    gp = meta["geometry_params"]
    r_canc = float(gp["radius_canc_bone"])
    r_cort = float(gp["radius_cort_bone"])
    r_musc = float(gp["radius_muscle"])
    r_fat = float(gp["radius_fat"])
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])

    # Tissue boundaries (inner, outer radius)
    tissues = {
        "cancellous": (0, r_canc, 0),
        "cortical": (r_canc, r_cort, 1),
        "muscle": (r_cort, r_musc, 2),
        "fat": (r_musc, r_fat, 3),
        "skin": (r_fat, r_skin, 4),
    }

    if weights is None:
        weights = {k: 1.0 for k in tissues}

    total_weight = sum(weights.values())

    all_points = []
    all_labels = []

    for tissue_name, (r_inner, r_outer, label) in tissues.items():
        w = weights.get(tissue_name, 1.0) / total_weight
        n_tissue = max(1, int(n_points * w))

        pts = _sample_annular_cylinder(
            r_inner, r_outer, length, n_tissue, rng
        )
        all_points.append(pts)
        all_labels.append(np.full(len(pts), label, dtype=np.int32))

    points = np.vstack(all_points)
    labels = np.concatenate(all_labels)

    # Shuffle and trim to exact count
    idx = rng.permutation(len(points))[:n_points]
    return points[idx], labels[idx]


def _sample_annular_cylinder(
    r_inner: float,
    r_outer: float,
    length: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniformly in an annular cylinder region."""
    points = []

    while len(points) < n_points:
        n_needed = (n_points - len(points)) * 2

        # Sample radius uniformly in area (r^2 distribution)
        r_sq = rng.uniform(r_inner**2, r_outer**2, n_needed)
        r = np.sqrt(r_sq)
        theta = rng.uniform(0, 2 * np.pi, n_needed)
        z = rng.uniform(0, length, n_needed)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        pts = np.stack([x, y, z], axis=1)
        points.extend(pts.tolist())

    return np.array(points[:n_points], dtype=np.float64)


def sample_points_surface_biased(
    meta: Dict,
    n_points: int,
    rng: np.random.Generator,
    surface_fraction: float = 0.5,
    surface_depth: float = 5.0,
) -> np.ndarray:
    """
    Sample with bias toward the skin surface.

    Parameters
    ----------
    meta : dict
        Mesh metadata.
    n_points : int
        Total number of points.
    rng : np.random.Generator
        Random number generator.
    surface_fraction : float
        Fraction of points in surface layer.
    surface_depth : float
        Depth of surface layer (mesh units).

    Returns
    -------
    np.ndarray
        Points array of shape (n_points, 3).
    """
    gp = meta["geometry_params"]
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])

    n_surface = int(n_points * surface_fraction)
    n_interior = n_points - n_surface

    # Interior points (uniform in inner region)
    interior_pts = sample_points_uniform_cylinder(
        r_skin - surface_depth, length, n_interior, rng
    )

    # Surface points (in annular shell)
    surface_pts = _sample_annular_cylinder(
        r_skin - surface_depth, r_skin, length, n_surface, rng
    )

    return np.vstack([interior_pts, surface_pts])


def sample_boundary_points(
    meta: Dict,
    n_points: int,
    rng: np.random.Generator,
    include_caps: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points on the domain boundary (for PINN Neumann BC).

    Parameters
    ----------
    meta : dict
        Mesh metadata.
    n_points : int
        Total number of boundary points.
    rng : np.random.Generator
        Random number generator.
    include_caps : bool
        Include top/bottom caps of cylinder.

    Returns
    -------
    points : np.ndarray
        Boundary points (n_points, 3).
    normals : np.ndarray
        Outward unit normals (n_points, 3).
    """
    gp = meta["geometry_params"]

    if _is_complex_geometry(gp):
        from .geometry import sample_boundary_points_general
        return sample_boundary_points_general(meta, n_points, rng, include_caps)

    # Fast path: simple circular cylinder
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])

    if include_caps:
        n_lateral = int(n_points * 0.7)
        n_cap = (n_points - n_lateral) // 2
        n_bottom = n_cap
        n_top = n_points - n_lateral - n_bottom
    else:
        n_lateral = n_points
        n_bottom = n_top = 0

    points = []
    normals = []

    # Lateral surface (r = r_skin)
    theta = rng.uniform(0, 2 * np.pi, n_lateral)
    z = rng.uniform(0, length, n_lateral)
    x = r_skin * np.cos(theta)
    y = r_skin * np.sin(theta)

    lateral_pts = np.stack([x, y, z], axis=1)
    lateral_normals = np.stack([np.cos(theta), np.sin(theta), np.zeros(n_lateral)], axis=1)

    points.append(lateral_pts)
    normals.append(lateral_normals)

    if n_bottom > 0:
        r_sq = rng.uniform(0, r_skin**2, n_bottom)
        r = np.sqrt(r_sq)
        theta = rng.uniform(0, 2 * np.pi, n_bottom)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(n_bottom)

        bottom_pts = np.stack([x, y, z], axis=1)
        bottom_normals = np.tile([0, 0, -1], (n_bottom, 1)).astype(np.float64)

        points.append(bottom_pts)
        normals.append(bottom_normals)

    if n_top > 0:
        r_sq = rng.uniform(0, r_skin**2, n_top)
        r = np.sqrt(r_sq)
        theta = rng.uniform(0, 2 * np.pi, n_top)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full(n_top, length)

        top_pts = np.stack([x, y, z], axis=1)
        top_normals = np.tile([0, 0, 1], (n_top, 1)).astype(np.float64)

        points.append(top_pts)
        normals.append(top_normals)

    return np.vstack(points), np.vstack(normals)


def sample_points(
    meta: Dict,
    n_points: int,
    strategy: str = "uniform",
    rng: Optional[np.random.Generator] = None,
    seed: int = 42,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    High-level point sampling function.

    Parameters
    ----------
    meta : dict
        Mesh metadata with geometry_params.
    n_points : int
        Number of interior points to sample.
    strategy : str
        Sampling strategy: "uniform", "near_electrode", "stratified_tissue",
        "surface_biased", "mixed".
    rng : np.random.Generator, optional
        Random number generator. Created from seed if not provided.
    seed : int
        Random seed (used if rng not provided).
    **kwargs : dict
        Strategy-specific parameters:
        - near_electrode: electrode_position, ground_position,
          concentration_radius (default: 10.0), concentration_fraction (default: 0.3)
        - stratified_tissue: weights (dict)
        - surface_biased: surface_fraction (default: 0.5), surface_depth (default: 5.0)
        - mixed: combines multiple strategies

    Returns
    -------
    dict
        Dictionary with "points" and optionally "tissue_labels".
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    gp = meta["geometry_params"]
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])

    # Detect if we need geometry-aware sampling
    _complex = _is_complex_geometry(gp)

    result = {}

    if strategy == "uniform":
        if _complex:
            from .geometry import sample_points_uniform_general
            points = sample_points_uniform_general(meta, n_points, rng)
        else:
            points = sample_points_uniform_cylinder(r_skin, length, n_points, rng)
        result["points"] = points

    elif strategy == "near_electrode":
        electrode_position = kwargs.get("electrode_position")
        if electrode_position is None:
            raise ValueError("near_electrode strategy requires electrode_position")

        if _complex:
            from .geometry import sample_points_uniform_general
            # Near-electrode: concentrated fraction near electrode, rest uniform
            frac = kwargs.get("concentration_fraction", 0.3)
            c_radius = kwargs.get("concentration_radius", 10.0)
            n_conc = int(n_points * frac)
            n_uni = n_points - n_conc

            uni_pts = sample_points_uniform_general(meta, n_uni, rng)
            conc_pts = _sample_sphere_clipped_general(
                center=electrode_position, radius=c_radius,
                n_points=n_conc, meta=meta, rng=rng,
            )
            ground_position = kwargs.get("ground_position")
            if ground_position is not None:
                n_ground = n_conc // 3
                ground_pts = _sample_sphere_clipped_general(
                    center=ground_position, radius=c_radius,
                    n_points=n_ground, meta=meta, rng=rng,
                )
                conc_pts = np.vstack([conc_pts[:n_conc - n_ground], ground_pts])
            points = np.vstack([uni_pts, conc_pts])
        else:
            points = sample_points_near_electrode(
                r_skin=r_skin,
                length=length,
                n_points=n_points,
                electrode_position=electrode_position,
                concentration_radius=kwargs.get("concentration_radius", 10.0),
                concentration_fraction=kwargs.get("concentration_fraction", 0.3),
                rng=rng,
                ground_position=kwargs.get("ground_position"),
            )
        result["points"] = points

    elif strategy == "stratified_tissue":
        if _complex:
            # For complex geometries, sample uniformly then label
            from .geometry import sample_points_uniform_general, compute_tissue_labels_general
            points = sample_points_uniform_general(meta, n_points, rng)
            labels = compute_tissue_labels_general(meta, points)
            result["points"] = points
            result["tissue_labels"] = labels
        else:
            points, labels = sample_points_stratified_tissue(
                meta=meta,
                n_points=n_points,
                rng=rng,
                weights=kwargs.get("weights"),
            )
            result["points"] = points
            result["tissue_labels"] = labels

    elif strategy == "surface_biased":
        if _complex:
            from .geometry import sample_points_uniform_general
            points = sample_points_uniform_general(meta, n_points, rng)
        else:
            points = sample_points_surface_biased(
                meta=meta,
                n_points=n_points,
                rng=rng,
                surface_fraction=kwargs.get("surface_fraction", 0.5),
                surface_depth=kwargs.get("surface_depth", 5.0),
            )
        result["points"] = points

    elif strategy == "mixed":
        # Combine strategies: 40% uniform, 30% near electrode, 30% surface
        n_uniform = int(n_points * 0.4)
        n_electrode = int(n_points * 0.3)
        n_surface = n_points - n_uniform - n_electrode

        if _complex:
            from .geometry import sample_points_uniform_general
            uniform_pts = sample_points_uniform_general(meta, n_uniform, rng)
        else:
            uniform_pts = sample_points_uniform_cylinder(r_skin, length, n_uniform, rng)

        electrode_position = kwargs.get("electrode_position")
        if electrode_position is not None:
            if _complex:
                electrode_pts = _sample_sphere_clipped_general(
                    electrode_position, kwargs.get("concentration_radius", 10.0),
                    n_electrode, meta, rng,
                )
            else:
                electrode_pts = sample_points_near_electrode(
                    r_skin, length, n_electrode, electrode_position,
                    kwargs.get("concentration_radius", 10.0), 1.0, rng,
                    kwargs.get("ground_position"),
                )
        else:
            if _complex:
                from .geometry import sample_points_uniform_general
                electrode_pts = sample_points_uniform_general(meta, n_electrode, rng)
            else:
                electrode_pts = sample_points_uniform_cylinder(r_skin, length, n_electrode, rng)

        if _complex:
            from .geometry import sample_points_uniform_general
            surface_pts = sample_points_uniform_general(meta, n_surface, rng)
        else:
            surface_pts = sample_points_surface_biased(
                meta, n_surface, rng,
                surface_fraction=1.0,
                surface_depth=kwargs.get("surface_depth", 5.0),
            )

        result["points"] = np.vstack([uniform_pts, electrode_pts, surface_pts])

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return result


def _is_complex_geometry(gp: Dict) -> bool:
    """Check if geometry requires general (non-circular) sampling."""
    if gp.get("shape", "circle") == "ellipse":
        return True
    if gp.get("z_profile", "constant") == "taper":
        return True
    if int(gp.get("n_bones", gp.get("bone_count", 1)) or 1) > 1:
        return True
    if "bone1_center_x" in gp:
        cx = float(gp["bone1_center_x"])
        cy = float(gp["bone1_center_y"])
        if abs(cx) > 1e-6 or abs(cy) > 1e-6:
            return True
    return False


def _sample_sphere_clipped_general(
    center: np.ndarray,
    radius: float,
    n_points: int,
    meta: Dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points in a sphere, clipped to general domain bounds."""
    from .geometry import point_inside_domain

    points = []
    max_attempts = n_points * 30
    attempts = 0

    while len(points) < n_points and attempts < max_attempts:
        n_needed = (n_points - len(points)) * 4
        offset = rng.uniform(-radius, radius, (n_needed, 3))
        dist = np.linalg.norm(offset, axis=1)
        mask_sphere = dist <= radius

        pts = center + offset[mask_sphere]
        if len(pts) == 0:
            attempts += n_needed
            continue

        mask_domain = point_inside_domain(pts[:, 0], pts[:, 1], pts[:, 2], meta)
        points.extend(pts[mask_domain].tolist())
        attempts += n_needed

    if len(points) < n_points:
        from .geometry import sample_points_uniform_general
        extra = sample_points_uniform_general(meta, n_points - len(points), rng)
        points.extend(extra.tolist())

    return np.array(points[:n_points], dtype=np.float64)


__all__ = [
    "SamplingStrategy",
    "sample_points",
    "sample_points_uniform_cylinder",
    "sample_points_near_electrode",
    "sample_points_stratified_tissue",
    "sample_points_surface_biased",
    "sample_boundary_points",
]
