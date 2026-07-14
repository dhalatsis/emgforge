"""
Geometry helpers for complex cross-sections.

Supports: circular, elliptical, off-center bone, two bones, tapering.
All functions read geometry type from metadata and dispatch accordingly.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional

from emgforge.fem.geometry import _ellipse_radius, _taper_scale


# ---------------------------------------------------------------------------
# Skin boundary radius at a given (theta, z)
# ---------------------------------------------------------------------------

def skin_radius_at(meta: Dict, theta: float, z: float = 0.0) -> float:
    """Return the skin-surface radius at angle theta (rad) and axial position z.

    For circular: constant r_skin.
    For elliptical: r(theta) = a*b / sqrt((b*cos)^2 + (a*sin)^2).
    For tapered: linearly interpolated scale along z.
    """
    gp = meta["geometry_params"]
    shape = gp.get("shape", "circle")
    length = float(gp["length"])

    if shape == "ellipse":
        a = float(gp["a_skin"])
        b = float(gp["b_skin"])
        r = _ellipse_radius(a, b, theta)
    else:
        r = float(gp["radius_skin"])

    # Apply taper if present
    z_profile = gp.get("z_profile", "constant")
    if z_profile == "taper":
        scale = _taper_scale(gp, z, length)
        r *= scale

    return r


def layer_radius_at(meta: Dict, layer_key: str, theta: float, z: float = 0.0) -> float:
    """Return the outer radius of a tissue layer at (theta, z).

    layer_key: one of 'canc', 'cort', 'muscle', 'fat', 'skin'.
    """
    gp = meta["geometry_params"]
    shape = gp.get("shape", "circle")
    length = float(gp["length"])

    radius_keys = {
        "canc": "radius_canc_bone",
        "cort": "radius_cort_bone",
        "muscle": "radius_muscle",
        "fat": "radius_fat",
        "skin": "radius_skin",
    }
    ellipse_keys = {
        "canc": ("a_canc_bone", "b_canc_bone"),
        "cort": ("a_cort_bone", "b_cort_bone"),
        "muscle": ("a_muscle", "b_muscle"),
        "fat": ("a_fat", "b_fat"),
        "skin": ("a_skin", "b_skin"),
    }

    if shape == "ellipse" and layer_key in ellipse_keys:
        ak, bk = ellipse_keys[layer_key]
        a = float(gp[ak])
        b = float(gp[bk])
        r = _ellipse_radius(a, b, theta)
    else:
        r = float(gp[radius_keys[layer_key]])

    z_profile = gp.get("z_profile", "constant")
    if z_profile == "taper":
        r *= _taper_scale(gp, z, length)

    return r


# ---------------------------------------------------------------------------
# Bounding box for rejection sampling
# ---------------------------------------------------------------------------

def bounding_box(meta: Dict) -> Tuple[float, float, float]:
    """Return (half_x, half_y, length) bounding box for the geometry."""
    gp = meta["geometry_params"]
    shape = gp.get("shape", "circle")
    length = float(gp["length"])

    if shape == "ellipse":
        hx = float(gp["a_skin"])
        hy = float(gp["b_skin"])
    else:
        hx = hy = float(gp["radius_skin"])

    # Taper can only shrink, so bbox at z=0 (scale=1) is the max
    return hx, hy, length


# ---------------------------------------------------------------------------
# Point-inside-domain test
# ---------------------------------------------------------------------------

def point_inside_domain(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        meta: Dict) -> np.ndarray:
    """Test whether points (x,y,z) are inside the outer skin boundary.

    Returns boolean mask of shape (N,).
    """
    gp = meta["geometry_params"]
    length = float(gp["length"])

    # z bounds
    mask = (z >= 0) & (z <= length)

    # Cross-section test
    shape = gp.get("shape", "circle")
    z_profile = gp.get("z_profile", "constant")

    if shape == "circle" and z_profile == "constant":
        # Fast path: simple circular cylinder
        r_skin = float(gp["radius_skin"])
        r = np.sqrt(x**2 + y**2)
        mask &= (r <= r_skin)
    elif shape == "ellipse" and z_profile == "constant":
        a = float(gp["a_skin"])
        b = float(gp["b_skin"])
        mask &= ((x / a)**2 + (y / b)**2 <= 1.0)
    else:
        # General case: per-point check with taper
        for i in range(len(x)):
            if not mask[i]:
                continue
            theta_i = np.arctan2(y[i], x[i])
            r_bound = skin_radius_at(meta, theta_i, z[i])
            r_i = np.sqrt(x[i]**2 + y[i]**2)
            if r_i > r_bound:
                mask[i] = False

    return mask


# ---------------------------------------------------------------------------
# Tissue label assignment (general geometry)
# ---------------------------------------------------------------------------

def compute_tissue_labels_general(meta: Dict, points: np.ndarray) -> np.ndarray:
    """Compute tissue labels for arbitrary geometry.

    Handles: circular, elliptical, off-center bone, two bones, taper.

    Returns
    -------
    np.ndarray of int32, shape (N,).
        0=cancellous, 1=cortical, 2=muscle, 3=fat, 4=skin, -1=outside.
    """
    gp = meta["geometry_params"]
    shape = gp.get("shape", "circle")
    z_profile = gp.get("z_profile", "constant")
    n_bones = int(gp.get("n_bones", gp.get("bone_count", 1)) or 1)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    N = len(points)
    labels = np.full(N, -1, dtype=np.int32)

    if (shape == "circle" and z_profile == "constant" and n_bones == 1
            and "bone1_center_x" not in gp):
        # Fast path: original concentric circular geometry
        r = np.sqrt(x**2 + y**2)
        r_canc = float(gp["radius_canc_bone"])
        r_cort = float(gp["radius_cort_bone"])
        r_musc = float(gp["radius_muscle"])
        r_fat = float(gp["radius_fat"])
        r_skin = float(gp["radius_skin"])

        labels[r <= r_canc] = 0
        labels[(r > r_canc) & (r <= r_cort)] = 1
        labels[(r > r_cort) & (r <= r_musc)] = 2
        labels[(r > r_musc) & (r <= r_fat)] = 3
        labels[(r > r_fat) & (r <= r_skin)] = 4
        return labels

    # General case: work outward from skin, then handle bones separately
    length = float(gp["length"])

    # First, assign outer layers (skin, fat, muscle) based on normalized distance
    for i in range(N):
        zi = z[i]
        if zi < 0 or zi > length:
            continue

        theta_i = np.arctan2(y[i], x[i])
        r_i = np.sqrt(x[i]**2 + y[i]**2)

        r_skin_i = layer_radius_at(meta, "skin", theta_i, zi)
        if r_i > r_skin_i:
            continue  # outside

        r_fat_i = layer_radius_at(meta, "fat", theta_i, zi)
        if r_i > r_fat_i:
            labels[i] = 4  # skin
            continue

        r_musc_i = layer_radius_at(meta, "muscle", theta_i, zi)
        if r_i > r_musc_i:
            labels[i] = 3  # fat
            continue

        # Inside muscle boundary — default to muscle, bones override below
        labels[i] = 2  # muscle

    # Now handle bone(s) — these are potentially off-center
    bone_centers = []
    bone_radii = []  # (r_canc, r_cort) per bone

    if n_bones >= 1:
        cx1 = float(gp.get("bone1_center_x", 0.0))
        cy1 = float(gp.get("bone1_center_y", 0.0))
        if shape == "ellipse":
            rc1 = float(gp.get("a_canc_bone", gp.get("radius_canc_bone", 0)))
            rk1 = float(gp.get("a_cort_bone", gp.get("radius_cort_bone", 0)))
            # Use average of a and b for distance-based assignment
            rc1_b = float(gp.get("b_canc_bone", rc1))
            rk1_b = float(gp.get("b_cort_bone", rk1))
            bone_centers.append((cx1, cy1))
            bone_radii.append(((rc1, rc1_b), (rk1, rk1_b)))
        else:
            rc1 = float(gp["radius_canc_bone"])
            rk1 = float(gp["radius_cort_bone"])
            bone_centers.append((cx1, cy1))
            bone_radii.append(((rc1, rc1), (rk1, rk1)))

    if n_bones >= 2:
        cx2 = float(gp.get("bone2_center_x", 0.0))
        cy2 = float(gp.get("bone2_center_y", 0.0))
        if shape == "ellipse":
            rc2 = float(gp.get("a_canc_bone_2", gp.get("radius_canc_bone_2", 0)))
            rk2 = float(gp.get("a_cort_bone_2", gp.get("radius_cort_bone_2", 0)))
            rc2_b = float(gp.get("b_canc_bone_2", rc2))
            rk2_b = float(gp.get("b_cort_bone_2", rk2))
            bone_centers.append((cx2, cy2))
            bone_radii.append(((rc2, rc2_b), (rk2, rk2_b)))
        else:
            rc2 = float(gp.get("radius_canc_bone_2", gp["radius_canc_bone"]))
            rk2 = float(gp.get("radius_cort_bone_2", gp["radius_cort_bone"]))
            bone_centers.append((cx2, cy2))
            bone_radii.append(((rc2, rc2), (rk2, rk2)))

    # For centered single bone without offset, use original radii
    if n_bones == 1 and "bone1_center_x" not in gp:
        bone_centers = [(0.0, 0.0)]
        if shape == "ellipse":
            rc = float(gp.get("a_canc_bone", gp["radius_canc_bone"]))
            rk = float(gp.get("a_cort_bone", gp["radius_cort_bone"]))
            rc_b = float(gp.get("b_canc_bone", rc))
            rk_b = float(gp.get("b_cort_bone", rk))
            bone_radii = [((rc, rc_b), (rk, rk_b))]
        else:
            rc = float(gp["radius_canc_bone"])
            rk = float(gp["radius_cort_bone"])
            bone_radii = [((rc, rc), (rk, rk))]

    # Assign bone labels
    for bc, br in zip(bone_centers, bone_radii):
        cx, cy = bc
        (rc_a, rc_b), (rk_a, rk_b) = br

        dx = x - cx
        dy = y - cy

        # Use elliptical distance if a != b, otherwise circular
        if abs(rc_a - rc_b) < 1e-6:
            # Circular bone
            dist = np.sqrt(dx**2 + dy**2)
            canc_mask = (dist <= rc_a) & (labels >= 0)
            cort_mask = (dist > rc_a) & (dist <= rk_a) & (labels >= 0)
        else:
            # Elliptical bone
            norm_canc = (dx / rc_a)**2 + (dy / rc_b)**2
            norm_cort = (dx / rk_a)**2 + (dy / rk_b)**2
            canc_mask = (norm_canc <= 1.0) & (labels >= 0)
            cort_mask = (~canc_mask) & (norm_cort <= 1.0) & (labels >= 0)

        labels[canc_mask] = 0
        labels[cort_mask] = 1

    return labels


# ---------------------------------------------------------------------------
# Boundary point sampling (general geometry)
# ---------------------------------------------------------------------------

def sample_boundary_points_general(
    meta: Dict,
    n_points: int,
    rng: np.random.Generator,
    include_caps: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points on the domain boundary for general geometries.

    Returns (points, normals) arrays.
    """
    gp = meta["geometry_params"]
    shape = gp.get("shape", "circle")
    z_profile = gp.get("z_profile", "constant")
    length = float(gp["length"])

    if include_caps:
        n_lateral = int(n_points * 0.7)
        n_cap = (n_points - n_lateral) // 2
        n_bottom = n_cap
        n_top = n_points - n_lateral - n_bottom
    else:
        n_lateral = n_points
        n_bottom = n_top = 0

    # --- Lateral surface ---
    theta = rng.uniform(0, 2 * np.pi, n_lateral)
    z_lat = rng.uniform(0, length, n_lateral)

    if shape == "ellipse":
        a = float(gp["a_skin"])
        b = float(gp["b_skin"])
    else:
        a = b = float(gp["radius_skin"])

    # Compute surface points and normals
    lat_pts = np.zeros((n_lateral, 3))
    lat_normals = np.zeros((n_lateral, 3))

    for i in range(n_lateral):
        zi = z_lat[i]
        ti = theta[i]

        # Local radius (with taper)
        scale = 1.0
        if z_profile == "taper":
            scale = _taper_scale(gp, zi, length)

        ai, bi = a * scale, b * scale
        lat_pts[i] = [ai * np.cos(ti), bi * np.sin(ti), zi]

        # Outward normal for ellipse: ∇((x/a)² + (y/b)² - 1)
        nx = lat_pts[i, 0] / (ai**2)
        ny = lat_pts[i, 1] / (bi**2)
        norm = np.sqrt(nx**2 + ny**2)
        if norm > 1e-12:
            lat_normals[i] = [nx / norm, ny / norm, 0.0]
        else:
            lat_normals[i] = [np.cos(ti), np.sin(ti), 0.0]

    all_pts = [lat_pts]
    all_normals = [lat_normals]

    # --- Bottom cap (z=0) ---
    if n_bottom > 0:
        scale_bot = 1.0
        if z_profile == "taper":
            scale_bot = _taper_scale(gp, 0.0, length)
        ab, bb = a * scale_bot, b * scale_bot
        cap_pts, cap_normals = _sample_ellipse_cap(ab, bb, n_bottom, z=0.0,
                                                    normal_z=-1.0, rng=rng)
        all_pts.append(cap_pts)
        all_normals.append(cap_normals)

    # --- Top cap (z=length) ---
    if n_top > 0:
        scale_top = 1.0
        if z_profile == "taper":
            scale_top = _taper_scale(gp, length, length)
        at, bt = a * scale_top, b * scale_top
        cap_pts, cap_normals = _sample_ellipse_cap(at, bt, n_top, z=length,
                                                    normal_z=1.0, rng=rng)
        all_pts.append(cap_pts)
        all_normals.append(cap_normals)

    points = np.vstack(all_pts).astype(np.float64)
    normals = np.vstack(all_normals).astype(np.float64)
    return points, normals


# ---------------------------------------------------------------------------
# Uniform interior sampling (general geometry)
# ---------------------------------------------------------------------------

def sample_points_uniform_general(
    meta: Dict,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points uniformly inside the domain for any geometry."""
    hx, hy, length = bounding_box(meta)

    points = []
    while len(points) < n_points:
        n_needed = (n_points - len(points)) * 2
        x = rng.uniform(-hx, hx, n_needed)
        y = rng.uniform(-hy, hy, n_needed)
        z = rng.uniform(0, length, n_needed)

        mask = point_inside_domain(x, y, z, meta)
        valid = np.stack([x[mask], y[mask], z[mask]], axis=1)
        points.extend(valid.tolist())

    return np.array(points[:n_points], dtype=np.float64)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# _ellipse_radius / _taper_scale now live in emgforge.fem.geometry (imported at
# top) — shared with fem.sanity, kept in core so nothing there reaches into this
# ML/dataset layer.


def _sample_ellipse_cap(a: float, b: float, n: int, z: float,
                         normal_z: float, rng: np.random.Generator
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly on an elliptical cap at given z."""
    pts = []
    while len(pts) < n:
        n_try = (n - len(pts)) * 2
        x = rng.uniform(-a, a, n_try)
        y = rng.uniform(-b, b, n_try)
        mask = (x / a)**2 + (y / b)**2 <= 1.0
        for xi, yi in zip(x[mask], y[mask]):
            pts.append([xi, yi, z])
            if len(pts) >= n:
                break

    pts = np.array(pts[:n])
    normals = np.zeros_like(pts)
    normals[:, 2] = normal_z
    return pts, normals
