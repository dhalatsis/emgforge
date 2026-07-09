"""
Per-muscle fiber direction estimation and conductivity tensor rotation.

Each muscle has a fiber direction defined by either:
  1. PCA of voxel coordinates (automatic)
  2. Manual start/end attachment points (from JSON config)

The fiber direction determines how the anisotropic conductivity tensor
is rotated from the default z-axis alignment for each muscle.

Usage:
    from mri.core.fiber_directions import MuscleFiberModel

    model = MuscleFiberModel("mri/data/PD_PROPELLER_5MM_FATS_FLX_0012/full.nii.gz")
    model.estimate_fibers()              # PCA-based
    model.save_config("mri/mesh/muscle_fibers.json")  # edit manually if needed
    model.load_config("mri/mesh/muscle_fibers.json")  # reload with edits

    # Get per-cell conductivity tensor for the FEM mesh
    sigma = model.get_conductivity_tensor(label=7)  # 3x3 matrix
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Conductivity constants (from src/emgop/fem/constants.py)
# ---------------------------------------------------------------------------
SIGMA_MUSCLE_CROSS = 0.2455    # S/m, transverse to fiber
SIGMA_MUSCLE_FIBER = 1.2275    # S/m, along fiber (5x ratio)
SIGMA_FAT_SKIN = 0.0379        # S/m, isotropic
SIGMA_CONNECTIVE = 0.2         # S/m, isotropic

# Default z-aligned muscle conductivity tensor
SIGMA_MUSCLE_Z = np.diag([SIGMA_MUSCLE_CROSS, SIGMA_MUSCLE_CROSS, SIGMA_MUSCLE_FIBER])

# Muscle labels (non-connective, non-fat, non-background)
# Labels 15, 22 = connective; 25 = fat_skin; 0 = background
CONNECTIVE_LABELS = {15, 22}
FAT_SKIN_LABELS = {25}


def rodrigues_rotation(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Rotation matrix that rotates unit vector v_from to v_to.

    Uses Rodrigues' formula. If vectors are (anti-)parallel, returns
    identity or negation matrix.

    Parameters
    ----------
    v_from, v_to : (3,) unit vectors

    Returns
    -------
    R : (3, 3) rotation matrix
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    sin_angle = np.linalg.norm(cross)

    if sin_angle < 1e-10:
        # Vectors are parallel or anti-parallel
        if dot > 0:
            return np.eye(3)
        else:
            # 180° rotation — pick any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
            perp = perp - np.dot(perp, v_from) * v_from
            perp = perp / np.linalg.norm(perp)
            return 2 * np.outer(perp, perp) - np.eye(3)

    # Skew-symmetric cross-product matrix
    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])

    R = np.eye(3) + K + K @ K * ((1 - dot) / (sin_angle ** 2))
    return R


def _apply_pennation(tangent: np.ndarray, pennation_deg: float) -> np.ndarray:
    """Tilt a fiber tangent by pennation_deg around an axis perpendicular
    to the tangent. Picks the axis as the cross-product of tangent and +z,
    so the tilt is in the (tangent, +z) plane → fiber pitches in xy.

    For tangent ≈ +z, this falls back to rotation around +x (deterministic).
    """
    t = tangent / max(np.linalg.norm(tangent), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(t, z)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        # tangent ≈ ±z → fall back to +x axis
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / norm
    ang = np.radians(pennation_deg)
    # Rodrigues' rotation of t around axis by ang
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R @ t


def rotate_conductivity(sigma_z: np.ndarray, fiber_dir: np.ndarray) -> np.ndarray:
    """Rotate conductivity tensor from z-alignment to given fiber direction.

    Parameters
    ----------
    sigma_z : (3, 3) conductivity tensor with fiber along z-axis
    fiber_dir : (3,) unit vector of actual fiber direction

    Returns
    -------
    sigma_rotated : (3, 3) rotated conductivity tensor
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    R = rodrigues_rotation(z_axis, fiber_dir)
    return R @ sigma_z @ R.T


class MuscleCrossSection:
    """Per-slice radial boundary distances for a muscle cross-section.

    Stores the distance from centroid to muscle boundary at angular samples
    for each z-slice. Used by fiber_path_morphing() to map normalized polar
    coordinates to physical positions that always stay inside the muscle.

    Parameters
    ----------
    z_mm : (K,) array
        z-coordinates of slices (mm), must be monotonically increasing.
    theta_deg : (M,) array
        Angular samples in degrees, e.g. [0, 5, 10, ..., 355].
    R_boundary : (K, M) array
        Radial distance from centroid to boundary (mm) at each (z, theta).
    r_inset : float
        Safety margin factor (0-1). Fibers at r_norm=1 map to r_inset * R_boundary.
    """

    def __init__(
        self,
        z_mm: np.ndarray,
        theta_deg: np.ndarray,
        R_boundary: np.ndarray,
        r_inset: float = 0.95,
    ):
        self.z_mm = np.asarray(z_mm, dtype=np.float64)
        self.theta_deg = np.asarray(theta_deg, dtype=np.float64)
        self.R_boundary = np.asarray(R_boundary, dtype=np.float64)
        self.r_inset = float(r_inset)

        assert self.R_boundary.shape == (len(self.z_mm), len(self.theta_deg))

        # Extend theta for periodicity: append theta=360 with same values as theta=0
        theta_ext = np.concatenate([self.theta_deg, [360.0]])
        R_ext = np.column_stack([self.R_boundary, self.R_boundary[:, 0]])

        self._interp = RegularGridInterpolator(
            (self.z_mm, theta_ext),
            R_ext,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate
        )

    def boundary_radius(self, z: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
        """Interpolate boundary radius at given (z, theta) pairs.

        Parameters
        ----------
        z : (N,) array — z-coordinates in mm
        theta_deg : (N,) array — angles in degrees [0, 360)

        Returns
        -------
        R : (N,) array — boundary radius in mm
        """
        z = np.asarray(z, dtype=np.float64)
        theta_deg = np.asarray(theta_deg, dtype=np.float64) % 360.0
        pts = np.column_stack([z, theta_deg])
        return self._interp(pts)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "z_mm": [round(float(z), 3) for z in self.z_mm],
            "theta_deg": [round(float(t), 2) for t in self.theta_deg],
            "R_boundary": [[round(float(r), 4) for r in row] for row in self.R_boundary],
            "r_inset": round(self.r_inset, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> MuscleCrossSection:
        """Deserialize from dict."""
        return cls(
            z_mm=np.array(d["z_mm"]),
            theta_deg=np.array(d["theta_deg"]),
            R_boundary=np.array(d["R_boundary"]),
            r_inset=d.get("r_inset", 0.95),
        )


class MuscleCenterline:
    """Piecewise-linear muscle fiber centerline from per-slice centroids.

    Stores (x, y) centroid waypoints at each z-slice and fits a cubic spline
    for smooth interpolation. The spline tangent at any z gives the local
    fiber direction.

    Parameters
    ----------
    z_mm : (K,) array
        z-coordinates of slice centroids (mm), must be monotonically increasing.
    cx_mm, cy_mm : (K,) arrays
        x, y centroid coordinates at each z-slice (mm).
    """

    def __init__(self, z_mm: np.ndarray, cx_mm: np.ndarray, cy_mm: np.ndarray):
        self.z_mm = np.asarray(z_mm, dtype=np.float64)
        self.cx_mm = np.asarray(cx_mm, dtype=np.float64)
        self.cy_mm = np.asarray(cy_mm, dtype=np.float64)

        # Sort by z
        order = np.argsort(self.z_mm)
        self.z_mm = self.z_mm[order]
        self.cx_mm = self.cx_mm[order]
        self.cy_mm = self.cy_mm[order]

        self.z_min = float(self.z_mm[0])
        self.z_max = float(self.z_mm[-1])

        # Build interpolators
        if len(self.z_mm) >= 4:
            self._spline_x = CubicSpline(self.z_mm, self.cx_mm, bc_type="natural")
            self._spline_y = CubicSpline(self.z_mm, self.cy_mm, bc_type="natural")
            self._use_spline = True
        else:
            # Fall back to linear interpolation for < 4 waypoints
            self._use_spline = False

    def position(self, z: float) -> np.ndarray:
        """Evaluate centerline position at z (clamped to range).

        Returns
        -------
        pos : (3,) array — [x, y, z] in mm
        """
        z_c = np.clip(z, self.z_min, self.z_max)
        if self._use_spline:
            x = float(self._spline_x(z_c))
            y = float(self._spline_y(z_c))
        else:
            x = float(np.interp(z_c, self.z_mm, self.cx_mm))
            y = float(np.interp(z_c, self.z_mm, self.cy_mm))
        return np.array([x, y, z_c])

    def tangent(self, z: float) -> np.ndarray:
        """Unit tangent vector at z (clamped to range).

        The tangent is normalize([dx/dz, dy/dz, 1.0]).

        Returns
        -------
        t : (3,) unit vector
        """
        z_c = np.clip(z, self.z_min, self.z_max)
        if self._use_spline:
            dxdz = float(self._spline_x(z_c, 1))
            dydz = float(self._spline_y(z_c, 1))
        else:
            # Linear: finite difference from endpoints
            dz = self.z_max - self.z_min
            if dz < 1e-6:
                return np.array([0.0, 0.0, 1.0])
            dxdz = (self.cx_mm[-1] - self.cx_mm[0]) / dz
            dydz = (self.cy_mm[-1] - self.cy_mm[0]) / dz

        t = np.array([dxdz, dydz, 1.0])
        norm = np.linalg.norm(t)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        t = t / norm
        # Canonical: positive z
        if t[2] < 0:
            t = -t
        return t

    def fiber_path(self, dx: float, dy: float, z_values: np.ndarray) -> np.ndarray:
        """Trace a fiber at constant lateral offset from centerline.

        Parameters
        ----------
        dx, dy : float
            Lateral offset from centerline in mm.
        z_values : (N,) array
            z-coordinates to evaluate along.

        Returns
        -------
        path : (N, 3) array — fiber coordinates in mm
        """
        z_values = np.asarray(z_values)
        z_c = np.clip(z_values, self.z_min, self.z_max)

        if self._use_spline:
            xs = self._spline_x(z_c) + dx
            ys = self._spline_y(z_c) + dy
        else:
            xs = np.interp(z_c, self.z_mm, self.cx_mm) + dx
            ys = np.interp(z_c, self.z_mm, self.cy_mm) + dy

        return np.column_stack([xs, ys, z_c])

    def fiber_path_morphing(
        self,
        r_norm: float,
        theta_deg: float,
        z_values: np.ndarray,
        cross_section: MuscleCrossSection,
        taper=None,
    ) -> np.ndarray:
        """Trace a fiber using morphing-disk coordinates.

        Maps normalized polar coordinates (r_norm, theta_deg) to physical
        positions at each z-slice. The fiber is guaranteed to stay inside
        the muscle boundary as long as r_norm <= 1.

        Parameters
        ----------
        r_norm : float
            Normalized radial position in [0, 1]. 0 = centerline, 1 = boundary.
        theta_deg : float
            Angular position in degrees [0, 360).
        z_values : (N,) array
            z-coordinates to evaluate along.
        cross_section : MuscleCrossSection
            Cross-section boundary data for this muscle.
        taper : None | float | callable
            Optional **bundle-wide tendon taper** applied as a multiplicative
            scale on ``r_norm`` along z. The muscle's anatomical boundary
            R(z) is already morphed; ``taper`` further pinches the bundle
            at the tendons to reproduce the spindle profile observed in
            Neurodec's MRI-FEM reference (RMS-r at z₀/z_end ≈ 0.3-0.4 ×
            peak RMS-r at z-mid).

            - ``None`` (default): no extra taper (legacy behaviour).
            - ``float in (0, 1)``: symmetric spindle with tendon-end ratio
              equal to this value. Profile:
              ``s(z_frac) = ratio + (1-ratio) · sin(π · z_frac)``.
              Peaks at 1.0 at z_frac=0.5, equal to ``ratio`` at the ends.
              ``taper=0.35`` reproduces the ECU/ECRL/PL spindle within
              ~5%. Pass ``taper=1.0`` for no taper (cylinder).
            - ``callable``: arbitrary ``f(z_frac in [0,1]) -> scale in
              [0, 1]``. Vectorised: receives an ndarray, returns one.
              Use this for asymmetric spindles (FCU has peak at z_frac=0.74).

        Returns
        -------
        path : (N, 3) array — fiber coordinates in mm
        """
        z_values = np.asarray(z_values)
        z_c = np.clip(z_values, self.z_min, self.z_max)
        n = len(z_c)

        # Centerline positions at each z
        if self._use_spline:
            cx = self._spline_x(z_c)
            cy = self._spline_y(z_c)
        else:
            cx = np.interp(z_c, self.z_mm, self.cx_mm)
            cy = np.interp(z_c, self.z_mm, self.cy_mm)

        if r_norm == 0:
            return np.column_stack([cx, cy, z_c])

        # Boundary radius at each z for this theta
        theta_arr = np.full(n, theta_deg % 360.0)
        R = cross_section.boundary_radius(z_c, theta_arr)

        # Map to physical offset
        theta_rad = np.radians(theta_deg)
        scale = r_norm * cross_section.r_inset * R

        # Optional bundle-wide tendon taper
        if taper is not None:
            z_extent = max(self.z_max - self.z_min, 1e-9)
            z_frac = (z_c - self.z_min) / z_extent
            if callable(taper):
                taper_scale = np.asarray(taper(z_frac), dtype=np.float64)
            else:
                ratio = float(taper)
                taper_scale = ratio + (1.0 - ratio) * np.sin(np.pi * z_frac)
            scale = scale * taper_scale

        xs = cx + scale * np.cos(theta_rad)
        ys = cy + scale * np.sin(theta_rad)

        return np.column_stack([xs, ys, z_c])

    def fiber_tangent_morphing(
        self,
        r_norm: float,
        theta_deg: float,
        z_values: np.ndarray,
        cross_section: MuscleCrossSection,
        h_mm: float = 0.5,
    ) -> np.ndarray:
        """Unit tangent vectors along a morphing-disk fiber path.

        Used to rotate the anisotropic muscle conductivity tensor along each
        fiber: σ_fiber gets aligned with this tangent at every point.

        Computed via central finite differences on
        :meth:`fiber_path_morphing`. At the z-extent endpoints the difference
        degenerates to a one-sided forward / backward difference; result is
        still a valid local tangent.

        Parameters
        ----------
        r_norm, theta_deg, z_values, cross_section
            Same meaning as :meth:`fiber_path_morphing`.
        h_mm : float
            Finite-difference step in mm. Default 0.5 — well below the 6mm
            slice spacing and below the ~9mm smoothing kernel, so derivatives
            sample the smoothed boundary field, not voxel noise.

        Returns
        -------
        t : (N, 3) array — unit tangent vector at each z in ``z_values``.
            Canonicalized to positive z-component.
        """
        z_values = np.atleast_1d(np.asarray(z_values, dtype=np.float64))
        z_plus = z_values + h_mm
        z_minus = z_values - h_mm

        p_plus = self.fiber_path_morphing(r_norm, theta_deg, z_plus, cross_section)
        p_minus = self.fiber_path_morphing(r_norm, theta_deg, z_minus, cross_section)

        delta = p_plus - p_minus  # (N, 3); z-component = 2h_mm interior, h_mm at ends

        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        t = delta / norms

        # Canonical: positive z-component
        flip_mask = t[:, 2] < 0
        if np.any(flip_mask):
            t[flip_mask] = -t[flip_mask]
        return t

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "z_mm": [round(float(z), 3) for z in self.z_mm],
            "cx_mm": [round(float(x), 3) for x in self.cx_mm],
            "cy_mm": [round(float(y), 3) for y in self.cy_mm],
        }

    @classmethod
    def from_dict(cls, d: dict) -> MuscleCenterline:
        """Deserialize from dict."""
        return cls(
            z_mm=np.array(d["z_mm"]),
            cx_mm=np.array(d["cx_mm"]),
            cy_mm=np.array(d["cy_mm"]),
        )


class MuscleInfo:
    """Fiber direction and properties for a single muscle."""

    def __init__(self, label: int, tissue_type: str = "muscle"):
        self.label = label
        self.tissue_type = tissue_type

        # Spatial properties (computed from segmentation)
        self.n_voxels: int = 0
        self.centroid_mm: np.ndarray = np.zeros(3)
        self.z_min_mm: float = 0.0
        self.z_max_mm: float = 0.0
        self.z_span_mm: float = 0.0
        self.n_slices: int = 0

        # Fiber direction (unit vector)
        self.fiber_direction: np.ndarray = np.array([0.0, 0.0, 1.0])
        self.fiber_angle_from_z_deg: float = 0.0

        # PCA results
        self.pca_eigenvalues: np.ndarray = np.zeros(3)
        self.pca_elongation: float = 1.0  # sqrt(λ1/λ2)

        # Piecewise-linear centerline (None = straight fiber)
        self.centerline: MuscleCenterline | None = None

        # Cross-section boundary data (None = not computed)
        self.cross_section: MuscleCrossSection | None = None

        # Manual override
        self.start_point_mm: np.ndarray | None = None  # proximal attachment
        self.end_point_mm: np.ndarray | None = None    # distal attachment
        self.direction_source: str = "pca"  # "pca" or "manual"

        # Pennation: fiber direction tilts away from centerline tangent
        # by this angle (degrees). Real pennate muscles: 5-30°.
        # 0 = parallel-fibered (sartorius-style).
        # The tilt axis is the local boundary-normal vector at the fiber's
        # cross-section position (so fibers pitch outward / inward).
        self.pennation_deg: float = 0.0

    @property
    def conductivity_tensor(self) -> np.ndarray:
        """3x3 conductivity tensor rotated to this muscle's fiber direction."""
        if self.tissue_type == "connective":
            return SIGMA_CONNECTIVE * np.eye(3)
        elif self.tissue_type == "fat_skin":
            return SIGMA_FAT_SKIN * np.eye(3)
        else:
            return rotate_conductivity(SIGMA_MUSCLE_Z, self.fiber_direction)

    def conductivity_tensor_at_z(self, z_mm: float) -> np.ndarray:
        """3x3 conductivity tensor with position-dependent fiber direction.

        If a centerline exists, uses its tangent at z_mm as the local fiber
        direction. Otherwise falls back to the global fiber_direction.

        Applies pennation_deg tilt if set: rotates the tangent by that angle
        around an axis perpendicular to it (chosen as the +x direction in
        the cell's local frame, which gives the simplest pitched-fiber
        approximation).
        """
        if self.tissue_type != "muscle" or self.centerline is None:
            return self.conductivity_tensor
        local_dir = self.centerline.tangent(z_mm)
        if self.pennation_deg != 0.0:
            local_dir = _apply_pennation(local_dir, self.pennation_deg)
        return rotate_conductivity(SIGMA_MUSCLE_Z, local_dir)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        d = {
            "label": self.label,
            "tissue_type": self.tissue_type,
            "n_voxels": self.n_voxels,
            "centroid_mm": self.centroid_mm.tolist(),
            "z_min_mm": round(self.z_min_mm, 1),
            "z_max_mm": round(self.z_max_mm, 1),
            "z_span_mm": round(self.z_span_mm, 1),
            "n_slices": self.n_slices,
            "fiber_direction": [round(x, 6) for x in self.fiber_direction.tolist()],
            "fiber_angle_from_z_deg": round(self.fiber_angle_from_z_deg, 1),
            "pca_elongation": round(self.pca_elongation, 1),
            "direction_source": self.direction_source,
            "pennation_deg": round(self.pennation_deg, 2),
        }
        if self.start_point_mm is not None:
            d["start_point_mm"] = [round(x, 1) for x in self.start_point_mm.tolist()]
        if self.end_point_mm is not None:
            d["end_point_mm"] = [round(x, 1) for x in self.end_point_mm.tolist()]
        if self.centerline is not None:
            d["centerline"] = self.centerline.to_dict()
        if self.cross_section is not None:
            d["cross_section"] = self.cross_section.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> MuscleInfo:
        """Deserialize from dict."""
        m = cls(d["label"], d.get("tissue_type", "muscle"))
        m.n_voxels = d.get("n_voxels", 0)
        m.centroid_mm = np.array(d.get("centroid_mm", [0, 0, 0]))
        m.z_min_mm = d.get("z_min_mm", 0)
        m.z_max_mm = d.get("z_max_mm", 0)
        m.z_span_mm = d.get("z_span_mm", 0)
        m.n_slices = d.get("n_slices", 0)
        m.fiber_direction = np.array(d.get("fiber_direction", [0, 0, 1]))
        m.fiber_angle_from_z_deg = d.get("fiber_angle_from_z_deg", 0)
        m.pca_elongation = d.get("pca_elongation", 1)
        m.direction_source = d.get("direction_source", "pca")
        m.pennation_deg = d.get("pennation_deg", 0.0)
        if "start_point_mm" in d:
            m.start_point_mm = np.array(d["start_point_mm"])
        if "end_point_mm" in d:
            m.end_point_mm = np.array(d["end_point_mm"])
        if "centerline" in d:
            m.centerline = MuscleCenterline.from_dict(d["centerline"])
        if "cross_section" in d:
            m.cross_section = MuscleCrossSection.from_dict(d["cross_section"])
        return m


class MuscleFiberModel:
    """Per-muscle fiber direction model for MRI forearm segmentation.

    Computes or loads fiber direction vectors for each muscle, then
    provides rotated anisotropic conductivity tensors for the FEM solver.
    """

    def __init__(self, nifti_path: str | None = None):
        self.muscles: dict[int, MuscleInfo] = {}

        if nifti_path is not None:
            self._load_segmentation(nifti_path)

    def _load_segmentation(self, nifti_path: str):
        """Load NIfTI and extract label coordinates."""
        import nibabel as nib

        img = nib.load(nifti_path)
        self.seg_data = img.get_fdata().astype(int)
        self.voxel_size = np.array(img.header.get_zooms())

        labels = np.unique(self.seg_data[self.seg_data > 0])

        for label in labels:
            label = int(label)
            if label in FAT_SKIN_LABELS:
                tissue = "fat_skin"
            elif label in CONNECTIVE_LABELS:
                tissue = "connective"
            else:
                tissue = "muscle"

            m = MuscleInfo(label, tissue)

            mask = self.seg_data == label
            coords = np.argwhere(mask) * self.voxel_size

            m.n_voxels = len(coords)
            m.centroid_mm = coords.mean(axis=0)
            m.z_min_mm = float(coords[:, 2].min())
            m.z_max_mm = float(coords[:, 2].max())
            m.z_span_mm = m.z_max_mm - m.z_min_mm
            m.n_slices = len(np.unique(np.argwhere(mask)[:, 2]))

            self.muscles[label] = m

    def estimate_fibers(self, method: str = "pca"):
        """Estimate fiber direction for each muscle.

        Parameters
        ----------
        method : str
            "pca" — principal component analysis of voxel coordinates
            "endpoints" — connect z-extent start/end centroids
        """
        for label, m in self.muscles.items():
            if m.tissue_type != "muscle":
                continue

            if m.direction_source == "manual" and m.start_point_mm is not None:
                # Already has manual override, skip PCA
                continue

            mask = self.seg_data == label
            coords = np.argwhere(mask) * self.voxel_size

            if method == "pca":
                self._estimate_pca(m, coords)
            elif method == "endpoints":
                self._estimate_endpoints(m, coords)

    def _estimate_pca(self, m: MuscleInfo, coords: np.ndarray):
        """Fiber direction from PCA of voxel coordinates."""
        if len(coords) < 10:
            m.fiber_direction = np.array([0.0, 0.0, 1.0])
            m.direction_source = "pca"
            return

        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        pc1 = eigenvectors[:, 0]

        # Canonical: positive z-component
        if pc1[2] < 0:
            pc1 = -pc1

        m.fiber_direction = pc1 / np.linalg.norm(pc1)
        m.pca_eigenvalues = eigenvalues
        m.pca_elongation = float(np.sqrt(eigenvalues[0] / max(eigenvalues[1], 1e-10)))

        # Angle from z
        cos_angle = abs(np.dot(m.fiber_direction, np.array([0, 0, 1])))
        m.fiber_angle_from_z_deg = float(np.degrees(np.arccos(np.clip(cos_angle, 0, 1))))
        m.direction_source = "pca"

    def _estimate_endpoints(self, m: MuscleInfo, coords: np.ndarray):
        """Fiber direction from centroids at z-extent start and end.

        Computes the centroid of voxels in the bottom 10% and top 10%
        of the muscle's z-range, then connects them.
        """
        z = coords[:, 2]
        z_range = z.max() - z.min()
        if z_range < 1e-3:
            m.fiber_direction = np.array([0.0, 0.0, 1.0])
            m.direction_source = "endpoints"
            return

        # Bottom 10% and top 10%
        z_lo = z.min() + 0.1 * z_range
        z_hi = z.max() - 0.1 * z_range

        start_mask = z <= z_lo
        end_mask = z >= z_hi

        start_centroid = coords[start_mask].mean(axis=0)
        end_centroid = coords[end_mask].mean(axis=0)

        direction = end_centroid - start_centroid
        length = np.linalg.norm(direction)

        if length < 1e-6:
            m.fiber_direction = np.array([0.0, 0.0, 1.0])
        else:
            m.fiber_direction = direction / length

        # Canonical: positive z-component
        if m.fiber_direction[2] < 0:
            m.fiber_direction = -m.fiber_direction

        m.start_point_mm = start_centroid
        m.end_point_mm = end_centroid

        cos_angle = abs(np.dot(m.fiber_direction, np.array([0, 0, 1])))
        m.fiber_angle_from_z_deg = float(np.degrees(np.arccos(np.clip(cos_angle, 0, 1))))
        m.direction_source = "endpoints"

    def set_manual_fiber(
        self, label: int,
        start_mm: np.ndarray | list | None = None,
        end_mm: np.ndarray | list | None = None,
        direction: np.ndarray | list | None = None,
    ):
        """Manually set fiber direction for a muscle.

        Either provide start_mm + end_mm (attachment points), or
        direction (unit vector) directly.
        """
        if label not in self.muscles:
            raise ValueError(f"Label {label} not found in segmentation")

        m = self.muscles[label]

        if direction is not None:
            d = np.asarray(direction, dtype=float)
            m.fiber_direction = d / np.linalg.norm(d)
            if m.fiber_direction[2] < 0:
                m.fiber_direction = -m.fiber_direction
        elif start_mm is not None and end_mm is not None:
            s = np.asarray(start_mm, dtype=float)
            e = np.asarray(end_mm, dtype=float)
            d = e - s
            length = np.linalg.norm(d)
            if length < 1e-6:
                raise ValueError("Start and end points are identical")
            m.fiber_direction = d / length
            if m.fiber_direction[2] < 0:
                m.fiber_direction = -m.fiber_direction
            m.start_point_mm = s
            m.end_point_mm = e
        else:
            raise ValueError("Provide either (start_mm, end_mm) or direction")

        cos_angle = abs(np.dot(m.fiber_direction, np.array([0, 0, 1])))
        m.fiber_angle_from_z_deg = float(np.degrees(np.arccos(np.clip(cos_angle, 0, 1))))
        m.direction_source = "manual"

    def estimate_centerlines(self, min_slices: int = 3, smooth_sigma: float = 1.0):
        """Compute piecewise-linear centerlines for each muscle.

        For each muscle label, groups voxels by z-slice, computes the (x, y)
        centroid per slice, and creates a MuscleCenterline from the waypoints.

        Requires that _load_segmentation() has been called (seg_data exists).

        Parameters
        ----------
        min_slices : int
            Minimum number of z-slices required to create a centerline.
            Muscles with fewer slices keep their global fiber direction only.
        smooth_sigma : float
            Gaussian smoothing sigma (in slice bins) applied to the per-slice
            (cx, cy) centroid waypoints before constructing the spline.
            1 bin = one z-slice (typically 6 mm for this MRI). Smooths
            voxel-quantization wiggle so the centerline spline isn't forced
            through noisy waypoints. Set to 0 to disable.
        """
        if not hasattr(self, "seg_data"):
            raise RuntimeError(
                "Segmentation data not loaded. "
                "Initialize MuscleFiberModel with a nifti_path first."
            )

        n_created = 0
        for label, m in self.muscles.items():
            if m.tissue_type != "muscle":
                continue

            mask = self.seg_data == label
            voxel_indices = np.argwhere(mask)  # (N, 3) in voxel coords

            if len(voxel_indices) < 10:
                continue

            # Group by z-slice index
            z_slices = np.unique(voxel_indices[:, 2])
            if len(z_slices) < min_slices:
                continue

            # Compute centroid per z-slice in physical coords
            z_mm_list = []
            cx_mm_list = []
            cy_mm_list = []

            for z_idx in z_slices:
                slice_mask = voxel_indices[:, 2] == z_idx
                slice_voxels = voxel_indices[slice_mask]
                # Convert to physical coordinates
                phys = slice_voxels * self.voxel_size
                z_mm_list.append(float(phys[0, 2]))  # all same z
                cx_mm_list.append(float(phys[:, 0].mean()))
                cy_mm_list.append(float(phys[:, 1].mean()))

            z_arr = np.array(z_mm_list)
            cx_arr = np.array(cx_mm_list)
            cy_arr = np.array(cy_mm_list)

            # Smooth the centroid waypoints along slice axis to remove
            # voxel-quantization jitter. mode='nearest' preserves endpoints.
            if smooth_sigma > 0 and len(cx_arr) >= 3:
                cx_arr = gaussian_filter1d(cx_arr, sigma=smooth_sigma, mode="nearest")
                cy_arr = gaussian_filter1d(cy_arr, sigma=smooth_sigma, mode="nearest")

            m.centerline = MuscleCenterline(
                z_mm=z_arr,
                cx_mm=cx_arr,
                cy_mm=cy_arr,
            )
            n_created += 1

        print(f"Estimated centerlines for {n_created} muscles "
              f"(min_slices={min_slices}, smooth_sigma={smooth_sigma})")

    def estimate_cross_sections(
        self,
        n_theta: int = 72,
        step_mm: float | None = None,
        r_inset: float = 0.95,
        smooth_sigma: float = 1.5,
        z_smooth_sigma: float = 1.5,
        min_voxels_per_slice: int = 5,
        mask_smooth_xy_mm: float = 0.0,
        mask_smooth_z_mm: float = 0.0,
        mask_dilate_iters: int = 0,
        constrain_dilation: bool = True,
    ):
        """Compute radial boundary distances for each muscle cross-section.

        Uses ray-casting from the centroid at each z-slice: march outward
        at n_theta angles and record the distance at the first exit from
        the muscle mask. Requires centerlines to be estimated first.

        Parameters
        ----------
        n_theta : int
            Number of angular samples (default 72 = 5 deg spacing).
        step_mm : float or None
            Ray-marching step size. If None, uses half the smaller xy voxel dim.
        r_inset : float
            Safety margin factor for fiber mapping (default 0.95).
        smooth_sigma : float
            Gaussian smoothing sigma (in angular bins) applied to R_boundary
            along theta to reduce voxel staircase noise. 1 bin = 5 deg.
        z_smooth_sigma : float
            Gaussian smoothing sigma (in slice bins) applied to R_boundary
            along z, AFTER theta smoothing. 1 bin = one z-slice = ``voxel_size[2]`` mm
            (typically 6 mm for this MRI). Reduces slice-to-slice jitter so
            morphing-disk fibers vary smoothly along the fiber axis.
            Set to 0 to disable.
        min_voxels_per_slice : int
            Minimum voxels in a slice to compute boundary (skip degenerate endpoints).
        mask_smooth_xy_mm : float, default 0
            If > 0, BEFORE ray-casting, 3D-smooth the binary muscle mask with
            a Gaussian (σ = this many mm in xy) and re-threshold at 0.5.
            Tends to preserve outer extent better than R-smoothing — the
            boundary doesn't shrink inward toward gaps between adjacent
            muscles, because the blur sees the mask, not a 1D radial signal.
        mask_smooth_z_mm : float, default 0
            Same idea, σ in mm along z. Only used if mask_smooth_xy_mm > 0.
            For the 6mm-anisotropic MRI a z σ ≈ 6mm is reasonable.
        mask_dilate_iters : int, default 0
            If > 0, apply binary dilation N times to the mask BEFORE smoothing.
            Useful to fill small concavities and to extend a muscle outward
            into the void between it and its neighbours, simulating the
            "muscles touching" property of real anatomy.
        constrain_dilation : bool, default True (when mask_dilate_iters > 0)
            If True, dilation only adds voxels that are NOT currently
            assigned to any other muscle (effectively a Voronoi-style
            expansion into the gaps). Prevents one muscle from bleeding
            into a neighbour.
        """
        if not hasattr(self, "seg_data"):
            raise RuntimeError(
                "Segmentation data not loaded. "
                "Initialize MuscleFiberModel with a nifti_path first."
            )

        if step_mm is None:
            step_mm = min(self.voxel_size[0], self.voxel_size[1]) * 0.5

        theta_deg = np.linspace(0, 360, n_theta, endpoint=False)
        theta_rad = np.radians(theta_deg)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        from scipy.ndimage import gaussian_filter, binary_dilation

        # Pre-compute "other muscle voxels" for constrained dilation
        if mask_dilate_iters > 0 and constrain_dilation:
            other_muscle_voxels = np.zeros_like(self.seg_data, dtype=bool)
            for other_lab, other_m in self.muscles.items():
                if other_m.tissue_type == "muscle":
                    other_muscle_voxels |= (self.seg_data == other_lab)

        n_created = 0
        for label, m in self.muscles.items():
            if m.tissue_type != "muscle":
                continue
            if m.centerline is None:
                continue

            mask = self.seg_data == label
            # Optional mask-based smoothing: dilate then 3D-blur then
            # threshold. Preserves outer extent and fills small concavities,
            # so the smoothed boundary touches neighbouring muscles rather
            # than leaving gaps.
            if mask_dilate_iters > 0:
                dilated = binary_dilation(
                    mask, iterations=int(mask_dilate_iters))
                if constrain_dilation:
                    # Only add voxels that aren't another muscle
                    not_other = ~(other_muscle_voxels & ~mask)
                    mask = mask | (dilated & not_other)
                else:
                    mask = dilated
            if mask_smooth_xy_mm > 0:
                sigmas = (
                    mask_smooth_xy_mm / max(self.voxel_size[0], 1e-6),
                    mask_smooth_xy_mm / max(self.voxel_size[1], 1e-6),
                    max(mask_smooth_z_mm, 0) / max(self.voxel_size[2], 1e-6),
                )
                blurred = gaussian_filter(mask.astype(float), sigmas)
                mask = blurred > 0.5
            cl = m.centerline

            # Determine which z-slices to use (same as centerline)
            voxel_indices = np.argwhere(mask)
            z_slice_indices = np.unique(voxel_indices[:, 2])

            z_mm_list = []
            R_boundary_list = []

            for z_idx in z_slice_indices:
                slice_mask = mask[:, :, z_idx]
                n_vox = np.sum(slice_mask)
                if n_vox < min_voxels_per_slice:
                    continue

                z_mm = float(z_idx * self.voxel_size[2])

                # Get centroid at this z from centerline
                pos = cl.position(z_mm)
                cx, cy = pos[0], pos[1]

                # Check if centroid is inside mask; if not, use mask center-of-mass
                cx_vox = int(round(cx / self.voxel_size[0]))
                cy_vox = int(round(cy / self.voxel_size[1]))
                cx_vox = np.clip(cx_vox, 0, self.seg_data.shape[0] - 1)
                cy_vox = np.clip(cy_vox, 0, self.seg_data.shape[1] - 1)

                if not slice_mask[cx_vox, cy_vox]:
                    # Centroid outside mask — use mask center-of-mass
                    vox_2d = np.argwhere(slice_mask)
                    cx = float(vox_2d[:, 0].mean()) * self.voxel_size[0]
                    cy = float(vox_2d[:, 1].mean()) * self.voxel_size[1]

                # Ray-cast at each angle
                R_row = np.zeros(n_theta)
                # Max possible radius (diagonal of slice)
                max_r = np.sqrt(
                    (self.seg_data.shape[0] * self.voxel_size[0]) ** 2
                    + (self.seg_data.shape[1] * self.voxel_size[1]) ** 2
                )

                for ti in range(n_theta):
                    # March outward from centroid
                    r = step_mm
                    while r < max_r:
                        px = cx + r * cos_theta[ti]
                        py = cy + r * sin_theta[ti]
                        # Convert to voxel
                        ix = int(round(px / self.voxel_size[0]))
                        iy = int(round(py / self.voxel_size[1]))
                        if (ix < 0 or ix >= self.seg_data.shape[0]
                                or iy < 0 or iy >= self.seg_data.shape[1]):
                            # Hit image boundary
                            R_row[ti] = r - step_mm
                            break
                        if not slice_mask[ix, iy]:
                            # First exit from muscle
                            R_row[ti] = r - step_mm
                            break
                        r += step_mm
                    else:
                        R_row[ti] = r - step_mm

                    # Ensure minimum radius
                    R_row[ti] = max(R_row[ti], step_mm)

                # Smooth along theta (circular) to reduce voxel staircase
                if smooth_sigma > 0:
                    R_row = gaussian_filter1d(R_row, sigma=smooth_sigma, mode="wrap")

                z_mm_list.append(z_mm)
                R_boundary_list.append(R_row)

            if len(z_mm_list) < 2:
                continue

            R_boundary = np.array(R_boundary_list)

            # Smooth along z (per-theta column) to remove slice-to-slice jitter
            # caused by 6mm anisotropic voxel spacing + per-slice ray-cast noise.
            # mode='nearest' preserves endpoint values (no false taper toward 0).
            if z_smooth_sigma > 0 and R_boundary.shape[0] >= 3:
                R_boundary = gaussian_filter1d(
                    R_boundary, sigma=z_smooth_sigma, axis=0, mode="nearest"
                )
                # Floor to avoid degenerate R after smoothing
                R_boundary = np.maximum(R_boundary, step_mm)

            m.cross_section = MuscleCrossSection(
                z_mm=np.array(z_mm_list),
                theta_deg=theta_deg,
                R_boundary=R_boundary,
                r_inset=r_inset,
            )
            n_created += 1

        print(f"Estimated cross-sections for {n_created} muscles "
              f"(n_theta={n_theta}, r_inset={r_inset}, "
              f"theta_sigma={smooth_sigma}, z_sigma={z_smooth_sigma})")

    def get_conductivity_tensor(self, label: int) -> np.ndarray:
        """Get the 3x3 conductivity tensor for a given label (global direction)."""
        if label in self.muscles:
            return self.muscles[label].conductivity_tensor
        elif label in FAT_SKIN_LABELS or label == 0:
            return SIGMA_FAT_SKIN * np.eye(3)
        elif label in CONNECTIVE_LABELS:
            return SIGMA_CONNECTIVE * np.eye(3)
        else:
            # Unknown label — treat as muscle with z-aligned fiber
            return SIGMA_MUSCLE_Z.copy()

    def get_conductivity_tensor_at(self, label: int, z_mm: float) -> np.ndarray:
        """Get 3x3 conductivity tensor at a specific z-position.

        For muscles with centerlines, uses position-dependent fiber direction.
        Otherwise falls back to get_conductivity_tensor().
        """
        if label in self.muscles:
            return self.muscles[label].conductivity_tensor_at_z(z_mm)
        return self.get_conductivity_tensor(label)

    def get_all_tensors(self) -> dict[int, np.ndarray]:
        """Get conductivity tensors for all labels (global direction)."""
        return {label: m.conductivity_tensor for label, m in self.muscles.items()}

    # ---- I/O ----

    def save_config(self, path: str):
        """Save fiber configuration to JSON for manual editing."""
        config = {
            "description": (
                "Per-muscle fiber direction configuration for MRI FEM model. "
                "Edit start_point_mm / end_point_mm to manually set fiber "
                "directions, then set direction_source to 'manual'."
            ),
            "conductivity": {
                "sigma_muscle_cross": SIGMA_MUSCLE_CROSS,
                "sigma_muscle_fiber": SIGMA_MUSCLE_FIBER,
                "sigma_fat_skin": SIGMA_FAT_SKIN,
                "sigma_connective": SIGMA_CONNECTIVE,
                "anisotropy_ratio": SIGMA_MUSCLE_FIBER / SIGMA_MUSCLE_CROSS,
            },
            "muscles": {
                str(label): m.to_dict()
                for label, m in sorted(self.muscles.items())
            },
        }

        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved fiber config to {path} ({len(self.muscles)} labels)")

    def load_config(self, path: str):
        """Load fiber configuration from JSON (possibly manually edited)."""
        with open(path) as f:
            config = json.load(f)

        for key, d in config["muscles"].items():
            label = int(d["label"])
            m = MuscleInfo.from_dict(d)

            # If manual override with start/end, recompute direction
            if m.direction_source == "manual" and m.start_point_mm is not None and m.end_point_mm is not None:
                direction = m.end_point_mm - m.start_point_mm
                length = np.linalg.norm(direction)
                if length > 1e-6:
                    m.fiber_direction = direction / length
                    if m.fiber_direction[2] < 0:
                        m.fiber_direction = -m.fiber_direction

            self.muscles[label] = m

        print(f"Loaded fiber config from {path} ({len(self.muscles)} labels)")

    # ---- Summary ----

    def summary(self) -> str:
        """Print summary table of muscle fiber directions."""
        lines = []
        lines.append(f"{'Label':>5} {'Type':>12} {'Slices':>6} {'z_span':>7} "
                      f"{'Fiber (x,y,z)':>22} {'Angle°':>7} {'Elong':>6} {'Source':>8} {'CL':>3}")
        lines.append("-" * 90)

        for label in sorted(self.muscles):
            m = self.muscles[label]
            fd = m.fiber_direction
            cl_flag = "yes" if m.centerline is not None else ""
            lines.append(
                f"{label:>5} {m.tissue_type:>12} {m.n_slices:>6} {m.z_span_mm:>7.0f} "
                f"({fd[0]:>+.3f},{fd[1]:>+.3f},{fd[2]:>+.3f}) "
                f"{m.fiber_angle_from_z_deg:>7.1f} {m.pca_elongation:>6.1f} {m.direction_source:>8} {cl_flag:>3}"
            )

        return "\n".join(lines)


__all__ = [
    "MuscleCenterline",
    "MuscleCrossSection",
    "MuscleFiberModel",
    "MuscleInfo",
    "rotate_conductivity",
    "rodrigues_rotation",
    "SIGMA_MUSCLE_Z",
    "SIGMA_MUSCLE_CROSS",
    "SIGMA_MUSCLE_FIBER",
]
