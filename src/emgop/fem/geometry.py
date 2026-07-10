"""Parametric layered geometry: mesh construction + electrode / fibre placement.

``ParametricGeometry`` is the shared geometry behind the cylinder and ellipse FEM
tiers, which previously duplicated their mesh-parameter dict, electrode formula, and
fibre-sampling math. The y-axis (b) uses the given radii; for an ellipse the x-axis
(a) is scaled by ``ellipse_ratio`` (1.0 = circle).

It knows only geometry — radii, length, angles, depths. It does not solve; pair it
with ``FEMModel`` (build the mesh, place an electrode, sample a solution along a
fibre line).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ParametricGeometry:
    r_bone: float        # cortical bone radius = inner muscle boundary (y-axis)
    r_muscle: float
    r_fat: float
    r_skin: float
    length: float
    ellipse_ratio: float = 1.0   # a/b; 1.0 → circle
    r_canc: float = 9.0          # thin cancellous core (5-layer FEM extra)

    @property
    def shape(self) -> str:
        return "circle" if self.ellipse_ratio == 1.0 else "ellipse"

    @property
    def a_skin(self) -> float:   # x-axis skin radius
        return self.r_skin * self.ellipse_ratio

    @property
    def b_skin(self) -> float:   # y-axis skin radius
        return self.r_skin

    # -- mesh -----------------------------------------------------------------

    def mesh_params(self) -> dict:
        """The ``emgop.meshing.build_one_mesh`` parameter dict for this geometry."""
        base = dict(bone_count=1, z_profile="constant",
                    radius_multiplicative_factor=40.0, length=self.length)
        if self.shape == "circle":
            return {**base, "shape": "circle",
                    "radius_canc_bone": self.r_canc, "radius_cort_bone": self.r_bone,
                    "radius_muscle": self.r_muscle, "radius_fat": self.r_fat,
                    "radius_skin": self.r_skin}
        r = self.ellipse_ratio
        b = dict(canc=self.r_canc, cort=self.r_bone, muscle=self.r_muscle,
                 fat=self.r_fat, skin=self.r_skin)
        return {**base, "shape": "ellipse", "ellipse_ratio": r,
                "bone_offset_x": 0.0, "bone_offset_y": 0.0,
                "a_canc_bone": r * b["canc"], "b_canc_bone": b["canc"],
                "a_cort_bone": r * b["cort"], "b_cort_bone": b["cort"],
                "a_muscle": r * b["muscle"], "b_muscle": b["muscle"],
                "a_fat": r * b["fat"], "b_fat": b["fat"],
                "a_skin": r * b["skin"], "b_skin": b["skin"]}

    def build(self, out_msh, out_json, char_length: float = 0.3, force: bool = False):
        """Build (and cache) the mesh via gmsh. Returns ``(msh_path, json_path)``."""
        out_msh, out_json = Path(out_msh), Path(out_json)
        if out_msh.exists() and not force:
            return out_msh, out_json
        out_msh.parent.mkdir(parents=True, exist_ok=True)
        import gmsh

        from emgop.meshing import build_one_mesh

        gmsh.initialize()
        try:
            build_one_mesh(self.mesh_params(), out_msh=out_msh, out_json=out_json,
                           mesh_char_length_factor=char_length, refine_on="skin")
        finally:
            gmsh.finalize()
        return out_msh, out_json

    # -- placement ------------------------------------------------------------

    def electrode_on_skin(self, theta_deg: float, z_mm: float) -> np.ndarray:
        """Electrode on the skin surface at parametric angle θ, height z."""
        th = np.deg2rad(theta_deg)
        return np.array([self.a_skin * np.cos(th), self.b_skin * np.sin(th), z_mm],
                        dtype=np.float64)

    def r_skin_at(self, theta_deg: float) -> float:
        """Radial distance from the centre to the skin boundary at angle θ (mm)."""
        th = np.deg2rad(theta_deg)
        return float(np.hypot(self.a_skin * np.cos(th), self.b_skin * np.sin(th)))

    def fibre_xy_radial(self, depth_mm: float, theta_deg: float):
        """(x, y) of a fibre at radial distance ``depth_mm`` from the centre, angle θ."""
        th = np.deg2rad(theta_deg)
        return depth_mm * np.cos(th), depth_mm * np.sin(th)

    def fibre_xy_below_skin(self, depth_below_mm: float, theta_deg: float):
        """(x, y) of a fibre ``depth_below_mm`` under the skin along the ray at angle θ."""
        th = np.deg2rad(theta_deg)
        frac = 1.0 - depth_below_mm / self.r_skin_at(theta_deg)
        return self.a_skin * np.cos(th) * frac, self.b_skin * np.sin(th) * frac

    def fibre_points(self, x0: float, y0: float, z_centroid_mm: float,
                     nz: int, dz: float, clip: float = 0.5):
        """A z-aligned fibre polyline at (x0, y0), centred at z_centroid.
        Returns ``(points (nz, 3), z_grid (nz,))`` with z_grid centred on 0."""
        z_grid = (np.arange(nz) - (nz - 1) / 2.0) * dz
        z_abs = np.clip(z_centroid_mm + z_grid, clip, self.length - clip)
        pts = np.column_stack([np.full(nz, x0), np.full(nz, y0), z_abs])
        return pts, z_grid
