"""
FEM solver adapted for MRI-derived forearm meshes (v2).

Solves the conductivity equation:
    ∇·(Σ(x)∇u) = f(x)   in Ω
    (Σ∇u)·n = 0          on ∂Ω

on a tetrahedral mesh generated from MRI segmentation, with per-muscle
anisotropic conductivity tensors rotated to match individual fiber directions.

v2 changes:
  - Per-muscle conductivity tensors (from fiber_directions.py)
  - Each muscle gets its own rotated 3x3 anisotropic tensor
  - Connective/fat remain isotropic

Usage:
    from mri.core.fem_solver import MRIFEMModel

    # v1: uniform z-aligned muscle anisotropy (backwards compatible)
    model = MRIFEMModel("mri/mesh/forearm.msh")

    # v2: per-muscle fiber directions
    model = MRIFEMModel(
        "mri/mesh/forearm.msh",
        fiber_config="mri/mesh/muscle_fibers.json",
    )

    uh = model.solve_for_point(source_point, source_sigma=5.0)
    vals = model.evaluate_solution_at_points(probe_points)

Requires fenicsx-env conda environment (dolfinx, petsc4py, mpi4py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, geometry, io
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType as default_scalar_type
from ufl import dx, ds


# ---------------------------------------------------------------------------
# MRI tissue mapping (from build_mesh.py)
# ---------------------------------------------------------------------------
# Physical tags in the .msh file (tissue_id + 1):
#   1 = background (tets at mesh boundary, outside segmentation)
#   2 = fat_skin (label 25)
#   3 = connective (labels 15, 22)
#   4 = muscle (all other non-zero labels)

TAG_TO_MATERIAL = {
    1: "fat_skin",     # background tets → treat as fat
    2: "fat_skin",
    3: "connective",
    4: "muscle",
}

# Default conductivity tensors (v1: uniform muscle anisotropy along z).
# ANISOTROPY_RATIO / SIGMA_MUSCLE_CROSS come from the single source of truth
# (emgop.tissue, dolfinx-free) so this table cannot drift from the FEM/analytical ones.
from emgop.tissue import ANISOTROPY_RATIO, SIGMA_MUSCLE_CROSS

CONDUCTIVITY = {
    "fat_skin": 0.0379,         # isotropic (using fat value)
    "connective": 0.2,          # isotropic
    "skin": 4.55e-4,            # isotropic (used by skin_shell_mm > 0)
    "muscle": np.diag([
        SIGMA_MUSCLE_CROSS,
        SIGMA_MUSCLE_CROSS,
        ANISOTROPY_RATIO * SIGMA_MUSCLE_CROSS,
    ]),
}


class MRIFEMModel:
    """FEM solver for MRI-derived forearm meshes.

    Mirrors the interface of ``emgop.fem.solver.FEMModel`` but adapted for:
    - MRI-derived tissue labels (fat_skin, connective, muscle)
    - Irregular non-cylindrical geometry
    - Per-muscle fiber direction (v2: via fiber_config)
    """

    default_options = {
        "source_sigma": 5.0,        # Gaussian source width (mm)
        "source_degree": 1,         # source function space degree
        "boundary_value": 0,        # Neumann BC value
        "solver_type": "gmres",     # PETSc KSP type
        "preconditioner": "ilu",    # PETSc PC type
        "rtol": 1e-8,
        "atol": 1e-10,
        "max_iter": 5000,
    }

    def __init__(
        self,
        msh_file_path: str,
        fiber_config: str | None = None,
        nifti_path: str | None = None,
        gdim: int = 3,
        skin_shell_mm: float = 0.0,
        sigma_mode: str = "centerline",
        **options,
    ):
        """
        Parameters
        ----------
        msh_file_path : str
            Path to Gmsh .msh mesh file.
        fiber_config : str, optional
            Path to muscle_fibers.json for per-muscle conductivity.
            If None, uses uniform z-aligned muscle anisotropy (v1).
        nifti_path : str, optional
            Path to NIfTI segmentation for per-cell label lookup.
            Required if fiber_config is provided (to map cells to muscles).
            If not provided, tries to find it from mesh metadata.
        gdim : int
            Geometric dimension (default 3).
        skin_shell_mm : float, default 0.0
            If > 0, retag cells in fat_skin tissue whose centroid is within
            this distance (mm) of the outer mesh boundary as 'skin' with
            σ=4.55e-4 S/m (much lower than fat 0.0379 S/m). Label 25 in
            the segmentation is "fat+skin" combined; this lets you
            simulate the real skin layer without re-meshing.
        sigma_mode : str, default "centerline"
            How to orient the anisotropic muscle σ tensor:
              "constant"   — z-aligned everywhere (no rotation)
              "global"     — global PCA direction per muscle (one
                             direction for the whole muscle)
              "centerline" — centerline tangent at the cell's z
                             (uniform across the cross-section, varies
                             along z — the v3 default)
              "morphing"   — per-cell fiber tangent from
                             fiber_tangent_morphing at the cell's
                             (r_norm, θ, z) within its muscle's
                             cross-section. Most physical but most
                             expensive.
        **options
            Override default solver options.
        """
        self.mesh, self.cell_markers, self.facet_markers = io.gmshio.read_from_msh(
            msh_file_path, MPI.COMM_WORLD, gdim=gdim
        )
        self.mesh_geometry = self.mesh.geometry
        self.mesh_topology = self.mesh.topology
        self.msh_path = Path(msh_file_path)

        self.options = self.default_options.copy()
        self.options.update(options)

        # Per-muscle fiber model (v2)
        self.fiber_model = None
        self._seg_labels_per_cell = None
        self.skin_shell_mm = float(skin_shell_mm)
        self._is_skin_cell = None  # bool array per cell
        if sigma_mode not in ("constant", "global", "centerline", "morphing"):
            raise ValueError(f"unknown sigma_mode {sigma_mode!r}")
        self.sigma_mode = sigma_mode
        self._cell_centroids = None  # populated by _compute_seg_labels

        if fiber_config is not None:
            self._load_fiber_config(fiber_config, nifti_path)

        self.build_model()

        if self.skin_shell_mm > 0:
            self._apply_skin_shell()

    def _load_fiber_config(self, fiber_config: str, nifti_path: str | None):
        """Load per-muscle fiber directions and segmentation for cell lookup."""
        from mri.core.fiber_directions import MuscleFiberModel

        self.fiber_model = MuscleFiberModel()
        self.fiber_model.load_config(fiber_config)

        # Need segmentation to map each cell to its muscle label
        if nifti_path is None:
            # Try to find from mesh metadata
            meta_path = self.msh_path.with_name(
                self.msh_path.stem + "_metadata.json"
            )
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                nifti_path = meta.get("source_nifti")

        if nifti_path is None:
            raise ValueError(
                "nifti_path required for per-muscle conductivity. "
                "Provide it explicitly or ensure forearm_metadata.json exists."
            )

        self._compute_seg_labels(nifti_path)

    def _compute_seg_labels(self, nifti_path: str):
        """Map each mesh cell to its segmentation label via centroid lookup."""
        import nibabel as nib
        from scipy.ndimage import zoom

        img = nib.load(nifti_path)
        seg_data = img.get_fdata().astype(int)
        voxel_size = np.array(img.header.get_zooms())

        # Resample z if needed (same as build_mesh.py)
        target_z = 1.0
        if voxel_size[2] > target_z * 1.5:
            z_factor = voxel_size[2] / target_z
            seg_data = zoom(seg_data.astype(np.float32), (1, 1, z_factor), order=0).astype(int)
            voxel_size = np.array([voxel_size[0], voxel_size[1], target_z])

        # Compute cell centroids
        n_cells = self.mesh.topology.index_map(3).size_local
        self.mesh_topology.create_connectivity(3, 0)
        c2v = self.mesh_topology.connectivity(3, 0)

        centroids = np.zeros((n_cells, 3))
        for i in range(n_cells):
            verts = c2v.links(i)
            centroids[i] = self.mesh_geometry.x[verts].mean(axis=0)

        self._cell_centroids = centroids  # save for sigma_mode == "morphing"

        # Convert physical coords → voxel indices
        voxel_idx = (centroids / voxel_size).astype(int)
        for d in range(3):
            voxel_idx[:, d] = np.clip(voxel_idx[:, d], 0, seg_data.shape[d] - 1)

        self._seg_labels_per_cell = seg_data[
            voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]
        ]

    # ---- Setup ----

    def build_model(self):
        """Create function spaces, bounding box tree, conductivity map."""
        self.tree = geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        n_cells = self.mesh.topology.index_map(3).size_local
        self.midpoints = geometry.create_midpoint_tree(
            self.mesh, self.mesh.topology.dim,
            np.arange(n_cells, dtype=np.int32),
        )

        self.mesh_topology.create_connectivity(self.mesh.topology.dim, 0)

        self.V_tensor = fem.functionspace(
            self.mesh, ("DG", 0, (self.mesh.topology.dim, self.mesh.topology.dim))
        )
        self.V_scalar = fem.functionspace(self.mesh, ("CG", 1))
        self.V_source = fem.functionspace(
            self.mesh, ("CG", self.options["source_degree"])
        )

        self.uh = fem.Function(self.V_scalar)

        if self.fiber_model is not None:
            self._build_conductivity_map_v2()
        else:
            self._build_conductivity_map_v1()

    def _build_conductivity_map_v1(self):
        """v1: Assign per-cell conductivity from tissue type (uniform muscle)."""
        self.sigma_anisotropic = fem.Function(self.V_tensor)

        with self.sigma_anisotropic.vector.localForm() as loc:
            for cell_idx, marker in enumerate(self.cell_markers.values):
                material = TAG_TO_MATERIAL.get(int(marker), "fat_skin")
                sigma_val = CONDUCTIVITY[material]

                if isinstance(sigma_val, np.ndarray):
                    loc.setValuesBlocked([cell_idx], sigma_val.flatten())
                else:
                    tensor = sigma_val * np.eye(3)
                    loc.setValuesBlocked([cell_idx], tensor.flatten())

    def _build_conductivity_map_v2(self):
        """Assign per-cell σ honoring self.sigma_mode.

        Dispatch:
          'constant'   — z-aligned diag(σx, σx, σz) everywhere muscle
          'global'     — global PCA direction per muscle (uniform within muscle)
          'centerline' — centerline tangent at cell z (uniform across xy)
          'morphing'   — per-tet fiber tangent from morphing-disk mapping
        """
        from mri.core.fiber_directions import (
            SIGMA_MUSCLE_Z, rotate_conductivity, FAT_SKIN_LABELS,
            CONNECTIVE_LABELS,
        )

        self.sigma_anisotropic = fem.Function(self.V_tensor)

        n_cells = self.mesh.topology.index_map(3).size_local
        n_per_muscle = {}

        # Pre-compute centroids if not already done
        if self._cell_centroids is None:
            c2v = self.mesh_topology.connectivity(3, 0)
            centroids = np.zeros((n_cells, 3))
            for i in range(n_cells):
                verts = c2v.links(i)
                centroids[i] = self.mesh_geometry.x[verts].mean(axis=0)
            self._cell_centroids = centroids
        centroids = self._cell_centroids

        # Cache global tensors per muscle (used by 'global', and as fallback
        # for non-centerline muscles in 'centerline'/'morphing')
        global_tensor = {}
        for label, muscle in self.fiber_model.muscles.items():
            global_tensor[label] = muscle.conductivity_tensor

        # Track which morphing-eligible muscles we have
        morph_labels = set()
        for label, muscle in self.fiber_model.muscles.items():
            if (muscle.tissue_type == "muscle"
                    and muscle.centerline is not None
                    and muscle.cross_section is not None):
                morph_labels.add(label)

        with self.sigma_anisotropic.vector.localForm() as loc:
            for cell_idx in range(n_cells):
                seg_label = int(self._seg_labels_per_cell[cell_idx])
                cx, cy, cz = centroids[cell_idx]
                muscle = self.fiber_model.muscles.get(seg_label)

                # Non-muscle dispatch
                if muscle is None:
                    if seg_label == 0:
                        tensor = CONDUCTIVITY["fat_skin"] * np.eye(3)
                    else:
                        tensor = CONDUCTIVITY["muscle"]
                elif muscle.tissue_type != "muscle":
                    tensor = global_tensor[seg_label]
                # Muscle dispatch by mode
                elif self.sigma_mode == "constant":
                    tensor = SIGMA_MUSCLE_Z.copy()
                elif self.sigma_mode == "global":
                    tensor = global_tensor[seg_label]
                elif self.sigma_mode == "centerline":
                    if muscle.centerline is not None:
                        tensor = muscle.conductivity_tensor_at_z(cz)
                    else:
                        tensor = global_tensor[seg_label]
                elif self.sigma_mode == "morphing":
                    if seg_label in morph_labels:
                        tensor = self._sigma_at_cell_morphing(
                            muscle, cx, cy, cz,
                        )
                    elif muscle.centerline is not None:
                        tensor = muscle.conductivity_tensor_at_z(cz)
                    else:
                        tensor = global_tensor[seg_label]
                else:
                    tensor = global_tensor[seg_label]

                loc.setValuesBlocked([cell_idx], tensor.flatten())
                n_per_muscle[seg_label] = n_per_muscle.get(seg_label, 0) + 1

        n_aniso = sum(
            count for label, count in n_per_muscle.items()
            if label in self.fiber_model.muscles
            and self.fiber_model.muscles[label].tissue_type == "muscle"
        )
        n_morph_used = sum(
            count for label, count in n_per_muscle.items()
            if label in morph_labels
        )
        print(f"σ map [{self.sigma_mode}]: {n_aniso} muscle cells "
              f"({n_morph_used} with cross-section info), "
              f"{len(n_per_muscle)} unique labels")

    def _sigma_at_cell_morphing(self, muscle, cx, cy, cz):
        """Per-tet σ using fiber_tangent_morphing at the cell's (r_norm, θ, z).

        Maps the cell's (cx, cy) to (r_norm, θ) inside the muscle's
        cross-section at z=cz, then uses the morphing-fiber tangent
        there as the σ orientation.

        Robust to cz outside the muscle's centerline z-range (clamps z)
        and to degenerate cases (returns z-aligned σ).
        """
        from mri.core.fiber_directions import (
            SIGMA_MUSCLE_Z, rotate_conductivity,
        )
        cl = muscle.centerline
        cs = muscle.cross_section

        # Clamp cz to the centerline's valid range with a 1mm buffer so the
        # central-FD tangent doesn't degenerate at the very edge.
        cz_use = float(np.clip(cz, cl.z_min + 1.0, cl.z_max - 1.0))
        if cl.z_max - cl.z_min < 3.0:
            return SIGMA_MUSCLE_Z.copy()  # muscle too thin → fall back

        # Centerline position at cz_use
        pos = cl.position(cz_use)
        mx, my = pos[0], pos[1]
        dx, dy = cx - mx, cy - my
        R_rel = (dx * dx + dy * dy) ** 0.5

        if R_rel < 1e-3:
            r_norm = 0.0
            theta_deg = 0.0
        else:
            theta_deg = float(np.degrees(np.arctan2(dy, dx))) % 360.0
            R_b = float(cs.boundary_radius(
                np.array([cz_use]), np.array([theta_deg]),
            )[0])
            if R_b < 1e-3:
                r_norm = 0.0
            else:
                r_norm = R_rel / max(cs.r_inset * R_b, 1e-6)
                # Cells outside the inset boundary use r_norm=1 (extrapolation).
                r_norm = float(np.clip(r_norm, 0.0, 1.0))

        tangent = cl.fiber_tangent_morphing(
            r_norm, theta_deg, np.array([cz_use]), cs,
        )[0]
        # Guard against degenerate tangent (zero norm)
        if not np.all(np.isfinite(tangent)) or np.linalg.norm(tangent) < 1e-6:
            return SIGMA_MUSCLE_Z.copy()
        return rotate_conductivity(SIGMA_MUSCLE_Z, tangent)

    # ---- Skin shell ----

    def _apply_skin_shell(self):
        """Retag cells near the outer mesh boundary as 'skin' (low σ).

        Identifies cells whose centroid is within ``skin_shell_mm`` of the
        outer mesh boundary AND whose current tag is fat_skin (tags 1 or 2,
        or per-muscle-label fat_skin). Overrides their conductivity tensor
        with the skin value (4.55e-4 S/m, isotropic).

        Boundary distance is computed against mesh boundary VERTICES (cheap
        KDTree). For our forearm mesh this approximates the skin surface
        well because the outer mesh boundary IS the skin.
        """
        from scipy.spatial import cKDTree

        # 1. Cell centroids
        n_cells = self.mesh.topology.index_map(3).size_local
        self.mesh_topology.create_connectivity(3, 0)
        c2v = self.mesh_topology.connectivity(3, 0)
        centroids = np.zeros((n_cells, 3))
        for i in range(n_cells):
            verts = c2v.links(i)
            centroids[i] = self.mesh_geometry.x[verts].mean(axis=0)

        # 2. Boundary vertices (vertices on facets that bound only one cell)
        self.mesh_topology.create_connectivity(2, 3)  # facet -> cell
        f2c = self.mesh_topology.connectivity(2, 3)
        n_facets = self.mesh_topology.index_map(2).size_local
        boundary_verts = set()
        self.mesh_topology.create_connectivity(2, 0)  # facet -> vertex
        f2v = self.mesh_topology.connectivity(2, 0)
        for fi in range(n_facets):
            if len(f2c.links(fi)) == 1:  # boundary facet
                for v in f2v.links(fi):
                    boundary_verts.add(int(v))
        boundary_pts = self.mesh_geometry.x[sorted(boundary_verts)]
        print(f"  skin shell: {len(boundary_pts)} boundary vertices")

        # 3. Distance from each cell centroid to nearest boundary vertex
        tree = cKDTree(boundary_pts)
        dists, _ = tree.query(centroids, k=1)

        # 4. Find fat_skin cells (tag 1, 2, or fat_skin label in segmentation)
        from mri.core.fiber_directions import FAT_SKIN_LABELS
        is_fat_skin = np.zeros(n_cells, dtype=bool)
        if self._seg_labels_per_cell is not None:
            for ci in range(n_cells):
                lab = int(self._seg_labels_per_cell[ci])
                if lab in FAT_SKIN_LABELS or lab == 0:
                    is_fat_skin[ci] = True
        else:
            for ci, marker in enumerate(self.cell_markers.values):
                if int(marker) in (1, 2):
                    is_fat_skin[ci] = True

        # 5. Mark skin cells
        is_skin = is_fat_skin & (dists < self.skin_shell_mm)
        n_skin = int(is_skin.sum())
        n_fat_skin = int(is_fat_skin.sum())
        print(f"  skin shell: {n_skin} / {n_fat_skin} fat_skin cells "
              f"within {self.skin_shell_mm}mm of boundary "
              f"({n_skin / max(n_fat_skin, 1) * 100:.1f}%)")

        # 6. Override σ for skin cells
        skin_sigma = CONDUCTIVITY["skin"] * np.eye(3)
        with self.sigma_anisotropic.vector.localForm() as loc:
            for ci in np.where(is_skin)[0]:
                loc.setValuesBlocked([int(ci)], skin_sigma.flatten())
        self._is_skin_cell = is_skin

    # ---- Source ----

    @staticmethod
    def _gaussian_nd(x, center, sigma):
        """3D Gaussian blob."""
        squared_dist = np.sum((x.T - center) ** 2, axis=-1)
        norm = (2 * np.pi * sigma**2) ** 1.5
        return (1 / norm) * np.exp(-squared_dist / (2 * sigma**2))

    def assign_source_to_point(self, point: np.ndarray, source_sigma: float | None = None):
        """Create a zero-integral Gaussian source centered at `point`."""
        self.point = np.asarray(point, dtype=np.float64).reshape(3)
        sigma = source_sigma or self.options["source_sigma"]

        self.source_function = fem.Function(self.V_source)

        # Interpolate Gaussian
        self.source_function.interpolate(
            lambda x: self._gaussian_nd(x, self.point, sigma)
        )

        # Compute volume and enforce zero integral
        u_one = fem.Constant(self.mesh, default_scalar_type(1.0))
        self.volume = fem.assemble_scalar(fem.form(u_one * dx))

        integral = fem.assemble_scalar(
            fem.form(self.source_function * dx)
        )
        mean_f = integral / self.volume

        self.source_function.interpolate(
            lambda x: self._gaussian_nd(x, self.point, sigma) - mean_f
        )

    # ---- Solve ----

    def solve_for_point(
        self,
        point: np.ndarray,
        source_sigma: float | None = None,
    ) -> Function:
        """Solve the conductivity equation for a Gaussian source at `point`.

        Parameters
        ----------
        point : (3,) array
            Source location in physical coordinates (mm).
        source_sigma : float, optional
            Gaussian source width in mm. Default from options.

        Returns
        -------
        uh : dolfinx.fem.Function
            Solution potential (CG1).
        """
        self.assign_source_to_point(point, source_sigma)

        u = ufl.TrialFunction(self.V_scalar)
        v = ufl.TestFunction(self.V_scalar)
        g = fem.Constant(self.mesh, default_scalar_type(self.options["boundary_value"]))

        a = ufl.dot(ufl.dot(self.sigma_anisotropic, ufl.grad(u)), ufl.grad(v)) * dx
        L = self.source_function * v * dx + g * v * ds

        # Assemble
        A = assemble_matrix(form(a))
        A.assemble()

        b = create_vector(form(L))
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(b, form(L))

        # Nullspace: constant
        nullspace = PETSc.NullSpace().create(constant=True)
        A.setNullSpace(nullspace)
        nullspace.remove(b)

        # Solve
        self.uh = fem.Function(self.V_scalar)
        solver = PETSc.KSP().create(A.getComm())
        solver.setOperators(A)
        solver.setType(self.options["solver_type"])
        solver.getPC().setType(self.options["preconditioner"])
        solver.setTolerances(
            rtol=self.options["rtol"],
            atol=self.options["atol"],
            max_it=self.options["max_iter"],
        )
        solver.solve(b, self.uh.vector)

        self._last_solver = solver
        return self.uh

    # ---- Evaluation ----

    def evaluate_solution_at_points(
        self,
        points: np.ndarray,
        uh: Function | None = None,
    ) -> np.ndarray:
        """Evaluate the FEM solution at arbitrary points.

        Parameters
        ----------
        points : (N, 3) array
            Evaluation points in physical coordinates (mm).
        uh : Function, optional
            Solution to evaluate. Defaults to last solve result.

        Returns
        -------
        values : (N,) array
            Potential at each point.
        """
        uh = self.uh if uh is None else uh
        cell_ids = geometry.compute_closest_entity(
            self.tree, self.midpoints, self.mesh, points
        ).squeeze()
        vals = uh.eval(points, cell_ids)
        return np.asarray(vals).reshape(-1)

    # ---- Mesh info ----

    @property
    def n_cells(self) -> int:
        return self.mesh.topology.index_map(3).size_local

    @property
    def n_vertices(self) -> int:
        return self.mesh.topology.index_map(0).size_local

    @property
    def n_dofs(self) -> int:
        return self.V_scalar.dofmap.index_map.size_local

    def get_muscle_cells(self) -> np.ndarray:
        """Return indices of cells tagged as muscle."""
        return np.where(self.cell_markers.values == 4)[0]

    def get_cell_centroid(self, cell_idx: int) -> np.ndarray:
        """Get the centroid of a cell in physical coordinates."""
        c2v = self.mesh_topology.connectivity(3, 0)
        verts = c2v.links(cell_idx)
        return self.mesh_geometry.x[verts].mean(axis=0)

    def get_skin_surface_point(self, theta_deg: float, z_frac: float) -> np.ndarray:
        """Get a point on the outer mesh surface.

        Approximates skin surface placement for electrode positioning.

        Parameters
        ----------
        theta_deg : float
            Angle from +x axis in degrees (0-360).
        z_frac : float
            Fractional z-position (0=bottom, 1=top).

        Returns
        -------
        point : (3,) array
            Closest mesh boundary point.
        """
        coords = self.mesh_geometry.x
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        z_target = z_min + z_frac * (z_max - z_min)

        # Forearm centroid at target z
        z_band = np.abs(coords[:, 2] - z_target) < 3.0  # 3mm band
        if z_band.sum() == 0:
            z_band = np.abs(coords[:, 2] - z_target) < 10.0
        band_coords = coords[z_band]
        cx, cy = band_coords[:, 0].mean(), band_coords[:, 1].mean()

        # Ray from centroid at theta
        theta = np.deg2rad(theta_deg)
        dx_dir, dy_dir = np.cos(theta), np.sin(theta)

        # Find outermost mesh vertex along this ray
        vecs = band_coords[:, :2] - np.array([cx, cy])
        projections = vecs[:, 0] * dx_dir + vecs[:, 1] * dy_dir
        angular_dist = np.abs(
            np.arctan2(vecs[:, 1], vecs[:, 0]) - theta
        )
        angular_dist = np.minimum(angular_dist, 2*np.pi - angular_dist)

        # Select vertices within 10° of target angle, pick outermost
        angle_mask = angular_dist < np.deg2rad(10)
        if angle_mask.sum() == 0:
            angle_mask = angular_dist < np.deg2rad(30)

        candidates = band_coords[angle_mask]
        proj_cands = projections[angle_mask]
        best = np.argmax(proj_cands)

        return candidates[best]


__all__ = ["MRIFEMModel"]
