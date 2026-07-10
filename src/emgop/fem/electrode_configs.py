"""
Electrode configuration utilities for FEM solver.

Provides:
- compute_ground_position(): return-current ground placement from mesh metadata
- ElectrodeFEMSolver: Wrapper around FEMModel with configurable return current
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType as default_scalar_type
from ufl import dx, ds

from .solver import FEMModel


def _get_geometry_param(meta: dict, key: str):
    """Get geometry parameter from meta, handling nested geometry_params structure."""
    # Try direct access first
    if key in meta:
        return meta[key]
    # Try nested geometry_params
    if "geometry_params" in meta and key in meta["geometry_params"]:
        return meta["geometry_params"][key]
    return None


def compute_ground_position(
    meta: dict,
    source_point: np.ndarray,
    ground_mode: str = "opposite",
    ground_cyl: tuple = None,
    inset: float = 0.0,
) -> np.ndarray:
    """
    Compute ground electrode position.

    Parameters
    ----------
    meta : dict
        Mesh metadata containing r_skin, length, etc. Can be flat or have nested
        'geometry_params' structure.
    source_point : np.ndarray
        Source electrode position (x, y, z)
    ground_mode : str
        Placement mode:
        - "opposite": same z, theta + 180 degrees, at r_skin
        - "distal": same theta, at z=0 or z=L (whichever is farther), at r_skin
        - "cylindrical": explicit normalized coordinates from ground_cyl
    ground_cyl : tuple, optional
        (r_norm, theta_deg, z_norm) for cylindrical mode
    inset : float
        Distance to inset from boundaries (radial and axial). Use this to ensure
        sampled electrode points stay inside the mesh. Should be >= electrode_radius.

    Returns
    -------
    np.ndarray
        Ground electrode position (x, y, z)
    """
    source_point = np.asarray(source_point).reshape(3)
    r_skin = _get_geometry_param(meta, "radius_skin")
    length = _get_geometry_param(meta, "length")

    # Effective radius for placement (inset from skin surface)
    r_effective = r_skin - inset

    if ground_mode == "opposite":
        # Same z, rotate theta by 180 degrees, inset from skin
        theta_source = np.arctan2(source_point[1], source_point[0])
        theta_ground = theta_source + np.pi
        z_ground = np.clip(source_point[2], inset, length - inset)
        return np.array([
            r_effective * np.cos(theta_ground),
            r_effective * np.sin(theta_ground),
            z_ground,
        ])

    elif ground_mode == "distal":
        # Same theta, at whichever z-end is farther, inset from boundaries
        theta_source = np.arctan2(source_point[1], source_point[0])
        z_source = source_point[2]
        # Pick z=inset or z=length-inset, whichever is farther from source
        z_ground = inset if z_source > length / 2 else (length - inset)
        return np.array([
            r_effective * np.cos(theta_source),
            r_effective * np.sin(theta_source),
            z_ground,
        ])

    elif ground_mode == "cylindrical" and ground_cyl is not None:
        r_norm, theta_deg, z_norm = ground_cyl
        theta_rad = np.deg2rad(theta_deg)
        r = r_norm * r_effective
        z = inset + z_norm * (length - 2 * inset)  # Scale z within inset bounds
        return np.array([
            r * np.cos(theta_rad),
            r * np.sin(theta_rad),
            z,
        ])

    else:
        raise ValueError(f"Unknown ground_mode: {ground_mode}")


class ElectrodeFEMSolver:
    """
    FEM solver with configurable electrode return current.

    Wraps FEMModel to provide:
    - Volumetric sink (default, existing behavior)
    - Localized ground electrode (bipolar configuration)

    Parameters
    ----------
    msh_file_path : str
        Path to Gmsh mesh file
    return_mode : str
        "volumetric" (uniform sink) or "localized" (ground electrode)
    ground_position : np.ndarray, optional
        Position of ground electrode (required for localized mode)
    ground_radius : float
        Radius for distributed ground electrode sampling
    ground_points : int
        Number of sample points for distributed ground electrode
    use_native_point_source : bool
        If True (default), use the native NativePointSource reimplementation
        (no external scifem dependency). Set False to use scifem.PointSource
        (scifem must be installed).
    **fem_options
        Additional options passed to FEMModel

    Attributes
    ----------
    A : PETSc.Mat or None
        Stiffness matrix from last solve (localized mode only)
    b_vec : PETSc.Vec or None
        RHS vector from last solve (localized mode only)
    ksp_iters : int
        Number of KSP iterations from last solve
    source_pts : np.ndarray or None
        Source electrode sample points from last solve
    ground_pts : np.ndarray or None
        Ground electrode sample points from last solve
    use_native_point_source : bool
        Whether to use native point source implementation
    """

    def __init__(
        self,
        msh_file_path: str,
        return_mode: str = "volumetric",
        ground_position: np.ndarray = None,
        ground_radius: float = 5.0,
        ground_points: int = 256,
        use_native_point_source: bool = True,
        source_mode: str = "point",
        source_sigma: float = 5.0,
        **fem_options,
    ):
        self.source_mode = source_mode
        self.source_sigma = source_sigma
        if source_mode == "point":
            fem_options["point_source"] = True
        else:
            fem_options["point_source"] = False
            fem_options["source_sigma"] = source_sigma
        self.model = FEMModel(msh_file_path, **fem_options)
        self.return_mode = return_mode
        self.ground_position = ground_position
        self.ground_radius = ground_radius
        self.ground_points = ground_points
        self.use_native_point_source = use_native_point_source

        # Storage for diagnostics (populated after solve)
        self.A = None
        self.b_vec = None
        self.ksp_iters = -1
        self.source_pts = None
        self.ground_pts = None

    def solve(self, source_point: np.ndarray, source_radius: float = 5.0, source_points: int = 256) -> Function:
        """
        Solve FEM with configured return current mode.

        Parameters
        ----------
        source_point : np.ndarray
            Position of current injection electrode
        source_radius : float
            Radius for distributed source electrode sampling
        source_points : int
            Number of sample points for source electrode

        Returns
        -------
        Function
            Solved potential field
        """
        source_point = np.asarray(source_point).reshape(3)

        if self.return_mode == "volumetric":
            # Use existing FEMModel behavior with point source + volumetric sink
            self.model.options["electrode_radius"] = source_radius
            self.model.options["sampled_electrode_points"] = source_points
            return self.model.solve_for_point(source_point)
        else:
            return self._solve_with_localized_ground(source_point, source_radius, source_points)

    def _sample_points_in_mesh(
        self,
        center: np.ndarray,
        radius: float,
        num_points: int,
        max_attempts: int = 10,
    ) -> np.ndarray:
        """
        Sample points in a sphere, keeping only those inside the mesh.

        Uses dolfinx geometry to verify points are within mesh cells.
        """
        from dolfinx import geometry

        center = center.reshape(3)
        valid_points = []
        attempts = 0

        while len(valid_points) < num_points and attempts < max_attempts:
            # Generate candidate points in sphere
            n_candidates = (num_points - len(valid_points)) * 3  # Oversample
            candidates = []
            while len(candidates) < n_candidates:
                rp = center + np.random.uniform(-radius, radius, size=(3,))
                if np.linalg.norm(rp - center) <= radius:
                    candidates.append(rp)
            candidates = np.array(candidates, dtype=np.float64)

            # Check which points are inside the mesh using collision detection
            tree = self.model.tree
            mesh = self.model.mesh

            # First pass: find candidate cells for each point
            cell_candidates = geometry.compute_collisions_points(tree, candidates)

            # Second pass: verify actual cell collision for all points at once
            colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, candidates)

            # Keep points that have at least one colliding cell
            for i in range(len(candidates)):
                if len(valid_points) >= num_points:
                    break
                if len(colliding_cells.links(i)) > 0:
                    valid_points.append(candidates[i])

            attempts += 1

        if len(valid_points) < num_points:
            print(f"Warning: Only found {len(valid_points)}/{num_points} points inside mesh after {max_attempts} attempts")

        return np.array(valid_points, dtype=np.float64)

    def _solve_with_localized_ground(
        self,
        source_point: np.ndarray,
        source_radius: float,
        source_points: int,
    ) -> Function:
        """
        Solve with bipolar point sources: injection (+I) and extraction (-I).

        Uses scifem.PointSource or NativePointSource for both electrodes with
        opposite magnitudes, maintaining pure Neumann formulation.

        Stores A, b_vec, source_pts, ground_pts, and ksp_iters for diagnostics.
        """
        if self.use_native_point_source:
            from .point_source import NativePointSource as PointSourceClass
        else:
            import scifem
            PointSourceClass = scifem.PointSource

        if self.ground_position is None:
            raise ValueError("ground_position required for localized return mode")

        ground_position = np.asarray(self.ground_position).reshape(3)

        # Store source point for reference
        self.model.point = source_point

        # Set up the weak form with zero volumetric source
        u = ufl.TrialFunction(self.model.V_scalar)
        v = ufl.TestFunction(self.model.V_scalar)
        g = fem.Constant(self.model.mesh, default_scalar_type(0.0))

        if self.model.options["build_conductivity_map"]:
            a = ufl.dot(ufl.dot(self.model.sigma_anisotropic, ufl.grad(u)), ufl.grad(v)) * dx
        else:
            a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

        # Zero volumetric source, only point sources
        L = g * v * ds

        # Assemble system
        A = assemble_matrix(form(a))
        A.assemble()

        b = create_vector(form(L))
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(b, form(L))

        b_fun = fem.Function(self.model.V_scalar)
        b_fun.vector.array = b.array

        # Sample electrode points, ensuring they're inside the mesh
        source_pts = self._sample_points_in_mesh(source_point, source_radius, source_points)
        ground_pts = self._sample_points_in_mesh(ground_position, self.ground_radius, self.ground_points)

        if len(source_pts) == 0:
            raise ValueError(f"No valid source points found inside mesh around {source_point}")
        if len(ground_pts) == 0:
            raise ValueError(f"No valid ground points found inside mesh around {ground_position}")

        # Store sample points for diagnostics
        self.source_pts = source_pts
        self.ground_pts = ground_pts

        # Add source electrode (+1 total current, distributed)
        gamma_source = 1.0 / source_pts.shape[0]
        source_ps = PointSourceClass(self.model.V_scalar, source_pts, magnitude=gamma_source)
        source_ps.apply_to_vector(b_fun)

        # Add ground electrode (-1 total current, distributed)
        gamma_ground = -1.0 / ground_pts.shape[0]
        ground_ps = PointSourceClass(self.model.V_scalar, ground_pts, magnitude=gamma_ground)
        ground_ps.apply_to_vector(b_fun)

        # Store A and b for residual diagnostics (before nullspace removal from b)
        self.A = A
        self.b_vec = b_fun.vector.copy()

        # Solve with nullspace handling
        uh = Function(self.model.V_scalar)
        solver = PETSc.KSP().create(A.getComm())
        solver.setOperators(A)

        # Configure KSP for tight convergence
        solver.setType("gmres")
        solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=1000)
        solver.getPC().setType("gamg")

        nullspace = PETSc.NullSpace().create(constant=True, comm=MPI.COMM_WORLD)
        A.setNullSpace(nullspace)
        nullspace.remove(b_fun.vector)

        solver.solve(b_fun.vector, uh.vector)

        # Store iteration count
        self.ksp_iters = solver.getIterationNumber()

        self.model.uh = uh
        return uh

    def compute_residual(self, uh: Function = None) -> dict:
        """
        Compute the true residual using stored A and b from the last solve.

        Parameters
        ----------
        uh : Function, optional
            Solution to check. If None, uses self.model.uh from last solve.

        Returns
        -------
        dict
            Residual diagnostics:
            - b_norm: ||b||
            - u_norm: ||u||
            - r_norm: ||Au - b||
            - rel_residual: ||Au - b|| / ||b||
            - ksp_iters: Number of KSP iterations
        """
        if self.A is None or self.b_vec is None:
            raise RuntimeError("No stored A/b. Run solve() first with localized mode.")

        if uh is None:
            uh = self.model.uh

        # Compute r = Au - b
        r = self.A.createVecRight()
        self.A.mult(uh.vector, r)
        r.axpy(-1.0, self.b_vec)

        r_norm = r.norm(PETSc.NormType.NORM_2)
        b_norm = self.b_vec.norm(PETSc.NormType.NORM_2)
        u_norm = uh.vector.norm(PETSc.NormType.NORM_2)

        r.destroy()

        rel_residual = r_norm / b_norm if b_norm > 1e-15 else float("nan")

        return {
            "b_norm": b_norm,
            "u_norm": u_norm,
            "r_norm": r_norm,
            "rel_residual": rel_residual,
            "ksp_iters": self.ksp_iters,
        }


__all__ = [
    "ElectrodeType",
    "DetectionMode",
    "ElectrodeConfig",
    "sample_electrode_area",
    "compute_electrode_centers",
    "compute_ground_position",
    "evaluate_detection",
    "ElectrodeFEMSolver",
]
