from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem, geometry, io
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType as default_scalar_type
from ufl import dx, ds

from .constants import CONDUCTIVITY, GROUP_NAMES
from .leadfield import GaussianSource, KSPConfig, UniformSink
from .rotation import rotate_point_in_cylinder
from .sigma import LayeredSigma


class ConstrainedLinearProblem:
    """
    Solve Neumann PDE with constant nullspace:
    - set PETSc NullSpace(constant=True)
    - remove nullspace component from RHS
    """

    def __init__(self, a, L, V):
        self.a = a
        self.L = L
        self.V = V

        self.A = assemble_matrix(form(self.a))
        self.A.assemble()

        b = create_vector(form(self.L))
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(b, form(self.L))

        self.b_fun = fem.Function(self.V)
        self.b_fun.vector.array = b.array

    def add_point_source(self, points: np.ndarray):
        # Native (scifem-free) reimplementation — see point_source.NativePointSource.
        from .point_source import NativePointSource

        gamma = 1.0 / points.shape[0]
        point_source = NativePointSource(self.V, points, magnitude=gamma)
        point_source.apply_to_vector(self.b_fun)

    def solve(self, ksp: "KSPConfig | None" = None) -> Function:
        uh = Function(self.V)

        solver = PETSc.KSP().create(self.A.getComm())
        solver.setOperators(self.A)
        # Explicit KSP config (previously relied on PETSc defaults — rtol ~1e-5,
        # and no convergence check). Defaults mirror MRIFEMModel's settings.
        (ksp or KSPConfig()).apply_to(solver)

        nullspace = PETSc.NullSpace().create(constant=True, comm=MPI.COMM_WORLD)
        self.A.setNullSpace(nullspace)

        nullspace.remove(self.b_fun.vector)
        solver.solve(self.b_fun.vector, uh.vector)

        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(
                f"KSP failed to converge: reason={reason}, "
                f"iterations={solver.getIterationNumber()}"
            )
        return uh


class FEMModel:
    default_options = {
        "build_conductivity_map": True,
        "source_sigma": 1,  # std dev in mesh units for Gaussian source
        "source_degree": 1,
        "boundary_value": 0,
        "point_source": False,  # default to Gaussian volumetric source
        "point_electrode": False,
        "electrode_radius": 10,
        "sampled_electrode_points": 1024,
    }

    def __init__(self, msh_file_path: str, gdim=3, conductivity: dict | None = None, **options):
        self.mesh, self.cell_markers, self.facet_markers = io.gmshio.read_from_msh(
            msh_file_path, MPI.COMM_WORLD, gdim=gdim
        )
        self.mesh_geometry = self.mesh.geometry
        self.mesh_topology = self.mesh.topology

        self.height = float(self.mesh_geometry.x.max(axis=0)[2])

        self.options = self.default_options.copy()
        self.options.update(options)

        # Per-tissue conductivity table. `conductivity` overrides the module
        # defaults key-by-key (matching the old monkeypatch semantics) without
        # mutating shared global state.
        self.conductivity = {**CONDUCTIVITY, **(conductivity or {})}

        self.build_model()

    @staticmethod
    def _sample_points_in_sphere(center: np.ndarray, radius: float, num_points: int) -> np.ndarray:
        pts = []
        center = center.reshape(3,)
        while len(pts) < num_points:
            rp = center + np.random.uniform(-radius, radius, size=(3,))
            if np.linalg.norm(rp - center) <= radius:
                pts.append(rp)
        return np.array(pts, dtype=np.float64)

    def build_model(self):
        self.tree = geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        self.midpoints = geometry.create_midpoint_tree(
            self.mesh, self.mesh.topology.dim, np.arange(self.cell_markers.values.size, dtype=np.int32)
        )

        self.mesh_topology.create_connectivity(self.mesh.topology.dim, 0)
        self.cell_to_vertex = self.mesh_topology.connectivity(self.mesh.topology.dim, 0)

        self.V_scalar = fem.functionspace(self.mesh, ("CG", 1))
        self.V_pol = fem.functionspace(self.mesh, ("CG", self.options["source_degree"]))

        self.uh = fem.Function(self.V_scalar)

        if self.options["build_conductivity_map"]:
            self.build_conductivity_map()

    def build_conductivity_map(self):
        self.sigma_anisotropic = LayeredSigma(self.conductivity)(self.mesh, self.cell_markers)

    def apply_pinnation(self, theta: float):
        """
        Apply pennation angle rotation to muscle conductivity tensors.

        Parameters
        ----------
        theta : float
            Pennation angle in degrees. Positive angles tilt muscle fibers
            away from the longitudinal (Z) axis.
        """
        material_map = {v: k for k, v in GROUP_NAMES.items()}
        with self.sigma_anisotropic.vector.localForm() as loc_aniso:
            for cell_index, marker in enumerate(self.cell_markers.values):
                material = material_map[int(marker)]
                if material == "Muscle":
                    conductivity_tensor = self.conductivity[material]
                    vertices = self.mesh_topology.connectivity(3, 0).links(cell_index)
                    coordinates = self.mesh_geometry.x[vertices]
                    centroid = np.mean(coordinates, axis=0)
                    rotated = rotate_point_in_cylinder(centroid, conductivity_tensor, theta)
                    loc_aniso.setValuesBlocked([cell_index], rotated.flatten())

    def evaluate_solution_at_points(self, points: np.ndarray, uh: Function | None = None) -> np.ndarray:
        uh = self.uh if uh is None else uh
        cell_ids = geometry.compute_closest_entity(self.tree, self.midpoints, self.mesh, points).squeeze()
        vals = uh.eval(points, cell_ids)
        return np.asarray(vals).reshape(-1)

    def assign_source_to_point(self, point: np.ndarray, source_value: float = 1.0):
        self.point = point

        self.source_function = fem.Function(self.V_pol)
        with self.source_function.vector.localForm() as local:
            local.set(0.0)

        u_constant = fem.Constant(self.mesh, default_scalar_type(1.0))
        self.volume = fem.assemble_scalar(fem.form(u_constant * dx))

        if self.options["point_source"]:
            self.source_function = UniformSink().as_constant(self.mesh, self.volume)
        else:
            sigma = float(self.options.get("source_sigma", self.options.get("variance", 0.1)))
            GaussianSource(sigma).assign(self.source_function, point, self.volume)

    def solve_for_point(self, point: np.ndarray, source_value: float = 1.0) -> Function:
        self.assign_source_to_point(point, source_value)

        u = ufl.TrialFunction(self.V_scalar)
        v = ufl.TestFunction(self.V_scalar)
        g = fem.Constant(self.mesh, default_scalar_type(self.options["boundary_value"]))

        if self.options["build_conductivity_map"]:
            a = ufl.dot(ufl.dot(self.sigma_anisotropic, ufl.grad(u)), ufl.grad(v)) * dx
        else:
            a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

        L = self.source_function * v * dx + g * v * ds

        problem = ConstrainedLinearProblem(a, L, self.V_scalar)

        if self.options["point_source"]:
            if self.options["point_electrode"]:
                problem.add_point_source(self.point)
            else:
                points = self._sample_points_in_sphere(
                    self.point,
                    self.options["electrode_radius"],
                    self.options["sampled_electrode_points"],
                )
                problem.add_point_source(points)

        self.uh = problem.solve()
        return self.uh


__all__ = ["ConstrainedLinearProblem", "FEMModel"]

