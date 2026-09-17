"""Reusable pieces of the reciprocity solve: source RHS + KSP configuration.

These are the composable value objects the future ``LeadField`` engine will use.
Introduced first (L1) so they can be extracted from ``FEMModel`` with no behaviour
change and verified in isolation, before the engine itself is built.

`LeadField` and the σ-builders land in later steps (see `_refactor/10_leadfield.md`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem, geometry
from dolfinx.fem import Function
from petsc4py.PETSc import ScalarType as default_scalar_type
from ufl import ds, dx


@dataclass
class KSPConfig:
    """PETSc KSP settings for the constrained Neumann solve.

    Defaults are the values established in P3 (previously hardcoded in
    ``ConstrainedLinearProblem.solve``); they also match ``MRIFEMModel``.
    """
    solver_type: str = "gmres"
    pc_type: str = "ilu"
    rtol: float = 1e-8
    atol: float = 1e-10
    max_it: int = 5000

    def apply_to(self, ksp) -> None:
        ksp.setType(self.solver_type)
        ksp.getPC().setType(self.pc_type)
        ksp.setTolerances(rtol=self.rtol, atol=self.atol, max_it=self.max_it)


@dataclass
class GaussianSource:
    """A zero-mean Gaussian current source of width ``sigma`` (mesh units).

    ``assign`` fills a source function with the Gaussian, then subtracts its mean so
    the RHS integrates to zero (required for the pure-Neumann problem).

    ``mean_after_assemble`` selects the exact zero-mean arithmetic: False (default)
    is ``FEMModel``'s form ``assemble(f/vol)``; True is ``MRIFEMModel``'s
    ``assemble(f)/vol``. They are mathematically identical but differ by a rounding
    ULP, so each backend keeps its own to stay byte-identical.
    """
    sigma: float
    mean_after_assemble: bool = False

    @staticmethod
    def _gaussian_nd(x, center, sigma_val):
        x = x.T
        squared_dist = np.sum((x - center) ** 2, axis=-1)
        return 1 / ((2 * np.pi * sigma_val**2) ** (x.shape[-1] / 2)) * np.exp(
            -squared_dist / (2 * sigma_val**2)
        )

    def assign(self, source_function, point: np.ndarray, volume: float):
        """Fill ``source_function`` (a CG Function) with the zero-mean Gaussian at ``point``."""
        source_function.interpolate(lambda x: self._gaussian_nd(x, point, self.sigma))
        if self.mean_after_assemble:
            mean = fem.assemble_scalar(fem.form(source_function * dx)) / volume
        else:
            mean = fem.assemble_scalar(fem.form(source_function / volume * dx))
        source_function.interpolate(lambda x: self._gaussian_nd(x, point, self.sigma) - mean)
        return source_function


class LeadField:
    """The shared reciprocity solve: Gaussian source at an electrode → φ along a fibre.

    Owns the machinery that ``FEMModel`` and ``MRIFEMModel`` duplicate — function
    spaces, source RHS, weak-form assembly, the constant-nullspace KSP solve, and
    point evaluation. The volume conductor enters ONLY through the ``sigma`` field
    (a DG0 tensor ``Function``), which each backend builds its own way
    (``LayeredSigma`` for the cylinder/ellipse tiers; the MRI v1/v2 + skin-shell
    methods for the forearm).
    """

    def __init__(self, mesh, sigma: "Function", *, source_degree: int = 1,
                 boundary_value: float = 0.0, ksp: "KSPConfig | None" = None):
        self.mesh = mesh
        self.sigma = sigma
        self.boundary_value = float(boundary_value)
        self.ksp = ksp or KSPConfig()

        dim = mesh.topology.dim
        mesh.topology.create_connectivity(dim, 0)
        self.tree = geometry.bb_tree(mesh, dim)
        n_cells = mesh.topology.index_map(dim).size_local
        self.midpoints = geometry.create_midpoint_tree(
            mesh, dim, np.arange(n_cells, dtype=np.int32))

        self.V_scalar = fem.functionspace(mesh, ("CG", 1))
        self.V_source = fem.functionspace(mesh, ("CG", source_degree))
        self.uh = Function(self.V_scalar)

    def solve(self, point: np.ndarray, source: GaussianSource) -> "Function":
        """Reciprocity solve for a source at ``point``. Returns the potential φ (CG1)."""
        from .solver import ConstrainedLinearProblem  # avoid an import cycle at module load

        sf = Function(self.V_source)
        u_one = fem.Constant(self.mesh, default_scalar_type(1.0))
        volume = fem.assemble_scalar(fem.form(u_one * dx))
        source.assign(sf, np.asarray(point, dtype=np.float64).reshape(3), volume)

        u = ufl.TrialFunction(self.V_scalar)
        v = ufl.TestFunction(self.V_scalar)
        g = fem.Constant(self.mesh, default_scalar_type(self.boundary_value))
        a = ufl.dot(ufl.dot(self.sigma, ufl.grad(u)), ufl.grad(v)) * dx
        L = sf * v * dx + g * v * ds

        self.uh = ConstrainedLinearProblem(a, L, self.V_scalar).solve(self.ksp)
        return self.uh

    def locate(self, points: np.ndarray) -> np.ndarray:
        """The cell containing (or nearest to) each point.

        Depends on the mesh only, not on the solve — so for a fixed set of sample points
        (a fibre bed) it can be computed once and handed to :meth:`phi` for every
        electrode. On the WR forearm bed (127 k points) the lookup is ~100× the cost of
        the evaluation itself.
        """
        return geometry.compute_closest_entity(
            self.tree, self.midpoints, self.mesh, points).squeeze()

    def phi(self, points: np.ndarray, uh: "Function | None" = None,
            cells: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the potential at arbitrary points (``cells``: a :meth:`locate` result
        for the same points, to skip the lookup)."""
        uh = self.uh if uh is None else uh
        cell_ids = self.locate(points) if cells is None else cells
        return np.asarray(uh.eval(points, cell_ids)).reshape(-1)


@dataclass
class UniformSink:
    """A constant ``-1/volume`` sink — the volumetric term used with a point/sampled
    electrode source (``FEMModel`` ``point_source=True`` path)."""

    def as_constant(self, mesh, volume: float):
        return fem.Constant(mesh, default_scalar_type(-1.0 / volume))
