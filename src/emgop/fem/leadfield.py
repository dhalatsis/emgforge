"""Reusable pieces of the reciprocity solve: source RHS + KSP configuration.

These are the composable value objects the future ``LeadField`` engine will use.
Introduced first (L1) so they can be extracted from ``FEMModel`` with no behaviour
change and verified in isolation, before the engine itself is built.

`LeadField` and the σ-builders land in later steps (see `_refactor/10_leadfield.md`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from dolfinx import fem
from petsc4py.PETSc import ScalarType as default_scalar_type
from ufl import dx


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
    the RHS integrates to zero (required for the pure-Neumann problem). The math is
    lifted verbatim from ``FEMModel.assign_source_to_point`` — byte-identical.
    """
    sigma: float

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
        integral = fem.assemble_scalar(fem.form(source_function / volume * dx))
        source_function.interpolate(lambda x: self._gaussian_nd(x, point, self.sigma) - integral)
        return source_function


@dataclass
class UniformSink:
    """A constant ``-1/volume`` sink — the volumetric term used with a point/sampled
    electrode source (``FEMModel`` ``point_source=True`` path)."""

    def as_constant(self, mesh, volume: float):
        return fem.Constant(mesh, default_scalar_type(-1.0 / volume))
