#!/usr/bin/env python3
"""
FEM Residual Troubleshooting Script

Tests multiple FEM formulations to diagnose why the relative residual
is ~1e-1 instead of the expected ~1e-7.

Key experiments:
1. Test A: Volumetric mode (FEMModel.solve_for_point with Gaussian source)
2. Test B: Volumetric mode with point source + volumetric sink
3. Test C: Localized mode (ElectrodeFEMSolver) with stored b
4. Test D: Localized mode with KSP configuration tuning
5. Test E: Direct ConstrainedLinearProblem reference

Usage:
    cd /path/to/emgforge
    export PYTHONPATH=src
    python tests/fem_residual_debug/test_residual_formulations.py [--mesh PATH] [--meta PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem, geometry
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType as default_scalar_type
from ufl import ds, dx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from emgop.fem import FEMModel
from emgop.fem.electrode_configs import ElectrodeFEMSolver, compute_ground_position
from emgop.fem.solver import ConstrainedLinearProblem


PASS_THRESHOLD = 1e-6
WARN_THRESHOLD = 1e-3


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_residual_result(
    name: str,
    b_norm: float,
    u_norm: float,
    r_norm: float,
    ksp_iters: int = -1,
    extra: str = "",
):
    """Print residual diagnostic in standard format."""
    rel_res = r_norm / b_norm if b_norm > 1e-15 else float("nan")

    status = "PASS" if rel_res < PASS_THRESHOLD else ("WARN" if rel_res < WARN_THRESHOLD else "FAIL")
    status_marker = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")

    print(f"\n--- {name} ---")
    print(f"||b||:      {b_norm:.6e}")
    print(f"||u||:      {u_norm:.6e}")
    print(f"||Au-b||:   {r_norm:.6e}")
    print(f"rel_res:    {rel_res:.6e}  [{status}] {status_marker}")
    if ksp_iters >= 0:
        print(f"KSP iters:  {ksp_iters}")
    if extra:
        print(f"Note: {extra}")

    return rel_res, status


def compute_residual(A: PETSc.Mat, u: PETSc.Vec, b: PETSc.Vec) -> tuple[float, float, float]:
    """Compute ||Au - b||, ||b||, ||u||."""
    r = A.createVecRight()
    A.mult(u, r)
    r.axpy(-1.0, b)

    r_norm = r.norm(PETSc.NormType.NORM_2)
    b_norm = b.norm(PETSc.NormType.NORM_2)
    u_norm = u.norm(PETSc.NormType.NORM_2)

    r.destroy()
    return b_norm, u_norm, r_norm


class ResidualTrackingSolver:
    """
    FEM solver that stores A and b for residual analysis.

    This is a modified version of ElectrodeFEMSolver that keeps
    the actual A and b used during the solve.
    """

    def __init__(
        self,
        model: FEMModel,
        ground_position: np.ndarray,
        ground_radius: float = 5.0,
        ground_points: int = 256,
    ):
        self.model = model
        self.ground_position = ground_position
        self.ground_radius = ground_radius
        self.ground_points = ground_points

        # Will be set during solve
        self.A = None
        self.b_vec = None
        self.ksp_iters = -1

    def _sample_points_in_sphere(self, center: np.ndarray, radius: float, num_points: int) -> np.ndarray:
        """Sample points in a sphere (simple rejection sampling)."""
        center = center.reshape(3)
        pts = []
        while len(pts) < num_points:
            rp = center + np.random.uniform(-radius, radius, size=(3,))
            if np.linalg.norm(rp - center) <= radius:
                pts.append(rp)
        return np.array(pts, dtype=np.float64)

    def solve_localized(
        self,
        source_point: np.ndarray,
        source_radius: float,
        source_points: int,
        ksp_type: str = "gmres",
        pc_type: str = "gamg",
        rtol: float = 1e-10,
        atol: float = 1e-14,
        max_it: int = 1000,
    ) -> Function:
        """
        Solve with localized ground, storing A and b.
        """
        import scifem

        source_point = np.asarray(source_point).reshape(3)
        ground_position = np.asarray(self.ground_position).reshape(3)

        # Build weak form
        u = ufl.TrialFunction(self.model.V_scalar)
        v = ufl.TestFunction(self.model.V_scalar)
        g = fem.Constant(self.model.mesh, default_scalar_type(0.0))

        if self.model.options.get("build_conductivity_map", False):
            a = ufl.dot(ufl.dot(self.model.sigma_anisotropic, ufl.grad(u)), ufl.grad(v)) * dx
        else:
            a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

        # Zero volumetric source
        L = g * v * ds

        # Assemble system
        self.A = assemble_matrix(form(a))
        self.A.assemble()

        b = create_vector(form(L))
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(b, form(L))

        b_fun = fem.Function(self.model.V_scalar)
        b_fun.vector.array[:] = b.array[:]

        # Sample electrode points
        source_pts = self._sample_points_in_sphere(source_point, source_radius, source_points)
        ground_pts = self._sample_points_in_sphere(ground_position, self.ground_radius, self.ground_points)

        # Apply source (+1 distributed)
        gamma_source = 1.0 / source_pts.shape[0]
        source_ps = scifem.PointSource(self.model.V_scalar, source_pts, magnitude=gamma_source)
        source_ps.apply_to_vector(b_fun)

        # Apply ground (-1 distributed)
        gamma_ground = -1.0 / ground_pts.shape[0]
        ground_ps = scifem.PointSource(self.model.V_scalar, ground_pts, magnitude=gamma_ground)
        ground_ps.apply_to_vector(b_fun)

        # Store b for residual analysis
        self.b_vec = b_fun.vector.copy()

        # Create solver with configurable options
        uh = Function(self.model.V_scalar)
        solver = PETSc.KSP().create(self.A.getComm())
        solver.setOperators(self.A)

        # Configure KSP
        solver.setType(ksp_type)
        solver.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
        solver.getPC().setType(pc_type)

        # Set nullspace
        nullspace = PETSc.NullSpace().create(constant=True, comm=MPI.COMM_WORLD)
        self.A.setNullSpace(nullspace)
        nullspace.remove(b_fun.vector)

        # Solve
        solver.solve(b_fun.vector, uh.vector)
        self.ksp_iters = solver.getIterationNumber()

        self.model.uh = uh
        return uh


def test_gaussian_volumetric(model: FEMModel, source_point: np.ndarray):
    """
    Test A: Gaussian source with no point source (pure volumetric).

    This should have low residual since source is balanced analytically.
    """
    print_header("Test A: Gaussian Volumetric Source")

    # Temporarily configure for Gaussian source
    old_opts = model.options.copy()
    model.options["point_source"] = False
    model.options["source_sigma"] = 1.0  # Wide Gaussian

    # Solve
    uh = model.solve_for_point(source_point)

    # Reconstruct A and b
    u = ufl.TrialFunction(model.V_scalar)
    v = ufl.TestFunction(model.V_scalar)
    g = fem.Constant(model.mesh, default_scalar_type(model.options["boundary_value"]))

    if model.options.get("build_conductivity_map", False):
        a = ufl.dot(ufl.dot(model.sigma_anisotropic, ufl.grad(u)), ufl.grad(v)) * dx
    else:
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

    L = model.source_function * v * dx + g * v * ds

    A = assemble_matrix(form(a))
    A.assemble()

    b = create_vector(form(L))
    with b.localForm() as b_loc:
        b_loc.set(0)
    assemble_vector(b, form(L))

    # Compute residual
    b_norm, u_norm, r_norm = compute_residual(A, uh.vector, b)
    rel_res, status = print_residual_result("Gaussian volumetric", b_norm, u_norm, r_norm)

    # Restore options
    model.options = old_opts

    return rel_res, status


def test_point_source_volumetric_sink(model: FEMModel, source_point: np.ndarray):
    """
    Test B: Point source + volumetric sink (ConstrainedLinearProblem).

    This is the existing FEMModel.solve_for_point with point_source=True.
    """
    print_header("Test B: Point Source + Volumetric Sink")

    # Configure for point source with volumetric compensation
    old_opts = model.options.copy()
    model.options["point_source"] = True
    model.options["point_electrode"] = True  # Single point

    # Solve
    uh = model.solve_for_point(source_point)

    # For point source mode, the source_function is the volumetric sink (-1/vol)
    # and point source is added via scifem.PointSource

    # Reconstruct the system
    u = ufl.TrialFunction(model.V_scalar)
    v = ufl.TestFunction(model.V_scalar)
    g = fem.Constant(model.mesh, default_scalar_type(model.options["boundary_value"]))

    if model.options.get("build_conductivity_map", False):
        a = ufl.dot(ufl.dot(model.sigma_anisotropic, ufl.grad(u)), ufl.grad(v)) * dx
    else:
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

    L = model.source_function * v * dx + g * v * ds

    A = assemble_matrix(form(a))
    A.assemble()

    b = create_vector(form(L))
    with b.localForm() as b_loc:
        b_loc.set(0)
    assemble_vector(b, form(L))

    # Add point source contribution (same as in ConstrainedLinearProblem)
    import scifem
    b_fun = fem.Function(model.V_scalar)
    b_fun.vector.array[:] = b.array[:]

    point_source = scifem.PointSource(model.V_scalar, source_point.reshape(1, 3), magnitude=1.0)
    point_source.apply_to_vector(b_fun)

    # Compute residual
    b_norm, u_norm, r_norm = compute_residual(A, uh.vector, b_fun.vector)
    rel_res, status = print_residual_result("Point source + volumetric sink", b_norm, u_norm, r_norm)

    model.options = old_opts

    return rel_res, status


def test_localized_ground_stored_b(
    model: FEMModel,
    source_point: np.ndarray,
    ground_position: np.ndarray,
    electrode_radius: float = 5.0,
    n_points: int = 256,
):
    """
    Test C: Localized ground with proper b tracking.

    Uses ResidualTrackingSolver to store actual A and b.
    """
    print_header("Test C: Localized Ground (Stored b)")

    solver = ResidualTrackingSolver(
        model,
        ground_position=ground_position,
        ground_radius=electrode_radius,
        ground_points=n_points,
    )

    uh = solver.solve_localized(
        source_point,
        source_radius=electrode_radius,
        source_points=n_points,
    )

    # Compute residual using stored A and b
    b_norm, u_norm, r_norm = compute_residual(solver.A, uh.vector, solver.b_vec)
    rel_res, status = print_residual_result(
        "Localized ground (stored b)",
        b_norm, u_norm, r_norm,
        ksp_iters=solver.ksp_iters,
    )

    return rel_res, status


def test_localized_ground_ksp_tuned(
    model: FEMModel,
    source_point: np.ndarray,
    ground_position: np.ndarray,
    electrode_radius: float = 5.0,
    n_points: int = 256,
):
    """
    Test D: Localized ground with tuned KSP settings.

    Tests if better solver configuration improves convergence.
    """
    print_header("Test D: Localized Ground (Tuned KSP)")

    solver = ResidualTrackingSolver(
        model,
        ground_position=ground_position,
        ground_radius=electrode_radius,
        ground_points=n_points,
    )

    # Try tighter tolerances and more iterations
    uh = solver.solve_localized(
        source_point,
        source_radius=electrode_radius,
        source_points=n_points,
        ksp_type="cg",  # CG is better for SPD systems with nullspace
        pc_type="gamg",
        rtol=1e-12,
        atol=1e-14,
        max_it=5000,
    )

    # Compute residual
    b_norm, u_norm, r_norm = compute_residual(solver.A, uh.vector, solver.b_vec)
    rel_res, status = print_residual_result(
        "Localized ground (tuned KSP: CG, tighter tol)",
        b_norm, u_norm, r_norm,
        ksp_iters=solver.ksp_iters,
    )

    return rel_res, status


def test_electrode_fem_solver_reconstructed_b(
    mesh_path: str,
    source_point: np.ndarray,
    ground_position: np.ndarray,
    electrode_radius: float = 5.0,
    n_points: int = 256,
):
    """
    Test E: Original ElectrodeFEMSolver with reconstructed b.

    This replicates the notebook's approach (which shows high residual).
    Should demonstrate the mismatch between solve-time and post-hoc b.
    """
    print_header("Test E: ElectrodeFEMSolver (Reconstructed b - Expected High Residual)")

    solver = ElectrodeFEMSolver(
        mesh_path,
        return_mode="localized",
        ground_position=ground_position,
        ground_radius=electrode_radius,
        ground_points=n_points,
    )

    uh = solver.solve(source_point, electrode_radius, n_points)
    model = solver.model

    # Reconstruct A
    u = ufl.TrialFunction(model.V_scalar)
    v = ufl.TestFunction(model.V_scalar)

    if model.options.get("build_conductivity_map", False):
        a = ufl.dot(ufl.dot(model.sigma_anisotropic, ufl.grad(u)), ufl.grad(v)) * dx
    else:
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

    A = assemble_matrix(form(a))
    A.assemble()

    # Reconstruct b with NEW point source samples (like notebook does)
    import scifem

    g = fem.Constant(model.mesh, default_scalar_type(0.0))
    L = g * v * ds

    b = create_vector(form(L))
    with b.localForm() as b_loc:
        b_loc.set(0)
    assemble_vector(b, form(L))

    b_fun = fem.Function(model.V_scalar)
    b_fun.vector.array[:] = b.array[:]

    # NEW samples (different from solve time!)
    source_pts = solver._sample_points_in_mesh(source_point.reshape(3), electrode_radius, n_points)
    ground_pts = solver._sample_points_in_mesh(ground_position.reshape(3), electrode_radius, n_points)

    gamma_source = 1.0 / len(source_pts)
    source_ps = scifem.PointSource(model.V_scalar, source_pts, magnitude=gamma_source)
    source_ps.apply_to_vector(b_fun)

    gamma_ground = -1.0 / len(ground_pts)
    ground_ps = scifem.PointSource(model.V_scalar, ground_pts, magnitude=gamma_ground)
    ground_ps.apply_to_vector(b_fun)

    # Compute residual (expected to be high due to b mismatch)
    b_norm, u_norm, r_norm = compute_residual(A, uh.vector, b_fun.vector)
    rel_res, status = print_residual_result(
        "ElectrodeFEMSolver (reconstructed b)",
        b_norm, u_norm, r_norm,
        extra="High residual expected: b differs from solve-time b!",
    )

    return rel_res, status


def main():
    parser = argparse.ArgumentParser(description="FEM Residual Troubleshooting")
    parser.add_argument(
        "--mesh",
        type=str,
        default="./data/generated_meshes/meshes/sample_000001.msh",
        help="Path to mesh file",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="./data/generated_meshes/metadata/sample_000001.json",
        help="Path to metadata file",
    )
    parser.add_argument(
        "--electrode-radius",
        type=float,
        default=3.0,
        help="Electrode sampling radius",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=256,
        help="Number of electrode sample points",
    )
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    meta_path = Path(args.meta)

    if not mesh_path.exists():
        print(f"ERROR: Mesh not found: {mesh_path}")
        sys.exit(1)

    # Load metadata
    if meta_path.exists():
        with open(meta_path) as f:
            meta_raw = json.load(f)
        meta = meta_raw.get("geometry_params", meta_raw)
    else:
        print(f"WARNING: Metadata not found, using defaults")
        meta = {"radius_skin": 40.0, "length": 240.0}

    print_header("FEM Residual Troubleshooting")
    print(f"Mesh: {mesh_path}")
    print(f"Meta: {meta_path}")
    print(f"r_skin: {meta.get('radius_skin', 40):.2f} mm")
    print(f"length: {meta.get('length', 240):.2f} mm")

    # Load model
    print("\nLoading FEMModel...")
    model = FEMModel(str(mesh_path), build_conductivity_map=True)
    print("Done.")

    # Define test points
    INSET = args.electrode_radius + 1.0
    r_electrode = meta.get("radius_skin", 40.0) - INSET
    z_electrode = meta.get("length", 240.0) / 2

    source_point = np.array([r_electrode, 0.0, z_electrode])
    ground_position = compute_ground_position(
        meta_raw if meta_path.exists() else meta,
        source_point,
        ground_mode="opposite",
        inset=INSET,
    )

    print(f"\nSource: {source_point}")
    print(f"Ground: {ground_position}")

    # Run tests
    results = {}

    # Test A: Gaussian volumetric
    try:
        rel_res, status = test_gaussian_volumetric(model, source_point)
        results["A_gaussian"] = (rel_res, status)
    except Exception as e:
        print(f"Test A failed: {e}")
        results["A_gaussian"] = (float("nan"), "ERROR")

    # Reload model for fresh state
    model = FEMModel(str(mesh_path), build_conductivity_map=True)

    # Test B: Point source + volumetric sink
    try:
        rel_res, status = test_point_source_volumetric_sink(model, source_point)
        results["B_point_vol"] = (rel_res, status)
    except Exception as e:
        print(f"Test B failed: {e}")
        results["B_point_vol"] = (float("nan"), "ERROR")

    # Reload model
    model = FEMModel(str(mesh_path), build_conductivity_map=True)

    # Test C: Localized with stored b
    try:
        rel_res, status = test_localized_ground_stored_b(
            model, source_point, ground_position,
            electrode_radius=args.electrode_radius,
            n_points=args.n_points,
        )
        results["C_localized_stored"] = (rel_res, status)
    except Exception as e:
        print(f"Test C failed: {e}")
        results["C_localized_stored"] = (float("nan"), "ERROR")

    # Reload model
    model = FEMModel(str(mesh_path), build_conductivity_map=True)

    # Test D: Localized with tuned KSP
    try:
        rel_res, status = test_localized_ground_ksp_tuned(
            model, source_point, ground_position,
            electrode_radius=args.electrode_radius,
            n_points=args.n_points,
        )
        results["D_localized_tuned"] = (rel_res, status)
    except Exception as e:
        print(f"Test D failed: {e}")
        results["D_localized_tuned"] = (float("nan"), "ERROR")

    # Test E: Original solver with reconstructed b
    try:
        rel_res, status = test_electrode_fem_solver_reconstructed_b(
            str(mesh_path), source_point, ground_position,
            electrode_radius=args.electrode_radius,
            n_points=args.n_points,
        )
        results["E_reconstructed"] = (rel_res, status)
    except Exception as e:
        print(f"Test E failed: {e}")
        results["E_reconstructed"] = (float("nan"), "ERROR")

    # Summary
    print_header("SUMMARY")
    print(f"{'Test':<30} {'Rel. Residual':<15} {'Status':<10}")
    print("-" * 55)
    for name, (rel_res, status) in results.items():
        status_mark = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        print(f"{name:<30} {rel_res:<15.2e} {status:<6} {status_mark}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    print("""
- If Test C passes but Test E fails:
  → The problem is reconstructed b mismatch (different random samples)
  → FIX: Store actual b during solve, not reconstruct

- If Test C also fails:
  → The KSP solver isn't converging properly
  → FIX: Tune KSP tolerances, check nullspace handling

- If Test A/B pass but C/D fail:
  → Point source formulation is numerically harder than volumetric
  → FIX: Consider adding small volumetric regularization

- If all tests fail:
  → Fundamental issue with matrix assembly or nullspace
  → Check conductivity map, mesh quality
""")


if __name__ == "__main__":
    main()
