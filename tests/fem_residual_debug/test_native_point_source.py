#!/usr/bin/env python3
"""
Test native point source implementation against scifem.

Verifies that NativePointSource produces identical results to scifem.PointSource.

Usage:
    cd /path/to/emgforge
    export PYTHONPATH=src
    python tests/fem_residual_debug/test_native_point_source.py [--mesh PATH] [--meta PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_native_vs_scifem_basic(V, points: np.ndarray, magnitude: float = 1.0):
    """
    Compare native implementation against scifem for a single call.

    Parameters
    ----------
    V : FunctionSpace
        FEniCSx function space
    points : np.ndarray
        Point coordinates, shape (n, 3)
    magnitude : float
        Point source magnitude

    Returns
    -------
    dict
        Comparison metrics
    """
    from dolfinx import fem
    import scifem
    from emgop.fem.point_source import NativePointSource

    # Create test vectors
    b_native = fem.Function(V)
    b_scifem = fem.Function(V)

    # Apply native
    native_ps = NativePointSource(V, points, magnitude=magnitude)
    native_ps.apply_to_vector(b_native)

    # Apply scifem
    scifem_ps = scifem.PointSource(V, points, magnitude=magnitude)
    scifem_ps.apply_to_vector(b_scifem)

    # Compare
    native_arr = b_native.x.array
    scifem_arr = b_scifem.x.array

    diff = np.abs(native_arr - scifem_arr)
    max_abs_diff = diff.max()
    scifem_norm = np.linalg.norm(scifem_arr)
    rel_diff = np.linalg.norm(diff) / scifem_norm if scifem_norm > 1e-15 else 0.0

    # Count non-zero entries
    native_nnz = np.sum(np.abs(native_arr) > 1e-15)
    scifem_nnz = np.sum(np.abs(scifem_arr) > 1e-15)

    return {
        "max_abs_diff": max_abs_diff,
        "rel_diff": rel_diff,
        "native_norm": np.linalg.norm(native_arr),
        "scifem_norm": scifem_norm,
        "native_nnz": native_nnz,
        "scifem_nnz": scifem_nnz,
        "n_points": len(points),
        "native_valid": native_ps.n_valid_points,
    }


def test_single_point(model, source_point: np.ndarray):
    """Test with a single point."""
    print_header("Test 1: Single Point")

    result = test_native_vs_scifem_basic(model.V_scalar, source_point.reshape(1, 3))

    print(f"Point:              {source_point}")
    print(f"Points valid:       {result['native_valid']}/{result['n_points']}")
    print(f"Max abs diff:       {result['max_abs_diff']:.2e}")
    print(f"Relative diff:      {result['rel_diff']:.2e}")
    print(f"Native norm:        {result['native_norm']:.6e}")
    print(f"Scifem norm:        {result['scifem_norm']:.6e}")
    print(f"Native non-zeros:   {result['native_nnz']}")
    print(f"Scifem non-zeros:   {result['scifem_nnz']}")

    passed = result['rel_diff'] < 1e-10
    print(f"\nStatus: {'PASS ✓' if passed else 'FAIL ✗'}")

    return passed, result


def test_multiple_points(model, center: np.ndarray, n_points: int = 256, radius: float = 5.0):
    """Test with multiple distributed points."""
    print_header("Test 2: Multiple Distributed Points")

    # Generate points in a sphere
    np.random.seed(42)  # For reproducibility
    points = []
    while len(points) < n_points:
        rp = center + np.random.uniform(-radius, radius, size=(3,))
        if np.linalg.norm(rp - center) <= radius:
            points.append(rp)
    points = np.array(points, dtype=np.float64)

    result = test_native_vs_scifem_basic(model.V_scalar, points)

    print(f"Center:             {center}")
    print(f"Radius:             {radius}")
    print(f"Points valid:       {result['native_valid']}/{result['n_points']}")
    print(f"Max abs diff:       {result['max_abs_diff']:.2e}")
    print(f"Relative diff:      {result['rel_diff']:.2e}")
    print(f"Native norm:        {result['native_norm']:.6e}")
    print(f"Scifem norm:        {result['scifem_norm']:.6e}")
    print(f"Native non-zeros:   {result['native_nnz']}")
    print(f"Scifem non-zeros:   {result['scifem_nnz']}")

    passed = result['rel_diff'] < 1e-10
    print(f"\nStatus: {'PASS ✓' if passed else 'FAIL ✗'}")

    return passed, result


def test_bipolar_sources(model, source_point: np.ndarray, ground_point: np.ndarray,
                         n_points: int = 256, radius: float = 5.0):
    """Test bipolar configuration (+1 source, -1 ground)."""
    print_header("Test 3: Bipolar Sources (±1)")

    from dolfinx import fem
    import scifem
    from emgop.fem.point_source import NativePointSource

    # Generate source points
    np.random.seed(42)
    source_pts = []
    while len(source_pts) < n_points:
        rp = source_point + np.random.uniform(-radius, radius, size=(3,))
        if np.linalg.norm(rp - source_point) <= radius:
            source_pts.append(rp)
    source_pts = np.array(source_pts, dtype=np.float64)

    # Generate ground points
    np.random.seed(43)
    ground_pts = []
    while len(ground_pts) < n_points:
        rp = ground_point + np.random.uniform(-radius, radius, size=(3,))
        if np.linalg.norm(rp - ground_point) <= radius:
            ground_pts.append(rp)
    ground_pts = np.array(ground_pts, dtype=np.float64)

    gamma_source = 1.0 / n_points
    gamma_ground = -1.0 / n_points

    # Create test vectors
    b_native = fem.Function(model.V_scalar)
    b_scifem = fem.Function(model.V_scalar)

    # Apply native
    native_src = NativePointSource(model.V_scalar, source_pts, magnitude=gamma_source)
    native_gnd = NativePointSource(model.V_scalar, ground_pts, magnitude=gamma_ground)
    native_src.apply_to_vector(b_native)
    native_gnd.apply_to_vector(b_native)

    # Apply scifem
    scifem_src = scifem.PointSource(model.V_scalar, source_pts, magnitude=gamma_source)
    scifem_gnd = scifem.PointSource(model.V_scalar, ground_pts, magnitude=gamma_ground)
    scifem_src.apply_to_vector(b_scifem)
    scifem_gnd.apply_to_vector(b_scifem)

    # Compare
    native_arr = b_native.x.array
    scifem_arr = b_scifem.x.array

    diff = np.abs(native_arr - scifem_arr)
    max_abs_diff = diff.max()
    scifem_norm = np.linalg.norm(scifem_arr)
    rel_diff = np.linalg.norm(diff) / scifem_norm if scifem_norm > 1e-15 else 0.0

    # Check balance (sum should be ~0)
    native_sum = native_arr.sum()
    scifem_sum = scifem_arr.sum()

    print(f"Source center:      {source_point}")
    print(f"Ground center:      {ground_point}")
    print(f"Points per electrode: {n_points}")
    print(f"Source valid:       {native_src.n_valid_points}/{n_points}")
    print(f"Ground valid:       {native_gnd.n_valid_points}/{n_points}")
    print(f"Max abs diff:       {max_abs_diff:.2e}")
    print(f"Relative diff:      {rel_diff:.2e}")
    print(f"Native sum:         {native_sum:.2e} (should be ~0)")
    print(f"Scifem sum:         {scifem_sum:.2e} (should be ~0)")

    passed = rel_diff < 1e-10
    print(f"\nStatus: {'PASS ✓' if passed else 'FAIL ✗'}")

    return passed, {
        "max_abs_diff": max_abs_diff,
        "rel_diff": rel_diff,
        "native_sum": native_sum,
        "scifem_sum": scifem_sum,
    }


def test_full_solver_comparison(mesh_path: str, source_point: np.ndarray,
                                 ground_position: np.ndarray,
                                 electrode_radius: float = 5.0,
                                 n_points: int = 256):
    """Compare full FEM solutions with native vs scifem."""
    print_header("Test 4: Full Solver Comparison")

    from emgop.fem.electrode_configs import ElectrodeFEMSolver

    # Solve with scifem
    print("Solving with scifem...")
    solver_scifem = ElectrodeFEMSolver(
        mesh_path,
        return_mode="localized",
        ground_position=ground_position,
        ground_radius=electrode_radius,
        ground_points=n_points,
        use_native_point_source=False,
        build_conductivity_map=True,
    )
    np.random.seed(42)  # Same seed for same point sampling
    uh_scifem = solver_scifem.solve(source_point, electrode_radius, n_points)
    scifem_iters = solver_scifem.ksp_iters

    # Solve with native
    print("Solving with native...")
    solver_native = ElectrodeFEMSolver(
        mesh_path,
        return_mode="localized",
        ground_position=ground_position,
        ground_radius=electrode_radius,
        ground_points=n_points,
        use_native_point_source=True,
        build_conductivity_map=True,
    )
    np.random.seed(42)  # Same seed for same point sampling
    uh_native = solver_native.solve(source_point, electrode_radius, n_points)
    native_iters = solver_native.ksp_iters

    # Compare solutions
    native_arr = uh_native.x.array
    scifem_arr = uh_scifem.x.array

    diff = native_arr - scifem_arr
    max_abs_diff = np.abs(diff).max()
    scifem_norm = np.linalg.norm(scifem_arr)
    rel_diff = np.linalg.norm(diff) / scifem_norm if scifem_norm > 1e-15 else 0.0

    # Compare RHS vectors
    b_diff = solver_native.b_vec.array - solver_scifem.b_vec.array
    b_max_diff = np.abs(b_diff).max()
    b_rel_diff = np.linalg.norm(b_diff) / solver_scifem.b_vec.norm()

    # Compute residuals
    res_native = solver_native.compute_residual(uh_native)
    res_scifem = solver_scifem.compute_residual(uh_scifem)

    print(f"\nRHS vector comparison:")
    print(f"  Max abs diff:     {b_max_diff:.2e}")
    print(f"  Relative diff:    {b_rel_diff:.2e}")

    print(f"\nSolution comparison:")
    print(f"  Max abs diff:     {max_abs_diff:.2e}")
    print(f"  Relative diff:    {rel_diff:.2e}")
    print(f"  Native norm:      {np.linalg.norm(native_arr):.6e}")
    print(f"  Scifem norm:      {scifem_norm:.6e}")

    print(f"\nKSP iterations:")
    print(f"  Native:           {native_iters}")
    print(f"  Scifem:           {scifem_iters}")

    print(f"\nResiduals:")
    print(f"  Native rel_res:   {res_native['rel_residual']:.2e}")
    print(f"  Scifem rel_res:   {res_scifem['rel_residual']:.2e}")

    # RHS should match exactly, solution should be very close
    b_passed = b_rel_diff < 1e-10
    sol_passed = rel_diff < 1e-6

    print(f"\nRHS match:     {'PASS ✓' if b_passed else 'FAIL ✗'}")
    print(f"Solution match: {'PASS ✓' if sol_passed else 'FAIL ✗'}")

    return b_passed and sol_passed, {
        "b_rel_diff": b_rel_diff,
        "sol_rel_diff": rel_diff,
        "native_residual": res_native['rel_residual'],
        "scifem_residual": res_scifem['rel_residual'],
    }


def main():
    parser = argparse.ArgumentParser(description="Test native point source vs scifem")
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

    print_header("Native Point Source Test Suite")
    print(f"Mesh: {mesh_path}")
    print(f"Meta: {meta_path}")

    # Load model
    print("\nLoading FEMModel...")
    from emgop.fem import FEMModel
    model = FEMModel(str(mesh_path), build_conductivity_map=True)
    print("Done.")

    # Define test points
    INSET = args.electrode_radius + 1.0
    r_electrode = meta.get("radius_skin", 40.0) - INSET
    z_electrode = meta.get("length", 240.0) / 2

    source_point = np.array([r_electrode, 0.0, z_electrode])

    from emgop.fem.electrode_configs import compute_ground_position
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

    # Test 1: Single point
    passed1, res1 = test_single_point(model, source_point)
    results["single_point"] = (passed1, res1)

    # Test 2: Multiple points
    passed2, res2 = test_multiple_points(model, source_point, args.n_points, args.electrode_radius)
    results["multiple_points"] = (passed2, res2)

    # Test 3: Bipolar sources
    passed3, res3 = test_bipolar_sources(model, source_point, ground_position,
                                          args.n_points, args.electrode_radius)
    results["bipolar"] = (passed3, res3)

    # Test 4: Full solver comparison
    passed4, res4 = test_full_solver_comparison(
        str(mesh_path), source_point, ground_position,
        args.electrode_radius, args.n_points
    )
    results["full_solver"] = (passed4, res4)

    # Summary
    print_header("SUMMARY")
    all_passed = True
    for name, (passed, _) in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:<20}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed! Native implementation matches scifem.")
    else:
        print("Some tests failed. Check output above for details.")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
