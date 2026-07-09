#!/usr/bin/env python3
"""
Generate point cloud training dataset with varying electrode positions.

This script generates training data for neural field models and PINNs by:
1. Running FEM solves for multiple electrode positions on a single mesh
2. Sampling point clouds from each solution
3. Storing points, solution values, and conductivity for training

Output format suitable for:
- Neural field training: u(x, y, z | electrode_x, y, z)
- PINN training: With source field f(x) and boundary data

Usage:
    export PYTHONPATH=src
    python scripts/06_generate_pointcloud_electrode.py \\
        --mesh meshes/sample_000000.msh \\
        --meta metadata/sample_000000.json \\
        --out_dir ./datasets/pointcloud_electrode \\
        --n_electrodes 100 \\
        --n_points 10000 \\
        --n_boundary 2000 \\
        --sampling_strategy near_electrode \\
        --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure local src/ is on path
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emgop.fem import (
    ElectrodeFEMSolver,
    FEMModel,
    load_meta,
    compute_ground_position,
)
from emgop.fem.sanity import source_point_from_cyl_normalized
from emgop.pointcloud import (
    generate_pointcloud_sample,
    save_pointcloud_sample,
)


def sample_electrode_positions(
    n: int,
    z_margin: float = 0.1,
    r_norm: float = 1.0,
    seed: int = 42,
) -> list:
    """Generate n electrode positions using uniform random sampling."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 360, n)
    z_norm = rng.uniform(z_margin, 1 - z_margin, n)
    r_vals = np.full(n, r_norm)
    return list(zip(r_vals, theta, z_norm))


def generate_electrode_grid(
    n_z: int,
    n_theta: int,
    z_margin: float = 0.1,
    r_norm: float = 1.0,
) -> list:
    """
    Generate electrode positions on a regular grid.

    Creates a grid of n_z x n_theta electrode positions around the cylinder surface.
    Electrodes are evenly spaced along z (within margins) and around theta (0-360).

    Args:
        n_z: Number of electrodes along z-axis
        n_theta: Number of electrodes around circumference
        z_margin: Margin from z boundaries (normalized, 0-1)
        r_norm: Normalized radial position (1.0 = skin surface)

    Returns:
        List of (r_norm, theta_deg, z_norm) tuples
    """
    z_vals = np.linspace(z_margin, 1 - z_margin, n_z)
    theta_vals = np.linspace(0, 360, n_theta, endpoint=False)
    positions = []
    for z in z_vals:
        for theta in theta_vals:
            positions.append((r_norm, theta, z))
    return positions


def generate_electrode_pointcloud_dataset(
    mesh_path: Path,
    meta_path: Path,
    out_dir: Path,
    n_electrodes: int,
    n_points: int,
    n_boundary: int,
    sampling_strategy: str,
    seed: int,
    z_margin: float = 0.1,
    r_norm: float = 1.0,
    source_layer: str = "skin",
    source_margin: float = 1.0,
    return_mode: str = "localized",
    ground_mode: str = "opposite",
    ground_radius: float = 5.0,
    ground_points: int = 256,
    electrode_inset: float = 4.0,
    use_native_point_source: bool = True,
    include_pinn_data: bool = True,
    concentration_radius: float = 15.0,
    concentration_fraction: float = 0.3,
    electrode_layout: str = "random",
    grid_z: int = 8,
    grid_theta: int = 16,
    source_mode: str = "point",
    source_sigma: float = 5.0,
):
    """
    Generate point cloud dataset with varying electrode positions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Point Cloud Electrode Dataset Generator")
    print("=" * 80)
    print(f"Mesh: {mesh_path}")
    print(f"Meta: {meta_path}")
    print(f"Output: {out_dir}")
    print(f"Electrodes: {n_electrodes}")
    print(f"Points per sample: {n_points} interior, {n_boundary} boundary")
    print(f"Sampling strategy: {sampling_strategy}")
    print(f"Electrode layout: {electrode_layout}" +
          (f" ({grid_z}x{grid_theta})" if electrode_layout == "grid" else ""))
    print(f"Return mode: {return_mode}")
    print(f"Source mode: {source_mode}" +
          (f" (sigma={source_sigma}mm)" if source_mode == "gaussian" else ""))
    print()

    meta = load_meta(meta_path)

    # Sample electrode positions based on layout
    if electrode_layout == "grid":
        positions = generate_electrode_grid(
            n_z=grid_z, n_theta=grid_theta, z_margin=z_margin, r_norm=r_norm
        )
        # Override n_electrodes to actual grid size
        actual_n = len(positions)
        if actual_n != n_electrodes:
            print(f"[note] Grid layout produces {actual_n} electrodes "
                  f"(requested {n_electrodes})")
            n_electrodes = actual_n
    else:
        positions = sample_electrode_positions(
            n_electrodes, z_margin=z_margin, r_norm=r_norm, seed=seed
        )

    print(f"Generated {len(positions)} electrode positions")

    # Create solver (mesh loaded once)
    print("Loading mesh and creating solver...")
    start_time = time.time()

    first_source = source_point_from_cyl_normalized(
        meta, positions[0][0], positions[0][1], positions[0][2],
        layer=source_layer, margin=source_margin
    )

    initial_ground = None
    if return_mode == "localized":
        initial_ground = compute_ground_position(
            meta, first_source[0], ground_mode=ground_mode, inset=electrode_inset
        )

    solver = ElectrodeFEMSolver(
        str(mesh_path),
        return_mode=return_mode,
        ground_position=initial_ground,
        ground_radius=ground_radius,
        ground_points=ground_points,
        use_native_point_source=use_native_point_source,
        source_mode=source_mode,
        source_sigma=source_sigma,
        gdim=3,
        build_conductivity_map=True,
    )

    load_time = time.time() - start_time
    print(f"Mesh loaded in {load_time:.2f}s")

    manifest = []
    rng = np.random.default_rng(seed)

    for idx, (r_n, theta_deg, z_n) in enumerate(positions):
        sample_start = time.time()

        # Compute source position
        source_pt = source_point_from_cyl_normalized(
            meta, r_norm=r_n, theta_deg=theta_deg, z_norm=z_n,
            layer=source_layer, margin=source_margin
        )
        source_position = source_pt[0]

        # Compute ground position
        ground_position = None
        if return_mode == "localized":
            ground_position = compute_ground_position(
                meta, source_position, ground_mode=ground_mode, inset=electrode_inset
            )
            solver.ground_position = ground_position

        # Solve FEM
        uh = solver.solve(source_position, source_radius=electrode_inset, source_points=ground_points)

        # Generate point cloud sample
        sample = generate_pointcloud_sample(
            model=solver.model,
            uh=uh,
            meta=meta,
            source_position=source_position,
            n_interior_points=n_points,
            n_boundary_points=n_boundary if include_pinn_data else 0,
            sampling_strategy=sampling_strategy,
            seed=rng.integers(0, 2**31),
            ground_position=ground_position,
            pennation_angle=0.0,
            source_sigma=source_sigma,
            include_pinn_data=include_pinn_data,
            concentration_radius=concentration_radius,
            concentration_fraction=concentration_fraction,
        )

        # Add electrode cylindrical coords
        sample["electrode_cyl"] = np.array([r_n, theta_deg, z_n], dtype=np.float32)

        # Save sample
        sample_id = f"elec_{idx:06d}"
        out_path = out_dir / f"{sample_id}.npz"
        save_pointcloud_sample(sample, out_path)

        # Record in manifest
        manifest_entry = {
            "sample_id": sample_id,
            "file": f"{sample_id}.npz",
            "electrode_cyl": [float(r_n), float(theta_deg), float(z_n)],
            "electrode_position": source_position.tolist(),
            "n_points": n_points,
            "n_boundary": n_boundary if include_pinn_data else 0,
        }
        if ground_position is not None:
            manifest_entry["ground_position"] = ground_position.tolist()
        manifest.append(manifest_entry)

        sample_time = time.time() - sample_start
        if (idx + 1) % 10 == 0 or idx == 0:
            u_min, u_max = sample["u"].min(), sample["u"].max()
            print(f"  [{idx+1:4d}/{n_electrodes}] theta={theta_deg:6.1f}° z={z_n:.3f} "
                  f"u=[{u_min:.4f}, {u_max:.4f}] ({sample_time:.2f}s)")

    # Save manifest
    electrode_layout_info = {"type": electrode_layout}
    if electrode_layout == "grid":
        electrode_layout_info["grid_z"] = grid_z
        electrode_layout_info["grid_theta"] = grid_theta

    manifest_data = {
        "mesh": str(mesh_path),
        "meta": str(meta_path),
        "dataset_type": "pointcloud_electrode_sweep",
        "n_samples": len(manifest),
        "n_interior_points": n_points,
        "n_boundary_points": n_boundary if include_pinn_data else 0,
        "sampling_strategy": sampling_strategy,
        "electrode_layout": electrode_layout_info,
        "generation_params": {
            "seed": seed,
            "z_margin": z_margin,
            "r_norm": r_norm,
            "source_layer": source_layer,
            "source_margin": source_margin,
            "return_mode": return_mode,
            "ground_mode": ground_mode,
            "source_mode": source_mode,
            "source_sigma": source_sigma,
            "include_pinn_data": include_pinn_data,
            "concentration_radius": concentration_radius,
            "concentration_fraction": concentration_fraction,
        },
        "samples": manifest,
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print()
    print("=" * 80)
    print("Dataset generation complete!")
    print(f"  Samples: {len(manifest)}")
    print(f"  Points per sample: {n_points} interior" +
          (f", {n_boundary} boundary" if include_pinn_data else ""))
    print(f"  Output: {out_dir}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(
        description="Generate point cloud electrode sweep dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    ap.add_argument("--mesh", type=str, required=True, help="Path to mesh file (.msh)")
    ap.add_argument("--meta", type=str, required=True, help="Path to metadata file (.json)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # Dataset size
    ap.add_argument("--n_electrodes", type=int, default=100, help="Number of electrode positions")
    ap.add_argument("--n_points", type=int, default=10000, help="Interior points per sample")
    ap.add_argument("--n_boundary", type=int, default=2000, help="Boundary points per sample (for PINN)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Sampling
    ap.add_argument("--sampling_strategy", type=str, default="near_electrode",
                    choices=["uniform", "near_electrode", "stratified_tissue", "surface_biased", "mixed"],
                    help="Point sampling strategy")
    ap.add_argument("--concentration_radius", type=float, default=15.0,
                    help="Radius for concentrated sampling near electrode")
    ap.add_argument("--concentration_fraction", type=float, default=0.3,
                    help="Fraction of points concentrated near electrode")

    # Electrode placement
    ap.add_argument("--electrode_layout", type=str, default="random",
                    choices=["random", "grid"],
                    help="Electrode position layout: random sampling or regular grid")
    ap.add_argument("--grid_z", type=int, default=8,
                    help="Number of electrodes along z-axis (grid layout only)")
    ap.add_argument("--grid_theta", type=int, default=16,
                    help="Number of electrodes around circumference (grid layout only)")
    ap.add_argument("--z_margin", type=float, default=0.1, help="Z margin from boundaries")
    ap.add_argument("--r_norm", type=float, default=1.0, help="Normalized radial position")
    ap.add_argument("--source_layer", type=str, default="skin", choices=["skin", "muscle"])
    ap.add_argument("--source_margin", type=float, default=1.0, help="Margin from layer boundaries")

    # FEM configuration
    ap.add_argument("--return_mode", type=str, default="localized",
                    choices=["volumetric", "localized"])
    ap.add_argument("--ground_mode", type=str, default="opposite",
                    choices=["opposite", "distal", "cylindrical"])
    ap.add_argument("--ground_radius", type=float, default=5.0)
    ap.add_argument("--ground_points", type=int, default=256)
    ap.add_argument("--electrode_inset", type=float, default=4.0)
    ap.add_argument("--native_point_source", action="store_true", default=True)

    # Source configuration
    ap.add_argument("--source_mode", type=str, default="point",
                    choices=["point", "gaussian"],
                    help="Source type: 'point' (delta) or 'gaussian' (volumetric)")
    ap.add_argument("--source_sigma", type=float, default=5.0,
                    help="Gaussian source sigma in mm (only used if source_mode=gaussian)")

    # PINN options
    ap.add_argument("--no_pinn_data", action="store_true",
                    help="Skip PINN data (boundary points, source field)")

    args = ap.parse_args()

    generate_electrode_pointcloud_dataset(
        mesh_path=Path(args.mesh),
        meta_path=Path(args.meta),
        out_dir=Path(args.out_dir),
        n_electrodes=args.n_electrodes,
        n_points=args.n_points,
        n_boundary=args.n_boundary,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
        z_margin=args.z_margin,
        r_norm=args.r_norm,
        source_layer=args.source_layer,
        source_margin=args.source_margin,
        return_mode=args.return_mode,
        ground_mode=args.ground_mode,
        ground_radius=args.ground_radius,
        ground_points=args.ground_points,
        electrode_inset=args.electrode_inset,
        use_native_point_source=args.native_point_source,
        include_pinn_data=not args.no_pinn_data,
        concentration_radius=args.concentration_radius,
        concentration_fraction=args.concentration_fraction,
        electrode_layout=args.electrode_layout,
        grid_z=args.grid_z,
        grid_theta=args.grid_theta,
        source_mode=args.source_mode,
        source_sigma=args.source_sigma,
    )


if __name__ == "__main__":
    main()
