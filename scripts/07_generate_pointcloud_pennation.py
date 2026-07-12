#!/usr/bin/env python3
"""
Generate point cloud training dataset with varying pennation angles.

This script generates training data for neural field models and PINNs by:
1. Running FEM solves for multiple pennation angles on a single mesh
2. Sampling point clouds from each solution
3. Storing points, solution values, and rotated conductivity for training

Output format suitable for:
- Neural field training: u(x, y, z | pennation_angle)
- PINN training: With source field f(x) and boundary data

Usage:
    export PYTHONPATH=src
    python scripts/07_generate_pointcloud_pennation.py \\
        --mesh meshes/sample_000000.msh \\
        --meta metadata/sample_000000.json \\
        --out_dir ./datasets/pointcloud_pennation \\
        --n_angles 20 \\
        --angle_min 0 --angle_max 45 \\
        --n_points 10000 \\
        --n_boundary 2000 \\
        --sampling_strategy uniform \\
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

from emgforge.fem import FEMModel, load_meta
from emgforge.fem.sanity import source_point_from_cyl_normalized
from emgforge.pointcloud import (
    generate_pointcloud_sample,
    save_pointcloud_sample,
)


def generate_pennation_pointcloud_dataset(
    mesh_path: Path,
    meta_path: Path,
    out_dir: Path,
    n_angles: int,
    angle_min: float,
    angle_max: float,
    n_points: int,
    n_boundary: int,
    sampling_strategy: str,
    seed: int,
    source_cyl: tuple = (1.0, 0.0, 0.5),
    source_layer: str = "skin",
    source_margin: float = 1.0,
    source_mode: str = "gaussian",
    source_sigma: float = 0.1,
    include_pinn_data: bool = True,
    concentration_radius: float = 15.0,
    concentration_fraction: float = 0.3,
):
    """
    Generate point cloud dataset with varying pennation angles.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Point Cloud Pennation Dataset Generator")
    print("=" * 80)
    print(f"Mesh: {mesh_path}")
    print(f"Meta: {meta_path}")
    print(f"Output: {out_dir}")
    print(f"Angles: {n_angles} from {angle_min}° to {angle_max}°")
    print(f"Points per sample: {n_points} interior, {n_boundary} boundary")
    print(f"Sampling strategy: {sampling_strategy}")
    print(f"Source: r={source_cyl[0]}, theta={source_cyl[1]}°, z={source_cyl[2]}")
    print()

    meta = load_meta(meta_path)

    # Generate angle sweep
    if n_angles == 1:
        angles = np.array([angle_min])
    else:
        angles = np.linspace(angle_min, angle_max, n_angles)

    print(f"Generating {len(angles)} samples with angles: {angles[0]:.1f}° to {angles[-1]:.1f}°")

    # Compute fixed source position
    src = source_point_from_cyl_normalized(
        meta, source_cyl[0], source_cyl[1], source_cyl[2],
        layer=source_layer, margin=source_margin
    )
    source_position = src[0]
    print(f"Source world coords: {source_position}")

    manifest = []
    rng = np.random.default_rng(seed)

    for idx, angle in enumerate(angles):
        sample_start = time.time()

        # Create fresh model for each angle
        print(f"\n[{idx+1:4d}/{n_angles}] Loading mesh for angle={angle:.2f}°...")
        model = FEMModel(
            str(mesh_path),
            gdim=3,
            point_source=(source_mode == "point"),
            build_conductivity_map=True,
            source_sigma=source_sigma,
        )

        # Apply pennation angle
        if angle != 0.0:
            model.apply_pinnation(angle)

        # Solve FEM
        uh = model.solve_for_point(src)

        # Generate point cloud sample
        sample = generate_pointcloud_sample(
            model=model,
            uh=uh,
            meta=meta,
            source_position=source_position,
            n_interior_points=n_points,
            n_boundary_points=n_boundary if include_pinn_data else 0,
            sampling_strategy=sampling_strategy,
            seed=rng.integers(0, 2**31),
            ground_position=None,
            pennation_angle=angle,
            source_sigma=source_sigma,
            include_pinn_data=include_pinn_data,
            concentration_radius=concentration_radius,
            concentration_fraction=concentration_fraction,
        )

        # Add source cylindrical coords
        sample["source_cyl"] = np.array(source_cyl, dtype=np.float32)

        # Save sample
        sample_id = f"penn_{idx:06d}"
        out_path = out_dir / f"{sample_id}.npz"
        save_pointcloud_sample(sample, out_path)

        # Record in manifest
        manifest_entry = {
            "sample_id": sample_id,
            "file": f"{sample_id}.npz",
            "pennation_angle": float(angle),
            "source_position": source_position.tolist(),
            "source_cyl": list(source_cyl),
            "n_points": n_points,
            "n_boundary": n_boundary if include_pinn_data else 0,
        }
        manifest.append(manifest_entry)

        sample_time = time.time() - sample_start
        u_min, u_max = sample["u"].min(), sample["u"].max()
        print(f"  angle={angle:6.2f}° u=[{u_min:.6f}, {u_max:.6f}] ({sample_time:.2f}s)")

        # Clean up
        del model, uh

    # Save manifest
    manifest_data = {
        "mesh": str(mesh_path),
        "meta": str(meta_path),
        "dataset_type": "pointcloud_pennation_sweep",
        "n_samples": len(manifest),
        "n_interior_points": n_points,
        "n_boundary_points": n_boundary if include_pinn_data else 0,
        "sampling_strategy": sampling_strategy,
        "generation_params": {
            "seed": seed,
            "angle_min": angle_min,
            "angle_max": angle_max,
            "n_angles": n_angles,
            "source_cyl": list(source_cyl),
            "source_layer": source_layer,
            "source_margin": source_margin,
            "source_mode": source_mode,
            "source_sigma": source_sigma,
            "include_pinn_data": include_pinn_data,
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
        description="Generate point cloud pennation sweep dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    ap.add_argument("--mesh", type=str, required=True, help="Path to mesh file (.msh)")
    ap.add_argument("--meta", type=str, required=True, help="Path to metadata file (.json)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # Angle sweep
    ap.add_argument("--n_angles", type=int, default=20, help="Number of angles")
    ap.add_argument("--angle_min", type=float, default=0.0, help="Minimum angle (degrees)")
    ap.add_argument("--angle_max", type=float, default=45.0, help="Maximum angle (degrees)")

    # Dataset size
    ap.add_argument("--n_points", type=int, default=10000, help="Interior points per sample")
    ap.add_argument("--n_boundary", type=int, default=2000, help="Boundary points per sample (for PINN)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Sampling
    ap.add_argument("--sampling_strategy", type=str, default="uniform",
                    choices=["uniform", "near_electrode", "stratified_tissue", "surface_biased", "mixed"],
                    help="Point sampling strategy")
    ap.add_argument("--concentration_radius", type=float, default=15.0,
                    help="Radius for concentrated sampling near electrode")
    ap.add_argument("--concentration_fraction", type=float, default=0.3,
                    help="Fraction of points concentrated near electrode")

    # Source placement
    ap.add_argument("--source_cyl", type=str, default="1.0,0.0,0.5",
                    help="Source position: r_norm,theta_deg,z_norm")
    ap.add_argument("--source_layer", type=str, default="skin", choices=["skin", "muscle"])
    ap.add_argument("--source_margin", type=float, default=1.0, help="Margin from layer boundaries")
    ap.add_argument("--source_mode", type=str, default="gaussian", choices=["gaussian", "point"])
    ap.add_argument("--source_sigma", type=float, default=0.1, help="Gaussian source sigma")

    # PINN options
    ap.add_argument("--no_pinn_data", action="store_true",
                    help="Skip PINN data (boundary points, source field)")

    args = ap.parse_args()

    # Parse source_cyl
    try:
        source_cyl = tuple(float(x) for x in args.source_cyl.split(","))
        if len(source_cyl) != 3:
            raise ValueError
    except ValueError:
        raise SystemExit("--source_cyl must be three comma-separated floats: r_norm,theta_deg,z_norm")

    generate_pennation_pointcloud_dataset(
        mesh_path=Path(args.mesh),
        meta_path=Path(args.meta),
        out_dir=Path(args.out_dir),
        n_angles=args.n_angles,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        n_points=args.n_points,
        n_boundary=args.n_boundary,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
        source_cyl=source_cyl,
        source_layer=args.source_layer,
        source_margin=args.source_margin,
        source_mode=args.source_mode,
        source_sigma=args.source_sigma,
        include_pinn_data=not args.no_pinn_data,
        concentration_radius=args.concentration_radius,
        concentration_fraction=args.concentration_fraction,
    )


if __name__ == "__main__":
    main()
