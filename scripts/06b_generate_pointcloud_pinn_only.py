#!/usr/bin/env python3
"""
Generate point cloud PINN training dataset WITHOUT any FEM solve.

Uses only analytical computations:
  - Interior points: sampling.sample_points() (rejection sampling in cylinder)
  - Conductivity sigma(x): analytical.compute_conductivity_analytical() (radial tissue lookup)
  - Source field f(x): evaluate.compute_source_field() (Gaussian blob)
  - Boundary data: sampling.sample_boundary_points() (surface + cap sampling)
  - u field: zeros (dummy — not used by pure PINN with lambda_data=0)

No dependencies on: FEniCSx, DOLFINx, mpi4py, Gmsh, PETSc.
Runs in the lightweight 'emg' conda env.

Usage:
    python scripts/06b_generate_pointcloud_pinn_only.py \\
        --meta metadata/reference_avg.json \\
        --out_dir data/pinn_only_504 \\
        --electrode_layout grid --grid_z 8 --grid_theta 63 \\
        --source_mode gaussian --source_sigma 5.0 \\
        --n_points 5000 --n_boundary 1000 --seed 42
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

from emgforge.pointcloud.sampling import sample_points, sample_boundary_points
from emgforge.pointcloud.evaluate import compute_tissue_labels, compute_source_field
from emgforge.pointcloud.analytical import compute_conductivity_analytical
from emgforge.pointcloud.generate import save_pointcloud_sample


def load_meta(meta_path: Path) -> dict:
    """Load mesh metadata JSON file."""
    with open(meta_path) as f:
        return json.load(f)


def source_point_from_cyl_normalized(
    meta: dict,
    r_norm: float,
    theta_deg: float,
    z_norm: float,
) -> np.ndarray:
    """Map normalized cylindrical coords to world coords on skin surface."""
    gp = meta["geometry_params"]
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])

    r_target = r_norm * r_skin
    theta = np.deg2rad(theta_deg)
    z = z_norm * length
    x = r_target * np.cos(theta)
    y = r_target * np.sin(theta)
    return np.array([x, y, z], dtype=np.float64)


def generate_electrode_grid(
    n_z: int, n_theta: int, z_margin: float = 0.1, r_norm: float = 1.0,
) -> list:
    """Generate electrode positions on a regular grid."""
    z_vals = np.linspace(z_margin, 1 - z_margin, n_z)
    theta_vals = np.linspace(0, 360, n_theta, endpoint=False)
    return [(r_norm, theta, z) for z in z_vals for theta in theta_vals]


def sample_electrode_positions(
    n: int, z_margin: float = 0.1, r_norm: float = 1.0, seed: int = 42,
) -> list:
    """Generate n electrode positions using uniform random sampling."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 360, n)
    z_norm = rng.uniform(z_margin, 1 - z_margin, n)
    return list(zip(np.full(n, r_norm), theta, z_norm))


def main():
    ap = argparse.ArgumentParser(
        description="Generate FEM-free PINN point cloud dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    ap.add_argument("--meta", type=str, required=True,
                     help="Path to metadata file (.json)")
    ap.add_argument("--out_dir", type=str, required=True,
                     help="Output directory")

    # Dataset size
    ap.add_argument("--n_electrodes", type=int, default=100,
                     help="Number of electrode positions (ignored for grid layout)")
    ap.add_argument("--n_points", type=int, default=5000,
                     help="Interior points per sample")
    ap.add_argument("--n_boundary", type=int, default=1000,
                     help="Boundary points per sample (for PINN Neumann BC)")
    ap.add_argument("--seed", type=int, default=42,
                     help="Random seed")

    # Sampling
    ap.add_argument("--sampling_strategy", type=str, default="near_electrode",
                     choices=["uniform", "near_electrode", "stratified_tissue",
                              "surface_biased", "mixed"])
    ap.add_argument("--concentration_radius", type=float, default=15.0)
    ap.add_argument("--concentration_fraction", type=float, default=0.3)

    # Electrode placement
    ap.add_argument("--electrode_layout", type=str, default="random",
                     choices=["random", "grid"])
    ap.add_argument("--grid_z", type=int, default=8)
    ap.add_argument("--grid_theta", type=int, default=16)
    ap.add_argument("--z_margin", type=float, default=0.1)
    ap.add_argument("--r_norm", type=float, default=1.0)

    # Source configuration
    ap.add_argument("--source_mode", type=str, default="gaussian",
                     choices=["point", "gaussian"])
    ap.add_argument("--source_sigma", type=float, default=5.0,
                     help="Gaussian source sigma in mm")

    # Pennation
    ap.add_argument("--pennation_angle", type=float, default=0.0,
                     help="Muscle fiber pennation angle in degrees")

    args = ap.parse_args()

    meta_path = Path(args.meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(meta_path)
    gp = meta["geometry_params"]
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])
    volume = np.pi * r_skin ** 2 * length

    print("=" * 80)
    print("FEM-Free PINN Point Cloud Generator")
    print("=" * 80)
    print(f"Meta: {meta_path}")
    print(f"Output: {out_dir}")
    print(f"Geometry: r_skin={r_skin:.2f}mm, length={length:.1f}mm")
    print(f"Source mode: {args.source_mode}" +
          (f" (sigma={args.source_sigma}mm)" if args.source_mode == "gaussian" else ""))
    print(f"Points per sample: {args.n_points} interior, {args.n_boundary} boundary")
    print(f"Sampling strategy: {args.sampling_strategy}")
    if args.pennation_angle != 0.0:
        print(f"Pennation angle: {args.pennation_angle} deg")

    # Generate electrode positions
    if args.electrode_layout == "grid":
        positions = generate_electrode_grid(
            args.grid_z, args.grid_theta, args.z_margin, args.r_norm
        )
        n_electrodes = len(positions)
        print(f"Electrode layout: grid ({args.grid_z}x{args.grid_theta} = {n_electrodes})")
    else:
        positions = sample_electrode_positions(
            args.n_electrodes, args.z_margin, args.r_norm, args.seed
        )
        n_electrodes = len(positions)
        print(f"Electrode layout: random ({n_electrodes})")

    print()

    rng = np.random.default_rng(args.seed)
    manifest = []
    t_start = time.time()

    for idx, (r_n, theta_deg, z_n) in enumerate(positions):
        sample_start = time.time()

        # Compute source position in world coordinates
        source_position = source_point_from_cyl_normalized(
            meta, r_n, theta_deg, z_n
        )

        # Sample interior points
        sampling_result = sample_points(
            meta=meta,
            n_points=args.n_points,
            strategy=args.sampling_strategy,
            rng=rng,
            electrode_position=source_position,
            concentration_radius=args.concentration_radius,
            concentration_fraction=args.concentration_fraction,
        )
        points = sampling_result["points"].astype(np.float32)

        # Compute conductivity analytically (no FEM!)
        sigma = compute_conductivity_analytical(
            points, meta, pennation_angle=args.pennation_angle
        )

        # Compute tissue labels
        tissue_labels = compute_tissue_labels(meta, points)

        # Compute source field analytically
        if args.source_mode == "gaussian":
            source_field = compute_source_field(
                points, source_position,
                source_sigma=args.source_sigma,
                volume=volume, meta=meta,
            )
        else:
            # Point source: approximate as very narrow Gaussian
            source_field = compute_source_field(
                points, source_position,
                source_sigma=0.5,  # narrow approximation
                volume=volume, meta=meta,
            )

        # Dummy u field (zeros — not used by pure PINN)
        u = np.zeros(args.n_points, dtype=np.float32)

        # Sample boundary points
        boundary_points, boundary_normals = sample_boundary_points(
            meta=meta, n_points=args.n_boundary, rng=rng, include_caps=True,
        )

        # Build sample dict (same format as FEM-based generation)
        sample = {
            "points": points,
            "u": u,
            "sigma": sigma,
            "tissue_labels": tissue_labels,
            "electrode_position": source_position.astype(np.float32),
            "electrode_cyl": np.array([r_n, theta_deg, z_n], dtype=np.float32),
            "source_field": source_field,
            "boundary_points": boundary_points.astype(np.float32),
            "boundary_normals": boundary_normals.astype(np.float32),
            "pennation_angle": np.array([args.pennation_angle], dtype=np.float32),
            "r_skin": np.array([r_skin], dtype=np.float32),
            "length": np.array([length], dtype=np.float32),
        }

        # Save
        sample_id = f"elec_{idx:06d}"
        out_path = out_dir / f"{sample_id}.npz"
        save_pointcloud_sample(sample, out_path)

        manifest.append({
            "sample_id": sample_id,
            "file": f"{sample_id}.npz",
            "electrode_cyl": [float(r_n), float(theta_deg), float(z_n)],
            "electrode_position": source_position.tolist(),
            "n_points": args.n_points,
            "n_boundary": args.n_boundary,
        })

        sample_time = time.time() - sample_start
        if (idx + 1) % 50 == 0 or idx == 0 or idx == n_electrodes - 1:
            f_max = float(source_field.max())
            print(f"  [{idx+1:4d}/{n_electrodes}] theta={theta_deg:6.1f} deg "
                  f"z={z_n:.3f} f_max={f_max:.6f} ({sample_time:.3f}s)")

    # Save manifest
    manifest_data = {
        "dataset_type": "pointcloud_electrode_sweep",
        "generation_mode": "analytical_pinn_only",
        "meta": str(meta_path),
        "n_samples": len(manifest),
        "n_interior_points": args.n_points,
        "n_boundary_points": args.n_boundary,
        "sampling_strategy": args.sampling_strategy,
        "electrode_layout": {
            "type": args.electrode_layout,
            **({"grid_z": args.grid_z, "grid_theta": args.grid_theta}
               if args.electrode_layout == "grid" else {}),
        },
        "generation_params": {
            "seed": args.seed,
            "z_margin": args.z_margin,
            "r_norm": args.r_norm,
            "source_mode": args.source_mode,
            "source_sigma": args.source_sigma,
            "pennation_angle": args.pennation_angle,
            "return_mode": "volumetric",
            "include_pinn_data": True,
            "concentration_radius": args.concentration_radius,
            "concentration_fraction": args.concentration_fraction,
            "u_field": "zeros (no FEM solve)",
        },
        "samples": manifest,
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    total_time = time.time() - t_start
    print()
    print("=" * 80)
    print("FEM-Free dataset generation complete!")
    print(f"  Samples: {len(manifest)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/len(manifest):.3f}s per sample)")
    print(f"  Output: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
