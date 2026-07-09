#!/usr/bin/env python3
"""
Generate point cloud training dataset with varying fat (adipose) thickness.

Unlike electrode and pennation sweeps, this requires generating a separate
mesh for each fat thickness value because the geometry changes.

Pipeline per sample:
    1. Compute geometry params with overridden fat thickness
    2. Build mesh via gmsh
    3. Solve FEM with fixed electrode
    4. Sample point cloud

Output format: same .npz as scripts 06/07 with added 'fat_thickness' field.
Suitable for training: u(x, y, z | fat_thickness)

Usage:
    export PYTHONPATH=src
    python scripts/08_generate_pointcloud_fat.py \\
        --out_dir data/fat_sweep_10 \\
        --n_thicknesses 10 \\
        --fat_min 1.2 --fat_max 20.0 \\
        --n_points 5000 --n_boundary 1000
"""

import argparse
import json
import sys
import time
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def compute_params_with_fat(fat_thickness_mm: float, factor: float = 40.0, length_ratio: float = 6.0) -> dict:
    """
    Compute geometry parameters with a specific fat thickness.

    All other layers use average values from the LHS sampling ranges.
    fat_thickness_mm is the actual thickness in mm (already scaled).
    """
    ranges_normalized = {
        "radius_canc_bone": (0.12, 0.25),
        "thickness_cort_bone": (0.03, 0.11),
        "thickness_muscle": (0.35, 0.95),
        "thickness_skin": (0.01, 0.06),
    }

    avg_norm = {k: (lo + hi) / 2 for k, (lo, hi) in ranges_normalized.items()}
    p = {k: v * factor for k, v in avg_norm.items()}

    # Override fat thickness
    p["thickness_fat"] = fat_thickness_mm

    # Cumulative radii
    r1 = p["radius_canc_bone"]
    r2 = r1 + p["thickness_cort_bone"]
    r3 = r2 + p["thickness_muscle"]
    r4 = r3 + fat_thickness_mm
    r5 = r4 + p["thickness_skin"]

    length = factor * length_ratio

    p.update({
        "shape": "circle",
        "radius_multiplicative_factor": factor,
        "length_ratio": length_ratio,
        "ellipse_ratio": 1.0,
        "length": length,
        "radius_cort_bone": r2,
        "radius_muscle": r3,
        "radius_fat": r4,
        "radius_skin": r5,
    })

    return p


def main():
    ap = argparse.ArgumentParser(
        description="Generate point cloud fat thickness sweep dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # Fat sweep
    ap.add_argument("--n_thicknesses", type=int, default=10, help="Number of fat thicknesses")
    ap.add_argument("--fat_min", type=float, default=1.2, help="Minimum fat thickness (mm)")
    ap.add_argument("--fat_max", type=float, default=20.0, help="Maximum fat thickness (mm)")

    # Dataset
    ap.add_argument("--n_points", type=int, default=10000, help="Interior points per sample")
    ap.add_argument("--n_boundary", type=int, default=2000, help="Boundary points (for PINN)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Sampling
    ap.add_argument("--sampling_strategy", type=str, default="uniform",
                    choices=["uniform", "near_electrode", "stratified_tissue", "surface_biased", "mixed"])
    ap.add_argument("--concentration_radius", type=float, default=15.0)
    ap.add_argument("--concentration_fraction", type=float, default=0.3)

    # Source placement (normalized cylindrical)
    ap.add_argument("--source_cyl", type=str, default="1.0,0.0,0.5",
                    help="Source: r_norm,theta_deg,z_norm")
    ap.add_argument("--source_layer", type=str, default="skin", choices=["skin", "muscle"])
    ap.add_argument("--source_margin", type=float, default=1.0)
    ap.add_argument("--source_mode", type=str, default="gaussian", choices=["gaussian", "point"])
    ap.add_argument("--source_sigma", type=float, default=0.1)

    # Mesh
    ap.add_argument("--factor", type=float, default=40.0, help="Radius multiplicative factor")
    ap.add_argument("--mesh_char_factor", type=float, default=0.2, help="Mesh characteristic length factor")

    # PINN
    ap.add_argument("--no_pinn_data", action="store_true")

    args = ap.parse_args()

    source_cyl = tuple(float(x) for x in args.source_cyl.split(","))
    if len(source_cyl) != 3:
        raise SystemExit("--source_cyl must be three comma-separated floats")

    include_pinn_data = not args.no_pinn_data
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Temp directory for intermediate meshes
    tmp_mesh_dir = out_dir / "_tmp_meshes"
    tmp_mesh_dir.mkdir(exist_ok=True)

    # Fat thickness sweep
    thicknesses = np.linspace(args.fat_min, args.fat_max, args.n_thicknesses)

    print("=" * 80)
    print("Point Cloud Fat Thickness Dataset Generator")
    print("=" * 80)
    print(f"Output: {out_dir}")
    print(f"Fat thicknesses: {args.n_thicknesses} from {args.fat_min:.1f} to {args.fat_max:.1f} mm")
    print(f"Points per sample: {args.n_points} interior, {args.n_boundary} boundary")
    print(f"Source: r={source_cyl[0]}, theta={source_cyl[1]}°, z={source_cyl[2]}")
    print()

    # Import heavy deps after arg parsing
    import gmsh
    from emgop.meshing import build_one_mesh
    from emgop.fem import FEMModel, load_meta
    from emgop.fem.sanity import source_point_from_cyl_normalized
    from emgop.pointcloud import generate_pointcloud_sample, save_pointcloud_sample

    manifest = []
    rng = np.random.default_rng(args.seed)

    for idx, fat_mm in enumerate(thicknesses):
        sample_start = time.time()

        sample_id = f"fat_{idx:06d}"
        msh_path = tmp_mesh_dir / f"{sample_id}.msh"
        json_path = tmp_mesh_dir / f"{sample_id}.json"

        # 1. Compute geometry with this fat thickness
        p = compute_params_with_fat(fat_mm, factor=args.factor)

        print(f"\n[{idx+1:4d}/{args.n_thicknesses}] fat={fat_mm:.2f} mm  r_skin={p['radius_skin']:.2f}")

        # 2. Build mesh
        gmsh.initialize()
        meta = build_one_mesh(
            p,
            out_msh=msh_path,
            out_json=json_path,
            mesh_char_length_factor=args.mesh_char_factor,
            refine_on="skin",
            verbose=False,
        )
        gmsh.finalize()

        # 3. Create FEM model and solve
        model = FEMModel(
            str(msh_path),
            gdim=3,
            point_source=(args.source_mode == "point"),
            build_conductivity_map=True,
            source_sigma=args.source_sigma,
        )

        # Compute source position for this geometry
        src = source_point_from_cyl_normalized(
            meta, source_cyl[0], source_cyl[1], source_cyl[2],
            layer=args.source_layer, margin=args.source_margin
        )
        source_position = src[0]

        uh = model.solve_for_point(src)

        # 4. Generate point cloud
        sample = generate_pointcloud_sample(
            model=model,
            uh=uh,
            meta=meta,
            source_position=source_position,
            n_interior_points=args.n_points,
            n_boundary_points=args.n_boundary if include_pinn_data else 0,
            sampling_strategy=args.sampling_strategy,
            seed=rng.integers(0, 2**31),
            ground_position=None,
            pennation_angle=0.0,
            source_sigma=args.source_sigma,
            include_pinn_data=include_pinn_data,
            concentration_radius=args.concentration_radius,
            concentration_fraction=args.concentration_fraction,
        )

        # Add fat-specific metadata
        sample["fat_thickness"] = np.array([fat_mm], dtype=np.float32)

        # Save
        out_path = out_dir / f"{sample_id}.npz"
        save_pointcloud_sample(sample, out_path)

        manifest_entry = {
            "sample_id": sample_id,
            "file": f"{sample_id}.npz",
            "fat_thickness_mm": float(fat_mm),
            "r_skin": float(p["radius_skin"]),
            "r_fat": float(p["radius_fat"]),
            "source_position": source_position.tolist(),
            "source_cyl": list(source_cyl),
            "n_points": args.n_points,
            "n_boundary": args.n_boundary if include_pinn_data else 0,
        }
        manifest.append(manifest_entry)

        sample_time = time.time() - sample_start
        u_min, u_max = sample["u"].min(), sample["u"].max()
        print(f"  u=[{u_min:.6f}, {u_max:.6f}] ({sample_time:.2f}s)")

        del model, uh

    # Clean up temp meshes
    shutil.rmtree(tmp_mesh_dir, ignore_errors=True)

    # Save manifest
    manifest_data = {
        "dataset_type": "pointcloud_fat_sweep",
        "n_samples": len(manifest),
        "n_interior_points": args.n_points,
        "n_boundary_points": args.n_boundary if include_pinn_data else 0,
        "sampling_strategy": args.sampling_strategy,
        "generation_params": {
            "seed": args.seed,
            "fat_min": args.fat_min,
            "fat_max": args.fat_max,
            "n_thicknesses": args.n_thicknesses,
            "source_cyl": list(source_cyl),
            "source_layer": args.source_layer,
            "source_mode": args.source_mode,
            "source_sigma": args.source_sigma,
            "factor": args.factor,
            "include_pinn_data": include_pinn_data,
        },
        "samples": manifest,
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    print()
    print("=" * 80)
    print("Dataset generation complete!")
    print(f"  Samples: {len(manifest)}")
    print(f"  Fat range: {args.fat_min:.1f} — {args.fat_max:.1f} mm")
    print(f"  Output: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
