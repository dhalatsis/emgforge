#!/usr/bin/env python3
"""
Generate a reference mesh with average parameter values.

This creates a single mesh with layer thicknesses at the midpoint of their
respective ranges, suitable for neural field training on a fixed geometry.

Output:
    meshes/reference_avg.msh
    metadata/reference_avg.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure local src/ is on path
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def compute_average_parameters(
    factor: float = 40.0,
    length_ratio: float = 6.0,
) -> dict:
    """
    Compute average parameter values from the default LHS ranges.

    Default normalized ranges (from emgforge/sampling/parameters.py):
        radius_canc_bone:    (0.12, 0.25)
        thickness_cort_bone: (0.03, 0.11)
        thickness_muscle:    (0.35, 0.95)
        thickness_fat:       (0.03, 0.50)
        thickness_skin:      (0.01, 0.06)
    """
    # Normalized ranges
    ranges_normalized = {
        "radius_canc_bone": (0.12, 0.25),
        "thickness_cort_bone": (0.03, 0.11),
        "thickness_muscle": (0.35, 0.95),
        "thickness_fat": (0.03, 0.50),
        "thickness_skin": (0.01, 0.06),
    }

    # Compute average normalized values
    avg_norm = {k: (lo + hi) / 2 for k, (lo, hi) in ranges_normalized.items()}

    # Scale by factor
    p = {k: v * factor for k, v in avg_norm.items()}

    # Compute cumulative radii
    r1 = p["radius_canc_bone"]
    r2 = r1 + p["thickness_cort_bone"]
    r3 = r2 + p["thickness_muscle"]
    r4 = r3 + p["thickness_fat"]
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
        description="Generate a reference mesh with average parameter values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--out_dir", type=str,
        default=os.path.join(os.environ.get("DATA_ROOT", "./data"), "generated_meshes"),
        help="Output directory"
    )
    ap.add_argument("--name", type=str, default="reference_avg", help="Mesh name")
    ap.add_argument("--factor", type=float, default=40.0, help="Radius multiplicative factor")
    ap.add_argument("--length_ratio", type=float, default=6.0, help="Length/factor ratio")
    ap.add_argument("--mesh_char_factor", type=float, default=0.2, help="Mesh characteristic length factor")
    ap.add_argument("--refine_on", type=str, default="skin", choices=["skin", "all_interfaces"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    meshes_dir = out_dir / "meshes"
    meta_dir = out_dir / "metadata"

    msh_path = meshes_dir / f"{args.name}.msh"
    json_path = meta_dir / f"{args.name}.json"

    # Compute average parameters
    p = compute_average_parameters(factor=args.factor, length_ratio=args.length_ratio)

    print("=" * 60)
    print(f"Generating reference mesh: {args.name}")
    print("=" * 60)
    print(f"Factor: {args.factor}")
    print(f"Length ratio: {args.length_ratio}")
    print()
    print("Layer dimensions (mm):")
    print(f"  Cancellous bone radius: {p['radius_canc_bone']:.2f}")
    print(f"  Cortical bone radius:   {p['radius_cort_bone']:.2f} (thickness: {p['thickness_cort_bone']:.2f})")
    print(f"  Muscle radius:          {p['radius_muscle']:.2f} (thickness: {p['thickness_muscle']:.2f})")
    print(f"  Fat radius:             {p['radius_fat']:.2f} (thickness: {p['thickness_fat']:.2f})")
    print(f"  Skin radius:            {p['radius_skin']:.2f} (thickness: {p['thickness_skin']:.2f})")
    print(f"  Length:                 {p['length']:.2f}")
    print()

    # Import gmsh and build mesh
    import gmsh
    from emgforge.meshing import build_one_mesh

    gmsh.initialize()

    if msh_path.exists() and json_path.exists():
        print(f"[skip] Mesh already exists: {msh_path}")
        with open(json_path, "r") as f:
            meta = json.load(f)
    else:
        print(f"[gen ] Building mesh...")
        meta = build_one_mesh(
            p,
            out_msh=msh_path,
            out_json=json_path,
            mesh_char_length_factor=args.mesh_char_factor,
            refine_on=args.refine_on,
            verbose=args.verbose,
        )
        print(f"[done] Generated in {meta['generated_seconds']:.2f}s")

    gmsh.finalize()

    print()
    print(f"Mesh file:     {msh_path}")
    print(f"Metadata file: {json_path}")
    print(f"Nodes:         {meta['num_nodes']}")
    print(f"Elements (3D): {meta['num_elems_3d']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
