import argparse
import csv
import json
import os
import sys
from pathlib import Path

import gmsh

# Ensure local src/ is on path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emgforge.meshing import build_one_mesh  # type: ignore
from emgforge.sampling import sample_parameters  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="number of meshes to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--factor",
        type=float,
        default=40.0,
        help="radius_multiplicative_factor (e.g. 40 for upper limb)",
    )
    ap.add_argument("--length_ratio_min", type=float, default=6.0)
    ap.add_argument("--length_ratio_max", type=float, default=6.0)
    ap.add_argument("--shape", type=str, default="circle", choices=["circle", "ellipse"])
    ap.add_argument("--ellipse_ratio_min", type=float, default=1.0, help="Ellipse a/b minimum (ellipse only)")
    ap.add_argument("--ellipse_ratio_max", type=float, default=1.0, help="Ellipse a/b maximum (ellipse only)")
    ap.add_argument(
        "--bone_offset_frac_max",
        type=float,
        default=0.0,
        help="Max bone center offset as fraction of r_skin (0 = concentric). Applies to cancellous+cortical.",
    )
    ap.add_argument("--bone_count", type=int, default=1, choices=[1, 2], help="Number of bones (1 or 2)")
    ap.add_argument(
        "--interbone_frac_min",
        type=float,
        default=0.0,
        help="Min separation fraction (scaled by muscle size) for two-bone mode",
    )
    ap.add_argument(
        "--interbone_frac_max",
        type=float,
        default=0.0,
        help="Max separation fraction (scaled by muscle size) for two-bone mode",
    )
    ap.add_argument(
        "--bone2_scale_min",
        type=float,
        default=1.0,
        help="Bone2 cancellous radius scale min (two-bone mode)",
    )
    ap.add_argument(
        "--bone2_scale_max",
        type=float,
        default=1.0,
        help="Bone2 cancellous radius scale max (two-bone mode)",
    )
    ap.add_argument(
        "--z_profile",
        type=str,
        default="constant",
        choices=["constant", "taper"],
        help="Longitudinal profile. 'taper' linearly scales all radii/axes from z=0 to z=L.",
    )
    ap.add_argument("--taper_scale_min", type=float, default=1.0, help="Scale at z=L (taper mode)")
    ap.add_argument("--taper_scale_max", type=float, default=1.0, help="Scale at z=L (taper mode)")
    ap.add_argument("--mesh_char_factor", type=float, default=0.2)
    ap.add_argument(
        "--refine_on", type=str, default="skin", choices=["skin", "all_interfaces"]
    )
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(os.environ.get("DATA_ROOT", "./data"), "generated_meshes"))
    ap.add_argument(
        "--limit", type=int, default=None, help="for small pipeline test, generate only first K"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    meshes_dir = out_dir / "meshes"
    meta_dir = out_dir / "metadata"
    manifest_path = out_dir / "manifest.csv"

    params = sample_parameters(
        n=args.n,
        seed=args.seed,
        radius_multiplicative_factor=args.factor,
        length_ratio_range=(args.length_ratio_min, args.length_ratio_max),
        shape=args.shape,
        ellipse_ratio_range=(args.ellipse_ratio_min, args.ellipse_ratio_max),
        bone_offset_frac_range=(0.0, args.bone_offset_frac_max),
        bone_count=args.bone_count,
        interbone_frac_range=(args.interbone_frac_min, args.interbone_frac_max),
        bone2_scale_range=(args.bone2_scale_min, args.bone2_scale_max),
        z_profile=args.z_profile,
        taper_scale_range=(args.taper_scale_min, args.taper_scale_max),
    )

    if args.limit is not None:
        params = params[: args.limit]

    gmsh.initialize()

    manifest_fields = [
        "sample_id",
        "mesh_file",
        "meta_file",
        "num_nodes",
        "num_elems_3d",
        "generated_seconds",
        "radius_multiplicative_factor",
        "length_ratio",
        "length",
        "radius_canc_bone",
        "thickness_cort_bone",
        "thickness_muscle",
        "thickness_fat",
        "thickness_skin",
        "radius_cort_bone",
        "radius_muscle",
        "radius_fat",
        "radius_skin",
        "shape",
        "ellipse_ratio",
        "bone_offset_x",
        "bone_offset_y",
        "bone_count",
        "bone1_center_x",
        "bone1_center_y",
        "bone2_center_x",
        "bone2_center_y",
        "radius_canc_bone_2",
        "radius_cort_bone_2",
        "z_profile",
        "taper_scale_z1",
        "mesh_char_factor",
        "refine_on",
        "seed",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=manifest_fields)
        writer.writeheader()

        for i, p in enumerate(params):
            sample_id = f"sample_{i:06d}"
            msh_path = meshes_dir / f"{sample_id}.msh"
            json_path = meta_dir / f"{sample_id}.json"

            if msh_path.exists() and json_path.exists():
                if args.verbose:
                    print(f"[skip] {sample_id}")
                with open(json_path, "r") as jf:
                    meta = json.load(jf)
            else:
                if args.verbose:
                    print(f"[gen ] {sample_id}")
                meta = build_one_mesh(
                    p,
                    out_msh=msh_path,
                    out_json=json_path,
                    mesh_char_length_factor=args.mesh_char_factor,
                    refine_on=args.refine_on,
                    verbose=args.verbose,
                )

            row = {
                "sample_id": sample_id,
                "mesh_file": str(msh_path),
                "meta_file": str(json_path),
                "num_nodes": meta["num_nodes"],
                "num_elems_3d": meta["num_elems_3d"],
                "generated_seconds": meta["generated_seconds"],
                "mesh_char_factor": args.mesh_char_factor,
                "refine_on": args.refine_on,
                "seed": args.seed,
            }

            gp = meta["geometry_params"]
            row.update(
                {
                    "radius_multiplicative_factor": gp["radius_multiplicative_factor"],
                    "length_ratio": gp["length_ratio"],
                    "length": gp["length"],
                    "radius_canc_bone": gp["radius_canc_bone"],
                    "thickness_cort_bone": gp["thickness_cort_bone"],
                    "thickness_muscle": gp["thickness_muscle"],
                    "thickness_fat": gp["thickness_fat"],
                    "thickness_skin": gp["thickness_skin"],
                    "radius_cort_bone": gp["radius_cort_bone"],
                    "radius_muscle": gp["radius_muscle"],
                    "radius_fat": gp["radius_fat"],
                    "radius_skin": gp["radius_skin"],
                    "shape": gp.get("shape", "circle"),
                    "ellipse_ratio": gp.get("ellipse_ratio", 1.0),
                    "bone_offset_x": gp.get("bone_offset_x", 0.0),
                    "bone_offset_y": gp.get("bone_offset_y", 0.0),
                    "bone_count": gp.get("bone_count", 1),
                    "bone1_center_x": gp.get("bone1_center_x", gp.get("bone_offset_x", 0.0)),
                    "bone1_center_y": gp.get("bone1_center_y", gp.get("bone_offset_y", 0.0)),
                    "bone2_center_x": gp.get("bone2_center_x", 0.0),
                    "bone2_center_y": gp.get("bone2_center_y", 0.0),
                    "radius_canc_bone_2": gp.get("radius_canc_bone_2", 0.0),
                    "radius_cort_bone_2": gp.get("radius_cort_bone_2", 0.0),
                    "z_profile": gp.get("z_profile", "constant"),
                    "taper_scale_z1": gp.get("taper_scale_z1", 1.0),
                }
            )

            writer.writerow(row)

    gmsh.finalize()
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
