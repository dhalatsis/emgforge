import argparse
import sys
from pathlib import Path

# Ensure local src/ is on path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emgforge.fem import run_manifest_entry, run_one  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=str, default=None, help="Path to .msh (single run)")
    ap.add_argument("--meta", type=str, default=None, help="Path to metadata .json (optional)")
    ap.add_argument("--manifest", type=str, default=None, help="CSV manifest with mesh_file/meta_file columns")
    ap.add_argument("--index", type=int, default=0, help="Row index in manifest to run")
    ap.add_argument("--out_dir", type=str, default="./fem_out", help="Output directory for single run")
    ap.add_argument("--out_root", type=str, default="./fem_out_manifest", help="Output root (manifest mode)")
    ap.add_argument("--plots", action="store_true", help="Save slice plots")
    ap.add_argument("--angle", type=float, default=0.0, help="Source angle (radians)")
    ap.add_argument("--eps_mode", type=str, default="skin_frac", choices=["skin_frac", "abs", "r_frac"])
    ap.add_argument("--eps_value", type=float, default=0.15, help="Meaning depends on eps_mode")
    ap.add_argument(
        "--source_mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "point"],
        help="Source type in FEM. gaussian -> volumetric blob; point -> point/distributed electrode.",
    )
    ap.add_argument(
        "--source_sigma",
        type=float,
        default=0.1,
        help="Std dev (mesh units) for Gaussian source (used when source_mode=gaussian).",
    )
    ap.add_argument(
        "--source_cyl",
        type=str,
        default=None,
        help="Normalized cylindrical source r_norm,theta_deg,z_norm (comma-separated). If set, overrides angle/eps* placement.",
    )
    ap.add_argument(
        "--source_layer",
        type=str,
        default="skin",
        choices=["skin", "muscle"],
        help="Clamp source radial placement to this layer band (for source_cyl).",
    )
    ap.add_argument(
        "--source_margin",
        type=float,
        default=0.0,
        help="Margin (mesh units) to stay away from layer boundaries when using source_cyl.",
    )
    # Electrode return mode configuration (new in v2.0.0)
    ap.add_argument(
        "--return_mode",
        type=str,
        default="volumetric",
        choices=["volumetric", "localized"],
        help="Current return mode: volumetric (monopolar) or localized (bipolar). Default: volumetric.",
    )
    ap.add_argument(
        "--ground_mode",
        type=str,
        default="opposite",
        choices=["opposite", "distal", "cylindrical"],
        help="Ground electrode placement mode for localized return. Default: opposite.",
    )
    ap.add_argument(
        "--ground_cyl",
        type=str,
        default=None,
        help="Cylindrical ground coords: 'r_norm,theta_deg,z_norm' (comma-separated). Only used with --ground_mode cylindrical.",
    )
    ap.add_argument(
        "--ground_radius",
        type=float,
        default=5.0,
        help="Ground electrode sampling radius (mesh units). Default: 5.0.",
    )
    ap.add_argument(
        "--ground_points",
        type=int,
        default=256,
        help="Number of points for ground electrode sampling. Default: 256.",
    )
    ap.add_argument(
        "--electrode_inset",
        type=float,
        default=4.0,
        help="Inset from boundaries for electrode placement (mesh units). Default: 4.0.",
    )
    ap.add_argument(
        "--native_point_source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the native (scifem-free) FEniCSx point source. Default: enabled. "
             "Pass --no-native_point_source to use scifem instead (must be installed).",
    )
    # Pennation angle for muscle fiber orientation
    ap.add_argument(
        "--pennation_angle",
        type=float,
        default=0.0,
        help="Pennation angle in degrees for muscle fiber orientation. 0 = Z-aligned (default).",
    )
    args = ap.parse_args()

    if args.mesh is None and args.manifest is None:
        raise SystemExit("Provide either --mesh (single) or --manifest (manifest mode).")

    source_cyl_tuple = None
    if args.source_cyl:
        try:
            vals = [float(x) for x in args.source_cyl.split(",")]
            if len(vals) != 3:
                raise ValueError
            source_cyl_tuple = (vals[0], vals[1], vals[2])
        except ValueError:
            raise SystemExit("--source_cyl must be three comma-separated floats: r_norm,theta_deg,z_norm")

    # Parse ground_cyl if provided
    ground_cyl_tuple = None
    if args.ground_cyl:
        try:
            vals = [float(x) for x in args.ground_cyl.split(",")]
            if len(vals) != 3:
                raise ValueError
            ground_cyl_tuple = (vals[0], vals[1], vals[2])
        except ValueError:
            raise SystemExit("--ground_cyl must be three comma-separated floats: r_norm,theta_deg,z_norm")

    if args.mesh is not None:
        mesh_path = Path(args.mesh)
        meta_path = Path(args.meta) if args.meta else None
        out_dir = Path(args.out_dir)
        run_one(
            mesh_path,
            meta_path,
            out_dir,
            args.plots,
            args.angle,
            args.eps_mode,
            args.eps_value,
            source_mode=args.source_mode,
            source_sigma=args.source_sigma,
            source_cyl=source_cyl_tuple,
            source_layer=args.source_layer,
            source_margin=args.source_margin,
            return_mode=args.return_mode,
            ground_mode=args.ground_mode,
            ground_cyl=ground_cyl_tuple,
            ground_radius=args.ground_radius,
            ground_points=args.ground_points,
            electrode_inset=args.electrode_inset,
            use_native_point_source=args.native_point_source,
            pennation_angle=args.pennation_angle,
        )
        return

    manifest_path = Path(args.manifest)
    out_root = Path(args.out_root)
    run_manifest_entry(
        manifest_path=manifest_path,
        index=args.index,
        out_root=out_root,
        plots=args.plots,
        angle=args.angle,
        eps_mode=args.eps_mode,
        eps_value=args.eps_value,
        source_mode=args.source_mode,
        source_sigma=args.source_sigma,
        source_cyl=source_cyl_tuple,
        source_layer=args.source_layer,
        source_margin=args.source_margin,
        return_mode=args.return_mode,
        ground_mode=args.ground_mode,
        ground_cyl=ground_cyl_tuple,
        ground_radius=args.ground_radius,
        ground_points=args.ground_points,
        electrode_inset=args.electrode_inset,
        use_native_point_source=args.native_point_source,
        pennation_angle=args.pennation_angle,
    )


if __name__ == "__main__":
    main()

