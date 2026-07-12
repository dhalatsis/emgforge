import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure local src/ is on path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emgforge.voxel import voxelize_geometry_and_source  # type: ignore
from emgforge.voxel.voxelize import load_meta  # type: ignore


def parse_source(arg: str):
    vals = [float(x) for x in arg.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("source must be three comma-separated floats: x,y,z")
    return np.array(vals, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, required=True, help="Path to mesh metadata JSON (from mesh gen).")
    ap.add_argument(
        "--source",
        type=parse_source,
        required=False,
        help="Source point in world coords, comma-separated (x,y,z). If omitted, provide --fem_summary with source_point.",
    )
    ap.add_argument("--out", type=str, required=True, help="Output npz path.")
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--nz", type=int, default=192)
    ap.add_argument("--sigma_vox", type=float, default=1.5, help="Gaussian sigma in voxel units.")
    ap.add_argument("--fem_u", type=str, default=None, help="Optional path to FEM solution vector (u.npy).")
    ap.add_argument("--fem_summary", type=str, default=None, help="Optional FEM summary.json containing source_point.")
    ap.add_argument("--mesh", type=str, default=None, help="Optional mesh override; defaults to meta['mesh_file'].")
    ap.add_argument("--fem_source", type=str, default=None, help="Optional FEM source vector (source.npy) for grid sampling.")
    args = ap.parse_args()

    meta = load_meta(Path(args.meta))

    # Derive source from FEM summary if not provided
    source_point = args.source
    if source_point is None and args.fem_summary:
        with open(args.fem_summary, "r") as f:
            summary = json.load(f)
        if "source_point" not in summary:
            raise SystemExit("source_point not found in fem_summary; provide --source explicitly.")
        source_point = np.array(summary["source_point"], dtype=np.float32)
    if source_point is None:
        raise SystemExit("Provide --source or --fem_summary with source_point.")

    fem_u_path = Path(args.fem_u) if args.fem_u else None
    mesh_path = Path(args.mesh) if args.mesh else None
    fem_source_path = Path(args.fem_source) if args.fem_source else None

    if fem_source_path is None and args.fem_summary:
        with open(args.fem_summary, "r") as f:
            summary = json.load(f)
        if summary.get("source_mode") == "gaussian" and summary.get("fem_source"):
            fem_source_path = Path(summary["fem_source"])

    vox = voxelize_geometry_and_source(
        meta=meta,
        source_point=source_point,
        grid_shape=(args.nx, args.ny, args.nz),
        gaussian_sigma_vox=args.sigma_vox,
        fem_u_path=fem_u_path,
        mesh_override=mesh_path,
        fem_source_path=fem_source_path,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **vox,
        meta_path=str(Path(args.meta).resolve()),
        fem_u_path=str(fem_u_path.resolve()) if fem_u_path else "",
        fem_summary=str(Path(args.fem_summary).resolve()) if args.fem_summary else "",
    )

    print(f"Saved voxel grid to {out_path}")


if __name__ == "__main__":
    main()

