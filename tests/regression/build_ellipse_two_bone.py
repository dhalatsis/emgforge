"""Build an ELLIPSE + TWO-BONE mesh by combining the cross-section of
generated_meshes_ellipse/sample_000000 with the bone placement of
generated_meshes_two_bone/sample_000000.

Output:
  tests/regression/neumann_meshes/meshes/ellipse_two_bone.msh
  tests/regression/neumann_meshes/metadata/ellipse_two_bone.json

Usage:  PYTHONPATH=src python tests/regression/build_ellipse_two_bone.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))


def main():
    import gmsh
    from emgforge.meshing import build_one_mesh

    ELLIPSE_META = "./data/generated_meshes_ellipse/metadata/sample_000000.json"
    TWOBONE_META = "./data/generated_meshes_two_bone/metadata/sample_000000.json"

    with open(ELLIPSE_META) as f:
        ell = json.load(f)["geometry_params"]
    with open(TWOBONE_META) as f:
        tb = json.load(f)["geometry_params"]

    ratio = float(ell["ellipse_ratio"])  # 1.2

    # Compute bone-2 ellipse parameters from the two-bone (circular) radii,
    # applying the ellipse aspect ratio so they're consistent with the layers.
    r_canc2 = float(tb["radius_canc_bone_2"])
    r_cort2 = float(tb["radius_cort_bone_2"])
    a_canc2, b_canc2 = r_canc2 * ratio, r_canc2
    a_cort2, b_cort2 = r_cort2 * ratio, r_cort2

    p = {
        "shape": "ellipse",
        "bone_count": 2,
        "ellipse_ratio": ratio,
        "radius_multiplicative_factor": 40.0,
        "length_ratio": 6.0,
        "length": 240.0,

        # Layers — ellipse (a, b) from the ellipse sample
        "a_canc_bone": float(ell["a_canc_bone"]),
        "b_canc_bone": float(ell["b_canc_bone"]),
        "a_cort_bone": float(ell["a_cort_bone"]),
        "b_cort_bone": float(ell["b_cort_bone"]),
        "a_muscle":    float(ell["a_muscle"]),
        "b_muscle":    float(ell["b_muscle"]),
        "a_fat":       float(ell["a_fat"]),
        "b_fat":       float(ell["b_fat"]),
        "a_skin":      float(ell["a_skin"]),
        "b_skin":      float(ell["b_skin"]),

        # Bone 1 — keep at the ellipse layer's center (matches existing
        # ellipse sample_000000), aligned with z-axis through (0,0).
        "bone1_center_x": float(tb["bone1_center_x"]),
        "bone1_center_y": float(tb["bone1_center_y"]),

        # Bone 2 — borrowed placement from two_bone sample
        "bone2_center_x": float(tb["bone2_center_x"]),
        "bone2_center_y": float(tb["bone2_center_y"]),
        "a_canc_bone_2": a_canc2,
        "b_canc_bone_2": b_canc2,
        "a_cort_bone_2": a_cort2,
        "b_cort_bone_2": b_cort2,

        # Carry over diagnostic info
        "radius_canc_bone": float(ell["radius_canc_bone"]),
        "radius_cort_bone": float(ell["radius_cort_bone"]),
        "radius_muscle":    float(ell["radius_muscle"]),
        "radius_fat":       float(ell["radius_fat"]),
        "radius_skin":      float(ell["radius_skin"]),
        "radius_canc_bone_2": r_canc2,
        "radius_cort_bone_2": r_cort2,
    }

    out_dir = Path("tests/regression/neumann_meshes")
    msh = out_dir / "meshes" / "ellipse_two_bone.msh"
    js = out_dir / "metadata" / "ellipse_two_bone.json"

    if msh.exists() and js.exists():
        print(f"[skip] {msh} exists; nothing to do")
        return

    out_dir.joinpath("meshes").mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("metadata").mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        print("Building ellipse + two-bone mesh...")
        print(f"  layers: a_skin={p['a_skin']:.2f}, b_skin={p['b_skin']:.2f}, "
              f"a_musc={p['a_muscle']:.2f}, b_musc={p['b_muscle']:.2f}")
        print(f"  bone1: center=({p['bone1_center_x']:.2f}, {p['bone1_center_y']:.2f})  "
              f"a={p['a_canc_bone']:.2f}/{p['a_cort_bone']:.2f}")
        print(f"  bone2: center=({p['bone2_center_x']:.2f}, {p['bone2_center_y']:.2f})  "
              f"a={a_canc2:.2f}/{a_cort2:.2f}")
        meta_out = build_one_mesh(
            p, out_msh=msh, out_json=js,
            mesh_char_length_factor=0.25, refine_on="skin", verbose=True,
        )
        print(f"[done] {meta_out['generated_seconds']:.1f}s, "
              f"{meta_out['num_nodes']} nodes")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
