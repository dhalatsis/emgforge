"""Build a 400-mm mesh with sample_000000's cross-section (Workstream A.3).

Goal: isolate the length effect from the cross-section effect in
PROBE_L400 — the previous L400 used the reference-average cross-section
(r_skin=48.2), which is far from sample_000000 (r_skin=36.95) and from the
analytical reference (r_skin=40).

Output: tests/regression/neumann_meshes/{meshes,metadata}/sample000_L400.msh/.json

Reproduce: `PYTHONPATH=src python tests/regression/build_extended_sample.py`
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

    # Load sample_000000's geometry
    src_meta = "./data/generated_meshes/metadata/sample_000000.json"
    with open(src_meta) as f:
        meta = json.load(f)
    p = dict(meta["geometry_params"])

    # Swap length from 240 to 400 while keeping the cross-section
    p["length"] = 400.0
    p["length_ratio"] = 10.0   # purely informational

    out_dir = Path("tests/regression/neumann_meshes")
    msh = out_dir / "meshes" / "sample000_L400.msh"
    js = out_dir / "metadata" / "sample000_L400.json"

    if msh.exists() and js.exists():
        print(f"[skip] {msh} exists; nothing to do")
        return

    gmsh.initialize()
    try:
        print(f"Building extended-length sample_000000 mesh...")
        print(f"  r_skin={p['radius_skin']:.2f} mm, length={p['length']:.0f} mm")
        meta_out = build_one_mesh(
            p,
            out_msh=msh,
            out_json=js,
            mesh_char_length_factor=0.25,
            refine_on="skin",
            verbose=True,
        )
        print(f"[done] generated in {meta_out['generated_seconds']:.1f}s, "
              f"{meta_out['num_nodes']} nodes")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
