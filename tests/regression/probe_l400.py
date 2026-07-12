"""Quick probe: run the still-failing challenging cases through the L400 mesh
(built by Workstream B) and compare against the L240 sample_000000 result.

Question: does a wider, reference-cross-section FEM mesh recover any of the
17 FEM-side challenging failures that the pipeline-side fixes (adaptive_w +
auto-smoothing) couldn't?
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from emgforge.synthesis.api import generate_muap_from_phi, get_adaptive_config  # noqa: E402

from tests.regression.analytical_ref import AnalyticalCase, analytical_muap  # noqa: E402

MESH_L240 = "./data/generated_meshes/meshes/sample_000000.msh"
META_L240 = "./data/generated_meshes/metadata/sample_000000.json"
# Use the sample_000000-cross-section L400 mesh so length is the only delta.
MESH_L400 = "tests/regression/neumann_meshes/meshes/sample000_L400.msh"
META_L400 = "tests/regression/neumann_meshes/metadata/sample000_L400.json"

V_MS, FSAMP_HZ = 4.0, 4096.0
DZ_MM = V_MS * 1000.0 / FSAMP_HZ


def _make_solver(mesh_path: str, meta_path: str):
    from emgforge.fem import ElectrodeFEMSolver, load_meta
    from emgforge.fem.sanity import source_point_from_cyl_normalized

    meta = load_meta(meta_path)
    length = float(meta["geometry_params"]["length"])
    src = source_point_from_cyl_normalized(
        meta, r_norm=1.0, theta_deg=0.0, z_norm=0.5, layer="skin", margin=1.0,
    )
    solver = ElectrodeFEMSolver(
        mesh_path, return_mode="volumetric",
        source_mode="gaussian", source_sigma=5.0, gdim=3, build_conductivity_map=True,
    )
    t0 = time.time()
    uh = solver.solve(src[0], source_radius=4.0, source_points=256)
    print(f"  solve {time.time() - t0:.1f}s")
    return solver, uh, length


def _sample(solver, uh, depth_mm, length, n=256, v=4.0, fs=4096.0):
    dz = v * 1000.0 / fs
    z_center = length * 0.5
    # Sample as wide as the mesh allows, capped at 1024 samples
    n_max = min(1024, int((length - 1.0) / dz))
    if n_max % 2:
        n_max -= 1
    n_use = max(n, n_max)
    z_half = (n_use // 2) * dz
    z_fiber = np.linspace(z_center - z_half, z_center + z_half, n_use)
    z_fiber = np.clip(z_fiber, 0.5, length - 0.5)
    pts = np.zeros((n_use, 3))
    pts[:, 0] = depth_mm
    pts[:, 2] = z_fiber
    phi = solver.model.evaluate_solution_at_points(pts, uh=uh)
    return np.asarray(phi).flatten(), dz


# Failing-case definitions from the bench
FAILING = [
    # (name, depth_mm, L1, L2, v, fsamp)
    ("B_too_shallow_d5",      5.0, 60, 60, 4.0, 4096.0),
    ("B_too_shallow_d6",      6.0, 60, 60, 4.0, 4096.0),
    ("B_shallow_fs2048",     15.0, 60, 60, 4.0, 2048.0),
    ("B_shallow_fs8192",     15.0, 60, 60, 4.0, 8192.0),
    ("B_deep_d24",           24.0, 60, 60, 4.0, 4096.0),
    ("B_deep_d27",           27.0, 60, 60, 4.0, 4096.0),
    ("B_deep_d31",           31.0, 60, 60, 4.0, 4096.0),
    ("B_deep_d33",           33.0, 60, 60, 4.0, 4096.0),
    ("B_deep_d335h",         33.5, 60, 60, 4.0, 4096.0),
    ("B_xtreme_d35",         35.0, 60, 60, 4.0, 4096.0),
    ("B_xtreme_d355h",       35.5, 60, 60, 4.0, 4096.0),
    ("B_xtreme_d36",         36.0, 60, 60, 4.0, 4096.0),
    ("B_xtreme_d365h",       36.5, 60, 60, 4.0, 4096.0),
    ("B_deep_fs2048_d20",    20.0, 60, 60, 4.0, 2048.0),
    ("B_deep_fs8192_d20",    20.0, 60, 60, 4.0, 8192.0),
    ("B_deep_fs2048_d30",    30.0, 60, 60, 4.0, 2048.0),
    ("B_deep_fs8192_d30",    30.0, 60, 60, 4.0, 8192.0),
]


def _run_one(name, depth, L1, L2, v, fs, sol_240, uh_240, len_240,
             sol_400, uh_400, len_400):
    # Tier B is constrained to depth ≤ skin radius. The L400 mesh now shares
    # sample_000000's cross-section (r_skin ≈ 36.95 mm), so any depth that
    # works in L240 also works in L400.
    skin_400 = 36.95
    if depth > skin_400 - 1.0:
        return None

    cfg = get_adaptive_config()
    cfg.v = v
    cfg.fsamp = fs

    # L240: skip if depth > skin radius
    phi240 = None
    if depth <= 35.95:
        phi240, dz240 = _sample(sol_240, uh_240, depth, len_240, v=v, fs=fs)
        res240 = generate_muap_from_phi(phi240.reshape(1, -1), dz_mm=dz240, config=cfg)
        m240 = np.real(res240.muap).flatten()
        w240 = res240.config.w
    else:
        m240, w240 = None, None

    # L400
    phi400, dz400 = _sample(sol_400, uh_400, depth, len_400, v=v, fs=fs)
    res400 = generate_muap_from_phi(phi400.reshape(1, -1), dz_mm=dz400, config=cfg)
    m400 = np.real(res400.muap).flatten()
    w400 = res400.config.w

    # Analytical references at each respective w
    if w240 is not None:
        case_ref240 = AnalyticalCase(
            fiber_depth_mm=depth, L1_mm=float(L1), L2_mm=float(L2),
            v_m_per_s=v, fsamp_hz=fs, w=int(w240),
        )
        _, mref240 = analytical_muap(case_ref240)
        n = min(len(m240), len(mref240))
        r_240 = float(np.corrcoef(m240[:n], mref240[:n])[0, 1])
    else:
        r_240 = None

    case_ref400 = AnalyticalCase(
        fiber_depth_mm=depth, L1_mm=float(L1), L2_mm=float(L2),
        v_m_per_s=v, fsamp_hz=fs, w=int(w400),
    )
    _, mref400 = analytical_muap(case_ref400)
    n = min(len(m400), len(mref400))
    r_400 = float(np.corrcoef(m400[:n], mref400[:n])[0, 1])

    return {
        "name": name, "depth": depth,
        "w240": w240, "r_240": r_240,
        "w400": w400, "r_400": r_400,
        "delta": (r_400 - r_240) if (r_240 is not None) else None,
    }


def main():
    print("Spinning up L240 (sample_000000)...")
    sol_240, uh_240, len_240 = _make_solver(MESH_L240, META_L240)
    print(f"  len={len_240}, r_skin~36.95")
    print("\nSpinning up L400 (avg ref)...")
    sol_400, uh_400, len_400 = _make_solver(MESH_L400, META_L400)
    print(f"  len={len_400}, r_skin~48.2")

    print(f"\n{'name':<25s} {'depth':>6s} {'w240':>5s} {'r240':>8s} {'w400':>5s} {'r400':>8s} {'Δr':>8s}")
    results = []
    for spec in FAILING:
        r = _run_one(*spec, sol_240, uh_240, len_240, sol_400, uh_400, len_400)
        if r is None:
            print(f"{spec[0]:<25s} {spec[1]:>6.1f}  (depth exceeds L400 skin)")
            continue
        results.append(r)
        r240s = f"{r['r_240']:.4f}" if r['r_240'] is not None else "  N/A"
        d_s = f"{r['delta']:+.4f}" if r['delta'] is not None else "  N/A"
        print(f"{r['name']:<25s} {r['depth']:>6.1f} {str(r['w240']):>5s} {r240s:>8s} "
              f"{r['w400']:>5d} {r['r_400']:>8.4f} {d_s:>8s}")

    out = Path("tests/regression/PROBE_L400.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
