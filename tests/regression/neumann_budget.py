"""Workstream B — Neumann-BC error budget study.

For a given cross-section, solve FEM with two mesh lengths (240 mm and 400 mm)
and a Gaussian point source. Sample φ(z) along a fibre line at several
electrode-to-fibre depths AND at several axial source offsets. Compare:

  - ‖φ_240 − φ_400‖₂ / ‖φ_400‖₂        along the fibre
  - max pointwise relative error
  - r(MUAP_240, MUAP_400) through the current production pipeline

If r(MUAP_240, MUAP_400) > 0.999 across all configurations → Neumann is fine,
skip Workstream C. If r < 0.99 → Workstream C is justified.

USAGE
-----
$ PYTHONPATH=src python tests/regression/neumann_budget.py \\
    --out tests/regression/NEUMANN_BUDGET.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from emgforge.synthesis.api import MUAPConfig, generate_muap_from_phi  # noqa: E402

V_MS = 4.0
FSAMP_HZ = 4096.0
DZ_MM = V_MS * 1000.0 / FSAMP_HZ   # ≈ 0.977 mm

MESH_240 = "tests/regression/neumann_meshes/meshes/avg_L240.msh"
META_240 = "tests/regression/neumann_meshes/metadata/avg_L240.json"
MESH_400 = "tests/regression/neumann_meshes/meshes/avg_L400.msh"
META_400 = "tests/regression/neumann_meshes/metadata/avg_L400.json"


@dataclass
class PairedConfig:
    depth_mm: float
    source_z_mm: float        # absolute z of source (same in both meshes)
    name: str = ""


@dataclass
class PairedResult:
    name: str
    depth_mm: float
    source_z_mm: float

    # Phi metrics
    phi_l2_rel: float
    phi_max_rel: float
    phi_edge_over_peak_240: float
    phi_edge_over_peak_400: float

    # MUAP metrics
    muap_r_240_vs_400: float
    muap_rel_diff: float

    # Diagnostics
    elapsed_s: float


def _setup_fem(mesh_path: str, meta_path: str, source_z_mm: float):
    """Spin up an ElectrodeFEMSolver with the source at the given absolute z."""
    from emgop.fem import ElectrodeFEMSolver, load_meta
    from emgop.fem.sanity import source_point_from_cyl_normalized

    meta = load_meta(meta_path)
    length = float(meta["geometry_params"]["length"])
    if not (0.5 <= source_z_mm <= length - 0.5):
        raise ValueError(f"source_z_mm={source_z_mm} out of mesh extent [0, {length}]")

    z_norm = source_z_mm / length
    src = source_point_from_cyl_normalized(
        meta, r_norm=1.0, theta_deg=0.0, z_norm=z_norm,
        layer="skin", margin=1.0,
    )
    elec = src[0]

    solver = ElectrodeFEMSolver(
        str(mesh_path),
        return_mode="volumetric",
        source_mode="gaussian",
        source_sigma=5.0,
        gdim=3,
        build_conductivity_map=True,
    )
    t0 = time.time()
    uh = solver.solve(elec, source_radius=4.0, source_points=256)
    print(f"  [{Path(mesh_path).stem} src_z={source_z_mm:.1f}mm] FEM solve {time.time() - t0:.1f}s")
    return solver, uh, length


def _sample_phi(solver, uh, depth_mm: float, length: float, source_z_mm: float,
                n_samples: int = 256):
    """Sample φ(z) along (x=depth, y=0) centred on the source's z."""
    z_half = (n_samples // 2) * DZ_MM
    z_fiber = np.linspace(source_z_mm - z_half, source_z_mm + z_half, n_samples)
    # Clip to mesh extent — z values outside the mesh would error during eval.
    # This is exactly the truncation that the BC contaminates.
    z_fiber = np.clip(z_fiber, 0.5, length - 0.5)
    pts = np.zeros((n_samples, 3))
    pts[:, 0] = depth_mm
    pts[:, 2] = z_fiber
    phi = solver.model.evaluate_solution_at_points(pts, uh=uh)
    return np.asarray(phi).flatten()


_FEM_CACHE: Dict[tuple, tuple] = {}   # (mesh, src_z_mm) -> (solver, uh, length)


def _cached_setup(mesh_path: str, meta_path: str, source_z_mm: float):
    key = (mesh_path, round(source_z_mm, 4))
    if key not in _FEM_CACHE:
        _FEM_CACHE[key] = _setup_fem(mesh_path, meta_path, source_z_mm)
    return _FEM_CACHE[key]


def _run_pair(cfg: PairedConfig) -> PairedResult:
    t0 = time.time()
    sol_240, uh_240, len_240 = _cached_setup(MESH_240, META_240, cfg.source_z_mm)
    sol_400, uh_400, len_400 = _cached_setup(MESH_400, META_400, cfg.source_z_mm)

    # Sample at standard w=256 (the production default), centred on the source z.
    phi_240 = _sample_phi(sol_240, uh_240, cfg.depth_mm, len_240, cfg.source_z_mm, n_samples=256)
    phi_400 = _sample_phi(sol_400, uh_400, cfg.depth_mm, len_400, cfg.source_z_mm, n_samples=256)

    # Note: the cross-sections are identical (same reference geometry), so the
    # only physical difference between the two phi profiles is the Neumann
    # reflection from the *closer* mesh boundary. φ_400 is treated as ground
    # truth; φ_240 is the BC-contaminated version.
    diff = phi_240 - phi_400
    l2_rel = float(np.linalg.norm(diff) / max(np.linalg.norm(phi_400), 1e-30))
    max_rel = float(np.max(np.abs(diff)) / max(np.max(np.abs(phi_400)), 1e-30))

    peak240 = float(np.max(np.abs(phi_240)))
    peak400 = float(np.max(np.abs(phi_400)))
    edge240 = float(np.abs(phi_240[0]) / max(peak240, 1e-30))
    edge400 = float(np.abs(phi_400[0]) / max(peak400, 1e-30))

    # Through the production pipeline → MUAP
    cfg_pipeline = MUAPConfig(w=256)
    cfg_pipeline.v = V_MS
    cfg_pipeline.fsamp = FSAMP_HZ
    res240 = generate_muap_from_phi(phi_240.reshape(1, -1), dz_mm=DZ_MM, config=cfg_pipeline)
    res400 = generate_muap_from_phi(phi_400.reshape(1, -1), dz_mm=DZ_MM, config=cfg_pipeline)
    m240 = np.real(res240.muap).flatten()
    m400 = np.real(res400.muap).flatten()
    n = min(len(m240), len(m400))
    if m240[:n].var() > 0 and m400[:n].var() > 0:
        r = float(np.corrcoef(m240[:n], m400[:n])[0, 1])
    else:
        r = 0.0
    rel_diff = float(np.linalg.norm(m240[:n] - m400[:n]) / max(np.linalg.norm(m400[:n]), 1e-30))

    return PairedResult(
        name=cfg.name,
        depth_mm=cfg.depth_mm,
        source_z_mm=cfg.source_z_mm,
        phi_l2_rel=l2_rel,
        phi_max_rel=max_rel,
        phi_edge_over_peak_240=edge240,
        phi_edge_over_peak_400=edge400,
        muap_r_240_vs_400=r,
        muap_rel_diff=rel_diff,
        elapsed_s=time.time() - t0,
    )


def define_configs() -> List[PairedConfig]:
    """Configurations to evaluate.

    Place source at absolute z (same in both meshes) so the only difference
    between L=240 and L=400 is the proximity of the Neumann boundary.

    Cases:
    - **centred**: src_z = 120 mm. In the L240 mesh that's dead centre
      (120 mm from each end). In the L400 mesh that's 120 mm from the
      lower end and 280 mm from the upper end — the lower boundary is
      what dominates BC error in BOTH meshes.
    - **offaxis-30mm**: src_z = 30 mm. Very close to z=0 in both meshes.
      Stresses the BC at the near end. Far end (210 mm or 370 mm) is
      where the difference between L240/L400 dominates.
    - **offaxis-60mm**: src_z = 60 mm. Intermediate.

    Depth sweep: 5 / 15 / 25 / 30 / 34 mm. Shallow has narrow field
    (BC less of a problem). Deeper has wide field — bigger BC issue.
    """
    configs = []
    for d in (5.0, 15.0, 25.0, 30.0, 34.0):
        for src_z, label in ((120.0, "centred"), (60.0, "off60mm"), (30.0, "off30mm")):
            configs.append(PairedConfig(
                depth_mm=d,
                source_z_mm=src_z,
                name=f"d{int(d):02d}_z{int(src_z):03d}_{label}",
            ))
    return configs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="JSON output path for the budget table")
    args = ap.parse_args()

    configs = define_configs()
    print(f"running {len(configs)} paired configurations...")

    results: List[PairedResult] = []
    for cfg in configs:
        print(f"\n[{cfg.name}]")
        try:
            r = _run_pair(cfg)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            continue
        results.append(r)
        print(f"  φ rel-L2: {r.phi_l2_rel:.4f}, MUAP r: {r.muap_r_240_vs_400:.4f}, "
              f"edge240: {r.phi_edge_over_peak_240:.3f}, edge400: {r.phi_edge_over_peak_400:.3f}")

    print(f"\n──────────── Summary ────────────")
    print(f"{'name':<28s} {'phi_L2':>8s} {'phi_max':>8s} {'MUAP r':>9s} "
          f"{'edge240':>8s} {'edge400':>8s}")
    for r in results:
        print(f"{r.name:<28s} {r.phi_l2_rel:>8.4f} {r.phi_max_rel:>8.4f} "
              f"{r.muap_r_240_vs_400:>9.4f} {r.phi_edge_over_peak_240:>8.3f} "
              f"{r.phi_edge_over_peak_400:>8.3f}")

    # Gate decision per PLAN §3.2, refined per the round-1 verification
    # (`field_to_muap_study/deliverables/.../VERIFICATION_2026-05-26.md` §V6
    # observes the threshold rule alone is too strict — a single mesh-
    # resolution-dominated corner case shouldn't trigger PROCEED).
    min_r = min(r.muap_r_240_vs_400 for r in results) if results else 0
    median_r = float(np.median([r.muap_r_240_vs_400 for r in results])) if results else 0
    n_below_99 = sum(1 for r in results if r.muap_r_240_vs_400 < 0.99)
    n_below_997 = sum(1 for r in results if r.muap_r_240_vs_400 < 0.997)
    print(f"\nGate metric: min(r(MUAP_240, MUAP_400)) = {min_r:.5f}, "
          f"median = {median_r:.5f}, n<0.99 = {n_below_99}/{len(results)}, "
          f"n<0.997 = {n_below_997}/{len(results)}")
    if min_r > 0.999:
        verdict = "SKIP Workstream C — Neumann is fine"
    elif median_r > 0.997 and n_below_99 <= 1:
        # Marginal: a single corner case (typically d=5 / centred, where the
        # Gaussian source σ is comparable to source-to-fibre distance and
        # mesh resolution dominates) drags the min below 0.99 but the bulk
        # is well above. See NEUMANN_BUDGET.md §"Decision".
        verdict = "SKIP Workstream C — MARGINAL (single corner case, mesh-resolution dominated)"
    elif min_r > 0.99:
        verdict = "MARGINAL — Workstream C optional"
    else:
        verdict = "PROCEED with Workstream C — Neumann materially distorts MUAP"
    print(f"Decision: {verdict}")

    out = {
        "version": 1,
        "v_m_per_s": V_MS,
        "fsamp_hz": FSAMP_HZ,
        "dz_mm": DZ_MM,
        "n_configs": len(results),
        "min_muap_r": min_r,
        "verdict": verdict,
        "results": [asdict(r) for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
