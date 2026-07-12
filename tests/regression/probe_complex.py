"""Quick probe: run get_adaptive_config() vs MUAPConfig() on COMPLEX meshes.

Tests three non-circular variants that exist locally:
  - two_bone   (circular cross-section, 2 bones) — sample_000000
  - ellipse    (elliptic ratio 1.2, 1 bone)      — sample_000000
  - offcenter  (circular, 1 bone, offset)         — sample_000000

For each mesh:
  - Solve FEM once with the same Gaussian σ=5 source as the main bench
  - Sample φ at a handful of fibre depths
  - Run the production pipeline at (a) default config, (b) adaptive_w_auto
  - Compare both against the analytical 4-layer cylinder reference at the
    same depth (knowing the reference doesn't match the complex geometry —
    the metric is just "does adaptive_w_auto regress vs default?")

Each MUAP is also sanity-checked for well-formedness (PTP, duration, HF
ratio) so we can flag silently-broken outputs.

Usage: PYTHONPATH=src python tests/regression/probe_complex.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from emgforge.synthesis.api import MUAPConfig, generate_muap_from_phi, get_adaptive_config  # noqa: E402

from tests.regression.analytical_ref import AnalyticalCase, analytical_muap  # noqa: E402


V_MS, FSAMP_HZ = 4.0, 4096.0
DZ_MM = V_MS * 1000.0 / FSAMP_HZ

VARIANTS = [
    {
        "name": "two_bone",
        "mesh": "./data/generated_meshes_two_bone/meshes/sample_000000.msh",
        "meta": "./data/generated_meshes_two_bone/metadata/sample_000000.json",
    },
    {
        "name": "ellipse",
        "mesh": "./data/generated_meshes_ellipse/meshes/sample_000000.msh",
        "meta": "./data/generated_meshes_ellipse/metadata/sample_000000.json",
    },
    {
        "name": "offcenter",
        "mesh": "./data/generated_meshes_offcenter/meshes/sample_000000.msh",
        "meta": "./data/generated_meshes_offcenter/metadata/sample_000000.json",
    },
]


def _solve(mesh_path: str, meta_path: str):
    """Solve once at z_norm=0.5, theta=0, r_norm=1.0 (skin top)."""
    from emgforge.fem import ElectrodeFEMSolver, load_meta
    from emgforge.fem.sanity import source_point_from_cyl_normalized

    meta = load_meta(meta_path)
    length = float(meta["geometry_params"]["length"])
    r_skin = float(meta["geometry_params"]["radius_skin"])

    src = source_point_from_cyl_normalized(
        meta, r_norm=1.0, theta_deg=0.0, z_norm=0.5, layer="skin", margin=1.0,
    )
    solver = ElectrodeFEMSolver(
        mesh_path, return_mode="volumetric",
        source_mode="gaussian", source_sigma=5.0,
        gdim=3, build_conductivity_map=True,
    )
    t0 = time.time()
    uh = solver.solve(src[0], source_radius=4.0, source_points=256)
    print(f"  FEM solve {time.time() - t0:.1f}s, r_skin={r_skin:.2f}, length={length:.0f}")
    return solver, uh, length, r_skin


def _sample_phi(solver, uh, depth_mm: float, length: float,
                n: int = 256, theta_deg: float = 0.0,
                extra_extent: bool = False) -> Tuple[np.ndarray, float]:
    dz = DZ_MM
    z_center = length * 0.5
    if extra_extent:
        # Cover up to 1024 samples or the mesh extent
        n_max = min(1024, int((length - 1.0) / dz))
        if n_max % 2:
            n_max -= 1
        n_use = max(n, n_max)
    else:
        n_use = n
    z_half = (n_use // 2) * dz
    z = np.linspace(z_center - z_half, z_center + z_half, n_use)
    z = np.clip(z, 0.5, length - 0.5)
    theta = math.radians(theta_deg)
    x = depth_mm * math.cos(theta)
    y = depth_mm * math.sin(theta)
    pts = np.zeros((n_use, 3))
    pts[:, 0] = x
    pts[:, 1] = y
    pts[:, 2] = z
    phi = solver.model.evaluate_solution_at_points(pts, uh=uh)
    return np.asarray(phi).flatten(), dz


def _muap_metrics(muap: np.ndarray, dz_ms: float) -> dict:
    """Quick well-formedness check on an arbitrary MUAP."""
    ptp = float(muap.max() - muap.min())
    if ptp <= 0:
        return {"ptp": 0.0, "duration_ms": 0.0, "hf_ratio": 0.0, "ok": False}
    above = np.abs(muap) > 0.05 * ptp
    if above.any():
        first = int(np.argmax(above))
        last = len(above) - 1 - int(np.argmax(above[::-1]))
        dur = (last - first) * dz_ms
    else:
        dur = 0.0
    spec = np.abs(np.fft.rfft(muap - muap.mean()))
    f = np.fft.rfftfreq(len(muap), d=dz_ms / 1000.0)
    total = (spec ** 2).sum()
    hf = float((spec[f > 500] ** 2).sum() / total) if total > 0 else 0.0
    # "OK" heuristic: PTP > epsilon, duration between 2 and 20 ms, HF < 0.5.
    ok = (ptp > 1e-12) and (2.0 < dur < 25.0) and (hf < 0.5)
    return {"ptp": ptp, "duration_ms": dur, "hf_ratio": hf, "ok": bool(ok)}


@dataclass
class ComplexResult:
    variant: str
    depth_mm: float
    theta_deg: float

    # default config
    w_default: int
    r_default_vs_anal: float
    metrics_default: dict

    # adaptive_w_auto
    w_auto: int
    r_auto_vs_anal: float
    metrics_auto: dict

    # cross-comparison: default vs auto on the SAME geometry
    r_default_vs_auto: float


def _run_one(variant: dict, depth_mm: float, theta_deg: float,
              solver, uh, length: float) -> Optional[ComplexResult]:
    # Default: 256 samples, fixed w
    phi_def, dz = _sample_phi(solver, uh, depth_mm, length, n=256, theta_deg=theta_deg)
    cfg_def = MUAPConfig()
    cfg_def.v, cfg_def.fsamp = V_MS, FSAMP_HZ
    res_def = generate_muap_from_phi(phi_def.reshape(1, -1), dz_mm=dz, config=cfg_def)
    m_def = np.real(res_def.muap).flatten()
    w_def = int(res_def.config.w)

    # Auto: extended sampling, adaptive w
    phi_auto, dz = _sample_phi(solver, uh, depth_mm, length, n=256,
                               theta_deg=theta_deg, extra_extent=True)
    cfg_auto = get_adaptive_config()
    cfg_auto.v, cfg_auto.fsamp = V_MS, FSAMP_HZ
    res_auto = generate_muap_from_phi(phi_auto.reshape(1, -1), dz_mm=dz, config=cfg_auto)
    m_auto = np.real(res_auto.muap).flatten()
    w_auto = int(res_auto.config.w)

    # Analytical baselines at matched w (note: cylinder analytical; complex
    # geometry MUAP shouldn't match exactly, that's expected)
    case_ref_def = AnalyticalCase(fiber_depth_mm=depth_mm, w=w_def,
                                   v_m_per_s=V_MS, fsamp_hz=FSAMP_HZ)
    _, m_anal_def = analytical_muap(case_ref_def)

    case_ref_auto = AnalyticalCase(fiber_depth_mm=depth_mm, w=w_auto,
                                    v_m_per_s=V_MS, fsamp_hz=FSAMP_HZ)
    _, m_anal_auto = analytical_muap(case_ref_auto)

    def _r(a, b):
        n = min(len(a), len(b))
        if a[:n].var() <= 0 or b[:n].var() <= 0:
            return 0.0
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    r_def = _r(m_def, m_anal_def)
    r_auto = _r(m_auto, m_anal_auto)

    # Cross-compare default vs auto (on first w_def samples — same time grid)
    if w_def == w_auto:
        r_da = _r(m_def, m_auto)
    else:
        # auto MUAP is longer; align by first w_def samples vs auto[:w_def]
        r_da = _r(m_def[:w_def], m_auto[:w_def])

    dz_ms = 1000.0 / FSAMP_HZ
    return ComplexResult(
        variant=variant["name"],
        depth_mm=depth_mm,
        theta_deg=theta_deg,
        w_default=w_def, r_default_vs_anal=r_def,
        metrics_default=_muap_metrics(m_def, dz_ms),
        w_auto=w_auto, r_auto_vs_anal=r_auto,
        metrics_auto=_muap_metrics(m_auto, dz_ms),
        r_default_vs_auto=r_da,
    )


def main():
    results: List[ComplexResult] = []
    for variant in VARIANTS:
        print(f"\n=== {variant['name']} ===")
        solver, uh, length, r_skin = _solve(variant["mesh"], variant["meta"])

        depths = [d for d in (10.0, 15.0, 20.0, 25.0, 30.0) if d < r_skin - 1.0]
        thetas = (0.0, 45.0, 90.0)

        for d in depths:
            for th in thetas:
                try:
                    r = _run_one(variant, d, th, solver, uh, length)
                except Exception as e:
                    print(f"  d={d}mm th={th:.0f}deg FAILED: {e!r}")
                    continue
                if r is None:
                    continue
                results.append(r)
                ok_d = "✓" if r.metrics_default["ok"] else "✗"
                ok_a = "✓" if r.metrics_auto["ok"] else "✗"
                print(f"  d={d:>4.0f}mm th={th:>3.0f}deg  "
                      f"default(w={r.w_default}) r={r.r_default_vs_anal:+.3f} {ok_d}  "
                      f"auto(w={r.w_auto}) r={r.r_auto_vs_anal:+.3f} {ok_a}  "
                      f"d↔a r={r.r_default_vs_auto:+.3f}")

    print("\n──────────── Summary by variant ────────────")
    by_var = {}
    for r in results:
        by_var.setdefault(r.variant, []).append(r)
    for var, rs in sorted(by_var.items()):
        rd = np.array([r.r_default_vs_anal for r in rs])
        ra = np.array([r.r_auto_vs_anal for r in rs])
        ok_d = sum(1 for r in rs if r.metrics_default["ok"])
        ok_a = sum(1 for r in rs if r.metrics_auto["ok"])
        print(f"  {var:10s} n={len(rs):2d}  "
              f"default r mean {rd.mean():.3f} ±{rd.std():.3f}  ok={ok_d}/{len(rs)}  "
              f"auto r mean {ra.mean():.3f} ±{ra.std():.3f}  ok={ok_a}/{len(rs)}  "
              f"Δr mean {(ra - rd).mean():+.4f}")

    out = Path("tests/regression/PROBE_COMPLEX.json")
    with open(out, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
