"""Probe the new ellipse + two-bone mesh.

Two questions:

  Q1. Does adaptive_w_auto match (or improve on) the default config on
      this complex geometry?
  Q2. Does the old high-frequency jaggedness in the MUAP/SFAP — the
      Phase-3 sign-flip / wide-field-truncation symptom — still appear?
      Concretely: measure raw φ(z) at the boundary, MUAP roughness
      (d²/dz² magnitude), and HF energy fraction.

For each fibre depth × angle:

  - Default     (MUAPConfig() — w=256, Butterworth, no edge_taper)
  - Adaptive    (get_adaptive_config() — w adaptive, auto-smoothing)
  - No-smooth   (raw FEM, Butterworth off) — shows the underlying jaggedness
  - Edge-taper  (default + edge_taper=15) — historical crutch

Compare MUAP shapes, compute roughness and HF metrics, plot all.

Usage:
  PYTHONPATH=src python tests/regression/probe_ellipse_two_bone.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from emgforge.synthesis.api import (  # noqa: E402
    MUAPConfig,
    generate_muap_from_phi,
    get_adaptive_config,
    get_truncated_input_config,
)


MESH = "tests/regression/neumann_meshes/meshes/ellipse_two_bone.msh"
META = "tests/regression/neumann_meshes/metadata/ellipse_two_bone.json"
FIG_DIR = Path("tests/regression/figures/ellipse_two_bone")

V_MS, FSAMP_HZ = 4.0, 4096.0
DZ_MM = V_MS * 1000.0 / FSAMP_HZ


def _solve():
    from emgforge.fem import ElectrodeFEMSolver, load_meta
    from emgforge.fem.sanity import source_point_from_cyl_normalized

    meta = load_meta(META)
    length = float(meta["geometry_params"]["length"])
    src = source_point_from_cyl_normalized(
        meta, r_norm=1.0, theta_deg=0.0, z_norm=0.5,
        layer="skin", margin=1.0,
    )
    solver = ElectrodeFEMSolver(
        MESH, return_mode="volumetric",
        source_mode="gaussian", source_sigma=5.0,
        gdim=3, build_conductivity_map=True,
    )
    t0 = time.time()
    uh = solver.solve(src[0], source_radius=4.0, source_points=256)
    print(f"  FEM solve {time.time() - t0:.1f}s")
    return solver, uh, length


def _sample_phi(solver, uh, depth, theta_deg, length, n=256, extra=False):
    dz = DZ_MM
    if extra:
        n_max = min(1024, int((length - 1.0) / dz))
        if n_max % 2: n_max -= 1
        n_use = max(n, n_max)
    else:
        n_use = n
    z_half = (n_use // 2) * dz
    z_center = length * 0.5
    z = np.clip(np.linspace(z_center - z_half, z_center + z_half, n_use), 0.5, length - 0.5)
    th = math.radians(theta_deg)
    pts = np.zeros((n_use, 3))
    pts[:, 0] = depth * math.cos(th)
    pts[:, 1] = depth * math.sin(th)
    pts[:, 2] = z
    phi = solver.model.evaluate_solution_at_points(pts, uh=uh)
    return np.asarray(phi).flatten(), dz


def _roughness(y, dz):
    d2 = (y[2:] - 2 * y[1:-1] + y[:-2]) / (dz ** 2)
    return float(np.sqrt(np.mean(d2 ** 2)))


def _hf_frac(y, dt_ms, hf_hz=500.0):
    spec = np.abs(np.fft.rfft(y - y.mean()))
    f = np.fft.rfftfreq(len(y), d=dt_ms / 1000.0)
    tot = (spec ** 2).sum()
    if tot <= 0: return 0.0
    return float((spec[f > hf_hz] ** 2).sum() / tot)


CONFIGS = {
    "default":     lambda: MUAPConfig(),
    "adaptive":    lambda: get_adaptive_config(),
    "no_smooth":   lambda: MUAPConfig(denoise="none"),
    "edge_taper":  lambda: get_truncated_input_config(),
}


def _run(phi, dz, name):
    cfg = CONFIGS[name]()
    cfg.v, cfg.fsamp = V_MS, FSAMP_HZ
    res = generate_muap_from_phi(phi.reshape(1, -1), dz_mm=dz, config=cfg)
    m = np.real(res.muap).flatten()
    return m, np.asarray(res.t_ms), int(res.config.w)


def main():
    print(f"=== ellipse + two-bone probe ===")
    solver, uh, length = _solve()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    depths = [10.0, 15.0, 20.0, 25.0, 30.0]
    thetas = [0.0, 45.0, 90.0, 135.0, 180.0]
    dt_ms = 1000.0 / FSAMP_HZ

    summary = []

    for d in depths:
        # Skip if depth exceeds smallest semi-axis at any tested theta.
        # Layer b_muscle = 33.7; sample inside this for safety.
        if d >= 33.0:
            continue
        for th in thetas:
            # Use extended-extent sampling so adaptive can grow w
            phi_e, dz = _sample_phi(solver, uh, d, th, length, extra=True)
            phi_short, _ = _sample_phi(solver, uh, d, th, length, extra=False)

            results = {}
            for name in CONFIGS:
                # Default uses the short (256-sample) phi; adaptive/no_smooth use extended
                phi_use = phi_short if name in ("default", "edge_taper") else phi_e
                m, t, w = _run(phi_use, dz, name)
                results[name] = {
                    "muap": m, "t": t, "w": w,
                    "roughness": _roughness(m, dt_ms),
                    "hf_ratio_500hz": _hf_frac(m, dt_ms),
                    "ptp": float(m.max() - m.min()),
                }

            # phi diagnostics (raw)
            phi_peak = float(np.abs(phi_short).max())
            phi_edge = float(np.abs(phi_short[0]))
            phi_edge_ratio = phi_edge / max(phi_peak, 1e-30)
            phi_rough = _roughness(phi_short, dz)

            summary.append({
                "depth_mm": d, "theta_deg": th,
                "phi_edge_over_peak": phi_edge_ratio,
                "phi_roughness": phi_rough,
                "w_default":   results["default"]["w"],
                "w_adaptive":  results["adaptive"]["w"],
                "roughness_default":     results["default"]["roughness"],
                "roughness_adaptive":    results["adaptive"]["roughness"],
                "roughness_no_smooth":   results["no_smooth"]["roughness"],
                "roughness_edge_taper":  results["edge_taper"]["roughness"],
                "hf_default":     results["default"]["hf_ratio_500hz"],
                "hf_adaptive":    results["adaptive"]["hf_ratio_500hz"],
                "hf_no_smooth":   results["no_smooth"]["hf_ratio_500hz"],
                "hf_edge_taper":  results["edge_taper"]["hf_ratio_500hz"],
                "ptp_default":   results["default"]["ptp"],
                "ptp_adaptive":  results["adaptive"]["ptp"],
            })

            # Plot for this (depth, theta)
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))
            ax = axes[0, 0]
            z_axis = (np.arange(len(phi_short)) - len(phi_short) // 2) * dz
            ax.plot(z_axis, phi_short, "k-", lw=0.9)
            ax.set_title(f"raw φ(z)  edge/peak={phi_edge_ratio:.2f}, rough={phi_rough:.2e}",
                          fontsize=9)
            ax.set_xlabel("z (mm)"); ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            ax.plot(results["no_smooth"]["t"], results["no_smooth"]["muap"], "C3-", lw=0.8,
                    label=f"no_smooth (rough={results['no_smooth']['roughness']:.2e}, "
                          f"hf={results['no_smooth']['hf_ratio_500hz']:.3f})")
            ax.legend(fontsize=7); ax.set_title("MUAP — no smoothing"); ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            ax.plot(results["default"]["t"], results["default"]["muap"], "C1-", lw=0.9,
                    label=f"default w={results['default']['w']}")
            ax.plot(results["edge_taper"]["t"], results["edge_taper"]["muap"], "C0--", lw=0.9,
                    label=f"edge_taper w={results['edge_taper']['w']}")
            ax.legend(fontsize=7); ax.set_title("default vs edge_taper crutch"); ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.plot(results["default"]["t"], results["default"]["muap"], "C1-", lw=0.9,
                    label=f"default w={results['default']['w']} (rough={results['default']['roughness']:.2e})")
            ax.plot(results["adaptive"]["t"], results["adaptive"]["muap"], "C2-", lw=0.9,
                    label=f"adaptive w={results['adaptive']['w']} (rough={results['adaptive']['roughness']:.2e})")
            ax.legend(fontsize=7); ax.set_title("default vs adaptive_w_auto"); ax.grid(True, alpha=0.3)

            fig.suptitle(f"ellipse+2bone  d={d:.0f}mm θ={th:.0f}°", y=1.0, fontsize=11)
            fig.tight_layout()
            out = FIG_DIR / f"d{int(d):02d}_th{int(th):03d}.png"
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)

    # Summary table
    print(f"\n{'d':>3s} {'θ':>4s} {'edge/pk':>8s} {'rough_phi':>12s} "
          f"{'w_def':>6s} {'w_ad':>5s} "
          f"{'rough_def':>11s} {'rough_ad':>11s} {'rough_ns':>11s} "
          f"{'hf_def':>8s} {'hf_ad':>8s} {'hf_ns':>8s}")
    for s in summary:
        print(f"{s['depth_mm']:>3.0f} {s['theta_deg']:>4.0f} {s['phi_edge_over_peak']:>8.3f} "
              f"{s['phi_roughness']:>12.2e} "
              f"{s['w_default']:>6d} {s['w_adaptive']:>5d} "
              f"{s['roughness_default']:>11.2e} {s['roughness_adaptive']:>11.2e} {s['roughness_no_smooth']:>11.2e} "
              f"{s['hf_default']:>8.4f} {s['hf_adaptive']:>8.4f} {s['hf_no_smooth']:>8.4f}")

    # Aggregate
    rd = np.array([s["roughness_default"]   for s in summary])
    ra = np.array([s["roughness_adaptive"]  for s in summary])
    rn = np.array([s["roughness_no_smooth"] for s in summary])
    print(f"\nMUAP roughness, geometric mean: "
          f"no_smooth={np.exp(np.log(rn).mean()):.2e}  "
          f"default={np.exp(np.log(rd).mean()):.2e}  "
          f"adaptive={np.exp(np.log(ra).mean()):.2e}")
    print(f"smoothing reduction factor (no_smooth/default): "
          f"{(rn/rd).mean():.1f}x mean")

    out_json = Path("tests/regression/PROBE_ELLIPSE_TWO_BONE.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json}")
    print(f"plots in {FIG_DIR}/")


if __name__ == "__main__":
    main()
