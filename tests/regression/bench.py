"""Regression bench runner — snapshot and compare MUAP outputs across configs.

Drives Step 1.2 / 1.3 of the FEM worker integration plan.

USAGE
-----

# 1. Take the baseline snapshot of the CURRENT pipeline (no changes yet):
$ PYTHONPATH=src python tests/regression/bench.py snapshot \\
      --config default \\
      --out tests/regression/snapshots/baseline.json

# 2. After a code change, re-snapshot and compare:
$ PYTHONPATH=src python tests/regression/bench.py snapshot \\
      --config default \\
      --out tests/regression/snapshots/post_change.json
$ PYTHONPATH=src python tests/regression/bench.py compare \\
      --reference tests/regression/snapshots/baseline.json \\
      --new tests/regression/snapshots/post_change.json

# 3. Try a non-default config on the existing cases:
$ PYTHONPATH=src python tests/regression/bench.py snapshot \\
      --config edge_taper \\
      --out tests/regression/snapshots/edge_taper.json

CONFIGS
-------
Configs are named keys in CONFIGS below. Each one is a MUAPConfig flavour.

Run-time
--------
Tier A cases: ~200 ms each (no FEM). 65 cases → ~15s.
Tier B cases: ~50 ms each AFTER one ~10s FEM setup. 134 cases → ~20s.
Total cold run: ~45s.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sure src/ and repo root are importable
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from muap_generator.api import MUAPConfig, generate_muap_from_phi  # noqa: E402

from tests.regression.analytical_ref import (  # noqa: E402
    AnalyticalCase,
    analytical_muap,
    analytical_phi_along_fibre,
)
from tests.regression.cases import TestCase, define_cases  # noqa: E402


MESH_PATH = "./data/generated_meshes/meshes/sample_000000.msh"
META_PATH = "./data/generated_meshes/metadata/sample_000000.json"

# Default snapshot format version. Bump if format changes.
SNAPSHOT_VERSION = 1


# ---------------------------------------------------------------------------
# Configs we can run cases through
# ---------------------------------------------------------------------------

def _config_default() -> MUAPConfig:
    """Current production defaults (Butterworth c=0.03 o=2, w=256, no edge_taper)."""
    return MUAPConfig()


def _config_edge_taper(n: int = 15) -> MUAPConfig:
    """Production defaults + edge_taper (the recipe-with-crutch from Phase 5.6)."""
    cfg = MUAPConfig()
    cfg.edge_taper = n
    return cfg


def _config_no_smoothing() -> MUAPConfig:
    """No smoothing — bare Fourier pipeline. Baseline for ablation."""
    cfg = MUAPConfig()
    cfg.smoothing_method = "none"
    return cfg


def _config_no_smoothing_edge_taper(n: int = 15) -> MUAPConfig:
    cfg = _config_no_smoothing()
    cfg.edge_taper = n
    return cfg


def _config_adaptive_w() -> MUAPConfig:
    """Workstream A: adaptive w via field-decay heuristic. Smoothing kept on."""
    cfg = MUAPConfig()
    cfg.w = None       # triggers choose_w
    cfg.w_min = 256
    cfg.w_max = 1024
    return cfg


def _config_adaptive_w_taper() -> MUAPConfig:
    """Adaptive w + edge_taper. Crutch layered on top of principled fix."""
    cfg = _config_adaptive_w()
    cfg.edge_taper = 15
    return cfg


def _config_adaptive_w_no_smoothing() -> MUAPConfig:
    """Adaptive w + no smoothing. Isolates the w effect from smoothing."""
    cfg = _config_adaptive_w()
    cfg.smoothing_method = "none"
    return cfg


def _config_adaptive_w_auto() -> MUAPConfig:
    """Adaptive w + auto smoothing. HF-fraction-aware smoothing.

    Detects high-frequency content in the input phi: if HF > 1e-10, applies
    Butterworth (good for noisy FEM); else skips smoothing (clean analytical).
    Designed to recover Tier A operator-consistency (r→1.0) while keeping
    FEM resilience.
    """
    cfg = _config_adaptive_w()
    cfg.smoothing_method = "auto"
    return cfg


CONFIGS: Dict[str, "ConfigBuilder"] = {
    "default":                     _config_default,
    "edge_taper":                  _config_edge_taper,
    "no_smoothing":                _config_no_smoothing,
    "no_smoothing_taper":          _config_no_smoothing_edge_taper,
    "adaptive_w":                  _config_adaptive_w,
    "adaptive_w_taper":            _config_adaptive_w_taper,
    "adaptive_w_no_smoothing":     _config_adaptive_w_no_smoothing,
    "adaptive_w_auto":             _config_adaptive_w_auto,
}


ConfigBuilder = callable


def _to_muap_config(case: TestCase, base_cfg: MUAPConfig) -> MUAPConfig:
    """Apply case-specific overrides to the base config.

    Note: ``case.w`` is only applied when ``base_cfg.w`` is itself an integer.
    If the base config opted into adaptive selection (w=None), preserve that
    — the case's `w=256` is a legacy default, not an instruction.
    """
    cfg = MUAPConfig(**{**asdict(base_cfg)})
    cfg.v = case.v_m_per_s
    cfg.fsamp = case.fsamp_hz
    if base_cfg.w is not None:
        cfg.w = case.w
    cfg.len1_mm = case.L1_mm
    cfg.len2_mm = case.L2_mm
    return cfg


# ---------------------------------------------------------------------------
# FEM solver (cached, single global setup)
# ---------------------------------------------------------------------------

class FEMContext:
    """Cached one-time FEM setup. Use ``get_phi_at_depth`` per case."""

    def __init__(self):
        from emgop.fem import ElectrodeFEMSolver, load_meta
        from emgop.fem.sanity import source_point_from_cyl_normalized

        meta = load_meta(META_PATH)
        gp = meta["geometry_params"]
        self.length = float(gp["length"])
        self.r_skin = float(gp["radius_skin"])
        self.z_center = self.length * 0.5

        source_pt = source_point_from_cyl_normalized(
            meta, r_norm=1.0, theta_deg=0.0, z_norm=0.5,
            layer="skin", margin=1.0,
        )
        self.elec_pos = source_pt[0]

        self.solver = ElectrodeFEMSolver(
            str(MESH_PATH),
            return_mode="volumetric",
            source_mode="gaussian",
            source_sigma=5.0,
            gdim=3,
            build_conductivity_map=True,
        )

        t0 = time.time()
        self.uh = self.solver.solve(
            self.elec_pos, source_radius=4.0, source_points=256,
        )
        self.solve_time_s = time.time() - t0

    def get_phi_at_depth(
        self, depth_mm: float, w: int, v_m_per_s: float, fsamp_hz: float,
        theta_deg: float = 0.0,
        max_extent_mm: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Sample φ(z) along a fibre line at given (depth, theta).

        Returns ``n`` samples spaced by ``dz = v · 1000 / fsamp`` covering up
        to ``max_extent_mm`` (default: as much as the mesh allows, capped by
        ``w · dz`` if ``max_extent_mm`` is None).

        When ``max_extent_mm`` exceeds ``w·dz``, the returned array has MORE
        than ``w`` samples — the downstream pipeline will then ``resample_centered_line``
        from this denser native sampling onto the adaptive ``w``-sized grid.
        """
        dz_mm = v_m_per_s * 1000.0 / fsamp_hz
        # Default extent = the legacy w·dz window
        extent_target = float(w * dz_mm) if max_extent_mm is None else float(max_extent_mm)
        # Clip extent to what the mesh actually provides
        usable_half = self.z_center - 0.5  # margin from z=0
        extent_max = 2.0 * min(usable_half, self.length - self.z_center - 0.5)
        extent = min(extent_target, extent_max)
        n_samples = max(int(round(extent / dz_mm)), w)
        # Keep n even so the centered-line layout is clean
        if n_samples % 2 == 1:
            n_samples += 1
        z_half = (n_samples // 2) * dz_mm
        z_fiber = np.linspace(self.z_center - z_half, self.z_center + z_half, n_samples)
        z_fiber = np.clip(z_fiber, 0.5, self.length - 0.5)
        theta_rad = math.radians(theta_deg)
        x = depth_mm * math.cos(theta_rad)
        y = depth_mm * math.sin(theta_rad)
        pts = np.zeros((n_samples, 3))
        pts[:, 0] = x
        pts[:, 1] = y
        pts[:, 2] = z_fiber
        phi = self.solver.model.evaluate_solution_at_points(pts, uh=self.uh)
        return np.asarray(phi).flatten(), dz_mm


# ---------------------------------------------------------------------------
# Compute one case
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    tier: str
    category: str
    tag: str
    target_r: float
    config_name: str

    # Numerical outputs
    t_ms: List[float]
    muap_prod: List[float]    # production pipeline output
    muap_ref:  List[float]    # analytical reference
    phi_dz_mm: float
    phi_peak: float
    phi_edge_over_peak: float

    # Derived metrics
    r_vs_ref: float
    rmse_vs_ref: float
    rel_diff_vs_ref: float
    ptp_prod: float
    ptp_ref: float
    elapsed_s: float

    # Case parameters echo (for downstream inspection)
    case_params: dict = field(default_factory=dict)


def _metrics(muap_prod: np.ndarray, muap_ref: np.ndarray) -> dict:
    n = min(len(muap_prod), len(muap_ref))
    a = muap_prod[:n]
    b = muap_ref[:n]
    nrm_b = float(np.linalg.norm(b))
    rel_diff = float(np.linalg.norm(a - b) / nrm_b) if nrm_b > 0 else float("inf")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    var_a = a.var()
    var_b = b.var()
    if var_a > 0 and var_b > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    else:
        r = 0.0
    return {"r": r, "rmse": rmse, "rel_diff": rel_diff,
            "ptp_prod": float(a.max() - a.min()),
            "ptp_ref":  float(b.max() - b.min())}


def run_case(case: TestCase, base_cfg: MUAPConfig, fem_ctx: Optional[FEMContext]) -> CaseResult:
    """Run one case and return a CaseResult."""
    t0 = time.time()

    # Input φ(z) — analytical or FEM, depending on tier.
    if case.tier == "A":
        # When adaptive w is on, give the analytical extraction enough headroom
        # so it doesn't truncate the field; the pipeline picks its own w later.
        w_extract = case.w if base_cfg.w is not None else max(case.w, int(base_cfg.w_max))
        anal_extract = AnalyticalCase(
            fiber_depth_mm=case.fiber_depth_mm,
            L1_mm=case.L1_mm, L2_mm=case.L2_mm,
            v_m_per_s=case.v_m_per_s, fsamp_hz=case.fsamp_hz,
            w=w_extract, distfib_deg=case.distfib_deg,
            electrode_dim1_mm=case.electrode_dim1_mm,
        )
        phi, dz_mm = analytical_phi_along_fibre(anal_extract)
    else:
        if fem_ctx is None:
            raise RuntimeError(f"FEM context required for Tier B case {case.name}")
        max_extent = None
        if base_cfg.w is None:
            max_extent = float(base_cfg.w_max) * case.v_m_per_s * 1000.0 / case.fsamp_hz
        phi, dz_mm = fem_ctx.get_phi_at_depth(
            depth_mm=case.fiber_depth_mm,
            w=case.w,
            v_m_per_s=case.v_m_per_s,
            fsamp_hz=case.fsamp_hz,
            max_extent_mm=max_extent,
        )

    # Run the production pipeline.
    cfg = _to_muap_config(case, base_cfg)
    phi_mat = phi.reshape(1, -1)
    res = generate_muap_from_phi(phi_mat, dz_mm=dz_mm, config=cfg)
    muap_prod = np.real(res.muap).flatten()

    # Reference MUAP — match the production pipeline's chosen w so both
    # signals share the same time-periodicity convention.
    final_w = int(res.config.w)
    anal_ref = AnalyticalCase(
        fiber_depth_mm=case.fiber_depth_mm,
        L1_mm=case.L1_mm, L2_mm=case.L2_mm,
        v_m_per_s=case.v_m_per_s, fsamp_hz=case.fsamp_hz,
        w=final_w, distfib_deg=case.distfib_deg,
        electrode_dim1_mm=case.electrode_dim1_mm,
    )
    t_ref, muap_ref = analytical_muap(anal_ref)

    mets = _metrics(muap_prod, muap_ref)
    phi_peak = float(np.max(np.abs(phi)))
    edge_over_peak = float(np.abs(phi[0]) / max(phi_peak, 1e-30))

    return CaseResult(
        name=case.name,
        tier=case.tier,
        category=case.category,
        tag=case.tag,
        target_r=case.target_r,
        config_name="",  # filled by caller
        t_ms=list(map(float, res.t_ms)),
        muap_prod=list(map(float, muap_prod)),
        muap_ref=list(map(float, muap_ref)),
        phi_dz_mm=float(dz_mm),
        phi_peak=phi_peak,
        phi_edge_over_peak=edge_over_peak,
        r_vs_ref=mets["r"],
        rmse_vs_ref=mets["rmse"],
        rel_diff_vs_ref=mets["rel_diff"],
        ptp_prod=mets["ptp_prod"],
        ptp_ref=mets["ptp_ref"],
        elapsed_s=time.time() - t0,
        case_params=case.to_dict(),
    )


# ---------------------------------------------------------------------------
# Snapshot / load / compare
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    version: int
    config_name: str
    n_cases: int
    elapsed_s: float
    results: List[dict]   # CaseResult serialised


def write_snapshot(snap: Snapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(snap), f, indent=1, default=str)
    print(f"\nwrote {snap.n_cases} cases × {len(snap.results[0]['muap_prod'])} samples → {path}")


def read_snapshot(path: Path) -> Snapshot:
    with open(path) as f:
        d = json.load(f)
    return Snapshot(
        version=d.get("version", SNAPSHOT_VERSION),
        config_name=d.get("config_name", "<unknown>"),
        n_cases=d.get("n_cases", len(d.get("results", []))),
        elapsed_s=d.get("elapsed_s", 0.0),
        results=d.get("results", []),
    )


def snapshot(config_name: str, out_path: Path, filter_tier: Optional[str] = None) -> None:
    """Run all cases through the named config, write snapshot."""
    if config_name not in CONFIGS:
        raise SystemExit(f"unknown config {config_name!r}; options: {sorted(CONFIGS)}")

    base_cfg = CONFIGS[config_name]()
    cases = define_cases()
    if filter_tier:
        cases = [c for c in cases if c.tier == filter_tier]

    # Decide whether to spin up FEM
    needs_fem = any(c.tier == "B" for c in cases)
    fem_ctx: Optional[FEMContext] = None
    if needs_fem:
        print("Spinning up FEM (one-time setup)...")
        fem_ctx = FEMContext()
        print(f"  FEM solved in {fem_ctx.solve_time_s:.1f}s")

    t_all = time.time()
    results: List[CaseResult] = []
    for i, case in enumerate(cases):
        try:
            r = run_case(case, base_cfg, fem_ctx)
        except Exception as e:
            print(f"  [{i+1:3d}/{len(cases)}] {case.name:32s} FAILED: {e!r}")
            continue
        r.config_name = config_name
        results.append(r)
        flag = "✓" if r.r_vs_ref >= case.target_r else "✗"
        if (i + 1) % 20 == 0 or i == len(cases) - 1:
            print(f"  [{i+1:3d}/{len(cases)}] {case.name:32s} r={r.r_vs_ref:.4f} "
                  f"(target {case.target_r:.3f}) {flag}")

    elapsed = time.time() - t_all
    snap = Snapshot(
        version=SNAPSHOT_VERSION,
        config_name=config_name,
        n_cases=len(results),
        elapsed_s=elapsed,
        results=[asdict(r) for r in results],
    )
    print(f"\nSummary ({config_name}, elapsed {elapsed:.1f}s):")
    _print_summary(results)
    write_snapshot(snap, out_path)


def _print_summary(results: List[CaseResult]) -> None:
    """Print pass/fail breakdown vs target_r."""
    cats: Dict[Tuple[str, str], List[CaseResult]] = {}
    for r in results:
        key = (r.category, r.tier)
        cats.setdefault(key, []).append(r)

    for (cat, tier), grp in sorted(cats.items()):
        rs = [g.r_vs_ref for g in grp]
        passing = sum(1 for g in grp if g.r_vs_ref >= g.target_r)
        print(f"  {cat:11s} tier {tier}: n={len(grp):3d}  "
              f"pass {passing:3d}/{len(grp):3d}  "
              f"r mean {np.mean(rs):.4f} min {np.min(rs):.4f} max {np.max(rs):.4f}")

    # By tag too — useful for spotting regimes
    by_tag: Dict[str, List[CaseResult]] = {}
    for r in results:
        by_tag.setdefault(r.tag, []).append(r)
    print(f"\n  Per-tag (challenging only):")
    for tag in sorted(by_tag):
        grp = [g for g in by_tag[tag] if g.category == "challenging"]
        if not grp:
            continue
        rs = [g.r_vs_ref for g in grp]
        passing = sum(1 for g in grp if g.r_vs_ref >= g.target_r)
        print(f"    {tag:30s} n={len(grp):3d}  pass {passing:3d}/{len(grp):3d}  "
              f"r mean {np.mean(rs):.4f} min {np.min(rs):.4f}")


# ---------------------------------------------------------------------------
# Compare two snapshots
# ---------------------------------------------------------------------------

def compare_snapshots(ref_path: Path, new_path: Path) -> int:
    ref = read_snapshot(ref_path)
    new = read_snapshot(new_path)
    print(f"Reference: {ref_path}  (config={ref.config_name}, n={ref.n_cases})")
    print(f"New:       {new_path}  (config={new.config_name}, n={new.n_cases})")

    ref_by_name = {r["name"]: r for r in ref.results}
    new_by_name = {r["name"]: r for r in new.results}
    common = sorted(set(ref_by_name) & set(new_by_name))

    regressions: List[Tuple[str, float, float]] = []
    improvements: List[Tuple[str, float, float]] = []
    matched: List[float] = []
    for name in common:
        r_ref = ref_by_name[name]["r_vs_ref"]
        r_new = new_by_name[name]["r_vs_ref"]
        delta = r_new - r_ref
        matched.append(delta)
        if delta < -0.01:
            regressions.append((name, r_ref, r_new))
        elif delta > 0.01:
            improvements.append((name, r_ref, r_new))

    print(f"\nMatched cases: {len(common)}  "
          f"Mean Δr: {np.mean(matched):+.4f}  "
          f"Max Δr: {max(matched, default=0):+.4f}  "
          f"Min Δr: {min(matched, default=0):+.4f}")

    if improvements:
        print(f"\nImprovements ({len(improvements)}):")
        for name, r0, r1 in sorted(improvements, key=lambda x: -(x[2] - x[1]))[:25]:
            print(f"  {name:40s}  r {r0:.4f} → {r1:.4f}  (Δ {r1 - r0:+.4f})")

    if regressions:
        print(f"\nRegressions ({len(regressions)}):")
        for name, r0, r1 in sorted(regressions, key=lambda x: (x[2] - x[1]))[:25]:
            print(f"  {name:40s}  r {r0:.4f} → {r1:.4f}  (Δ {r1 - r0:+.4f})")
        return 1
    print("\n✓ no regressions (no case with Δr < −0.01)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="Run all cases through a config; save snapshot.")
    s.add_argument("--config", required=True, choices=sorted(CONFIGS) + ["all"])
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--tier", choices=["A", "B"], default=None,
                   help="Restrict to one tier (A: analytical-φ, B: FEM-φ).")

    c = sub.add_parser("compare", help="Compare two snapshot files.")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--new", type=Path, required=True)

    args = ap.parse_args(argv)

    if args.cmd == "snapshot":
        if args.config == "all":
            for name in sorted(CONFIGS):
                snapshot(name, args.out.parent / f"{args.out.stem}_{name}.json",
                         filter_tier=args.tier)
            return 0
        snapshot(args.config, args.out, filter_tier=args.tier)
        return 0
    if args.cmd == "compare":
        return compare_snapshots(args.reference, args.new)
    return 1


if __name__ == "__main__":
    sys.exit(main())
