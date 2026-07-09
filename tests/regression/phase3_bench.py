"""Phase 3 cross-geometry sign-flip regression — Tier C of the bench.

The original Phase 3 sign-flip finding paired electrodes from a complex
geometry (offgrid_complex) against electrodes from the circular reference,
matched by (theta, z) position. Truncated lead fields caused MUAP
correlations to flip sign on ~20% of pairs.

This harness reproduces that test against the production pipeline.

Inputs (already on disk under ./data/training/results/):
  - fem_fiber_leadfields/fem_fiber_leadfields.npz                 — circular reference
  - fem_fiber_offgrid_complex/fem_fiber_leadfields.npz             — Phase 3 worst geometry
  - fem_fiber_matched_depth_complex/fem_fiber_leadfields.npz       — broader coverage
  - fem_fiber_distance_sweep_complex/fem_fiber_leadfields.npz
  - fem_fiber_finedz_complex/fem_fiber_leadfields.npz

For each complex dataset:
  - Compute MUAPs for every electrode under each of three configs:
    default, adaptive_w_auto (post-Round 2), explicit edge_taper=15.
  - Pair complex electrode i to nearest circular electrode by (theta, z)
    (the original Phase 3 pairing).
  - Filter to non-silent pairs (PTP > 30% of complex median PTP).
  - Report: # flips, median signed r, median |r|, range.

ACCEPTANCE (Round 2 Definition of Done):
  - `default`               — ~33% flips (documents the bug)
  - `adaptive_w_auto`       — 0% flips (after R2.1+R2.2 fix)
  - `truncated_explicit`    — 0% flips (sanity matches adaptive)

Usage:
  PYTHONPATH=src python tests/regression/phase3_bench.py \\
      --out tests/regression/snapshots/phase3_round2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from muap_generator.api import (  # noqa: E402
    MUAPConfig,
    generate_muap_from_phi,
    get_adaptive_config,
)


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data")) / "training" / "results"
COMPLEX_DATASETS = [
    "fem_fiber_offgrid_complex",
    "fem_fiber_matched_depth_complex",
    "fem_fiber_distance_sweep_complex",
    "fem_fiber_finedz_complex",
]
CIRCULAR_DATASET = "fem_fiber_leadfields"


def _adaptive_cfg_fixed_w(w: int = 256) -> MUAPConfig:
    """`get_adaptive_config()` but with w pinned — keeps auto-smoothing AND
    the new auto-edge_taper while disabling the variable-w behaviour.

    For the Phase 3 cross-geometry pairing, we want both members of each
    pair to have the same time axis (the original P3 test was at w=256).
    The Round 2 fix that matters here is auto-edge_taper, not adaptive `w`.
    """
    cfg = get_adaptive_config()
    cfg.w = int(w)   # pin w
    return cfg


def _load(tag: str) -> Dict[str, np.ndarray]:
    """Load one Phase-3-style dataset.

    Returns dict with at least `lead_fields`, `z`, `electrode_cyl`.
    """
    path = DATA_ROOT / tag / "fem_fiber_leadfields.npz"
    if not path.exists():
        raise FileNotFoundError(f"phase 3 dataset not found at {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _muaps_for(ds: Dict[str, np.ndarray], config: MUAPConfig,
                target_len: int = 256) -> np.ndarray:
    """Compute pipeline MUAPs for every electrode in the dataset.

    Note: adaptive_w can produce variable-length outputs; resample (linear
    interp) to ``target_len`` for fair cross-electrode correlation. This
    matches the verify_signflip.py skeleton's resampling step.
    """
    from scipy.interpolate import interp1d

    z = ds["z"]
    lf = ds["lead_fields"]
    dz = float(z[1] - z[0])
    out = np.zeros((len(lf), target_len))
    for i in range(len(lf)):
        phi = lf[i:i + 1]
        res = generate_muap_from_phi(phi, dz_mm=dz, config=config)
        m = np.real(res.muap).flatten()
        if len(m) != target_len:
            xold = np.linspace(0, 1, len(m))
            xnew = np.linspace(0, 1, target_len)
            m = interp1d(xold, m, kind="linear")(xnew)
        out[i] = m
    return out


def _pair_by_position(ec_complex: np.ndarray, ec_circ: np.ndarray,
                       max_dtheta: float = 20.0, max_dz: float = 0.08
                       ) -> List[Tuple[int, int]]:
    """Pair each complex electrode to the nearest circular electrode by
    (theta_deg, z_norm). Mirrors the original Phase 3 pairing."""
    pairs = []
    for i in range(len(ec_complex)):
        th_a, z_a = ec_complex[i, 1], ec_complex[i, 2]
        dth = np.abs((ec_circ[:, 1] - th_a + 180) % 360 - 180)
        dz = np.abs(ec_circ[:, 2] - z_a)
        cost = (dth / max_dtheta) ** 2 + (dz / max_dz) ** 2
        j = int(np.argmin(cost))
        if dth[j] <= max_dtheta and dz[j] <= max_dz:
            pairs.append((i, j))
    return pairs


@dataclass
class Phase3Result:
    dataset: str
    config_name: str
    n_pairs: int
    n_filtered: int
    n_flipped: int
    pct_flipped: float
    median_r_signed: float
    median_r_abs: float
    r_min: float
    r_max: float
    per_pair_r: List[float]


def run_one(ds_tag: str, circ: Dict[str, np.ndarray],
            config_name: str, cfg: MUAPConfig,
            amplitude_filter_frac: float = 0.3,
            ) -> Phase3Result:
    """Run one (dataset, config) cell and return a Phase3Result."""
    comp = _load(ds_tag)
    m_circ = _muaps_for(circ, cfg)
    m_comp = _muaps_for(comp, cfg)
    pairs = _pair_by_position(comp["electrode_cyl"], circ["electrode_cyl"])

    # PTP-based amplitude filter (mirrors original P3)
    ptps_comp = [float(m.max() - m.min()) for m in m_comp]
    nonzero = [p for p in ptps_comp if p > 0]
    median_ptp = float(np.median(nonzero)) if nonzero else 0.0

    rs = []
    for i, j in pairs:
        a, b = m_comp[i], m_circ[j]
        if a.var() == 0 or b.var() == 0:
            continue
        if (a.max() - a.min()) < amplitude_filter_frac * median_ptp:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))

    arr = np.array(rs) if rs else np.array([0.0])
    return Phase3Result(
        dataset=ds_tag, config_name=config_name,
        n_pairs=len(pairs), n_filtered=len(rs),
        n_flipped=int((arr < 0).sum()),
        pct_flipped=float(100.0 * (arr < 0).sum() / max(len(arr), 1)),
        median_r_signed=float(np.median(arr)),
        median_r_abs=float(np.median(np.abs(arr))),
        r_min=float(arr.min()),
        r_max=float(arr.max()),
        per_pair_r=rs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                     help="JSON output path")
    ap.add_argument("--datasets", nargs="+", default=COMPLEX_DATASETS,
                     help="complex datasets to evaluate (default: all 4)")
    args = ap.parse_args()

    print("Loading circular reference dataset ...")
    circ = _load(CIRCULAR_DATASET)
    print(f"  {len(circ['lead_fields'])} electrodes, "
          f"fiber_r={circ['fiber_r']:.2f}, r_skin={circ['r_skin']:.2f}")

    # All configs lock w=256 so paired MUAPs share the same time axis
    # (the original Phase 3 test was conducted at fixed w=256). Adaptive_w
    # CAN pick larger w on densely-sampled inputs (e.g. matched_depth_complex
    # has n=1024 samples per electrode → adaptive picks 1024) but that
    # creates an apples-to-oranges comparison with the circular reference
    # which has n=401 → adaptive picks 256. The relevant Round 2 fix is
    # auto-edge_taper (whose behaviour we want to test); we fix w=256 so
    # only that fix matters for the comparison.
    configs = {
        "default": MUAPConfig(),
        "adaptive_w_auto": _adaptive_cfg_fixed_w(),
        "truncated_explicit": MUAPConfig(
            smoothing_method="butterworth", butterworth_cutoff=0.03,
            butterworth_order=2, edge_taper=15,
        ),
    }

    t0 = time.time()
    rows: List[Phase3Result] = []
    for ds in args.datasets:
        print(f"\n=== Dataset: {ds} ===")
        for cname, cfg in configs.items():
            t1 = time.time()
            res = run_one(ds, circ, cname, cfg)
            print(f"  [{cname:25s}] {res.n_flipped}/{res.n_filtered} flipped "
                  f"({res.pct_flipped:.1f}%), median signed r {res.median_r_signed:+.3f}, "
                  f"[{res.r_min:+.3f}, {res.r_max:+.3f}]  ({time.time() - t1:.1f}s)")
            rows.append(res)

    # Acceptance summary
    print(f"\n{'='*60}\nAcceptance check (Round 2 Definition of Done):")
    failures = []
    for r in rows:
        if r.config_name == "adaptive_w_auto" and r.pct_flipped > 0.0:
            failures.append(f"  {r.dataset}: adaptive_w_auto has {r.n_flipped}/{r.n_filtered} flips (≠ 0%)")
        if r.config_name == "default" and r.dataset == "fem_fiber_offgrid_complex":
            if r.pct_flipped < 10.0:
                failures.append(f"  {r.dataset}: default has only {r.pct_flipped:.1f}% flips "
                                f"— expected ~20–33% to confirm bench actually reproduces the bug")

    if failures:
        print("FAIL:")
        for f in failures: print(f)
    else:
        print(f"  PASS — adaptive_w_auto gives 0% flips on all "
              f"{len(args.datasets)} Phase 3 datasets")

    out = {
        "version": 1,
        "elapsed_s": time.time() - t0,
        "datasets": args.datasets,
        "configs": list(configs.keys()),
        "results": [
            {
                "dataset": r.dataset, "config": r.config_name,
                "n_pairs": r.n_pairs, "n_filtered": r.n_filtered,
                "n_flipped": r.n_flipped, "pct_flipped": r.pct_flipped,
                "median_r_signed": r.median_r_signed,
                "median_r_abs": r.median_r_abs,
                "r_min": r.r_min, "r_max": r.r_max,
                "per_pair_r": r.per_pair_r,
            }
            for r in rows
        ],
        "acceptance_failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
