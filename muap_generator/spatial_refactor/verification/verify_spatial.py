"""Verify the spatial engine against the GOLDEN cylindrical reference set.

For each golden case, run the spatial engine on the same φ/geometry and score
peak-aligned Pearson r vs the golden (Fourier) SFAP. A correct engine should hit
r ≳ 0.95 everywhere. Exit non-zero if any case is below THRESHOLD.

Run: python \
        muap_generator/spatial_refactor/verification/verify_spatial.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[3]  # repo root (muap_generator lives here)
sys.path.insert(0, str(REPO))
from muap_generator.spatial_refactor import compute_sfap_spatial, SpatialConfig

HERE = Path(__file__).resolve().parent
THRESHOLD = 0.95
FS, W = 2048.0, 256


def _norm(x):
    m = np.abs(x).max()
    return x / m if m > 0 else x


def align_r(tg, g, t, s, max_lag=20.0, dt=0.02):
    """Peak-aligned |r| (sign-blind: the spatial polarity convention is separate)."""
    gn, sn = _norm(g), _norm(s)
    coarse = tg[np.argmax(np.abs(gn))] - t[np.argmax(np.abs(sn))]
    grid = np.arange(tg.min(), tg.max(), dt)
    gg = np.interp(grid, tg, gn, left=0, right=0)
    best = -2.0
    for d in np.arange(-max_lag, max_lag + dt, dt):
        sg = np.interp(grid, t + coarse + d, sn, left=0, right=0)
        if sg.std() > 1e-9:
            best = max(best, abs(pearsonr(gg, sg)[0]))
    return best


def main():
    d = np.load(HERE / "golden_cylindrical.npz", allow_pickle=True)
    names = [str(n) for n in d["case_names"]]
    print(f"{'case':22s} {'L1':>4} {'L2':>4} {'v':>4} {'posz':>6}  {'r_align':>8}  status")
    print("-" * 64)
    rs = []
    for n in names:
        phi = d[f"{n}__phi"].astype(float)
        dz = float(d[f"{n}__dz_mm"]); L1 = float(d[f"{n}__L1"]); L2 = float(d[f"{n}__L2"])
        v = float(d[f"{n}__v"]); posz = float(d[f"{n}__posz"])
        tg, g = d[f"{n}__golden_t_ms"], d[f"{n}__golden_sfap"]
        # pre-roll big enough to give baseline; boxcar to match the Fourier sharp box
        cfg = SpatialConfig(v=v, fsamp=FS, w=W, csd_derivative=2, upsample_factor=2,
                            fiber_window="boxcar", smoothing=False,
                            edge_taper_left=0, edge_taper_right=0, t_start_ms=-40.0)
        t, s, _ = compute_sfap_spatial(phi, dz, len1_mm=L1, len2_mm=L2,
                                       posz_mm=posz, config=cfg)
        r = align_r(tg, g, t, s)
        rs.append(r)
        ok = "PASS" if r >= THRESHOLD else "FAIL"
        print(f"{n:22s} {L1:>4.0f} {L2:>4.0f} {v:>4.1f} {posz:>+6.1f}  {r:>+8.3f}  {ok}")
    rs = np.array(rs)
    n_pass = int((rs >= THRESHOLD).sum())
    print("-" * 64)
    print(f"mean r = {rs.mean():+.3f}   median = {np.median(rs):+.3f}   "
          f"PASS {n_pass}/{len(rs)}  (threshold {THRESHOLD})")
    return 0 if n_pass == len(rs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
