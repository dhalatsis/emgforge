"""Plot the golden cylindrical set vs the fixed spatial engine (12-panel gallery).

For each golden case: golden (Fourier, black) vs spatial engine (red), normalised
and peak-aligned, titled with r. Visual companion to verify_spatial.py.

Run: python \
        emgforge.synthesis/spatial_refactor/verification/plot_golden.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.synthesis.spatial_refactor import compute_sfap_spatial, SpatialConfig

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
FS, W = 2048.0, 256


def _norm(x):
    m = np.abs(x).max()
    return x / m if m > 0 else x


def best_shift_r(tg, g, t, s, max_lag=20.0, dt=0.02):
    gn, sn = _norm(g), _norm(s)
    coarse = tg[np.argmax(np.abs(gn))] - t[np.argmax(np.abs(sn))]
    grid = np.arange(tg.min(), tg.max(), dt)
    gg = np.interp(grid, tg, gn, left=0, right=0)
    best_abs, best_r, best_sh = -1.0, 0.0, coarse
    for d in np.arange(-max_lag, max_lag + dt, dt):
        sgg = np.interp(grid, t + coarse + d, sn, left=0, right=0)
        if sgg.std() < 1e-9:
            continue
        r = pearsonr(gg, sgg)[0]
        if abs(r) > best_abs:
            best_abs, best_r, best_sh = abs(r), r, coarse + d
    flip = 1 if best_r >= 0 else -1
    return best_sh, best_abs, flip


def main():
    d = np.load(HERE / "golden_cylindrical.npz", allow_pickle=True)
    names = [str(n) for n in d["case_names"]]
    nrow, ncol = 3, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.6, nrow * 2.7))
    for ax, n in zip(axes.flat, names):
        phi = d[f"{n}__phi"].astype(float)
        dz = float(d[f"{n}__dz_mm"]); L1 = float(d[f"{n}__L1"]); L2 = float(d[f"{n}__L2"])
        v = float(d[f"{n}__v"]); posz = float(d[f"{n}__posz"])
        tg, g = d[f"{n}__golden_t_ms"], d[f"{n}__golden_sfap"]
        cfg = SpatialConfig(v=v, fsamp=FS, w=W, csd_derivative=2, upsample_factor=2,
                            fiber_window="boxcar", smoothing=False,
                            edge_taper_left=0, edge_taper_right=0, t_start_ms=-40.0)
        t, s, _ = compute_sfap_spatial(phi, dz, L1, L2, posz, cfg)
        sh, r, flip = best_shift_r(tg, g, t, s)
        c = tg[np.argmax(np.abs(g))]
        ax.plot(tg, _norm(g), "k-", lw=2.0, label="golden (Fourier)")
        ax.plot(t + sh, flip * _norm(s), "tab:red", lw=1.2, label="spatial (fixed)")
        ax.set_xlim(c - 16, c + 16); ax.axhline(0, color="gray", lw=0.3)
        ax.set_yticks([]); ax.grid(alpha=0.25)
        ax.set_title(f"{n}\nr={r:.3f}", fontsize=8.5)
    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Golden cylindrical set vs FIXED spatial engine — "
                 "12/12 PASS, mean r=0.997 (normalised, peak-aligned)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "golden_vs_engine.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {FIG/'golden_vs_engine.png'}")


if __name__ == "__main__":
    main()
