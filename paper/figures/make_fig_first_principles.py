"""Figure 4 — the spatial engine against first principles (closed-form φ of a point
electrode in an infinite anisotropic medium): (a) engine vs oracle at three NMJ offsets,
(b) the Fourier engine vs the same oracle, (c) engine/oracle amplitude ratio vs CV
(≈ 1, v-independent, on the corrected engine; the pre-correction 1/v shown as reference),
(d) the monopole-free source identity. Laid out 2 × 2.

Run from the repo root:  python paper/figures/make_fig_first_principles.py   (~5 s)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "paper/figures")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import use, save, letter, COL, W1, W2               # noqa: F401
from f1_common import (DZ, FS, T_AX, W, Z, best_r, fourier_physical, golden_cfg, normed, phi_inf, record,
                       sfap, sfap_first_principles)
from emgforge.synthesis.engines.spatial import build_csd_matrix
from emgforge.synthesis.iap import ROSENFALCK_AMPLITUDE_V

use()
NUM = {}
phi0 = phi_inf(Z)
CFG0 = golden_cfg(fiber_window="boxcar", denoise="none")     # the engine with no FEM-specific processing
geoms = [(0.0, 60.0, 60.0), (-20.0, 40.0, 80.0), (-30.0, 30.0, 90.0)]

# (a) ------------------------------------------------------------------------
pairs = []
for posz, l1, l2 in geoms:
    t_r, s_r = sfap_first_principles(posz, l1, l2)
    t_s, s_s = sfap(phi0, DZ, l1, l2, posz, CFG0)
    r, lag, amp = best_r(t_r, s_r, t_s, s_s)
    pairs.append(dict(posz=posz, t_r=t_r, s_r=s_r, t_s=t_s, s_s=s_s, r=r, lag=lag, amp=amp))
NUM["a_spatial_vs_oracle"] = {f"nmj{p['posz']:+g}": dict(r=p["r"], lag_ms=p["lag"], amp_ratio=p["amp"]) for p in pairs}

# (b) ------------------------------------------------------------------------
posz, l1, l2 = geoms[1]
t_r, s_r = sfap_first_principles(posz, l1, l2)
t_f, s_f = fourier_physical(phi0, DZ, l1, l2, posz)
r_f, lag_f, _ = best_r(t_r, s_r, t_f, s_f, max_lag_ms=15)
t_m, s_m = sfap_first_principles(posz, l1, l2, mirror=True)
r_fm, lag_fm, _ = best_r(t_m, s_m, t_f, s_f, max_lag_ms=15)
NUM["b_fourier_vs_oracle_nmj-20"] = dict(signed_r=r_f, lag_ms=lag_f, r_vs_mirrored_iap=r_fm, lag_vs_mirrored_ms=lag_fm)

# (c) ------------------------------------------------------------------------
vs = (2.0, 3.0, 4.0, 5.0)
ratio = {}
for v_ in vs:
    t_r_, s_r_ = sfap_first_principles(0.0, 60, 60, v=v_)
    t_s_, s_s_ = sfap(phi0, DZ, 60, 60, 0.0, golden_cfg(fiber_window="boxcar", denoise="none", v=v_))
    ratio[v_] = best_r(t_r_, s_r_, t_s_, s_s_)[2]
NUM["c_amplitude_ratio_engine_over_oracle"] = {f"v{v_:g}": ratio[v_] for v_ in vs}
NUM["c_ratio_mean"] = float(np.mean([ratio[v_] for v_ in vs]))
NUM["c_ratio_spread"] = float(np.ptp([ratio[v_] for v_ in vs]))
# what the v1 engine (spurious `/v` in compute_sfap_spatial, removed 2026-09-15) gave: ratio / v
NUM["c_before_correction_v1_ratio_over_v"] = {f"v{v_:g}": ratio[v_] / v_ for v_ in vs}

# (d) ------------------------------------------------------------------------
net = {}
for win in ("boxcar", "one_sided"):
    csd = build_csd_matrix(Z, T_AX, -20.0, 40.0, 80.0, DZ, golden_cfg(fiber_window=win))
    net[win] = float(np.max(np.abs(csd.sum(1))) / np.max(np.abs(csd).sum(1)))
# contrast: the analytic per-point kernel V''(ξ) windowed to the fibre, i.e. the propagating
# tripoles WITHOUT the junction and tendon-end terms that the full-field derivative supplies
A = ROSENFALCK_AMPLITUDE_V
xi = 4.0 * T_AX[:, None] - np.abs(Z[None, :] + 20.0)
kern = np.where(xi >= 0, A * (6 * xi - 6 * xi ** 2 + xi ** 3) * np.exp(-np.clip(xi, 0, None)), 0.0)
kern *= ((Z >= -60.0) & (Z <= 60.0))[None, :]
net["tripoles only (no junction / end terms)"] = float(np.max(np.abs(kern.sum(1))) / np.max(np.abs(kern).sum(1)))
NUM["d_monopole_residual"] = net

# figure (2 × 2, ~1.5 column width so the type stays ≥ 7 pt at print size) -------------
fig = plt.figure(figsize=(W2 * 0.8, 5.0))
outer = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.42, left=0.1, right=0.985, top=0.96, bottom=0.085)
top = outer[0].subgridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.32)
bot = outer[1].subgridspec(1, 2, width_ratios=[1.3, 0.75], wspace=0.5)

ax = fig.add_subplot(top[0, 0])
for k, p in enumerate(pairs):
    off = -1.6 * k
    ax.plot(p["t_r"], normed(p["s_r"]) + off, color=COL["first"], lw=2.0, alpha=0.35,
            label="closed-form oracle" if k == 0 else None, solid_capstyle="round")
    ax.plot(p["t_s"], normed(p["s_s"]) + off, color=COL["direct"], lw=0.9, ls="--",
            label="spatial engine" if k == 0 else None)
    lag = 0.0 if abs(p["lag"]) < 0.005 else p["lag"]
    ax.text(29.5, off + 0.3, f"NMJ {p['posz']:+g} mm\nr = {p['r']:.4f}, lag {lag:.2f} ms",
            fontsize=6.5, ha="right", va="bottom")
ax.set_xlim(-2, 30); ax.set_yticks([]); ax.set_ylim(-4.95, 1.35)
ax.set_xlabel("t (ms), t = 0 at the NMJ"); ax.set_ylabel("SFAP (norm., offset)")
ax.legend(loc="lower right", handlelength=1.6, borderaxespad=0.2, fontsize=6.5, labelspacing=0.3)   # below the last trace
ax.spines["left"].set_visible(False)
letter(ax, "a", dx=-0.06)

ax = fig.add_subplot(top[0, 1])
ax.plot(t_r, normed(s_r), color=COL["first"], lw=2.0, alpha=0.35, label="closed-form oracle")
ax.plot(t_f, normed(s_f), color=COL["fourier"], lw=0.9, label="Fourier engine")
ax.axhline(0, color="0.85", lw=0.5, zorder=0)
ax.set_xlim(-2, 30); ax.set_ylim(-1.55, 1.45); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel("t (ms), t = 0 at the NMJ"); ax.set_ylabel("SFAP (norm.)")
ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.1, fontsize=6.5, labelspacing=0.3)
ax.text(0.03, 0.02, f"NMJ −20 mm\nsigned r = {r_f:+.2f} at lag {lag_f:+.1f} ms", transform=ax.transAxes,
        fontsize=6.5, ha="left", va="bottom")      # below the −1 troughs
letter(ax, "b", dx=-0.16)

ax = fig.add_subplot(bot[0, 0])
vv = np.linspace(1.6, 5.6, 100)
old = [ratio[v_] / v_ for v_ in vs]                       # the v1 engine's values (spurious /v)
ax.axhline(1.0, color="0.85", lw=0.6, zorder=0)
ax.plot(vv, 1 / vv, color="0.55", lw=0.9, ls="--", zorder=1)
ax.plot(vs, old, "o", color="0.55", ms=3.6, mfc="white", mew=0.9, zorder=2)
ax.plot(vs, [ratio[v_] for v_ in vs], "o", color=COL["direct"], ms=4.8, mec="white", mew=0.5, zorder=3)
for v_, o_ in zip(vs, old):
    ax.text(v_, ratio[v_] - 0.045, f"{ratio[v_]:.3f}", fontsize=6, ha="center", va="top", color=COL["direct"])
    ax.text(v_ + 0.12, o_ + 0.02, f"{o_:.3f}", fontsize=6, ha="left", va="bottom", color="0.4")
ax.set_xlim(1.6, 5.6); ax.set_ylim(0, 1.2); ax.set_xticks(vs); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("conduction velocity v (m/s)"); ax.set_ylabel("amplitude ratio engine / oracle")
ax.legend(handles=[Line2D([], [], color=COL["direct"], marker="o", ls="none", ms=4.8, mec="white", mew=0.5,
                          label="corrected engine (no 1/v)"),
                   Line2D([], [], color="0.55", marker="o", ls="--", lw=0.9, ms=3.6, mfc="white", mew=0.9,
                          label="before correction (v1): 1/v")],
          loc="center right", bbox_to_anchor=(1.0, 0.55), handlelength=1.8, borderaxespad=0.1, fontsize=6.5, labelspacing=0.3)
letter(ax, "c", dx=-0.2)

ax = fig.add_subplot(bot[0, 1])
names = ["boxcar", "one_sided", "tripoles only (no junction / end terms)"]
vals = [net[n] for n in names]
cols = [COL["first"], COL["direct"], COL["raw"]]
ax.bar([0, 1, 2], vals, color=cols, width=0.65, bottom=1e-18)
ax.set_yscale("log"); ax.set_ylim(1e-18, 3); ax.set_xlim(-0.55, 2.55)
ax.axhline(1e-6, color="0.5", lw=0.5, ls=(0, (2, 1.5))); ax.text(-0.5, 1.8e-6, "gate 10$^{-6}$", fontsize=6, ha="left", va="bottom", color="0.4")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["boxcar", "one-\nsided", "tripoles\nonly"], fontsize=6.5)
ax.set_ylabel("max$_t$ |Σ$_z$ i$_m$| / Σ$_z$ |i$_m$|", fontsize=7)
ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
for x_, v_ in zip([0, 1, 2], vals):
    ax.text(x_, v_ * 3, f"{v_:.0e}".replace("e-", "e−"), fontsize=6, ha="center", va="bottom")
letter(ax, "d", dx=-0.42)

save(fig, "fig_first_principles")
record("fig4_first_principles", NUM)
