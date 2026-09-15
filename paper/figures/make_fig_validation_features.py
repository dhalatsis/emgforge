"""Figure 9 — tier-B MUAP phenomenology in paper style (2 × 3): depth power law, fat
attenuation vs Kuiken 2003, EOF/propagating ratio vs depth, CV recovered vs set CV, the
SD array across the innervation zone, transverse 50 %-width profiles. The computations
mirror scripts/validation/tier_b_features.py (B5, B11, B4b, B2, B3, B6).

Run from the repo root:  python paper/figures/make_fig_validation_features.py   (~1.5 min)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "paper/figures")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt

from style import use, save, letter, COL, W1, W2               # noqa: F401
from f1_common import (CACHE, DZ, FS, T_AX, V, W, Z_ABS, ZE, ana_phi, cv_from_array, cyl_fem, farina, fem_phi,
                       golden_cfg, p2p, power_law, record, sfap, shift_phi)

use()
NUM = {}
T0 = time.time()
phi30 = ana_phi(30.0)[0]


def pipe_array(phi, zc, len1=60.0, len2=60.0, posz=0.0, cfg=None, montage="mono", ied=10.0):
    def one(z):
        return sfap(shift_phi(phi, DZ, z), DZ, len1, len2, posz, cfg)[1]
    if montage == "mono":
        return np.array([one(z) for z in zc])
    if montage == "SD":
        return np.array([one(z + ied / 2) - one(z - ied / 2) for z in zc])
    return np.array([one(z - ied) - 2 * one(z) + one(z + ied) for z in zc])


# B5 depth power law -----------------------------------------------------------------
depths = np.array([7.0, 10.0, 13.0, 15.0, 20.0, 25.0]); radii = 40.0 - depths
amp_m = np.array([p2p(sfap(ana_phi(r)[0], DZ, 100, 100, -20.0)[1]) for r in radii])
amp_s = np.array([p2p(pipe_array(ana_phi(r)[0], np.array([0.0]), 100, 100, -20.0, None, "SD")[0]) for r in radii])
amp_F = np.array([p2p(farina(depth=r, zi=-20.0, L1=100, L2=100)[1][0]) for r in radii])
amp_fem = np.array([p2p(sfap(fem_phi(r, 0.0), DZ, 100, 100, -20.0)[1]) for r in radii])
n_m, r2_m = power_law(depths, amp_m); n_s, r2_s = power_law(depths, amp_s)
n_F, r2_F = power_law(depths, amp_F); n_fem, r2_fem = power_law(depths, amp_fem)
NUM["depth_power_law"] = dict(depths_mm=depths, exponent=dict(pipeline_mono=n_m, pipeline_SD=n_s, farina_mono=n_F, fem_mono=n_fem),
                              r2=dict(pipeline_mono=r2_m, pipeline_SD=r2_s, farina_mono=r2_F, fem_mono=r2_fem),
                              p2p_rel=dict(pipeline_mono=amp_m / amp_m[0], pipeline_SD=amp_s / amp_s[0],
                                           farina=amp_F / amp_F[0], fem=amp_fem / amp_fem[0]))

# B11 fat -----------------------------------------------------------------------------
fats = (0.5, 3.0, 6.0, 9.0, 18.0)
def fat_case(f):
    return farina(depth=30.0, zi=-20.0, L1=100, L2=100, geo=dict(r_fat=35.0 + f, r_skin=37.0 + f))[1][0]
sf = [fat_case(f) for f in fats]
rms_f = np.array([np.sqrt((x ** 2).mean()) for x in sf]); rel = rms_f / rms_f[0]
kuiken = {3.0: 0.687, 9.0: 0.198, 18.0: 0.100}
def fem_fat_p2p(cfg):
    return np.array([p2p(sfap(cyl_fem.window(CACHE[f"fat{f}_phi"][0, 0], Z_ABS, ZE, W, DZ), DZ, 100, 100, -20.0, cfg)[1])
                     for f in (2, 4, 6, 8)])
fem_g, fem_b = fem_fat_p2p(golden_cfg()), fem_fat_p2p(golden_cfg(denoise="butterworth"))
ana_b = np.array([p2p(farina(depth=30.0, zi=-20.0, L1=100, L2=100, geo=dict(r_fat=35.0 + f, r_skin=37.0 + f))[1][0])
                  for f in (2, 4, 6, 8)])
NUM["fat"] = dict(fats_mm=fats, analytical_rms_retained=rel, kuiken_2003=kuiken,
                  kuiken_ratio_model_over_kuiken={f"{f:g}": float(rel[list(fats).index(f)] / k) for f, k in kuiken.items()},
                  fem_fats_mm=[2, 4, 6, 8], fem_p2p_retained_golden=fem_g / fem_g[0],
                  fem_p2p_retained_butterworth=fem_b / fem_b[0], farina_p2p_retained_same_fat=ana_b / ana_b[0])

# B4b EOF / propagating vs depth ---------------------------------------------------------
def eof_parts(phi, zc_, montage="mono", posz=-20.0, L_near=40.0, l2=80.0):
    cfg = golden_cfg(fiber_window="boxcar")
    a = pipe_array(phi, zc_, L_near, l2, posz, cfg, montage)
    b = pipe_array(phi, zc_, L_near + 60.0, l2, posz, cfg, montage)
    return np.abs(a - b).max(1), np.array([p2p(x) for x in b]), a - b
d_eof = (7.0, 10.0, 13.0, 15.0, 20.0, 25.0)
ratio_d = []
for d in d_eof:
    e, p, _ = eof_parts(ana_phi(40.0 - d)[0], np.array([0.0]))
    ratio_d.append(float(e[0] / p[0]))
ratio_m = {}
for m in ("mono", "SD", "DD"):
    e, p, _ = eof_parts(phi30, np.array([0.0]), m)
    ratio_m[m] = float(e[0] / p[0])
NUM["eof_ratio"] = dict(depths_mm=d_eof, ratio=ratio_d, montage_at_10mm=ratio_m)

# B2 CV --------------------------------------------------------------------------------
cvs = {}
zc = np.array([15.0, 25.0, 35.0, 45.0])
for v_ in (3.0, 4.0, 5.0):
    pa = ana_phi(30.0, v=v_)[0]
    st = pipe_array(pa, zc, 100.0, 100.0, 0.0, golden_cfg(v=v_), "SD", 10.0)
    cv_p, r2_p, _ = cv_from_array(st, 10.0, FS)
    try:
        _, s_F, _ = farina(depth=30.0, zi=0.0, L1=100.0, L2=100.0, det_type=2, channels=9, dint=10.0, v=v_)
        cv_F, r2_F_, _ = cv_from_array(s_F[5:], 10.0, FS)
    except ValueError:
        cv_F, r2_F_ = np.nan, np.nan
    cvs[v_] = dict(pipeline=cv_p, pipeline_r2=r2_p, farina=cv_F, farina_r2=r2_F_)
NUM["cv"] = {f"v{v_:g}": c for v_, c in cvs.items()}

# B3 IZ array -----------------------------------------------------------------------------
zc9 = np.arange(-40.0, 40.1, 10.0)
mono = pipe_array(phi30, zc9, 60.0, 60.0, 0.0)
sd = pipe_array(phi30, zc9, 60.0, 60.0, 0.0, None, "SD", 10.0)
sd_iz = float(p2p(sd[4]) / max(p2p(s) for s in sd))
rev = [float(np.corrcoef(sd[4 + k], sd[4 - k])[0, 1]) for k in (1, 2, 3)]
sym = [float(np.corrcoef(mono[4 + k], mono[4 - k])[0, 1]) for k in (1, 2, 3)]
NUM["iz_array"] = dict(sd_at_iz_over_max=sd_iz, sd_r_plus_minus=rev, mono_r_plus_minus=sym)

# B6 transverse -----------------------------------------------------------------------------
arc = lambda th, rs=40.0: np.deg2rad(th) * rs                    # arc length on the skin (mm)
ths = np.arange(0.0, 90.1, 3.0)
def transverse(r, montage):
    cfg = golden_cfg(fiber_window="boxcar")
    if montage == "mono":
        return np.array([p2p(sfap(ana_phi(r, distfib=th)[0], DZ, 40, 80, -20.0, cfg)[1]) for th in ths])
    return np.array([p2p(pipe_array(ana_phi(r, distfib=th)[0], np.array([0.0]), 40, 80, -20.0, cfg, "SD")[0]) for th in ths])
def width50(prof):
    x = arc(ths); half = prof[0] / 2
    k = np.where(prof < half)[0][0]
    return 2 * float(np.interp(half, [prof[k], prof[k - 1]], [x[k], x[k - 1]]))
prof = {(d, m): transverse(40 - d, m) for d in (15.0, 20.0) for m in ("mono", "SD")}
w_m = {d: width50(prof[(d, "mono")]) for d in (15.0, 20.0)}
w_s = {d: width50(prof[(d, "SD")]) for d in (15.0, 20.0)}
NUM["transverse"] = dict(width50_mm=dict(mono_15=w_m[15.0], mono_20=w_m[20.0], sd_15=w_s[15.0], sd_20=w_s[20.0]),
                         mono_over_sd=dict(d15=w_m[15.0] / w_s[15.0], d20=w_m[20.0] / w_s[20.0]),
                         roeleveld_depth_over_0p2width=dict(d15=15 / (0.2 * w_m[15.0]), d20=20 / (0.2 * w_m[20.0])))
NUM["runtime_s"] = time.time() - T0

# figure ------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(W2, 4.5), gridspec_kw=dict(hspace=0.5, wspace=0.42, left=0.075, right=0.985, top=0.95, bottom=0.1))
MS = 3.2

ax = axes[0, 0]
ax.loglog(depths, amp_m / amp_m[0], "o-", color=COL["analytical"], ms=MS, mec="white", mew=0.4, label=f"pipeline mono, n = {n_m:.2f}")
ax.loglog(depths, amp_s / amp_s[0], "s--", color=COL["analytical"], ms=MS, mec="white", mew=0.4, label=f"pipeline SD, n = {n_s:.2f}")
ax.loglog(depths, amp_fem / amp_fem[0], "x-", color=COL["fem"], ms=MS + 0.5, mew=0.9, label=f"FEM pipeline mono, n = {n_fem:.2f}")
ax.loglog(depths, amp_F / amp_F[0], "^:", color=COL["fourier"], ms=MS, mec="white", mew=0.4, label=f"Farina 2004 mono, n = {n_F:.2f}")
ax.set_xlabel("depth below skin (mm)"); ax.set_ylabel("p2p / p2p(7 mm)")
ax.set_xticks([7, 10, 15, 20, 25]); ax.set_xticklabels(["7", "10", "15", "20", "25"]); ax.minorticks_off()
ax.legend(loc="lower left", fontsize=5.8, handlelength=1.8, borderaxespad=0.1, labelspacing=0.25)
ax.text(0.97, 0.97, "p2p ∝ d$^{-n}$", transform=ax.transAxes, fontsize=6.5, ha="right", va="top")
letter(ax, "a", dx=-0.28)

ax = axes[0, 1]
ax.plot(fats, rel, "o-", color=COL["fourier"], ms=MS, mec="white", mew=0.4, label="analytical, RMS")
ax.plot(list(kuiken), list(kuiken.values()), "^", color=COL["lit"], ms=5, label="Kuiken 2003 (FE)")
ax.plot([2, 4, 6, 8], fem_g / fem_g[0], "x-", color=COL["fem"], ms=MS + 0.5, mew=0.9, label="FEM golden, p2p")
ax.plot([2, 4, 6, 8], fem_b / fem_b[0], "x--", color=COL["fem"], ms=MS + 0.5, mew=0.9, alpha=0.55, label="FEM Butterworth-φ, p2p")
ax.plot([2, 4, 6, 8], ana_b / ana_b[0], "+:", color=COL["fourier"], ms=MS + 1, mew=0.9, label="analytical, p2p")
ax.set_xlabel("subcutaneous fat thickness (mm)"); ax.set_ylabel("retained (rel. to thinnest)")
ax.set_xlim(0, 19); ax.set_ylim(0, 1.08)
ax.legend(loc="upper right", fontsize=5.8, handlelength=1.8, borderaxespad=0.1, labelspacing=0.25,
          bbox_to_anchor=(1.03, 1.03))
letter(ax, "b", dx=-0.28)

ax = axes[0, 2]
ax.plot(d_eof, ratio_d, "o-", color=COL["analytical"], ms=MS, mec="white", mew=0.4)
ax.set_xlabel("depth below skin (mm)"); ax.set_ylabel("EOF / propagating amplitude")
ax.set_ylim(0, max(ratio_d) * 1.25); ax.set_xlim(5, 27)
ax.text(0.04, 0.96, "monopolar, NMJ −20 mm,\nnear tendon 40 mm\n\nat 10 mm: mono {mono:.3f}\nSD {SD:.3f}, DD {DD:.3f}".format(**ratio_m),
        transform=ax.transAxes, fontsize=6, ha="left", va="top", color="0.25")
letter(ax, "c", dx=-0.28)

ax = axes[1, 0]
vset = np.array([3.0, 4.0, 5.0])
ax.plot([2.5, 5.5], [2.5, 5.5], color="0.7", lw=0.7, ls="--", label="identity")
ax.plot(vset, [cvs[v]["pipeline"] for v in vset], "o", color=COL["analytical"], ms=4.5, mec="white", mew=0.5, label="pipeline (SD array)")
ax.plot(vset, [cvs[v]["farina"] for v in vset], "^", color=COL["fourier"], ms=4.5, mec="white", mew=0.5, label="Farina 2004 (SD array)")
for v in vset:
    ax.text(v + 0.08, cvs[v]["pipeline"] - 0.05, f"{cvs[v]['pipeline']:.2f}", fontsize=5.8, ha="left", va="top", color=COL["analytical"])
    if not np.isnan(cvs[v]["farina"]):
        ax.text(v - 0.08, cvs[v]["farina"] + 0.05, f"{cvs[v]['farina']:.2f}", fontsize=5.8, ha="right", va="bottom", color=COL["fourier"])
ax.set_xlim(2.5, 5.5); ax.set_ylim(2.5, 5.5); ax.set_xticks(vset); ax.set_yticks(vset)
ax.set_xlabel("set CV (m/s)"); ax.set_ylabel("recovered CV (m/s)")
ax.legend(loc="upper left", fontsize=5.8, handlelength=1.8, borderaxespad=0.1, labelspacing=0.25)
ax.text(0.97, 0.03, "IED 10 mm, 4 channels\nz = +15 … +45 mm", transform=ax.transAxes, fontsize=6, ha="right", va="bottom", color="0.25")
letter(ax, "d", dx=-0.28)

ax = axes[1, 1]
sc_sd, sc_mo = 0.55 / np.abs(sd).max(), 0.55 / np.abs(mono).max()
for k in range(9):
    ax.axhline(k, color="0.92", lw=0.5, zorder=0)
    ax.plot(T_AX, mono[k] * sc_mo + k, color=COL["mono"], lw=0.6, alpha=0.6, label="monopolar" if k == 0 else None)
    ax.plot(T_AX, sd[k] * sc_sd + k, color=COL["sd"], lw=0.9, label="SD (IED 10 mm)" if k == 0 else None)
ax.set_xlim(-2, 24); ax.set_ylim(-1.6, 8.8); ax.set_yticks(range(9)); ax.set_yticklabels([f"{v:+.0f}" for v in zc9])
ax.set_xlabel("t (ms), t = 0 at the NMJ"); ax.set_ylabel("electrode z (mm)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(loc="lower right", fontsize=5.8, handlelength=1.6, borderaxespad=0.0, labelspacing=0.25, ncol=2,
          bbox_to_anchor=(1.02, 1.0), columnspacing=0.8, handletextpad=0.5)
ax.text(0.98, 0.01, f"SD at the IZ: {sd_iz:.3f} of max;  r(+z, −z) = {rev[0]:.3f}", transform=ax.transAxes,
        fontsize=6, ha="right", va="bottom", color="0.25")
letter(ax, "e", dx=-0.22)

ax = axes[1, 2]
x = arc(ths)
ax.plot(x, prof[(15.0, "mono")] / prof[(15.0, "mono")][0], "-", color=COL["mono"], lw=1.0, label=f"monopolar, 50 %-width {w_m[15.0]:.0f} mm")
ax.plot(x, prof[(15.0, "SD")] / prof[(15.0, "SD")][0], "--", color=COL["sd"], lw=1.0, label=f"SD, 50 %-width {w_s[15.0]:.0f} mm")
ax.axhline(0.5, color="0.8", lw=0.5, zorder=0)
ax.set_xlim(0, 63); ax.set_ylim(0, 1.05)
ax.set_xlabel("distance across the skin (mm)"); ax.set_ylabel("p2p / p2p(0)")
ax.legend(loc="upper right", fontsize=5.8, handlelength=1.8, borderaxespad=0.1, labelspacing=0.25)
ax.text(0.97, 0.62, "fibre 15 mm deep", transform=ax.transAxes, fontsize=6, ha="right", va="top", color="0.25")
letter(ax, "f", dx=-0.28)

save(fig, "fig_validation_features")
record("fig9_validation_features", NUM)
