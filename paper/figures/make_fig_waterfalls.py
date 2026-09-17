"""Fig — waterfall galleries: how the monopolar potential changes along ONE axis at a time.

Six panels; each sweeps one parameter and stacks the resulting waveforms with a vertical
offset (as Fig 5's waterfall). Cylinder panels (a)–(e): the analytical 4-layer cylinder
lead field (Farina 2004; bone 10 / muscle 35 / fat 38 / skin 40 mm unless swept), direct
line-source synthesis (``golden_cfg()``), monopolar Ø10 mm disc electrode at z = 0, NMJ
20 mm proximal of the electrode (z = −20), tendons 60 mm from the NMJ on both sides
(fibre z ∈ [−80, +40]), fibre 10 mm below the skin (r = 30 mm), v = 4 m/s, fs = 4096 Hz.
Panel (f): the released FCU pool (dataset D2, FEM lead field on the WR forearm MRI),
seven units of 74–102 fibres at increasing distance from the centre electrode.

  (a) depth below the skin        (d) conduction velocity (dz = v/fs kept)
  (b) electrode angle round the cylinder   (e) subcutaneous fat thickness
  (c) distal tendon distance L2   (f) real anatomy: MU depth (FCU pool)

Amplitude convention: the analytical lead field carries the Farina port's normalisation
(a.u.), so cylinder amplitudes are quoted relative to A0 = p2p of the reference fibre
(10 mm deep, θ = 0, L1 = L2 = 60, v = 4, fat 3 mm) — the trace that appears in every
cylinder panel; (f) is in µV. Panels whose p2p spans > 10× ((a), (b), (e)) draw each
trace scaled to its own p2p; (c), (d), (f) share one scale per panel (scale bar).

Run from the repo root:  python paper/figures/make_fig_waterfalls.py   (~40 s)
Writes fig_waterfalls.{pdf,png} and key_numbers_waterfalls.json beside this file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts/validation", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import numpy as np
import matplotlib.pyplot as plt

from style import use, save, letter, COL, W2                                   # noqa: E402
from harness import V, FS, W, ana_phi, sfap, golden_cfg, duration_ms, mnf, p2p, power_law  # noqa: E402
from tests.regression import analytical_ref as AR                              # noqa: E402

use()
T0 = time.time()
DZ = V * 1000.0 / FS                    # 0.977 mm — the recipe's coupled grid dz = v/fs
Z_NMJ = -20.0                           # NMJ 20 mm proximal of the electrode (electrode at z = 0)
L1, L2 = 60.0, 60.0                     # tendons 60 mm from the NMJ: fibre z ∈ [−80, +40]
R_DEF = 30.0                            # 10 mm below the skin (r_skin = 40)
G0 = dict(AR.ANAL_GEOMETRY)             # r_bone 10, r_muscle 35, r_fat 38, r_skin 40
KEY = HERE / "key_numbers_waterfalls.json"
D2 = ROOT / "_results/paper/datasets/forearm_fcu_mu_pool.npz"
NUM: dict = {}


# ----------------------------------------------------------------------------- helpers
def trace(phi, dz, len1=L1, len2=L2, cfg=None):
    return sfap(phi, dz, len1, len2, Z_NMJ, cfg)


def lobe_stats(t, x, t_lo, t_hi):
    """Time and full width at half depth of the negative (propagating) lobe of ``x``, searched
    in [t_lo, t_hi] so neither the generation deflection at t ≈ 0 nor an end-of-fibre
    potential can be mistaken for it."""
    m = (t >= t_lo) & (t <= t_hi)
    k = int(np.where(m)[0][np.argmin(x[m])]); half = 0.5 * x[k]
    lo = k
    while lo > 0 and x[lo] < half:
        lo -= 1
    hi = k
    while hi < len(x) - 1 and x[hi] < half:
        hi += 1
    return float(t[k]), float(t[hi] - t[lo])


def measure(phi, dz, len1=L1, len2=L2, cfg=None, v=V):
    """One cylinder trace + its numbers. The distal-tendon EOF is isolated by subtracting
    the same fibre with that tendon moved 60 mm further (fig 5 / tier B4 method); the
    propagating lobe is timed on that long fibre so no EOF can bias it. With the recipe's
    one-sided window the tendon term is spread over the last quarter of the semi-fibre, so
    its 5 %-onset precedes L2/v; the EOF *peak* is what tracks L2/v."""
    t, s = trace(phi, dz, len1, len2, cfg)
    _, b = trace(phi, dz, len1, len2 + 60.0, cfg)
    d = s - b
    # search from 2 ms before the wave front: for deep / far electrodes the broad lobe
    # merges with the generation deflection and its minimum precedes the front
    lobe, width = lobe_stats(t, b, (-Z_NMJ) / v - 2.0, 0.85 * len1 / v)
    k_on = int(np.where(np.abs(d) > 0.05 * np.abs(d).max())[0][0])
    return t, s, dict(p2p_au=p2p(s), duration10_ms=duration_ms(t, s, 0.1), lobe_fwhm_ms=width, mnf_hz=mnf(s, FS),
                      lobe_neg_peak_ms=lobe, eof_onset_ms=float(t[k_on]), eof_peak_ms=float(t[np.argmax(np.abs(d))]),
                      eof_expected_L2_over_v_ms=len2 / v, eof_share=float(np.abs(d).max() / p2p(b)))


def rel(rows, a0):
    for r in rows:
        r["p2p_rel_A0"] = r["p2p_au"] / a0
    return rows


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- reference
phi_ref, _ = ana_phi(R_DEF)
t_ref, s_ref, m_ref = measure(phi_ref, DZ)
A0 = m_ref["p2p_au"]
D0 = m_ref["lobe_neg_peak_ms"] - (-Z_NMJ) / V          # the negative lobe trails the wave front (~1 ms)
NUM["reference"] = dict(depth_mm=10.0, r_mm=R_DEF, angle_deg=0.0, L1_mm=L1, L2_mm=L2, v_m_per_s=V, fs_hz=FS,
                        fat_mm=3.0, skin_mm=2.0, electrode="Ø10 mm disc, monopolar, z = 0", nmj_z_mm=Z_NMJ,
                        A0_p2p_au=A0, lobe_lag_behind_front_ms=D0, **m_ref)

# (a) depth --------------------------------------------------------------------------
DEPTHS = [5.0, 7.0, 10.0, 13.0, 16.0, 20.0, 25.0]
tr_a, rows_a = [], []
for d in DEPTHS:
    t, s, m = measure(ana_phi(G0["r_skin"] - d)[0], DZ)
    tr_a.append(s); rows_a.append(dict(depth_mm=d, r_mm=G0["r_skin"] - d, **m))
rel(rows_a, A0)
n_a, r2_a = power_law(DEPTHS, [r["p2p_au"] for r in rows_a])
NUM["a_depth"] = dict(settings="fibre at r = 40 − depth; θ = 0; L1 = L2 = 60; v = 4; fat 3 mm", traces=rows_a,
                      power_law=dict(exponent=n_a, r2=r2_a, range_mm=[DEPTHS[0], DEPTHS[-1]]))

# (b) angle --------------------------------------------------------------------------
ANGLES = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0]
tr_b, rows_b = [], []
for th in ANGLES:
    t, s, m = measure(ana_phi(R_DEF, distfib=th)[0], DZ)
    tr_b.append(s); rows_b.append(dict(angle_deg=th, arc_on_skin_mm=np.deg2rad(th) * G0["r_skin"], **m))
rel(rows_b, A0)
pb = np.array([r["p2p_rel_A0"] for r in rows_b])
th50 = float(np.interp(0.5, pb[::-1], np.array(ANGLES)[::-1]))
NUM["b_angle"] = dict(settings="fibre r = 30 (10 mm deep); electrode meridian rotated by θ; L1 = L2 = 60; v = 4",
                      traces=rows_b, angle_at_half_p2p_deg=th50, arc_at_half_p2p_mm=np.deg2rad(th50) * G0["r_skin"])

# (c) distal tendon ------------------------------------------------------------------
L2S = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0]
tr_c, rows_c = [], []
for l2 in L2S:
    t, s, m = measure(phi_ref, DZ, L1, l2)
    tr_c.append(s); rows_c.append(dict(L2_mm=l2, tendon_z_from_electrode_mm=Z_NMJ + l2, **m))
rel(rows_c, A0)
NUM["c_distal_tendon"] = dict(settings="fibre r = 30, θ = 0; NMJ at z = −20; L1 = 60 (proximal tendon at z = −80); "
                                       "L2 swept (distal tendon at z = −20 + L2, i.e. under the electrode at L2 = 20)",
                              traces=rows_c,
                              eof_peak_minus_L2_over_v_ms=[r["eof_peak_ms"] - r["eof_expected_L2_over_v_ms"] for r in rows_c],
                              eof_onset_minus_L2_over_v_ms=[r["eof_onset_ms"] - r["eof_expected_L2_over_v_ms"] for r in rows_c],
                              lobe_spread_ms=float(np.ptp([r["lobe_neg_peak_ms"] for r in rows_c])),
                              p2p_rel_range=[float(min(r["p2p_rel_A0"] for r in rows_c)), float(max(r["p2p_rel_A0"] for r in rows_c))])

# (d) conduction velocity ------------------------------------------------------------
# The recipe couples the φ grid to the sampling (dz = v/fs), so φ is re-extracted at each v;
# w = 512 keeps the spatial window at 250 mm for the slow cases (256 × v/fs would be only
# 125 mm at v = 2). (The tier-A CV check keeps φ on the 0.977-mm grid and changes only cfg.v;
# the two agree to r > 0.9997 and 1 % in amplitude — the engine is discretisation-converged.)
VS = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
W_CV = 512
tr_d, t_d, rows_d = [], [], []
for v in VS:
    phi_v, dz_v = ana_phi(R_DEF, v=v, w=W_CV)
    t, s, m = measure(phi_v, dz_v, L1, L2, golden_cfg(v=v, w=W_CV), v=v)
    tr_d.append(s); t_d.append(t)
    rows_d.append(dict(v_m_per_s=v, dz_mm=dz_v, lobe_expected_front_ms=(-Z_NMJ) / v,
                       eof_expected_L_over_v_ms=L2 / v, **m))
rel(rows_d, A0)
delta_mm = np.array([r["lobe_neg_peak_ms"] * r["v_m_per_s"] for r in rows_d]) - (-Z_NMJ)   # lobe trails the front by δ mm
NUM["d_velocity"] = dict(settings=f"fibre r = 30, θ = 0; L1 = L2 = 60; φ re-extracted at dz = v/fs with w = {W_CV}",
                         traces=rows_d, lobe_lag_mm=dict(mean=float(delta_mm.mean()), spread=float(np.ptp(delta_mm))),
                         duration_times_v=[r["duration10_ms"] * r["v_m_per_s"] for r in rows_d],
                         mnf_over_v=[r["mnf_hz"] / r["v_m_per_s"] for r in rows_d])

# (e) fat ----------------------------------------------------------------------------
# As tier B11: r_muscle = 35 fixed, r_fat = 35 + fat, r_skin = r_fat + 2 (2 mm skin), fibre
# at r = 30 — 5 mm inside the muscle — so the depth below the skin is 7 + fat mm (8 … 25) and
# the 3-mm case IS the reference fibre. (A fibre 10 mm below the skin cannot exist under
# 9–18 mm of fat + 2 mm skin, so the fibre-in-muscle position is what is held fixed.)
FATS = [1.0, 3.0, 6.0, 9.0, 12.0, 18.0]
tr_e, rows_e = [], []
try:
    for f in FATS:
        AR.ANAL_GEOMETRY = dict(G0, r_fat=G0["r_muscle"] + f, r_skin=G0["r_muscle"] + f + 2.0)
        t, s, m = measure(ana_phi(R_DEF)[0], DZ)
        AR.ANAL_GEOMETRY = G0
        _, s_same, m_same = measure(ana_phi(G0["r_skin"] - (7.0 + f))[0], DZ)      # same depth, default 3-mm fat
        tr_e.append(s)
        rows_e.append(dict(fat_mm=f, r_fat_mm=G0["r_muscle"] + f, r_skin_mm=G0["r_muscle"] + f + 2.0,
                           depth_below_skin_mm=7.0 + f, depth_in_muscle_mm=5.0, **m,
                           same_depth_default_fat=dict(p2p_au=m_same["p2p_au"], p2p_rel_A0=m_same["p2p_au"] / A0,
                                                       mnf_hz=m_same["mnf_hz"])))
finally:
    AR.ANAL_GEOMETRY = G0
rel(rows_e, A0)
NUM["e_fat"] = dict(settings="r_bone 10, r_muscle 35 fixed; r_fat = 35 + fat; r_skin = r_fat + 2; fibre r = 30 "
                             "(5 mm inside the muscle, 7 + fat mm below the skin); θ = 0; L1 = L2 = 60; v = 4",
                    traces=rows_e)

# (f) real anatomy: FCU pool ---------------------------------------------------------
PICKS = [57, 52, 54, 60, 62, 58, 59]                    # 74–102 fibres, increasing distance from the centre electrode
d2 = np.load(D2)
t_f = np.asarray(d2["t_ms"], float); fs_f = float(d2["fs"])
e0 = d2["elec_grid_xyz"][2, 2]
depth_f = np.linalg.norm(d2["mu_centre_xy"] - e0[:2], axis=1)   # MU centre → electrode, in the electrode plane (as Fig 7)
muap = np.asarray(d2["muap_single"], float)             # µV, centre electrode (= muap_grid[:, 12])
assert np.allclose(muap, d2["muap_grid"][:, 12])
paths, fidx, blen = d2["bed_paths"], d2["mu_fibre_idx"], d2["bed_length_mm"]
tr_f, rows_f = [], []
for i in PICKS:
    s = muap[i]; fib = fidx[i][fidx[i] >= 0]
    Lf = float(blen[fib].mean())
    # IZ at 0.305 of the fibre: which end the paths start from is not recorded, so take the
    # candidate (0.305 or 0.695 of the path) whose |z_IZ − z_electrode| / v + lobe lag matches
    # the measured lobe; the fibre-end times follow from the semi-lengths 0.305 L / 0.695 L
    lobe, width = lobe_stats(t_f, s, 3.0, 30.0)
    cands = []
    for frac in (0.305, 0.695):
        z_iz = paths[fib, int(round(frac * (paths.shape[1] - 1))), 2].mean()
        cands.append((abs(abs(z_iz - e0[2]) / V + D0 - lobe), frac, float(abs(z_iz - e0[2]))))
    _, frac_used, dz_iz = min(cands)
    tr_f.append(s)
    rows_f.append(dict(mu=int(i), n_fibres=int(d2["mu_sizes"][i]), depth_mm=float(depth_f[i]),
                       p2p_uV=p2p(s), p2p_per_fibre_uV=p2p(s) / int(d2["mu_sizes"][i]),
                       duration10_ms=duration_ms(t_f, s, 0.1), lobe_fwhm_ms=width, mnf_hz=mnf(s, fs_f), lobe_neg_peak_ms=lobe,
                       fibre_length_mm=Lf, iz_to_electrode_along_z_mm=dz_iz, iz_path_fraction_used=frac_used,
                       lobe_expected_ms=dz_iz / V + D0,
                       eof_expected_ms=[0.305 * Lf / V, 0.695 * Lf / V], eof_onset_ms=None))
n_f, r2_f = power_law([r["depth_mm"] for r in rows_f], [r["p2p_uV"] for r in rows_f])
n_f1, r2_f1 = power_law([r["depth_mm"] for r in rows_f], [r["p2p_per_fibre_uV"] for r in rows_f])
NUM["f_fcu_pool"] = dict(settings="dataset D2 forearm_fcu_mu_pool.npz, muap_single (centre electrode 12 of the 5×5 grid), "
                                  "FEM lead field on the WR forearm, direct recipe, fs 2048 Hz; depth = |MU centre − "
                                  "electrode| in the electrode plane", electrode_xyz=e0.tolist(), traces=rows_f,
                         power_law_p2p=dict(exponent=n_f, r2=r2_f), power_law_p2p_per_fibre=dict(exponent=n_f1, r2=r2_f1))

# ----------------------------------------------------------------------------- figure
STEP = 1.0
Y_FOOT = -0.92                    # guide labels / scale bar sit here, clear of the lowest trace (min −0.52·STEP)
fig, axes = plt.subplots(2, 3, figsize=(W2, 6.0),
                         gridspec_kw=dict(wspace=0.42, hspace=0.32, left=0.085, right=0.99, top=0.96, bottom=0.075))


def waterfall(ax, ts, traces, ticks, col, mode, xlim, right, ylabel, tag, guides=(), scale_bar=None):
    """Stack traces top→bottom (first on top). mode 'each': every trace to its own p2p;
    'common': one scale per panel. guides: (kind, x, label, side) with kind 'v' (vertical
    dotted) or 'd' (dashed line through per-trace x values); the label sits at the bottom,
    to the left ('l') or right ('r') of the line."""
    n = len(traces); ys = (n - 1 - np.arange(n)) * STEP
    pk = [p2p(s) for s in traces]
    sc = [0.75 * STEP / a for a in pk] if mode == "each" else [0.75 * STEP / max(pk)] * n
    for k, (t, s) in enumerate(zip(ts, traces)):
        ax.axhline(ys[k], color="0.92", lw=0.5, zorder=0)
        ax.plot(t, s * sc[k] + ys[k], color=col, lw=0.9)
        ax.text(xlim[1] - 0.3, ys[k] + 0.07, right[k], fontsize=5.5, ha="right", va="bottom", color="0.3")
    for kind, x, lab, side in guides:
        if kind == "v":
            ax.axvline(x, color="0.35", lw=0.6, ls=(0, (1, 1.2)), zorder=1); xb = x
        else:
            ax.plot(x, ys, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=1); xb = x[-1]
        ax.text(xb + (0.25 if side == "r" else -0.25), Y_FOOT, lab, fontsize=5.5, ha="left" if side == "r" else "right",
                va="bottom", color="0.3")
    if scale_bar is not None:
        amp, lab = scale_bar
        x0 = xlim[1] - 0.6; h = amp * sc[0]
        ax.plot([x0, x0], [Y_FOOT, Y_FOOT + h], color="k", lw=1.0, solid_capstyle="butt")
        ax.text(x0 - 0.4, Y_FOOT + h / 2, lab, fontsize=5.5, ha="right", va="center")
    ax.set_yticks(ys); ax.set_yticklabels(ticks, fontsize=6.5)
    ax.set_xlim(*xlim); ax.set_ylim(Y_FOOT - 0.06, (n - 1) * STEP + 1.25)  # foot row for labels, headroom for the tag
    ax.set_ylabel(ylabel, fontsize=7)
    ax.text(0.01, 0.995, tag, transform=ax.transAxes, fontsize=5.5, ha="left", va="top", color="0.25", linespacing=1.15)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


LOBE = m_ref["lobe_neg_peak_ms"]                        # (Δz + δ)/v of the reference fibre, δ = D0·v ≈ 4.4 mm
dmean = float(delta_mm.mean())
g_fixed = [("v", LOBE, "(Δz+δ)/v", "l"), ("v", L2 / V, "L/v", "r")]
XA = (-2.0, 24.0)
blue, orange = COL["analytical"], COL["fem"]

waterfall(axes[0, 0], [t_ref] * len(tr_a), tr_a, [f"{d:g}" for d in DEPTHS], blue, "each", XA,
          [f"{r['p2p_rel_A0']:.2f} · EOF {100 * r['eof_share']:.0f} %" for r in rows_a],
          "depth below the skin (mm)",
          f"depth · each trace to its own p2p\np2p ∝ d$^{{-{n_a:.1f}}}$ · right: p2p/A₀ · EOF share", g_fixed)
waterfall(axes[0, 1], [t_ref] * len(tr_b), tr_b, [f"{a:g}°\n({r['arc_on_skin_mm']:.0f} mm)" for a, r in zip(ANGLES, rows_b)],
          blue, "each", XA, [f"{r['p2p_rel_A0']:.2f}" for r in rows_b], "electrode angle θ (arc on the skin)",
          f"angle θ · each trace to its own p2p\nhalf amplitude at θ ≈ {th50:.0f}° · right: p2p/A₀", g_fixed)
waterfall(axes[0, 2], [t_ref] * len(tr_c), tr_c, [f"{l:g}" for l in L2S], blue, "common", (-2.0, 26.0),
          [f"{r['p2p_rel_A0']:.2f}" for r in rows_c], "distal tendon L₂ from the NMJ (mm)",
          "distal tendon L₂ · common scale\nlobe fixed, EOF at L₂/v · right: p2p/A₀",
          [("v", LOBE, "(Δz+δ)/v", "l"), ("v", L1 / V, "L₁/v", "r"), ("d", np.array(L2S) / V, "L₂/v", "r")],
          scale_bar=(A0, "A₀"))
waterfall(axes[1, 0], t_d, tr_d, [f"{v:g}" for v in VS], blue, "common", (-4.0, 40.0),
          [f"{r['p2p_rel_A0']:.2f} · {r['duration10_ms']:.1f} ms" for r in rows_d], "conduction velocity v (m/s)",
          f"velocity · common scale · dz = v/fs\nδ = {dmean:.1f} mm · right: p2p/A₀ · 10 %-duration",
          [("d", ((-Z_NMJ) + dmean) / np.array(VS), "(Δz+δ)/v", "l"), ("d", L2 / np.array(VS), "L/v", "r")],
          scale_bar=(A0, "A₀"))
waterfall(axes[1, 1], [t_ref] * len(tr_e), tr_e, [f"{f:g}\n(d {7 + f:g})" for f in FATS], blue, "each", XA,
          [f"{r['p2p_rel_A0']:.2f} · {r['mnf_hz']:.0f} Hz" for r in rows_e], "fat thickness (mm)  (depth below skin)",
          "fat · fibre 5 mm in the muscle · own p2p\nright: p2p/A₀ · mean frequency", g_fixed)
waterfall(axes[1, 2], [t_f] * len(tr_f), tr_f, [f"{r['depth_mm']:.1f}\n({r['n_fibres']})" for r in rows_f], orange, "common",
          (-2.0, 36.0), [f"MU {r['mu']} · {r['p2p_uV']:.0f} µV" for r in rows_f], "MU centre to electrode d (mm)  (fibres)",
          f"FCU pool, MRI/FEM · common scale\np2p ∝ d$^{{-{n_f:.1f}}}$ (MU centres) · right: MU · p2p",
          scale_bar=(20.0, "20 µV"))
for ax, s in zip(axes.ravel(), "abcdef"):
    letter(ax, s, dx=-0.27, dy=1.0)
for ax in axes[0]:
    ax.set_xlabel("t (ms)", labelpad=1)
for ax in axes[1]:
    ax.set_xlabel("t (ms), t = 0 at the NMJ", labelpad=1)
save(fig, "fig_waterfalls")

# ----------------------------------------------------------------------------- summary
ra, rb, rc, rd, re, rf = rows_a, rows_b, rows_c, rows_d, rows_e, rows_f
def rng(rows, key, fmt=".1f"):
    return f"{min(r[key] for r in rows):{fmt}}–{max(r[key] for r in rows):{fmt}}"


lines = [
    f"(a) Depth 5→25 mm: p2p falls {ra[0]['p2p_rel_A0']:.2f}→{ra[-1]['p2p_rel_A0']:.3f} A0 "
    f"({ra[0]['p2p_au'] / ra[-1]['p2p_au']:.0f}×, p2p ∝ d^-{n_a:.2f}, r² {r2_a:.2f}), the negative lobe widens "
    f"{ra[0]['lobe_fwhm_ms']:.1f}→{ra[-1]['lobe_fwhm_ms']:.1f} ms (FWHM) and the end-of-fibre share grows "
    f"{100 * ra[0]['eof_share']:.0f}→{100 * ra[-1]['eof_share']:.0f} %; the lobe minimum sits at (Δz+δ)/v = "
    f"{ra[2]['lobe_neg_peak_ms']:.1f} ms for the shallow fibres and drifts to {ra[-1]['lobe_neg_peak_ms']:.1f} ms at 25 mm as the "
    f"broad far-field lobe merges with the generation deflection, and the EOF peak from {ra[0]['eof_peak_ms']:.1f} ms near the "
    f"surface (L/v = 15; 5 %-onset {rng(ra, 'eof_onset_ms')} ms, inside the one-sided window's taper 0.75–1 L/v) to "
    f"{ra[-1]['eof_peak_ms']:.1f} ms at depth.",
    f"(b) Electrode 0→60° round the cylinder (0→{rb[-1]['arc_on_skin_mm']:.0f} mm on the skin, fibre 10 mm deep): p2p falls to "
    f"{rb[-1]['p2p_rel_A0']:.3f} A0 (half amplitude at θ ≈ {th50:.0f}°, {np.deg2rad(th50) * 40:.0f} mm arc), the lobe FWHM "
    f"{rb[0]['lobe_fwhm_ms']:.1f}→{rb[-1]['lobe_fwhm_ms']:.1f} ms, EOF share {100 * rb[0]['eof_share']:.0f}→{100 * rb[-1]['eof_share']:.0f} %; "
    f"the wave's arrival is unchanged (lobe minimum {rb[0]['lobe_neg_peak_ms']:.1f} ms up to 30°, drifting to "
    f"{rb[-1]['lobe_neg_peak_ms']:.1f} ms at 60° as the far-field lobe broadens; EOF peak {rng(rb, 'eof_peak_ms')} ms).",
    f"(c) Distal tendon L2 = 20→80 mm (junction fixed 20 mm from the electrode): the lobe stays at "
    f"{rng(rc, 'lobe_neg_peak_ms')} ms and the EOF peak walks {rc[0]['eof_peak_ms']:.1f}→{rc[-1]['eof_peak_ms']:.1f} ms against "
    f"L2/v = {rc[0]['eof_expected_L2_over_v_ms']:.0f}→{rc[-1]['eof_expected_L2_over_v_ms']:.0f} ms (peak − L2/v within "
    f"{max(abs(x) for x in NUM['c_distal_tendon']['eof_peak_minus_L2_over_v_ms']):.1f} ms; 5 %-onsets "
    f"{rc[0]['eof_onset_ms']:.1f}→{rc[-1]['eof_onset_ms']:.1f} ms); p2p "
    f"{NUM['c_distal_tendon']['p2p_rel_range'][0]:.2f}–{NUM['c_distal_tendon']['p2p_rel_range'][1]:.2f} A0, lowest with the tendon under the electrode.",
    f"(d) Velocity 2→6 m/s (dz = v/fs kept): the lobe arrives at (Δz + {dmean:.1f} mm)/v = "
    f"{rd[0]['lobe_neg_peak_ms']:.1f}→{rd[-1]['lobe_neg_peak_ms']:.1f} ms, the EOF peak at {rd[0]['eof_peak_ms']:.1f}→"
    f"{rd[-1]['eof_peak_ms']:.1f} ms (L/v = {rd[0]['eof_expected_L_over_v_ms']:.0f}→{rd[-1]['eof_expected_L_over_v_ms']:.0f}), "
    f"the 10 %-duration stretches as 1/v ({rd[0]['duration10_ms']:.1f}→{rd[-1]['duration10_ms']:.1f} ms; duration·v = "
    f"{min(NUM['d_velocity']['duration_times_v']):.0f}–{max(NUM['d_velocity']['duration_times_v']):.0f} mm) and MNF ∝ v "
    f"({rd[0]['mnf_hz']:.0f}→{rd[-1]['mnf_hz']:.0f} Hz); p2p {rng(rd, 'p2p_rel_A0', '.2f')} A0.",
    f"(e) Fat 1→18 mm (fibre 5 mm inside the muscle, so 8→25 mm below the skin): p2p "
    f"{re[0]['p2p_rel_A0']:.2f}→{re[-1]['p2p_rel_A0']:.2f} A0 ({re[0]['p2p_au'] / re[-1]['p2p_au']:.0f}×) and MNF "
    f"{re[0]['mnf_hz']:.0f}→{re[-1]['mnf_hz']:.0f} Hz, lobe FWHM {re[0]['lobe_fwhm_ms']:.1f}→{re[-1]['lobe_fwhm_ms']:.1f} ms; "
    f"timing unchanged (lobe {rng(re, 'lobe_neg_peak_ms')} ms, EOF peak {rng(re, 'eof_peak_ms')} ms). "
    f"At equal depth below the skin the same fibre under the default 3-mm fat gives "
    f"{re[-1]['same_depth_default_fat']['p2p_rel_A0']:.3f} A0 at 25 mm (fat spreads the potential, muscle shunts it).",
    f"(f) FCU pool, real anatomy: MUs {', '.join(str(r['mu']) for r in rf)} ({min(r['n_fibres'] for r in rf)}–"
    f"{max(r['n_fibres'] for r in rf)} fibres) at d = {rf[0]['depth_mm']:.1f}→{rf[-1]['depth_mm']:.1f} mm from the centre electrode: "
    f"p2p {rf[0]['p2p_uV']:.0f}→{rf[-1]['p2p_uV']:.0f} µV (p2p ∝ d^-{n_f:.1f}, r² {r2_f:.2f}; per fibre d^-{n_f1:.1f}), "
    f"lobe FWHM {rf[0]['lobe_fwhm_ms']:.1f}→{rf[-1]['lobe_fwhm_ms']:.1f} ms, 10 %-duration {rf[0]['duration10_ms']:.1f}→{rf[-1]['duration10_ms']:.1f} ms, "
    f"MNF {rf[0]['mnf_hz']:.0f}→{rf[-1]['mnf_hz']:.0f} Hz; the lobe at {rng(rf, 'lobe_neg_peak_ms')} ms is the common "
    f"{rf[0]['iz_to_electrode_along_z_mm']:.0f}-mm IZ–electrode distance at 4 m/s ({rf[0]['iz_to_electrode_along_z_mm'] / V:.1f} ms) "
    f"plus a lag that grows with d.",
]
NUM["summary"] = lines
NUM["runtime_s"] = time.time() - T0
KEY.write_text(json.dumps(_jsonable(NUM), indent=1))
print(f"[key numbers → {KEY.name}]  ({NUM['runtime_s']:.0f} s)")
print("\n".join(lines))
