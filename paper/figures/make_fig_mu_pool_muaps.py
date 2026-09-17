"""Fig 7 — the 100-MU Henneman pool of the FCU and its MUAPs (direct line-source synthesis): territories, size
distribution / recruitment order, small–medium–large MUAPs at the centre grid electrode,
and MUAP peak-to-peak vs fibre count.

Reads the cached lead fields + MUAP tensor built by f2_common (the single-channel MUAPs
are the centre electrode of the 5×5 grid — one lead-field set serves every panel).

Run: /home/dc23/miniconda3/envs/fenicsx-env/bin/python paper/figures/make_fig_mu_pool_muaps.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "paper/figures")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

from style import use, save, letter, COL, W2
import f2_common as C

use()
T0 = time.time()
KN = {}

# --------------------------------------------------------------------------- data
fm = C.load_fibre_model()
bed = C.poisson_bed(fm)
pool = C.henneman_pool(bed)
D = C.load_muaps()                                     # F2_TENSOR = new | legacy42 | auto
t_ms = D["t_ms"]
e0 = D["elec_xyz"][C.M // 2, C.M // 2]                 # centre electrode (row 2, col 2)
muap = D["muap_single"]                                # (100, 256) single-channel direct-method MUAPs at e0
dt = float(t_ms[1] - t_ms[0])
KN["tensor_source"] = D["source"]; KN["grid_note"] = D["grid_note"]
print(f"MUAP source: {D['source']} — {D['grid_note']}")

sizes = np.array([m.size for m in pool])
cxy = np.array([m.centre_xy for m in pool])
rad = np.array([m.territory_radius_mm for m in pool])
depth = np.linalg.norm(cxy - e0[:2], axis=1)           # MU centre → electrode (in the electrode plane)
p2p = np.ptp(muap, axis=1)


def duration_ms(m):
    """The library's MUAP duration metric (api._compute_metrics): |m| > 10 % of p2p."""
    above = np.abs(m) > 0.1 * np.ptp(m)
    first = int(np.argmax(above)); last = len(above) - 1 - int(np.argmax(above[::-1]))
    return (last - first) * dt


dur = np.array([duration_ms(m) for m in muap])
from emgforge.activation import MotoneuronPool
rte = MotoneuronPool(n_mu=len(pool), fs=C.FS).rte

# representative small / medium / large MUs: the median-p2p member of each size band
bands = {"small": sizes <= 10, "medium": (sizes >= 40) & (sizes <= 100), "large": sizes >= 200}
picks = {}
for nm, sel in bands.items():
    idx = np.where(sel)[0]
    picks[nm] = int(idx[np.argsort(p2p[idx])[len(idx) // 2]])
pk = [picks["small"], picks["medium"], picks["large"]]

# p2p ∝ size^k (log-log OLS) and the depth-controlled version
X = np.log10(sizes); Y = np.log10(p2p)
k_slope, k_icpt = np.polyfit(X, Y, 1)
r_fit = float(np.corrcoef(X, Y)[0, 1])
A = np.c_[X, depth, np.ones_like(X)]
coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
# fibre-sharing statistics
count = np.zeros(len(bed.r_norms), int)
for m in pool:
    count[m.fiber_idxs] += 1

KN.update(
    n_mu=len(pool), n_bed_fibres=len(bed.r_norms), electrode_xyz=e0.tolist(),
    sizes=dict(min=int(sizes.min()), median=float(np.median(sizes)), max=int(sizes.max()), total=int(sizes.sum()),
               mean=float(sizes.mean())),
    territory_radius_mm=dict(min=rad.min(), median=np.median(rad), max=rad.max()),
    fibre_sharing=dict(frac_bed_fibres_in_any_mu=float((count > 0).mean()), mean_mus_per_fibre=float(count.mean()),
                       max_mus_per_fibre=int(count.max())),
    muap_p2p_uV=dict(min=p2p.min() * 1e6, median=np.median(p2p) * 1e6, max=p2p.max() * 1e6,
                     range_fold=float(p2p.max() / p2p.min())),
    muap_duration_ms=dict(min=dur.min(), median=np.median(dur), max=dur.max()),
    depth_mm=dict(min=depth.min(), median=np.median(depth), max=depth.max()),
    p2p_vs_size=dict(slope=k_slope, intercept=k_icpt, r=r_fit,
                     slope_depth_controlled=coef[0], depth_coef_per_mm=coef[1]),
    picks={nm: dict(mu=int(k), size=int(sizes[k]), p2p_uV=p2p[k] * 1e6, duration_ms=dur[k], depth_mm=depth[k],
                    rte=rte[k]) for nm, k in picks.items()},
    rte=dict(first=float(rte[0]), last=float(rte[-1])),
    p2p_of_first_last_uV=[p2p[0] * 1e6, p2p[-1] * 1e6],
)
print(f"pool: {len(pool)} MUs, sizes {sizes.min()}–{sizes.max()} (median {np.median(sizes):.0f}), "
      f"territory radius {rad.min():.1f}–{rad.max():.1f} mm")
print(f"MUAP p2p {p2p.min()*1e6:.3f}–{p2p.max()*1e6:.2f} µV (median {np.median(p2p)*1e6:.2f}), "
      f"duration median {np.median(dur):.1f} ms; p2p ∝ size^{k_slope:.2f} (r={r_fit:.2f}; "
      f"depth-controlled {coef[0]:.2f}, {coef[1]:+.3f} dex/mm)")
print("picks:", KN["picks"])

# --------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(W2, 4.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
                      left=0.07, right=0.98, top=0.95, bottom=0.09, wspace=0.32, hspace=0.42)
snorm = LogNorm(sizes.min(), sizes.max())
scol = lambda n: plt.cm.viridis(snorm(n))
pick_lab = {picks["small"]: "S", picks["medium"]: "M", picks["large"]: "L"}

# (a) territories ----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
zmid = 0.5 * (bed.z_vals[0] + bed.z_vals[-1])
outline = C.fcu_outline(fm, zmid)
for c in outline:
    ax.plot(c[:, 0], c[:, 1], color="k", lw=0.7, zorder=6)
clip = PathPatch(MPath(np.vstack([outline[0], outline[0][:1]]), closed=True), transform=ax.transData,
                 facecolor="none", edgecolor="none")
ax.add_patch(clip)
ax.scatter(bed.xy_mid[:, 0], bed.xy_mid[:, 1], s=1.2, c="0.75", lw=0)
for m in pool:                                          # territories, clipped to the muscle outline
    circ = Circle(m.centre_xy, m.territory_radius_mm, fill=False, ec=scol(m.size), lw=0.5, alpha=0.75)
    ax.add_patch(circ); circ.set_clip_path(clip)
for k in sorted(pk, key=lambda k: -sizes[k]):           # large → small so the small MU stays visible
    fx = bed.xy_mid[pool[k].fiber_idxs]
    ax.scatter(fx[:, 0], fx[:, 1], s=5 if sizes[k] < 100 else 3.5, color=scol(sizes[k]), edgecolor="k",
               lw=0.25, zorder=5, alpha=1.0 if sizes[k] < 100 else 0.7)
    ax.annotate(pick_lab[k], xy=pool[k].centre_xy, xytext=(4, 4), textcoords="offset points",
                fontsize=7, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="w", ec="none", alpha=0.7))
ax.scatter(*e0[:2], marker="v", s=30, facecolor="w", edgecolor="k", lw=0.8, zorder=6)
ax.text(e0[0] + 2.5, e0[1], "electrode", fontsize=6, va="center")
ax.set_aspect("equal"); ax.autoscale_view()
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
cb = fig.colorbar(ScalarMappable(norm=snorm, cmap="viridis"), ax=ax, fraction=0.04, pad=0.03)
cb.set_label("fibres per MU", fontsize=7); cb.ax.tick_params(labelsize=6)
letter(ax, "a")

# (b) size distribution + recruitment order ----------------------------------
gsb = gs[0, 1].subgridspec(1, 2, wspace=0.6)
axh = fig.add_subplot(gsb[0, 0])
bins = np.geomspace(sizes.min() * 0.95, sizes.max() * 1.05, 12)
axh.hist(sizes, bins=bins, color="0.6", lw=0)
axh.set_xscale("log"); axh.set_xlabel("fibres per MU"); axh.set_ylabel("# MUs")
# recruitment threshold vs recruitment order (the pool is recruited in size order by
# construction, so order-vs-size is monotone and carries nothing; the threshold curve does)
axr = fig.add_subplot(gsb[0, 1])
order = np.arange(len(pool))
axr.scatter(order, rte, s=7, c=sizes, cmap="viridis", norm=snorm, lw=0)
for k in pk:
    axr.annotate(pick_lab[k], xy=(k, rte[k]), xytext=(-8, 1), textcoords="offset points", fontsize=7, fontweight="bold")
axr.set_xlabel("recruitment order"); axr.set_ylabel("recruitment threshold\n(fraction of maximal drive)")
axr.set_xlim(-2, len(pool) + 1); axr.set_ylim(0, rte.max() * 1.08)
axr.text(0.03, 0.97, "colour: fibres per MU", transform=axr.transAxes, fontsize=6, va="top")
KN["rte"].update(median=float(np.median(rte)), n_below_0p1=int(np.sum(rte < 0.1)), n_below_0p5=int(np.sum(rte < 0.5)))
letter(axh, "b", dx=-0.4)

# (c) small / medium / large MUAPs -------------------------------------------
gsc = gs[1, 0].subgridspec(3, 1, hspace=0.12)
for r, k in enumerate(pk):
    ax = fig.add_subplot(gsc[r, 0])
    ax.axhline(0, color="0.7", lw=0.4)
    ax.plot(t_ms, muap[k] * 1e6, color=scol(sizes[k]), lw=1.0)
    ym = np.abs(muap[k]).max() * 1e6 * 1.15
    ax.set_ylim(-ym, ym); ax.set_xlim(0, 40)
    ax.text(0.99, 0.95, f"{pick_lab[k]}: MU {k}, {sizes[k]} fibres, d = {depth[k]:.0f} mm, {p2p[k]*1e6:.2f} µV",
            transform=ax.transAxes, fontsize=6, ha="right", va="top")
    if r < 2:
        ax.set_xticklabels([])
    if r == 1:
        ax.set_ylabel("MUAP (µV)")
    if r == 0:
        letter(ax, "c")
ax.set_xlabel("t (ms), 0 = NMJ firing")

# (d) p2p vs size -------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(sizes, p2p * 1e6, s=10, c=depth, cmap="magma_r", lw=0)
xx = np.array([sizes.min(), sizes.max()])
ax.plot(xx, 10 ** (k_icpt + k_slope * np.log10(xx)) * 1e6, color="k", lw=0.8, ls="--")
for k in pk:
    ax.annotate(pick_lab[k], xy=(sizes[k], p2p[k] * 1e6), xytext=(3, 3), textcoords="offset points",
                fontsize=7, fontweight="bold")
ax.text(0.03, 0.95, f"p2p ∝ n$^{{{k_slope:.2f}}}$  (r = {r_fit:.2f})", transform=ax.transAxes, fontsize=7, va="top")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("fibres per MU"); ax.set_ylabel("MUAP peak-to-peak (µV)")
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("MU centre depth below electrode (mm)", fontsize=7); cb.ax.tick_params(labelsize=6)
letter(ax, "d")

save(fig, "fig_mu_pool_muaps")
KN["total_s"] = time.time() - T0
C.update_key_numbers("fig7_mu_pool_muaps", KN)
