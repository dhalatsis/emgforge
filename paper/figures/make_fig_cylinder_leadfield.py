"""Figure 2 — cylinder lead fields: (a) cross-section schematic, (b) analytical vs FEM
φ(z) at three depths, (c) transverse profile at the fibre depth, (d) FEM lead-field map
on the electrode plane (one reciprocal solve).

Run from the repo root:  python paper/figures/make_fig_cylinder_leadfield.py   (~40 s)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "paper/figures")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LogNorm

from style import use, save, letter, COL, W1, W2               # noqa: F401
from f1_common import (CACHE, DZ, RADII, THETAS, W, Z, ZE, ana_phi, cyl_fem, dc_free,
                       fem_phi, fwhm, normed, record)

use()
R = dict(bone=10.0, muscle=35.0, fat=38.0, skin=40.0)
FIB_R = (33.0, 27.0, 20.0)                       # fibre radial positions (depth 7/13/20 mm)
NUM = {}

# ------------------------------------------------------------------ FEM: one solve
t0 = time.time()
geo = cyl_fem._geo()
model = cyl_fem._model(cyl_fem.MESH_CACHE / "cyl_10_35_38_40.msh",
                       cyl_fem.MESH_CACHE / "cyl_10_35_38_40.json", geo)
uh = model.solve_for_point(geo.electrode_on_skin(0.0, ZE))
NUM["fem_model_plus_solve_s"] = time.time() - t0
t0 = time.time()
g = np.linspace(-40, 40, 241)
X, Y = np.meshgrid(g, g)
pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, ZE)])
inside = np.hypot(pts[:, 0], pts[:, 1]) <= R["skin"] - 0.15
plane = np.full(X.size, np.nan)
plane[inside] = model.evaluate_solution_at_points(pts[inside], uh=uh)
plane = plane.reshape(X.shape)
th_arc = np.arange(-90.0, 90.01, 1.0)
arc_pts = np.column_stack([30 * np.cos(np.deg2rad(th_arc)), 30 * np.sin(np.deg2rad(th_arc)),
                           np.full(len(th_arc), ZE)])
arc_fem = model.evaluate_solution_at_points(arc_pts, uh=uh)
NUM["fem_plane_eval_s"] = time.time() - t0
del model, uh

# The FEM solution carries an arbitrary constant (pure-Neumann, zero-mean). Along-fibre
# curves are referenced to their far-z samples (the tier-A convention); the log map is
# referenced to the minimum over the cross-section so that log|φ| has no zero crossing.
pf30 = fem_phi(30.0, 0.0)
dc_fem = float(np.mean(np.r_[pf30[:12], pf30[-12:]]))
plane_min = float(np.nanmin(plane))
NUM["baseline"] = dict(far_z_line_value=dc_fem, plane_min=plane_min, difference=dc_fem - plane_min,
                       peak_minus_far_z=float(np.nanmax(plane) - dc_fem))

# ------------------------------------------------------------------ (b) φ(z) at 3 depths
curves = {}
for r in FIB_R:
    pa = dc_free(ana_phi(r)[0]); pf = dc_free(fem_phi(r, 0.0))
    core = np.abs(Z) <= 60.0
    curves[r] = dict(ana=normed(pa), fem=normed(pf), fwhm_ana=fwhm(Z, pa), fwhm_fem=fwhm(Z, pf),
                     r=float(np.corrcoef(pa[core], pf[core])[0, 1]))
NUM["phi_z"] = {f"r{r:g}_depth{40 - r:g}mm": dict(fwhm_ana_mm=c["fwhm_ana"], fwhm_fem_mm=c["fwhm_fem"],
                                                  r_core=c["r"]) for r, c in curves.items()}

# ------------------------------------------------------------------ (c) transverse profile
th_ana = np.arange(0.0, 90.01, 5.0)
peak_ana = np.array([dc_free(ana_phi(30.0, distfib=th)[0])[W // 2] for th in th_ana])
s_ana = np.deg2rad(th_ana) * 30.0
s_fem = np.deg2rad(th_arc) * 30.0
k0 = int(np.argmin(np.abs(th_arc)))
prof_fem = (arc_fem - dc_fem) / (arc_fem[k0] - dc_fem)                # far-z baseline
prof_fem_alt = (arc_fem - plane_min) / (arc_fem[k0] - plane_min)      # cross-section-minimum baseline
prof_ana = peak_ana / peak_ana[0]
s_ana2, prof_ana2 = np.r_[-s_ana[:0:-1], s_ana], np.r_[prof_ana[:0:-1], prof_ana]
NUM["transverse_r30"] = dict(
    fwhm_arc_fem_mm=fwhm(s_fem, prof_fem), fwhm_arc_fem_alt_baseline_mm=fwhm(s_fem, prof_fem_alt),
    fwhm_arc_ana_mm=fwhm(s_ana2, prof_ana2),
    fem_over_ana_at_deg={f"{t:g}": float(np.interp(t, th_arc, prof_fem) / np.interp(t, th_ana, prof_ana))
                         for t in (15, 30, 45, 60, 90)},
    profile_at_90deg=dict(fem=float(prof_fem[-1]), fem_alt_baseline=float(prof_fem_alt[-1]), ana=float(prof_ana[-1])),
)
NUM["leadfield_map"] = dict(phi_max_minus_min=float(np.nanmax(plane) - plane_min), decades_shown=2.5)
# diagnostic: is the narrow FEM transverse profile the σ = 5 mm electrode blob? Same lines, σ = 1 mm source
phi_s1 = cyl_fem.sigma1_lines()
i30 = int(np.argmin(np.abs(RADII - 30.0)))
sel = THETAS <= 90.0
pk = {}
for name, lines in (("sigma5", CACHE["phi"][i30]), ("sigma1", phi_s1[i30])):
    v = np.array([dc_free(cyl_fem.window(lines[j], CACHE["z_abs"], ZE, W, DZ))[W // 2] for j in np.where(sel)[0]])
    pk[name] = v / v[0]
NUM["transverse_r30"]["cache_lines_theta_deg"] = THETAS[sel]
NUM["transverse_r30"]["fem_peak_profile_sigma5_source"] = pk["sigma5"]
NUM["transverse_r30"]["fem_peak_profile_sigma1_source"] = pk["sigma1"]
NUM["transverse_r30"]["analytical_peak_profile_same_theta"] = np.interp(THETAS[sel], th_ana, prof_ana)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(W2, 5.3))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], wspace=0.28, hspace=0.32, left=0.02, right=0.985,
                      top=0.96, bottom=0.08)

# (a) schematic --------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
tissue = [("skin", R["skin"], "#c9b39f"), ("fat", R["fat"], "#f5e3a3"),
          ("muscle", R["muscle"], "#f0c2b0"), ("bone", R["bone"], "#e3e3e3")]
for name, rad, col in tissue:
    ax.add_patch(Circle((0, 0), rad, facecolor=col, edgecolor="0.35", lw=0.5))
ax.text(0, 0, "bone\n(r = 10)", fontsize=6.5, ha="center", va="center")
ax.text(-17, 17, "muscle\n(r = 35)", fontsize=6.5, ha="center", va="center")
for name, rad, ang, ty in (("fat (r = 38)", 36.5, 118, 52), ("skin (r = 40)", 39.2, 132, 45)):
    x, y = rad * np.cos(np.deg2rad(ang)), rad * np.sin(np.deg2rad(ang))
    ax.annotate(name, (x, y), (-55, ty), fontsize=6.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3", shrinkA=0, shrinkB=0))
# electrode (Ø10 mm disc seen edge-on) on the skin at θ = 0
ax.add_patch(Rectangle((R["skin"] - 0.3, -5), 1.5, 10, facecolor="k", edgecolor="none"))
ax.text(R["skin"] + 3.2, 0, "electrode\n(Ø10 mm)", fontsize=6.5, va="center", ha="left")
for rad, ty in zip(FIB_R, (-9.0, -9.0, -9.0)):
    ax.plot(rad, 0, "o", ms=3.4, color=COL["first"], mec="white", mew=0.4, zorder=5)
    ax.annotate(f"{40 - rad:g}", (rad, -0.8), (rad, ty), fontsize=6, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3", shrinkA=0, shrinkB=0))
ax.text(26.5, -14.5, "fibres: depth below skin (mm)", fontsize=6, ha="center", va="top")
ax.plot([-40, -30], [-47, -47], "k-", lw=1.2); ax.text(-35, -49.5, "10 mm", fontsize=6, ha="center", va="top")
ax.text(12, -47, "fibres run along z\n(out of the page)", fontsize=6, ha="center", va="center", color="0.3")
ax.set_xlim(-60, 64); ax.set_ylim(-58, 58); ax.set_aspect("equal"); ax.axis("off")
letter(ax, "a", dx=0.0, dy=0.96)

# (b) φ(z) analytical vs FEM -------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
for k, (r, c) in enumerate(curves.items()):
    ax.plot(Z, c["ana"], color=COL["analytical"], lw=1.0, label="analytical (Farina 2004)" if k == 0 else None)
    ax.plot(Z, c["fem"], color=COL["fem"], lw=0.9, ls="--", label="FEM (FEniCSx)" if k == 0 else None)
    yv = 0.5 * (np.interp(78, Z, c["ana"]) + np.interp(78, Z, c["fem"]))
    ax.text(80.5, yv, f"{40 - r:g} mm", fontsize=6, ha="left", va="center")
ax.axhline(0, color="0.8", lw=0.5, zorder=0)
ax.set_xlim(-80, 96); ax.set_ylim(-0.06, 1.1)
ax.set_xlabel("z along the fibre (mm)"); ax.set_ylabel("φ / max φ (baseline removed)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.8, borderaxespad=0.0, fontsize=6.5, labelspacing=0.3)
# FWHM table pinned to the top-right corner, right of the legend (the peak at z = 0 sits between them)
txt = "FWHM (mm)  ana / FEM\n" + "\n".join(f"{40 - r:>2.0f} mm deep: {c['fwhm_ana']:>3.0f} / {c['fwhm_fem']:>3.0f}"
                                           for r, c in curves.items())
ax.text(0.99, 0.99, txt, transform=ax.transAxes, fontsize=6, ha="right", va="top", family="DejaVu Sans Mono")
ax.text(0.99, 0.62, "electrode at z = 0, θ = 0", transform=ax.transAxes, fontsize=6, ha="right", va="top", color="0.35")
letter(ax, "b", dx=-0.14)

# (c) transverse ---------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.fill_between(s_fem, prof_fem, prof_fem_alt, color=COL["fem"], alpha=0.18, lw=0, label="FEM baseline uncertainty")
ax.plot(s_fem, prof_fem, color=COL["fem"], lw=1.0, label="FEM, σ = 5 mm source")
s_c = np.deg2rad(THETAS[sel]) * 30.0
ax.plot(np.r_[-s_c[:0:-1], s_c], np.r_[pk["sigma1"][:0:-1], pk["sigma1"]], "s", ms=2.6, mfc="none",
        color=COL["fem"], mew=0.8, label="FEM, σ = 1 mm source")
ax.plot(s_ana2, prof_ana2, "o", ms=2.8, mfc="none", color=COL["analytical"], mew=0.8, label="analytical, Ø10 mm disc")
ax.axhline(0.5, color="0.8", lw=0.5, zorder=0)
# head-room above the peak holds the two-column legend; the FWHM note sits below it at the left,
# above the flanks of the profiles — nothing overlaps the data
ax.set_xlim(-48, 48); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("transverse arc length s = rθ at r = 30 mm (mm)"); ax.set_ylabel("peak φ / peak φ(θ = 0)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6, borderaxespad=0.1, fontsize=6.2,
          labelspacing=0.3, columnspacing=1.2, handletextpad=0.5)
ax.text(0.02, 0.8, f"FWHM: FEM {NUM['transverse_r30']['fwhm_arc_fem_mm']:.0f} mm,\nanalytical "
        f"{NUM['transverse_r30']['fwhm_arc_ana_mm']:.0f} mm", transform=ax.transAxes, fontsize=6, ha="left", va="top", color="0.35")
sec = ax.secondary_xaxis("top", functions=(lambda s: np.rad2deg(s / 30.0), lambda d: np.deg2rad(d) * 30.0))
sec.set_xlabel("θ (deg)", fontsize=7, labelpad=1); sec.tick_params(labelsize=6)
letter(ax, "c", dx=-0.16, dy=1.12)

# (d) lead-field map -----------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
mag = plane - plane_min
vmax = np.nanmax(mag); vmin = vmax / 10 ** NUM["leadfield_map"]["decades_shown"]
im = ax.pcolormesh(X, Y, np.clip(mag, vmin, vmax), cmap="YlOrBr", norm=LogNorm(vmin, vmax),
                   shading="nearest", rasterized=True)
for name, rad, _ in tissue:
    ax.add_patch(Circle((0, 0), rad, facecolor="none", edgecolor="white" if name != "skin" else "0.3",
                        lw=0.5, ls="-" if name == "skin" else (0, (2, 1.5))))
ax.plot(R["skin"], 0, "s", ms=3.5, color="k", mec="white", mew=0.4)
for rad in FIB_R:
    ax.plot(rad, 0, "o", ms=2.6, color="k", mec="white", mew=0.3)
ax.set_xlim(-42, 42); ax.set_ylim(-42, 42); ax.set_aspect("equal"); ax.axis("off")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.8)
cb.set_label("φ − min φ (a.u., log scale)", fontsize=6.5, labelpad=2); cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)
ax.text(0, -44.5, "electrode plane z = z$_e$; ■ electrode, ● fibres of (b)", fontsize=6, ha="center", va="top", color="0.3")
letter(ax, "d", dx=-0.02, dy=0.96)

save(fig, "fig_cylinder_leadfield")
record("fig2_cylinder_leadfield", NUM)
