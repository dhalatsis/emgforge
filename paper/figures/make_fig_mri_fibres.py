"""Fig 6 — the MRI tier: WR forearm segmentation with the HD-EMG grid, the FCU fibre
beds (Poisson straight vs harmonic single-NMJ streamlines), the reciprocal lead field of
one grid electrode on the axial mesh slice, and φ along three fibres at different depths.

Run: /home/dc23/miniconda3/envs/fenicsx-env/bin/python paper/figures/make_fig_mri_fibres.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "paper/figures")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from style import use, save, letter, COL, W2
import f2_common as C

use()
T0 = time.time()
KN = {}

# --------------------------------------------------------------------------- data
t0 = time.time()
fm = C.load_fibre_model()
pbed = C.poisson_bed(fm)
KN["fibre_model_build_s"] = time.time() - t0
hbed_def = C.harmonic_bed(fm, grid_mm=2.0)                 # library default seeding
hbed = C.harmonic_bed(fm, grid_mm=C.HARM_GRID_MM)          # density-matched (plotted)
KN["harmonic_bed_build_s"] = dict(default_2mm=getattr(hbed_def, "build_seconds", None),
                                  dense=getattr(hbed, "build_seconds", None))
KN["harmonic_seed_grid_mm"] = dict(default=2.0, dense=C.HARM_GRID_MM)
fem = C.build_fem()
KN["fem_build_s"] = fem.build_seconds
elec, ginfo = C.grid_electrodes(fem, fm, pbed)
IE, JE = 2, 2                                            # the centre electrode
e0 = elec[IE, JE]
t0 = time.time(); fem.solve_for_point(e0, source_sigma=5.0); KN["fem_solve_s"] = time.time() - t0
vs = fm.voxel_size
KN.update(grid=ginfo, electrode_xyz=e0.tolist())

# --------------------------------------------------------------------------- bed statistics
N_p, N_h = len(pbed.r_norms), len(hbed.r_norms)
cont_p, _ = C.mask_containment(pbed.paths, fm)
cont_h, per_h = C.mask_containment(hbed.paths, fm)
arc_p_dz, L_p = C.bed_arc_geometry(pbed)
L_h = hbed.half1_mm + hbed.half2_mm
# longitudinal span fraction of each streamline in the muscle's PCA frame (the criterion
# harmonic_fibers.HarmonicFibreField uses; solve=False builds only the frame)
from emgforge.mri.core.harmonic_fibers import HarmonicFibreField
frame = HarmonicFibreField(fm.seg_data == C.FCU, vs, solve=False)
span_h = np.array([np.ptp(frame._long_fraction(np.asarray(p))) for p in hbed.paths])
span_p = np.array([np.ptp(frame._long_fraction(p)) for p in pbed.paths])
# NMJ points: harmonic = arc position half1 (index half1/dz); Poisson = IZ_FRAC of the path
nmj_h = np.array([np.asarray(p)[int(round(h1 / hbed.dz_mm))] for p, h1 in zip(hbed.paths, hbed.half1_mm)])
k_iz = int(round(C.IZ_FRAC * (pbed.paths.shape[1] - 1)))
nmj_p = pbed.paths[:, k_iz]
z_fcu = frame.z1 - frame.z0
# the library-default (2 mm seed) harmonic bed, for the record
N_hd = len(hbed_def.r_norms)
cont_hd, _ = C.mask_containment(hbed_def.paths, fm)
span_hd = np.array([np.ptp(frame._long_fraction(np.asarray(p))) for p in hbed_def.paths])
L_hd = hbed_def.half1_mm + hbed_def.half2_mm
KN["harmonic_default_2mm"] = dict(n_fibres=N_hd, containment=cont_hd, span_median=np.median(span_hd),
                                  length_mean_mm=L_hd.mean())
print(f"harmonic bed, library-default 2 mm seeds: {N_hd} fibres, containment {cont_hd*100:.1f} %, "
      f"span median {np.median(span_hd):.2f}, length {L_hd.mean():.0f} mm")
KN.update(
    n_fibres_poisson=N_p, n_fibres_harmonic=N_h,
    containment_poisson=cont_p, containment_harmonic=cont_h,
    harmonic_fraction_fibres_fully_inside=float((per_h >= 0.999).mean()),
    poisson_fibre_length_mm=dict(mean=L_p.mean(), min=L_p.min(), max=L_p.max(), half_mm=pbed.half_mm),
    harmonic_fibre_length_mm=dict(mean=L_h.mean(), median=np.median(L_h), min=L_h.min(), max=L_h.max()),
    harmonic_span_fraction=dict(median=np.median(span_h), q25=np.percentile(span_h, 25),
                                q75=np.percentile(span_h, 75), min=span_h.min(), max=span_h.max(),
                                frac_above_0p8=float((span_h > 0.8).mean())),
    poisson_span_fraction=dict(median=np.median(span_p), min=span_p.min(), max=span_p.max()),
    fcu_z_extent_mm=z_fcu,
    harmonic_nmj=dict(iz_fraction_mean=float(hbed.iz_fractions.mean()), iz_fraction_std=float(hbed.iz_fractions.std()),
                      z_mean=nmj_h[:, 2].mean(), z_std=nmj_h[:, 2].std(),
                      z_range=[nmj_h[:, 2].min(), nmj_h[:, 2].max()],
                      z_spread_fraction_of_muscle=float(np.ptp(nmj_h[:, 2]) / z_fcu)),
    poisson_nmj=dict(iz_fraction=C.IZ_FRAC, z_mean=nmj_p[:, 2].mean(), z_std=nmj_p[:, 2].std()),
    poisson_bed=dict(density_per_mm2=C.DENSITY, area_mm2=pbed.cross_section_area_mm2,
                     z_range=[pbed.z_vals[0], pbed.z_vals[-1]], dz_mm=float(pbed.z_vals[1] - pbed.z_vals[0])),
)
print(f"FCU beds: Poisson {N_p} fibres (containment {cont_p*100:.1f} %, length {L_p.mean():.0f} mm), "
      f"harmonic {N_h} fibres (containment {cont_h*100:.1f} %, length {L_h.mean():.0f} mm, "
      f"span median {np.median(span_h):.2f} [{np.percentile(span_h,25):.2f}–{np.percentile(span_h,75):.2f}])")
print(f"harmonic NMJ: iz {hbed.iz_fractions.mean():.3f}±{hbed.iz_fractions.std():.3f}, z {nmj_h[:,2].mean():.1f}±{nmj_h[:,2].std():.1f} mm "
      f"(range {np.ptp(nmj_h[:,2]):.1f} mm = {np.ptp(nmj_h[:,2])/z_fcu*100:.1f} % of muscle); "
      f"Poisson IZ z {nmj_p[:,2].mean():.1f}±{nmj_p[:,2].std():.1f} mm")
print(f"grid: {ginfo}")

# --------------------------------------------------------------------------- panel data
# (a) axial slice at the electrode z
kz = C.seg_slice_index(fm, e0[2])
sl = fm.seg_data[:, :, kz]
rgb = np.ones(sl.shape + (3,))
rgb[sl > 0] = to_rgb("#dcdcdc")                       # muscles: light grey
rgb[np.isin(sl, [15, 25])] = to_rgb("#f3e8d6")        # fat / skin: beige
rgb[np.isin(sl, [2, 3])] = to_rgb("#9a9a9a")          # bone: dark grey
rgb[sl == C.FCU] = to_rgb("#e6550d")                  # FCU highlight
ext = [-vs[0] / 2, (sl.shape[0] - 0.5) * vs[0], -vs[1] / 2, (sl.shape[1] - 0.5) * vs[1]]
tis = np.argwhere(sl > 0) * vs[:2]
bb = [tis[:, 0].min() - 4, tis[:, 0].max() + 4, tis[:, 1].min() - 4, tis[:, 1].max() + 4]
lab_pos = {l: (np.argwhere(sl == l) * vs[:2]).mean(0) for l in (2, 3, C.FCU) if (sl == l).any()}

# (c) lead field on the slice
t0 = time.time()
xs = np.linspace(bb[0], bb[1], 260); ys = np.linspace(bb[2], bb[3], 260)
XX, YY = np.meshgrid(xs, ys)
pts = np.c_[XX.ravel(), YY.ravel(), np.full(XX.size, e0[2])]
ci = np.round(pts / vs).astype(int)
inside = (np.all(ci >= 0, 1) & (ci[:, 0] < sl.shape[0]) & (ci[:, 1] < sl.shape[1]))
lab = np.zeros(len(pts), int); lab[inside] = sl[ci[inside, 0], ci[inside, 1]]
phi_sl = fem.evaluate_solution_at_points(pts)
phi_sl = np.where(lab > 0, phi_sl, np.nan).reshape(XX.shape)
KN["slice_eval_s"] = time.time() - t0
logphi = np.log10(np.abs(phi_sl) * 1e3 + 1e-12)           # φ in mV
vmax = np.nanmax(logphi); vmin = vmax - 3.5

# (d) φ along three fibres at different depths (distance electrode → fibre)
dist = np.array([np.linalg.norm(p - e0, axis=1).min() for p in pbed.paths])
order = np.argsort(dist)
pick = [order[0], order[len(order) // 2], order[-1]]
phi_fib = C.phi_along_paths(fem, pbed.paths[pick])
from emgforge.synthesis.preprocessing import denoise_field_n
KN["phi_fibres"] = [dict(fibre=int(i), depth_mm=float(dist[i]), phi_peak_mV=float(np.abs(phi_fib[k]).max() * 1e3),
                         xy_mid=pbed.xy_mid[i].tolist()) for k, i in enumerate(pick)]
print("φ along fibres:", [(round(dist[i], 1), round(float(np.abs(phi_fib[k]).max() * 1e3), 4)) for k, i in enumerate(pick)])

# side-view lateral coordinate: the one with the larger fibre excursion in the harmonic bed
exc = [np.mean([np.ptp(np.asarray(p)[:, d]) for p in hbed.paths]) for d in (0, 1)]
LAT = int(np.argmax(exc)); LATNAME = "xy"[LAT]
KN["side_view_axis"] = LATNAME
KN["harmonic_lateral_excursion_mm"] = dict(x=exc[0], y=exc[1])
KN["poisson_lateral_excursion_mm"] = dict(x=float(np.mean(np.ptp(pbed.paths[:, :, 0], 1))), y=float(np.mean(np.ptp(pbed.paths[:, :, 1], 1))))

# --------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(W2, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], height_ratios=[1.0, 1.0],
                      left=0.06, right=0.99, top=0.96, bottom=0.08, wspace=0.34, hspace=0.38)

# (a) --------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=ext, interpolation="nearest")
ax.scatter(elec[..., 0].ravel(), elec[..., 1].ravel(), s=5, c="k", zorder=5, lw=0)
ax.scatter(*e0[:2], s=26, facecolor="w", edgecolor="k", lw=0.8, zorder=6)
for l, nm in ((2, "radius"), (3, "ulna"), (C.FCU, "FCU")):
    if l in lab_pos:
        ax.text(*lab_pos[l], nm, ha="center", va="center", fontsize=6,
                color="w" if l == C.FCU else "0.2", fontweight="bold" if l == C.FCU else None)
ax.text(bb[0] + 2, bb[3] - 2, "skin + fat", fontsize=6, color="0.35", va="top")
ax.annotate("5×5 grid\n(10 mm IED)", xy=(elec[2, 0, 0], elec[2, 0, 1]), xytext=(bb[1] - 1, e0[1] - 4), fontsize=6,
            ha="right", va="top", arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3", shrinkB=2))
ax.set_xlim(bb[0], bb[1]); ax.set_ylim(bb[2], bb[3]); ax.set_aspect("equal")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
letter(ax, "a")

# (b) --------------------------------------------------------------------
gsb = gs[0, 1].subgridspec(2, 2, width_ratios=[1.0, 2.4], wspace=0.3, hspace=0.45)
zmid_p = 0.5 * (pbed.z_vals[0] + pbed.z_vals[-1])
outline = C.fcu_outline(fm, zmid_p)
fcu_vox = np.argwhere(fm.seg_data == C.FCU)
zs_sil = np.unique(fcu_vox[:, 2])
sil_lo = np.array([fcu_vox[fcu_vox[:, 2] == k, LAT].min() for k in zs_sil]) * vs[LAT]
sil_hi = np.array([fcu_vox[fcu_vox[:, 2] == k, LAT].max() for k in zs_sil]) * vs[LAT]
z_sil = zs_sil * vs[2]
axes_b = {}
SUB = 6                                                   # side view: every 6th fibre, so streamlines stay visible
for r, (name, bed, col, nmj) in enumerate((("Poisson", pbed, COL["poisson"], nmj_p),
                                            ("harmonic", hbed, COL["harmonic"], nmj_h))):
    axy = fig.add_subplot(gsb[r, 0]); axs = fig.add_subplot(gsb[r, 1])
    paths = [np.asarray(p) for p in bed.paths]
    # xy: the true cross-section at the muscle mid-z (each fibre's point nearest z-mid), not the projection
    for c in outline:
        axy.plot(c[:, 0], c[:, 1], color="k", lw=0.6)
    xs_mid = np.array([p[np.argmin(np.abs(p[:, 2] - zmid_p)), :2] for p in paths
                       if np.abs(p[:, 2] - zmid_p).min() < 3.0])
    axy.scatter(xs_mid[:, 0], xs_mid[:, 1], s=1.6, c=col, lw=0, alpha=0.9)
    # side view: muscle silhouette, a subsample of fibres, every NMJ
    axs.fill_between(z_sil, sil_lo, sil_hi, color="0.92", lw=0, step="mid")
    axs.add_collection(LineCollection([p[:, [2, LAT]] for p in paths[::SUB]], colors=col, linewidths=0.35, alpha=0.6))
    axs.scatter(nmj[:, 2], nmj[:, LAT], s=1.0, c="k", lw=0, zorder=4)
    axy.set_aspect("equal"); axy.autoscale_view()
    axs.set_xlim(z_sil.min() - 5, z_sil.max() + 5); axs.autoscale_view(scalex=False)
    axs.set_aspect("equal")
    axy.set_ylabel("y (mm)"); axs.set_ylabel(f"{LATNAME} (mm)")
    axs.text(0.0, 1.06, f"{name} · {len(paths)} fibres (1 in {SUB} drawn) · NMJ (·)", transform=axs.transAxes,
             fontsize=6.5, va="bottom", color=col, fontweight="bold")
    if r == 0:
        axy.set_xticklabels([]); axs.set_xticklabels([])
    else:
        axy.set_xlabel("x (mm)"); axs.set_xlabel("z along the arm (mm)")
    axes_b[(r, 0)] = axy; axes_b[(r, 1)] = axs
letter(axes_b[(0, 0)], "b", dx=-0.4)

# (c) --------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
im = ax.pcolormesh(xs, ys, logphi, cmap="magma", vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
for c in C.fcu_outline(fm, e0[2]):
    ax.plot(c[:, 0], c[:, 1], color="w", lw=0.6)
ax.scatter(*e0[:2], s=26, facecolor="w", edgecolor="k", lw=0.8, zorder=6)
cols3 = [plt.cm.viridis(v) for v in (0.85, 0.5, 0.15)]
for k, i in enumerate(pick):
    ax.scatter(*pbed.xy_mid[i], s=14, facecolor=cols3[k], edgecolor="w", lw=0.5, zorder=6)
ax.set_xlim(bb[0], bb[1]); ax.set_ylim(bb[2], bb[3]); ax.set_aspect("equal")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.ax.set_title("log$_{10}$|φ|\n(φ in mV)", fontsize=6, pad=3); cb.ax.tick_params(labelsize=6)
ax_c = ax

# (d) --------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
for k, i in enumerate(pick):
    z = pbed.paths[i, :, 2] - e0[2]
    ax.plot(z, phi_fib[k] * 1e3, color=cols3[k], lw=1.0, label=f"d = {dist[i]:.0f} mm")
    ax.plot(z, denoise_field_n(phi_fib[k], float(arc_p_dz[i]), n=3) * 1e3, color="k", lw=0.6, ls="--")
ax.plot([], [], color="k", lw=0.6, ls="--", label="3-monopole fit")
ax.axvline(0, color="0.6", lw=0.5, ls=":")
ax.set_xlabel("z along the fibre, from the electrode plane (mm)"); ax.set_ylabel("φ (mV)")
ax.legend(fontsize=6.5, loc="upper right", title="electrode → fibre", title_fontsize=6.5)
ax.set_xlim(z.min(), z.max())
# panel letters c/d at the same height (the equal-aspect map shrinks its own axes box)
yc = max(ax_c.get_position().y1, ax.get_position().y1) + 0.012
fig.text(ax_c.get_position().x0 - 0.045, yc, "c", fontsize=9, fontweight="bold", va="bottom")
fig.text(ax.get_position().x0 - 0.06, yc, "d", fontsize=9, fontweight="bold", va="bottom")

save(fig, "fig_mri_fibres")
KN["total_s"] = time.time() - T0
C.update_key_numbers("fig6_mri_fibres", KN)
