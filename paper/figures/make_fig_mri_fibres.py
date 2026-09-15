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
vs = fm.voxel_size
# the straight bed's NMJs: IZ_FRAC of each path from its proximal (low-z) end
k_iz = int(round(C.IZ_FRAC * (pbed.paths.shape[1] - 1)))
nmj_p = pbed.paths[:, k_iz]
z_iz_target = float(nmj_p[:, 2].mean())                   # the IZ plane both beds share
# the harmonic beds are cached at the library-default IZ (0.5 of the muscle); re-innervate
# them on the straight bed's IZ plane with the long_fibers placement rule
# (f2_common.harmonic_bed_at_iz). long_fibers takes the fraction in the muscle's PCA frame,
# whose axis sign is arbitrary — for the FCU it points distally, so the plane z = 126 mm
# (0.305 of the straight fibres from the proximal end) is frame fraction ≈ 0.68, and a
# literal 0.305 would innervate the far end of the muscle.
from emgforge.mri.core.harmonic_fibers import HarmonicFibreField
frame = C.harmonic_frame(fm)
IZ_FRAME = float(frame._long_fraction(nmj_p).mean())      # frame fraction of the IZ plane
hbed_def05 = C.harmonic_bed(fm, grid_mm=2.0)               # library default seeding, IZ 0.5
hbed05 = C.harmonic_bed(fm, grid_mm=C.HARM_GRID_MM)        # density-matched, IZ 0.5
hbed_def = C.harmonic_bed_at_iz(hbed_def05, frame, IZ_FRAME)
hbed = C.harmonic_bed_at_iz(hbed05, frame, IZ_FRAME)      # plotted
KN["harmonic_bed_build_s"] = dict(default_2mm=getattr(hbed_def05, "build_seconds", None),
                                  dense=getattr(hbed05, "build_seconds", None))
KN["harmonic_seed_grid_mm"] = dict(default=2.0, dense=C.HARM_GRID_MM)
# check: the lower-level builder that does take the fraction, on the cheap 2 mm seeding
t0 = time.time()
chk = HarmonicFibreField(fm.seg_data == C.FCU, vs, solve=True).long_fibers(
    grid_mm=2.0, dz_mm=1.0, iz_fraction=IZ_FRAME)
KN["harmonic_iz_check"] = dict(
    builder=f"HarmonicFibreField.long_fibers(grid_mm=2.0, iz_fraction={IZ_FRAME:.4f})",
    iz_fraction_frame=IZ_FRAME, frame_axis_p1=frame.p1.tolist(),
    frame_points_distally=bool(frame.p1[2] < 0),
    iz_fraction_from_proximal_end=(1.0 - IZ_FRAME) if frame.p1[2] < 0 else IZ_FRAME,
    literal_0p305_would_be_z_mm=float(frame.z1 - C.IZ_FRAC * (frame.z1 - frame.z0)) if frame.p1[2] < 0
    else float(frame.z0 + C.IZ_FRAC * (frame.z1 - frame.z0)),
    laplace_solve_s=time.time() - t0, n_fibres_builder=len(chk), n_fibres_cached=len(hbed_def.paths),
    paths_identical=bool(len(chk) == len(hbed_def.paths) and all(
        np.asarray(a.path).shape == np.asarray(b).shape and np.allclose(a.path, b) for a, b in zip(chk, hbed_def.paths))),
    max_abs_diff_half1_mm=float(np.max(np.abs(np.array([f.half1_mm for f in chk]) - hbed_def.half1_mm))),
    max_abs_diff_half2_mm=float(np.max(np.abs(np.array([f.half2_mm for f in chk]) - hbed_def.half2_mm))),
    max_abs_diff_iz_fraction=float(np.max(np.abs(np.array([f.iz_fraction for f in chk]) - hbed_def.iz_fractions))),
    build_muscle_beds_exposes_iz_fraction=False)
print(f"harmonic IZ check (long_fibers at frame fraction {IZ_FRAME:.4f}): {KN['harmonic_iz_check']}")
fem = C.build_fem()
KN["fem_build_s"] = fem.build_seconds
elec, ginfo = C.grid_electrodes(fem, fm, pbed)
IE, JE = 2, 2                                            # the centre electrode
e0 = elec[IE, JE]
t0 = time.time(); fem.solve_for_point(e0, source_sigma=5.0); KN["fem_solve_s"] = time.time() - t0
KN.update(grid=ginfo, electrode_xyz=e0.tolist())

# --------------------------------------------------------------------------- bed statistics
N_p, N_h = len(pbed.r_norms), len(hbed.r_norms)
cont_p, _ = C.mask_containment(pbed.paths, fm)
cont_h, per_h = C.mask_containment(hbed.paths, fm)
arc_p_dz, L_p = C.bed_arc_geometry(pbed)
L_h = hbed.half1_mm + hbed.half2_mm
# longitudinal span of each streamline in the muscle's PCA frame (the criterion
# harmonic_fibers.HarmonicFibreField uses) and whether it reaches the IZ at all
lfr_h = [frame._long_fraction(np.asarray(p)) for p in hbed.paths]
span_h = np.array([np.ptp(f) for f in lfr_h])
reach_h = np.array([f.min() <= IZ_FRAME <= f.max() for f in lfr_h])
span_p = np.array([np.ptp(frame._long_fraction(p)) for p in pbed.paths])
SPAN_CUT = 0.53                                          # the half-length cluster sits below this
# NMJ points: harmonic = arc position half1 (index half1/dz); Poisson = IZ_FRAC of the path
nmj_h = np.array([np.asarray(p)[int(round(h1 / hbed.dz_mm))] for p, h1 in zip(hbed.paths, hbed.half1_mm)])
nmj_h05 = np.array([np.asarray(p)[int(round(h1 / hbed05.dz_mm))] for p, h1 in zip(hbed05.paths, hbed05.half1_mm)])
z_fcu = frame.z1 - frame.z0


def nmj_stats(nmj, reach=None):
    z = nmj[:, 2]
    d = dict(z_mean=z.mean(), z_std=z.std(), z_median=float(np.median(z)), z_range=[z.min(), z.max()],
             z_spread_mm=float(np.ptp(z)), z_spread_fraction_of_muscle=float(np.ptp(z) / z_fcu))
    if reach is not None:                                # streamlines that contain the IZ
        d.update(z_mean_reaching_iz=z[reach].mean(), z_std_reaching_iz=z[reach].std(),
                 z_spread_mm_reaching_iz=float(np.ptp(z[reach])))
    return d


# the library-default (2 mm seed) harmonic bed, for the record
N_hd = len(hbed_def.r_norms)
cont_hd, _ = C.mask_containment(hbed_def.paths, fm)
span_hd = np.array([np.ptp(frame._long_fraction(np.asarray(p))) for p in hbed_def.paths])
L_hd = hbed_def.half1_mm + hbed_def.half2_mm
KN["harmonic_default_2mm"] = dict(n_fibres=N_hd, containment=cont_hd, span_median=np.median(span_hd),
                                  frac_span_below_0p53=float((span_hd < SPAN_CUT).mean()), length_mean_mm=L_hd.mean())
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
                                frac_above_0p8=float((span_h > 0.8).mean()),
                                frac_below_0p53=float((span_h < SPAN_CUT).mean()), span_cut=SPAN_CUT,
                                n_below_0p53=int((span_h < SPAN_CUT).sum())),
    poisson_span_fraction=dict(median=np.median(span_p), min=span_p.min(), max=span_p.max()),
    fcu_z_extent_mm=z_fcu, fcu_z_range_mm=[frame.z0, frame.z1],
    harmonic_nmj=dict(iz_plane_z_mm=z_iz_target, iz_fraction_frame=IZ_FRAME,
                      iz_fraction_frame_mean=float(hbed.iz_fractions.mean()),
                      iz_fraction_frame_std=float(hbed.iz_fractions.std()),
                      frac_streamlines_reaching_iz=float(reach_h.mean()), n_not_reaching_iz=int((~reach_h).sum()),
                      **nmj_stats(nmj_h, reach_h),
                      half1_mean_mm=float(hbed.half1_mm.mean()), half2_mean_mm=float(hbed.half2_mm.mean()),
                      note="NMJ of every streamline re-placed on the straight bed's IZ plane (0.305 of the straight "
                           "fibres from the proximal end) with the long_fibers rule; a streamline that does not "
                           "reach the plane gets its NMJ at its nearer end — those are the second cluster of dots "
                           "in panel b"),
    harmonic_nmj_library_default=dict(iz_fraction=0.5, **nmj_stats(nmj_h05)),
    poisson_nmj=dict(iz_fraction=C.IZ_FRAC, z_mean=nmj_p[:, 2].mean(), z_std=nmj_p[:, 2].std(),
                     long_fraction_in_muscle_frame=float(frame._long_fraction(nmj_p).mean())),
    poisson_bed=dict(density_per_mm2=C.DENSITY, area_mm2=pbed.cross_section_area_mm2,
                     z_range=[pbed.z_vals[0], pbed.z_vals[-1]], dz_mm=float(pbed.z_vals[1] - pbed.z_vals[0])),
)
print(f"FCU beds: Poisson {N_p} fibres (containment {cont_p*100:.1f} %, length {L_p.mean():.0f} mm), "
      f"harmonic {N_h} fibres (containment {cont_h*100:.1f} %, length {L_h.mean():.0f} mm, "
      f"span median {np.median(span_h):.2f} [{np.percentile(span_h,25):.2f}–{np.percentile(span_h,75):.2f}], "
      f"{(span_h < SPAN_CUT).mean()*100:.1f} % span < {SPAN_CUT})")
hn = KN["harmonic_nmj"]
print(f"harmonic NMJ on the IZ plane (frame fraction {IZ_FRAME:.3f}): iz {hn['iz_fraction_frame_mean']:.3f}±"
      f"{hn['iz_fraction_frame_std']:.3f}, z {hn['z_mean']:.1f}±{hn['z_std']:.1f} mm (spread {hn['z_spread_mm']:.1f} mm = "
      f"{hn['z_spread_fraction_of_muscle']*100:.1f} % of muscle; target z {z_iz_target:.1f}); "
      f"{reach_h.mean()*100:.1f} % of streamlines reach the IZ, those: z {hn['z_mean_reaching_iz']:.1f}±"
      f"{hn['z_std_reaching_iz']:.1f} mm; Poisson IZ z {nmj_p[:,2].mean():.1f}±{nmj_p[:,2].std():.1f} mm "
      f"(= {KN['poisson_nmj']['long_fraction_in_muscle_frame']:.3f} of the muscle frame); "
      f"library default 0.5: z {nmj_h05[:,2].mean():.1f}±{nmj_h05[:,2].std():.1f} mm")
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
# φ of the reciprocity solve (per 1 A injected) changes sign on the slice (mean-zero
# reference), so log|φ| shows a black band at the zero crossing: plot φ − min φ on a log
# colour scale instead, as Fig 2d does
phi_mV = phi_sl * 1e3
mag = phi_mV - np.nanmin(phi_mV)
DECADES = 3.5
vmax = np.nanmax(mag); vmin = vmax / 10 ** DECADES
KN["slice_phi_mV_per_A"] = dict(min=float(np.nanmin(phi_mV)), max=float(np.nanmax(phi_mV)),
                                frac_negative=float(np.nanmean(phi_mV < 0)), map="phi - min phi, log colour",
                                decades_shown=DECADES)

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
gsb = gs[0, 1].subgridspec(2, 2, width_ratios=[1.0, 2.4], wspace=0.55, hspace=0.5)
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
    axs.axvline(z_iz_target, color="k", lw=0.4, ls=":", zorder=3)          # the shared IZ (0.305)
    axy.set_aspect("equal"); axy.autoscale_view()
    axs.set_xlim(z_sil.min() - 5, z_sil.max() + 5); axs.autoscale_view(scalex=False)
    axs.set_aspect("equal")
    # cross-section is (x, y); the side view is the (z, LAT) projection — its vertical axis is LAT
    axy.set_ylabel("y (mm)", labelpad=1); axs.set_ylabel(f"{LATNAME} (mm)", labelpad=1)
    axs.text(-0.42, 1.06, f"{name} · {len(paths)} fibres (1 in {SUB} drawn) · NMJ (·) on the IZ plane (⋮)",
             transform=axs.transAxes, fontsize=6, va="bottom", color=col, fontweight="bold")
    if r == 0:
        axy.set_xticklabels([]); axs.set_xticklabels([])
    else:
        axy.set_xlabel("x (mm)"); axs.set_xlabel("z along the arm (mm)")
    axes_b[(r, 0)] = axy; axes_b[(r, 1)] = axs
letter(axes_b[(0, 0)], "b", dx=-0.4)

# (c) --------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
from matplotlib.colors import LogNorm
im = ax.pcolormesh(xs, ys, np.clip(mag, vmin, vmax), cmap="magma", norm=LogNorm(vmin, vmax), shading="auto",
                   rasterized=True)
for c in C.fcu_outline(fm, e0[2]):
    ax.plot(c[:, 0], c[:, 1], color="w", lw=0.6)
ax.scatter(*e0[:2], s=26, facecolor="w", edgecolor="k", lw=0.8, zorder=6)
cols3 = [plt.cm.viridis(v) for v in (0.85, 0.5, 0.15)]
for k, i in enumerate(pick):
    ax.scatter(*pbed.xy_mid[i], s=14, facecolor=cols3[k], edgecolor="w", lw=0.5, zorder=6)
ax.set_xlim(bb[0], bb[1]); ax.set_ylim(bb[2], bb[3]); ax.set_aspect("equal")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("φ − min φ (mV per A, log scale)", fontsize=6.5, labelpad=2); cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)
ax_c = ax

# (d) --------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
for k, i in enumerate(pick):
    z = pbed.paths[i, :, 2] - e0[2]
    ax.plot(z, phi_fib[k] * 1e3, color=cols3[k], lw=1.0, label=f"d = {dist[i]:.0f} mm")
    ax.plot(z, denoise_field_n(phi_fib[k], float(arc_p_dz[i]), n=3) * 1e3, color="k", lw=0.6, ls="--")
ax.plot([], [], color="k", lw=0.6, ls="--", label="3-monopole fit")
ax.axvline(0, color="0.6", lw=0.5, ls=":")
ax.set_xlabel("z along the fibre, from the electrode plane (mm)"); ax.set_ylabel("φ (mV per A injected)")
ax.legend(fontsize=6.5, loc="upper right", title="electrode → fibre", title_fontsize=6.5)
ax.set_xlim(z.min(), z.max())
# panel letters c/d at the same height (the equal-aspect map shrinks its own axes box)
yc = max(ax_c.get_position().y1, ax.get_position().y1) + 0.012
fig.text(ax_c.get_position().x0 - 0.045, yc, "c", fontsize=9, fontweight="bold", va="bottom")
fig.text(ax.get_position().x0 - 0.06, yc, "d", fontsize=9, fontweight="bold", va="bottom")

save(fig, "fig_mri_fibres")
KN["total_s"] = time.time() - T0
C.update_key_numbers("fig6_mri_fibres", KN)
