"""Study figure — crosstalk: motor units of the muscles around the flexor carpi ulnaris (FCU)
seen by the 5×5 HD-EMG grid over the FCU. (a) forearm cross-section at the grid-centre
height with the study muscles, the chosen unit centres, the grid and the units' own skin
electrodes; (b) monopolar (top) and single-differential (bottom) RMS maps of the largest
unit of every muscle; (c) size-matched crosstalk ratio vs the distance of the unit centre
(on the grid-centre plane) from the grid-centre electrode, per montage, the FCU units as
the reference; (d) monopolar MUAPs at the grid centre of the largest unit of every muscle
at one scale (the quoted distance is the same centre-to-grid-centre distance, not depth).

Data: ``crosstalk_common`` caches (lead fields + direct-method MUAPs of 7 units × 4
neighbouring muscles) and the cached 100-unit FCU tensor of ``f2_common``.
Numbers → ``paper/figures/key_numbers_study_crosstalk.json``.

Run: /home/dc23/miniconda3/envs/fenicsx-env/bin/python paper/figures/make_fig_crosstalk.py
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "paper/figures")
import crosstalk_common as X                      # noqa: E402  (thread env before numpy)
import numpy as np                                # noqa: E402
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.cm import ScalarMappable          # noqa: E402
from matplotlib.colors import LogNorm, to_rgb     # noqa: E402
from matplotlib.lines import Line2D               # noqa: E402

from style import use, save, letter, W2           # noqa: E402
import f2_common as C                             # noqa: E402

MK = {"mono": "o", "sd": "s", "dd": "^"}
LS = {"mono": "-", "sd": "--", "dd": ":"}
MLAB = {"mono": "monopolar", "sd": "single diff.", "dd": "double diff."}


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main():
    use(); T0 = time.time(); KN = {}
    LF = X.ensure_leadfields(); MU = X.ensure_muaps()
    fm, beds, pools = X.load_anatomy()
    D = C.load_muaps("new")                                   # the 100-unit FCU tensor (same recipe)
    W_fcu, t_ms, sizes_fcu = D["W"], D["t_ms"], np.asarray(D["sizes"])
    elec, zc = LF["elec_xyz"], float(LF["zc_mm"]); e0 = elec[C.M // 2, C.M // 2]
    E = elec.reshape(-1, 3)
    vs = fm.voxel_size; kz = C.seg_slice_index(fm, zc); S = fm.seg_data[:, :, kz]

    # ------------------------------------------------------------------ muscle table
    muscles = {}
    zs = np.arange(fm.seg_data.shape[2]) * vs[2]
    kin = np.where((zs >= E[:, 2].min() - 3) & (zs <= E[:, 2].max() + 3))[0]
    tissue_z = np.where((fm.seg_data > 0).any(axis=(0, 1)))[0]                 # the field of view (tissue) in slices
    for l in X.ORDER:
        bed = beds[l]; izf, note = X.iz_fraction(l)
        lab_z = np.where((fm.seg_data == l).any(axis=(0, 1)))[0]
        cut = [bool(lab_z.min() <= tissue_z.min()), bool(lab_z.max() >= tissue_z.max())]
        vox = np.argwhere(fm.seg_data[:, :, kin] == l).astype(float)
        pts = np.c_[vox[:, 0] * vs[0], vox[:, 1] * vs[1], zs[kin][vox[:, 2].astype(int)]]
        dv = np.linalg.norm(pts[:, None, :] - E[None], axis=2)
        cxy = np.argwhere(S == l).mean(0) * vs[:2]
        muscles[l] = dict(label=l, abbrev=X.MUSCLES[l][0], name=X.MUSCLES[l][1], n_bed_fibres=int(len(bed.r_norms)),
                          bed_area_mm2=float(bed.cross_section_area_mm2), slice_area_mm2=float((S == l).sum() * vs[0] * vs[1]),
                          bed_z_mm=[float(bed.z_vals[0]), float(bed.z_vals[-1])],
                          centroid_xy_at_zc=cxy.tolist(), centroid_dist_to_centre_mm=float(np.linalg.norm(cxy - e0[:2])),
                          min_dist_to_any_electrode_mm=float(dv.min()), min_dist_to_centre_mm=float(dv[:, C.E0].min()),
                          iz_fraction=izf, iz_note=note,
                          iz_z_mm=float(bed.z_vals[0] + izf * (bed.z_vals[-1] - bed.z_vals[0])),
                          label_z_mm=[float(zs[lab_z.min()]), float(zs[lab_z.max()])],
                          fov_cut=dict(proximal=cut[0], distal=cut[1],
                                       note="True = the label reaches that end of the imaged volume (muscle truncated there)"))
    KN["muscles"] = muscles
    KN["grid"] = dict(M=C.M, ied_mm=C.IED_MM, centre_xyz=e0.tolist(), zc_mm=zc, z_rows_mm=elec[:, 0, 2].tolist(),
                      note=D["grid_note"], n_units_per_neighbour=X.N_UNITS, own_electrode="skin point on the grid-centre "
                      "z-plane nearest to the unit's fibre centroid (one FEM solve per unit)")

    # ------------------------------------------------------------------ FCU reference (100 units)
    fcu = [X.amplitude(W_fcu[k]) for k in range(C.N_MU)]
    fits = {}
    for m in X.MONTAGES:
        A = np.array([f[m]["max"] for f in fcu]); En = np.array([f[m]["energy"] for f in fcu]); x = np.log10(sizes_fcu)
        b, a = np.polyfit(x, np.log10(A), 1); be, ae = np.polyfit(x, np.log10(En), 1)
        fits[m] = dict(amp=[a, b], energy=[ae, be], r_amp=float(np.corrcoef(x, np.log10(A))[0, 1]),
                       resid_dex_sd=float(np.std(np.log10(A) - (a + b * x))))
    ref_amp = lambda m, n: 10 ** (fits[m]["amp"][0] + fits[m]["amp"][1] * np.log10(n))
    ref_en = lambda m, n: 10 ** (fits[m]["energy"][0] + fits[m]["energy"][1] * np.log10(n))
    fcu_rows = []
    for k in range(C.N_MU):
        xy, zk = X.unit_xy_at(beds[X.FCU], pools[X.FCU][k], zc); p = np.r_[xy, zk]
        fcu_rows.append(dict(k=k, size=int(sizes_fcu[k]), d_centre=float(np.linalg.norm(p - e0)),
                             d_min=float(np.linalg.norm(E - p, axis=1).min()),
                             **{m: dict(max=fcu[k][m]["max"], centre=fcu[k][m]["centre"], energy=fcu[k][m]["energy"],
                                        R_size=fcu[k][m]["max"] / ref_amp(m, sizes_fcu[k])) for m in X.MONTAGES}))
    KN["fcu_reference"] = dict(size_law=fits, n_units=C.N_MU,
                               d_centre_mm=dict(min=min(r["d_centre"] for r in fcu_rows), median=float(np.median([r["d_centre"] for r in fcu_rows])),
                                                max=max(r["d_centre"] for r in fcu_rows)),
                               p2p_max_uV=dict(**{m: dict(min=min(r[m]["max"] for r in fcu_rows) * 1e6, max=max(r[m]["max"] for r in fcu_rows) * 1e6)
                                                  for m in X.MONTAGES}),
                               energy_ratio_median={f"{m}_over_mono": float(np.median([r[m]["energy"] / r["mono"]["energy"] for r in fcu_rows]))
                                                    for m in ("sd", "dd")},
                               p2p_ratio_median={f"{m}_over_mono": float(np.median([r[m]["max"] / r["mono"]["max"] for r in fcu_rows]))
                                                 for m in ("sd", "dd")})

    # ------------------------------------------------------------------ neighbouring units
    rows = []
    for l in X.NEIGHBOURS:
        for u in LF["muscles"][l]["units"]:
            k = u["k"]; W = MU["W"][(l, k)]; n = u["size"]
            amp = X.amplitude(W[:X.E_GRID]); own = float(np.ptp(W[X.E_GRID]))
            p = np.r_[u["xy_zc"], u["z_plane"]]
            row = dict(label=l, abbrev=X.MUSCLES[l][0], k=k, size=n, xy_zc=np.asarray(u["xy_zc"]).tolist(),
                       d_centre=float(np.linalg.norm(p - e0)), d_min=float(np.linalg.norm(E - p, axis=1).min()),
                       own_xyz=np.asarray(u["own_xyz"]).tolist(), d_own=float(np.linalg.norm(p - u["own_xyz"])),
                       own_p2p=own, R_own_centre=amp["mono"]["centre"] / own, R_own_max=amp["mono"]["max"] / own,
                       secs=float(MU["secs"][(l, k)]))
            for m in X.MONTAGES:
                Eref = ref_en(m, n); Aref = ref_amp(m, n)
                row[m] = dict(max=amp[m]["max"], centre=amp[m]["centre"], energy=amp[m]["energy"],
                              R_size=amp[m]["max"] / Aref, R_idx=amp[m]["max"] / fcu[k][m]["max"],
                              R_centre=amp[m]["centre"] / (fcu[k][m]["centre"] if fcu[k][m]["centre"] > 0 else np.nan),
                              E_ratio=amp[m]["energy"] / Eref, E_fraction=amp[m]["energy"] / (amp[m]["energy"] + Eref))
            row["amp"] = amp
            rows.append(row)
    d_all = np.array([r["d_centre"] for r in rows])

    # distance dependence per montage (pooled over the four muscles): log10 R = a + b·d
    dist = {}
    for m in X.MONTAGES:
        R = np.array([r[m]["R_size"] for r in rows]); b, a = np.polyfit(d_all, np.log10(R), 1)
        ok = R >= 0.1
        dist[m] = dict(slope_dex_per_mm=b, intercept=a, r=float(np.corrcoef(d_all, np.log10(R))[0, 1]),
                       d10_fit_mm=float((-1.0 - a) / b) if b < 0 else np.nan,
                       d_max_with_R_ge_0p1_mm=float(d_all[ok].max()) if ok.any() else None, n_units_R_ge_0p1=int(ok.sum()),
                       d_min_with_R_lt_0p1_mm=float(d_all[~ok].min()) if (~ok).any() else None,
                       note="log10 R = intercept + slope·d over the 28 neighbouring units (17–55 mm); d10_fit is an "
                            "extrapolation and meaningless when |r| is small (monopolar: no distance dependence)")
    per_muscle = {}
    for l in X.NEIGHBOURS:
        rr = [r for r in rows if r["label"] == l]; big = max(rr, key=lambda r: r["size"])
        per_muscle[X.MUSCLES[l][0]] = dict(
            d_centre_mm=[r["d_centre"] for r in rr], d_min_mm=[r["d_min"] for r in rr], sizes=[r["size"] for r in rr],
            R_size_median={m: float(np.median([r[m]["R_size"] for r in rr])) for m in X.MONTAGES},
            R_size_range={m: [min(r[m]["R_size"] for r in rr), max(r[m]["R_size"] for r in rr)] for m in X.MONTAGES},
            R_idx_median={m: float(np.median([r[m]["R_idx"] for r in rr])) for m in X.MONTAGES},
            E_fraction_median={m: float(np.median([r[m]["E_fraction"] for r in rr])) for m in X.MONTAGES},
            E_ratio_median={m: float(np.median([r[m]["E_ratio"] for r in rr])) for m in X.MONTAGES},
            energy_after_diff_median={f"{m}_over_mono": float(np.median([r[m]["energy"] / r["mono"]["energy"] for r in rr])) for m in ("sd", "dd")},
            R_own_centre_median=float(np.median([r["R_own_centre"] for r in rr])),
            R_own_max_median=float(np.median([r["R_own_max"] for r in rr])),
            d_own_mm_median=float(np.median([r["d_own"] for r in rr])),
            largest_unit=dict(k=big["k"], size=big["size"], d_centre_mm=big["d_centre"], own_p2p_uV=big["own_p2p"] * 1e6,
                              **{m: dict(p2p_max_uV=big[m]["max"] * 1e6, p2p_centre_uV=big[m]["centre"] * 1e6, R_size=big[m]["R_size"],
                                         R_idx=big[m]["R_idx"], fcu_same_index_p2p_max_uV=fcu[big["k"]][m]["max"] * 1e6) for m in X.MONTAGES}),
            secs_per_unit=[r["secs"] for r in rr])
    supp = {f"{m}_over_mono": dict(neighbours_median=float(np.median([r[m]["R_size"] / r["mono"]["R_size"] for r in rows])),
                                   neighbours_range=[min(r[m]["R_size"] / r["mono"]["R_size"] for r in rows),
                                                     max(r[m]["R_size"] / r["mono"]["R_size"] for r in rows)],
                                   energy_neighbours_median=float(np.median([r[m]["energy"] / r["mono"]["energy"] for r in rows])),
                                   energy_fcu_median=KN["fcu_reference"]["energy_ratio_median"][f"{m}_over_mono"])
            for m in ("sd", "dd")}
    KN.update(units=[{k: v for k, v in r.items() if k != "amp"} for r in rows], per_muscle=per_muscle,
              distance_dependence=dist, montage_suppression_of_R=supp,
              runtimes=dict(leadfield_wall_s=float(LF["wall_s"]), leadfield_solve_sample_s_mean=float(LF["solve_sample_s_mean"]),
                            muap_wall_s=float(MU["wall_s"]), muap_cpu_s=float(sum(MU["secs"].values())),
                            muap_s_per_unit_min_max=[float(min(MU["secs"].values())), float(max(MU["secs"].values()))]),
              synthesis=dict(recipe="direct line-source synthesis, f2_common.SPCFG", **C.spcfg_dict(),
                             iz_jitter_frac=C.IZ_JITTER, density_per_mm2=C.DENSITY))
    for m in X.MONTAGES:
        print(f"{MLAB[m]:>13}: R_size median per muscle " +
              ", ".join(f"{a} {v['R_size_median'][m]:.3f}" for a, v in per_muscle.items()) +
              f" | d10 fit {dist[m]['d10_fit_mm']:.0f} mm, slope {dist[m]['slope_dex_per_mm']:.3f} dex/mm")
    print("own-electrode ratio (grid centre / own) medians:", {a: round(v["R_own_centre_median"], 3) for a, v in per_muscle.items()})

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(W2, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], left=0.06, right=0.99, top=0.96, bottom=0.07, hspace=0.32)
    gtop = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.12)
    gbot = gs[1].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.28)

    # (a) cross-section ---------------------------------------------------------
    ax = fig.add_subplot(gtop[0])
    img = np.ones(S.shape + (3,))
    img[np.isin(S, [15, 25])] = to_rgb("#f2ece2")
    img[np.isin(S, [2, 3])] = to_rgb("#8f8f8f")
    other = (S > 3) & ~np.isin(S, [15, 25]) & ~np.isin(S, X.ORDER)
    img[other] = to_rgb("#d8d8d8")
    for l in X.ORDER:
        img[S == l] = 0.45 * np.array(to_rgb(X.COLOURS[l])) + 0.55
    ax.imshow(img.transpose(1, 0, 2), origin="lower", interpolation="nearest",      # pixel centres at index × voxel size
              extent=(-vs[0] / 2, (S.shape[0] - 0.5) * vs[0], -vs[1] / 2, (S.shape[1] - 0.5) * vs[1]))
    for l in X.ORDER:
        for c in C.fcu_outline(fm, zc, label=l):
            ax.plot(c[:, 0], c[:, 1], color=X.COLOURS[l], lw=0.6)
        cxy = np.array(muscles[l]["centroid_xy_at_zc"]) + (np.array([-9.0, -1.5]) if l == X.FCU else 0.0)
        ax.text(cxy[0], cxy[1], X.MUSCLES[l][0], fontsize=6, ha="center", va="center", fontweight="bold", color="k",
                bbox=dict(boxstyle="round,pad=0.1", fc="w", ec="none", alpha=0.6), zorder=7)
    for r in rows:                                          # own electrodes + unit centres
        ax.scatter(r["own_xyz"][0], r["own_xyz"][1], marker="s", s=7, facecolor="none", edgecolor=X.COLOURS[r["label"]], lw=0.5, zorder=5)
        ax.scatter(*r["xy_zc"], s=3 + 0.045 * r["size"], color=X.COLOURS[r["label"]], edgecolor="k", lw=0.3, zorder=6)
    for k in X.select_units(pools[X.FCU]):                  # the same seven size-matched FCU units
        xy, _ = X.unit_xy_at(beds[X.FCU], pools[X.FCU][k], zc)
        ax.scatter(*xy, s=3 + 0.045 * sizes_fcu[k], color=X.COLOURS[X.FCU], edgecolor="k", lw=0.3, zorder=6)
    KN["fcu_units_shown"] = X.select_units(pools[X.FCU])
    ax.scatter(elec[..., 0].ravel(), elec[..., 1].ravel(), s=4, c="k", zorder=7, lw=0)
    ax.scatter(e0[0], e0[1], s=16, facecolor="w", edgecolor="k", lw=0.7, zorder=8)
    ax.annotate("grid", xy=(e0[0], e0[1]), xytext=(e0[0] + 6, e0[1] + 9), fontsize=6, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="k"))
    lim = np.argwhere(S > 0); lo = lim.min(0) * vs[:2] - 4; hi = lim.max(0) * vs[:2] + 4
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.text(0.02, 0.02, f"axial slice, z = {zc:.0f} mm\n• unit centres (area ∝ fibres)\n□ own electrodes",
            transform=ax.transAxes, fontsize=5.5, va="bottom")
    letter(ax, "a", dx=-0.16)

    # (b)/(c) RMS maps of the largest unit per muscle --------------------------------
    gmap = gtop[1].subgridspec(2, len(X.ORDER) + 1, width_ratios=[1] * len(X.ORDER) + [0.08], wspace=0.15, hspace=0.0)
    largest = {}
    for l in X.ORDER:
        if l == X.FCU:
            k = int(np.argmax(sizes_fcu)); largest[l] = (k, W_fcu[k])
        else:
            u = max(LF["muscles"][l]["units"], key=lambda u: u["size"]); largest[l] = (u["k"], MU["W"][(l, u["k"])][:X.E_GRID])
    maps = {l: X.amplitude(W) for l, (k, W) in largest.items()}
    allv = np.concatenate([maps[l][m]["rms"].ravel() for l in X.ORDER for m in ("mono", "sd")]) * 1e6
    norm = LogNorm(max(allv.min(), allv.max() * 1e-3), allv.max())
    for r_, m in enumerate(("mono", "sd")):
        for c_, l in enumerate(X.ORDER):
            axm = fig.add_subplot(gmap[r_, c_])
            rms = maps[l][m]["rms"] * 1e6
            axm.imshow(rms, origin="lower", cmap="magma", norm=norm, interpolation="nearest", aspect="equal")
            axm.set_xticks([]); axm.set_yticks([])
            for s in axm.spines.values():
                s.set_edgecolor(X.COLOURS[l]); s.set_linewidth(1.0); s.set_visible(True)
            axm.set_title(f"{X.MUSCLES[l][0]}  {maps[l][m]['max']*1e6:.2g} µV", fontsize=6, pad=2, color="k")
            if c_ == 0:
                axm.set_ylabel("along arm →" if r_ == 0 else "", fontsize=6)
                axm.text(-0.55, 0.5, MLAB[m], transform=axm.transAxes, fontsize=7, rotation=90, va="center", ha="center")
                if r_ == 0:                                     # one letter for both map rows
                    letter(axm, "b", dx=-0.75, dy=1.12)
    cax = fig.add_subplot(gmap[:, -1])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="magma"), cax=cax)
    cb.set_label("channel RMS (µV)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    KN["largest_unit_maps"] = {X.MUSCLES[l][0]: dict(k=largest[l][0], **{m: dict(rms_max_uV=float(maps[l][m]["rms"].max() * 1e6),
                                                                              p2p_max_uV=maps[l][m]["max"] * 1e6) for m in X.MONTAGES})
                               for l in X.ORDER}

    # (d) crosstalk ratio vs distance -----------------------------------------------
    ax = fig.add_subplot(gbot[0])
    dF = np.array([r["d_centre"] for r in fcu_rows])
    for m in X.MONTAGES:
        ax.scatter(dF, [r[m]["R_size"] for r in fcu_rows], marker=MK[m], s=6, color="0.72", lw=0, alpha=0.8, zorder=2)
    for r in rows:
        for m in X.MONTAGES:
            ax.scatter(r["d_centre"], r[m]["R_size"], marker=MK[m], s=10 + 0.06 * r["size"], color=X.COLOURS[r["label"]],
                       edgecolor="k", lw=0.3, alpha=0.9, zorder=4)
    xx = np.linspace(min(dF.min(), d_all.min()), d_all.max() + 2, 50)
    for m in X.MONTAGES:
        ax.plot(xx, 10 ** (dist[m]["intercept"] + dist[m]["slope_dex_per_mm"] * xx), color="k", lw=0.7, ls=LS[m], zorder=3)
    ax.axhline(0.1, color="0.4", lw=0.5, ls="-."); ax.text(xx[0] + 0.5, 0.1, "10 %", fontsize=6, va="bottom", color="0.3")
    ax.axhline(1.0, color="0.4", lw=0.4)
    ax.text(0.98, 0.97, "10 % crossing of the fits:\n" + "\n".join(
        f"{MLAB[m]}: " + (f"{dist[m]['d10_fit_mm']:.0f} mm" if abs(dist[m]["r"]) >= 0.3 else f"none (r = {dist[m]['r']:.2f})")
        for m in X.MONTAGES), transform=ax.transAxes, fontsize=5.5, ha="right", va="top")
    ax.set_yscale("log"); ax.set_xlabel("distance of the unit centre from the grid centre (mm)")
    ax.set_ylabel("crosstalk ratio\n(grid-max p2p / FCU size law)")
    h1 = [Line2D([], [], marker="o", ls="", color=X.COLOURS[l], markeredgecolor="k", markeredgewidth=0.3, label=X.MUSCLES[l][0]) for l in X.ORDER]
    h2 = [Line2D([], [], marker=MK[m], ls=LS[m], color="k", lw=0.7, markersize=4, label=MLAB[m]) for m in X.MONTAGES]
    ax.legend(handles=h1 + h2, fontsize=5.5, loc="lower left", ncol=2, handlelength=1.8, columnspacing=0.8)
    letter(ax, "c")

    # (e) MUAPs at the grid centre ----------------------------------------------------
    ge = gbot[1].subgridspec(len(X.ORDER), 1, hspace=0.1)
    ymax = max(np.abs(largest[l][1][C.E0]).max() for l in X.ORDER) * 1e6 * 1.1
    for i, l in enumerate(X.ORDER):
        axe = fig.add_subplot(ge[i]); k, W = largest[l]
        axe.axhline(0, color="0.8", lw=0.4)
        axe.plot(t_ms, W[C.E0] * 1e6, color=X.COLOURS[l], lw=0.9)
        axe.set_ylim(-ymax, ymax); axe.set_xlim(-5, 60)
        d = (fcu_rows[k]["d_centre"] if l == X.FCU else next(r["d_centre"] for r in rows if r["label"] == l and r["k"] == k))
        axe.text(0.99, 0.92, f"{X.MUSCLES[l][0]}: {int(sizes_fcu[k])} fibres, {d:.0f} mm from grid centre, {np.ptp(W[C.E0])*1e6:.2g} µV",
                 transform=axe.transAxes, fontsize=6, ha="right", va="top")
        if i < len(X.ORDER) - 1:
            axe.set_xticklabels([])
        if i == 2:
            axe.set_ylabel("MUAP at the grid centre (µV)")
        if i == 0:
            letter(axe, "d", dx=-0.2)
    axe.set_xlabel("t (ms), 0 = NMJ firing")

    save(fig, "fig_crosstalk")
    KN["total_s"] = time.time() - T0
    with open(X.KEY_JSON, "w") as fh:
        json.dump(_clean(KN), fh, indent=2)
    print("key numbers →", X.KEY_JSON)


if __name__ == "__main__":
    main()
