"""Fig 8 — HD-EMG and the activation layer: one MU on the 5×5 grid (small multiples + a
waterfall down its largest-amplitude column with the fitted conduction velocity), the
interference HD-EMG plateau as an RMS footprint of the grid + one row of monopolar traces
and their single differentials, recruitment and onion-skin rate coding, a trapezoid
contraction (drive, raster, EMG, force) and an angle-modulated dynamic trial.

Uses the cached MUAP tensor (f2_common) and the same trial generators as dataset D3.

Run: /home/dc23/miniconda3/envs/fenicsx-env/bin/python paper/figures/make_fig_hdemg_activation.py
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "paper/figures")
import numpy as np
import matplotlib.pyplot as plt

from style import use, save, letter, COL, W2
import f2_common as C
from emgforge.activation import single_diff, rms_envelope

use()
T0 = time.time()
KN = {}
FS, M = C.FS, C.M

# --------------------------------------------------------------------------- data
D = C.load_muaps()                                     # F2_TENSOR = new | legacy42 | auto
W, t_ms = D["W"], D["t_ms"]                           # (n_grid, 25, 256) V, ms
sizes = D["sizes"]
elec = D["elec_xyz"]
ied_z, ied_t = D["ied_along"], D["ied_across"]
W1 = D["muap_single"]                                  # (100, 256): every MU at the centre electrode
n_grid = D["n_grid_mus"]
mn, tw = C.activation_models()
KN["tensor_source"] = D["source"]; KN["grid_note"] = D["grid_note"]; KN["n_grid_mus"] = n_grid
KN["ied_mm"] = dict(along=ied_z, across=ied_t, along_range=D["ied_along_range"], across_range=D["ied_across_range"])
print(f"MUAP source: {D['source']} ({n_grid} MUs on the grid) — {D['grid_note']}")


# (a) CV from propagation down the grid columns (scripts/sanity/simulator_sanity.py::column_cv)
def _sub_lag(a, b):
    cc = np.correlate(b, a, "full"); k = int(np.argmax(cc)); delta = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]; den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return (k + delta) - (len(a) - 1)


def fit_column(mu, j):
    """Lag of every row of column j against row 0 (sub-sample cross-correlation), a straight
    line through the lags → CV = IED / slope; returns None for a silent/flat column."""
    col = W[mu].reshape(M, M, -1)[:, j]
    if np.ptp(col) == 0:
        return None
    lags = np.array([0.0] + [_sub_lag(col[0], col[i]) for i in range(1, M)])   # samples, vs row 0
    A = np.vstack([np.arange(M), np.ones(M)]).T
    sol = np.linalg.lstsq(A, lags, rcond=None)[0]; slope, icpt = sol; fit = A @ sol
    r2 = 1 - np.sum((lags - fit) ** 2) / max(np.sum((lags - lags.mean()) ** 2), 1e-9)
    if abs(slope) < 1e-6:
        return None
    return dict(j=j, cv=ied_z / (abs(slope) * 1000.0 / FS), r2=float(r2), slope_ms=slope * 1000.0 / FS,
                icpt_ms=icpt * 1000.0 / FS, lags_ms=lags * 1000.0 / FS, energy=float(np.sqrt((col ** 2).mean())))


def column_cv(mu):
    """(score, cv, r2, column) of the best column (highest R² × RMS energy) of one MU."""
    fits = [f for j in range(M) if (f := fit_column(mu, j)) is not None]
    if not fits:
        return None
    b = max(fits, key=lambda f: f["r2"] * f["energy"])
    return (b["r2"] * b["energy"], b["cv"], b["r2"], b["j"])


cv_all = [(mu, *c) for mu in range(W.shape[0]) if (c := column_cv(mu)) is not None]
cv_ok = [c for c in cv_all if c[3] > 0.9]
cvs = np.array([c[2] for c in cv_ok])
mu_rep = max(cv_ok, key=lambda c: c[1])[0]
col_rep = [c for c in cv_ok if c[0] == mu_rep][0][4]
cv_rep = [c for c in cv_ok if c[0] == mu_rep][0][2]
grid_rms = np.sqrt((W[mu_rep].reshape(M, M, -1) ** 2).mean(-1))
# the waterfall column of panel (a): the one holding the largest amplitude, with its own fit
col_amp = int(np.argmax(np.abs(W[mu_rep].reshape(M, M, -1)).max(axis=(0, 2))))
fw = fit_column(mu_rep, col_amp)
KN["cv"] = dict(median_m_per_s=float(np.median(cvs)), min=float(cvs.min()), max=float(cvs.max()),
                n_mus_r2_above_0p9=int(len(cvs)), n_mus_total=int(W.shape[0]),
                corr_size_cv=float(np.corrcoef(sizes[[c[0] for c in cv_ok]], cvs)[0, 1]),
                set_cv=C.CV, mu_rep=int(mu_rep), mu_rep_size=int(sizes[mu_rep]), mu_rep_cv=float(cv_rep),
                mu_rep_column=int(col_rep),
                mu_rep_selectivity_peak_over_mean_rms=float(grid_rms.max() / grid_rms.mean()),
                mu_rep_p2p_uV_range=[float(np.ptp(W[mu_rep], 1).min() * 1e6), float(np.ptp(W[mu_rep], 1).max() * 1e6)],
                waterfall=dict(column=col_amp, cv_m_per_s=float(fw["cv"]), r2=fw["r2"], lags_ms=fw["lags_ms"].tolist(),
                               slope_ms_per_row=float(fw["slope_ms"]), icpt_ms=float(fw["icpt_ms"]),
                               p2p_uV_per_row=(np.ptp(W[mu_rep].reshape(M, M, -1)[:, col_amp], 1) * 1e6).tolist()))
print(f"CV from the grid: median {np.median(cvs):.2f} m/s ({cvs.min():.2f}–{cvs.max():.2f}, {len(cvs)}/{W.shape[0]} MUs with R²>0.9; "
      f"set {C.CV}); representative MU {mu_rep} ({sizes[mu_rep]} fibres, CV {cv_rep:.2f}, column {col_rep}); IED {ied_z:.1f}×{ied_t:.1f} mm")
print(f"waterfall: column {col_amp} (largest amplitude), CV {fw['cv']:.2f} m/s (R² {fw['r2']:.3f}), lags {np.round(fw['lags_ms'], 2)} ms")

# (c) recruitment / rate coding
E_ax = np.linspace(0, 1, 201)
n_act = np.array([int(np.sum(mn.rte < e)) for e in E_ax])
rate_units = [0, 25, 50, 75, 99]
rates = mn.firing_rate(E_ax)
KN["recruitment"] = dict(n_active_at_levels={str(l): int(np.sum(mn.rte < l)) for l in C.LEVELS},
                         rte_first=float(mn.rte[0]), rte_last=float(mn.rte[-1]),
                         peak_rate_first_hz=float(mn.peak_fr[0]), peak_rate_last_hz=float(mn.peak_fr[-1]),
                         min_rate_first_hz=float(mn.min_fr[0]), min_rate_last_hz=float(mn.min_fr[-1]))

# (b, d) trapezoid trials — same generator as dataset D3 (grid EMG only where needed)
t0 = time.time()
trials = {lv: C.trapezoid_trial(lv, mn, tw, W1, Wg=W if lv in (0.35,) else None) for lv in C.LEVELS}
KN["trial_gen_s"] = time.time() - t0
T = len(trials[0.5]["drive"]); pl = C.plateau_slice(T)
rms = {lv: float(np.sqrt((tr["emg_single"][pl] ** 2).mean())) for lv, tr in trials.items()}
frc = {lv: dict(mean=float(tr["force"][pl].mean() * 100), p95=float(np.percentile(tr["force"], 95) * 100))
       for lv, tr in trials.items()}
nact = {lv: int(sum(len(s) > 0 for s in tr["spikes"])) for lv, tr in trials.items()}
KN["rms_vs_drive_uV"] = {str(lv): rms[lv] * 1e6 for lv in C.LEVELS}
KN["force_vs_drive_pct_mvc"] = {str(lv): frc[lv] for lv in C.LEVELS}
KN["n_active_vs_drive"] = {str(lv): nact[lv] for lv in C.LEVELS}
KN["trapezoid"] = dict(**C.TRAP, common_drive=C.COMMON_DRIVE, fs=FS, n_samples=int(T))
print("plateau RMS (µV):", {lv: round(rms[lv] * 1e6, 3) for lv in C.LEVELS})
print("plateau force (%MVC, mean):", {lv: round(frc[lv]["mean"], 1) for lv in C.LEVELS})
print("active MUs:", nact)

# (e) dynamic trial
t0 = time.time()
dyn = C.dynamic_trial(mn, W1)
KN["dynamic_gen_s"] = time.time() - t0
ang = dyn["angle"]; flex, ext = ang > 0.75, ang < 0.25
rms_flex = float(np.sqrt((dyn["emg"][flex] ** 2).mean())); rms_ext = float(np.sqrt((dyn["emg"][ext] ** 2).mean()))
KN["dynamic"] = dict(**C.DYN, rms_flexed_uV=rms_flex * 1e6, rms_extended_uV=rms_ext * 1e6,
                     rms_flexed_over_extended=rms_flex / max(rms_ext, 1e-12),
                     n_active=int(sum(len(s) > 0 for s in dyn["spikes"])),
                     drive_range=[float(dyn["drive"].min()), float(dyn["drive"].max())])
print(f"dynamic: RMS flexed/extended = {rms_flex/rms_ext:.2f} ({rms_flex*1e6:.3f} vs {rms_ext*1e6:.3f} µV)")

# --------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(W2, 7.6))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], left=0.07, right=0.985, top=0.965, bottom=0.06, hspace=0.42)
top = outer[0].subgridspec(1, 3, width_ratios=[1.55, 1.5, 0.78], wspace=0.32)
bot = outer[1].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.25)


def small_multiples(gspec, ncol, nrow, r0=0, c0=0):
    """nrow×ncol tick-less cells of a (sub)gridspec, starting at cell (r0, c0)."""
    axs = np.empty((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            ax = fig.add_subplot(gspec[r0 + i, c0 + j]); axs[i, j] = ax
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(0.3); s.set_color("0.75")
    return axs


# (a) one MU on the grid + waterfall down the largest-amplitude column ---------
gsa_outer = top[0].subgridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.42)
gsa = gsa_outer[0].subgridspec(M, M, wspace=0.08, hspace=0.08)
axa = small_multiples(gsa, M, M)
Wr = W[mu_rep].reshape(M, M, -1) * 1e6
ym = np.abs(Wr).max() * 1.05
tsel = (t_ms >= -5) & (t_ms <= 60)
for i in range(M):
    for j in range(M):
        ax = axa[i, j]
        ax.plot(t_ms[tsel], Wr[i, j, tsel], lw=0.6, color=COL["direct"] if j == col_amp else "0.25")
        ax.set_ylim(-ym, ym); ax.set_xlim(-5, 60)
axa[0, M // 2].set_title(f"MU {mu_rep} ({sizes[mu_rep]} fibres)", fontsize=6, pad=3)
axa[M - 1, 0].plot([44, 44], [-ym * 0.95, -ym * 0.95 + ym], color="k", lw=1.0)
axa[M - 1, 0].text(36, -ym * 0.95 + ym + ym * 0.05, f"{ym:.0f} µV", fontsize=5, ha="center", va="bottom")
axa[M - 1, 0].plot([0, 20], [-ym * 0.95, -ym * 0.95], color="k", lw=1.0)
axa[M - 1, 0].text(10, -ym * 0.9, "20 ms", fontsize=5, ha="center", va="bottom")
axa[M - 1, M // 2].set_xlabel(f"→ around the arm ({ied_t:.0f} mm)", fontsize=6.5, labelpad=2)
axa[M // 2, 0].set_ylabel(f"↓ along the arm ({ied_z:.0f} mm)", fontsize=6.5, labelpad=2)
# waterfall: the five rows of the highlighted column, offset by their z along the arm, so the
# constant-velocity guide (dashed) has slope = CV in these (ms, mm) axes
axw = fig.add_subplot(gsa_outer[1])
T0W, T1W = 0.0, 30.0
wsel = (t_ms >= T0W) & (t_ms <= T1W)
colw = Wr[:, col_amp]                                              # (M, T) µV
amp_w = np.abs(colw[:, wsel]).max()
scale = 0.42 * ied_z / amp_w                                       # mm per µV: a trace spans < one IED
for i in range(M):
    axw.plot(t_ms[wsel], i * ied_z + colw[i, wsel] * scale, lw=0.7, color=COL["direct"], zorder=3)
i_pk = int(np.argmax(np.abs(colw[0, wsel]))); t_pk0 = float(t_ms[wsel][i_pk])
zz = np.array([-0.45, M - 1 + 0.45]) * ied_z
axw.plot(t_pk0 + fw["icpt_ms"] + fw["slope_ms"] * zz / ied_z, zz, ls="--", lw=0.7, color="k", zorder=2)
axw.text(0.98, 0.985, f"CV {fw['cv']:.2f} m/s", transform=axw.transAxes, fontsize=6, ha="right", va="top")
sb = 50.0 if amp_w > 80 else 20.0                                  # µV scale bar, bottom-right (row M-1 is flat there)
y_sb = (M - 1) * ied_z + 0.2 * ied_z
axw.plot([T1W - 1.5] * 2, [y_sb, y_sb + sb * scale], color="k", lw=1.0)
axw.text(T1W - 2.3, y_sb + 0.5 * sb * scale, f"{sb:.0f} µV", fontsize=5, ha="right", va="center")
axw.set_xlim(T0W, T1W); axw.set_ylim(zz[1] + 0.05 * ied_z, zz[0] - 0.05 * ied_z)   # row 0 on top, as in the grid
axw.set_yticks(np.arange(M) * ied_z); axw.set_yticklabels([f"{i * ied_z:.0f}" for i in range(M)])
axw.tick_params(labelsize=6, length=2)
axw.set_xlabel("t (ms), 0 = NMJ firing", fontsize=6.5, labelpad=2)
axw.set_ylabel("z along the arm (mm)", fontsize=6.5, labelpad=1)
axw.set_title(f"column {col_amp}", fontsize=6, pad=3, color=COL["direct"])

# (b) interference HD-EMG at the plateau: RMS footprint + one row of traces ----
tr = trials[0.35]
grid = tr["emg_grid"].reshape(M, M, -1) * 1e6
a, b = int(1.4 * FS), int(2.0 * FS)
tw_s = np.arange(a, b) / FS - a / FS
rms_map = np.sqrt((grid[:, :, pl] ** 2).mean(-1))                 # (M, M) plateau RMS, µV
row_rep = int(np.argmax(rms_map.mean(1)))                          # the row with the largest RMS
col_sd = int(np.argmax(rms_map.mean(0)))
gsb_outer = top[1].subgridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.85)
from matplotlib.patches import Rectangle
axm = fig.add_subplot(gsb_outer[0])
axm.set_anchor("N")                                                # square map at the top of its tall cell
im = axm.imshow(rms_map, cmap="viridis", origin="upper", aspect="equal", interpolation="nearest")
axm.add_patch(Rectangle((-0.5, row_rep - 0.5), M, 1, fill=False, ec="w", lw=1.0, zorder=5))
axm.set_xticks([0, M // 2, M - 1]); axm.set_yticks(range(M))
axm.set_xticklabels([f"{-(M // 2) * ied_t:.0f}", "0", f"+{(M // 2) * ied_t:.0f}"])
axm.set_yticklabels([f"{i * ied_z:.0f}" for i in range(M)])
axm.tick_params(labelsize=6, length=2)
axm.set_xlabel("→ around the arm (mm)", fontsize=6, labelpad=2)
axm.set_ylabel("↓ along the arm (mm)", fontsize=6, labelpad=1)
for s in axm.spines.values():
    s.set_visible(True); s.set_linewidth(0.3)
cax = axm.inset_axes([0.0, -0.5, 1.0, 0.07])                        # horizontal colourbar under the x-label
cbm = fig.colorbar(im, cax=cax, orientation="horizontal")
cbm.set_label("plateau RMS (µV)", fontsize=6.5, labelpad=2); cbm.ax.tick_params(labelsize=6, length=2)
cbm.outline.set_linewidth(0.4)
axm.set_title(f"drive 0.35\n{nact[0.35]}/{len(W1)} MUs active"
              + (f" ({n_grid} on grid)" if n_grid < len(W1) else ""), fontsize=6, pad=3)
# the marked row: five monopolar traces (offset by their position around the arm) and their SD
gsb2 = gsb_outer[1].subgridspec(2, 1, height_ratios=[M, M - 1], hspace=0.3)
axr = fig.add_subplot(gsb2[0]); axsd = fig.add_subplot(gsb2[1])
row = grid[row_rep, :, a:b]                                        # (M, T)
sd = single_diff(row, axis=0)                                      # (M-1, T)
ymono = np.abs(row).max(); ysd = np.abs(sd).max()
scale_b = 0.48 * ied_t / max(ymono, ysd)                           # mm per µV, shared by both stacks
pos = (np.arange(M) - M // 2) * ied_t
for j in range(M):
    axr.plot(tw_s, pos[j] + row[j] * scale_b, lw=0.35, color=COL["mono"] if j == col_sd else "0.25")
axr.set_yticks(pos); axr.set_yticklabels([f"{p:+.0f}" if p else "0" for p in pos])
axr.set_ylim(pos[-1] + 0.55 * ied_t, pos[0] - 0.55 * ied_t)        # left-most electrode on top
axr.set_xlim(tw_s[0], tw_s[-1]); axr.set_xticklabels([])
axr.set_ylabel("around the arm (mm)", fontsize=6.5, labelpad=1); axr.tick_params(labelsize=6, length=2)
axr.set_title(f"row at {row_rep * ied_z:.0f} mm · monopolar", fontsize=6, pad=3)
sbb = 100.0
axr.plot([tw_s[-1] - 0.02] * 2, [pos[-1] + 0.5 * ied_t - sbb * scale_b, pos[-1] + 0.5 * ied_t], color="k", lw=1.0,
         clip_on=False)
axr.text(tw_s[-1] - 0.035, pos[-1] + 0.5 * ied_t - 0.5 * sbb * scale_b, f"{sbb:.0f} µV", fontsize=5, ha="right", va="center")
pos_sd = 0.5 * (pos[1:] + pos[:-1])
for i in range(M - 1):
    axsd.plot(tw_s, pos_sd[i] + sd[i] * scale_b, lw=0.35, color=COL["sd"])
axsd.set_yticks(pos_sd); axsd.set_yticklabels([f"{p:+.0f}" for p in pos_sd])
axsd.set_ylim(pos_sd[-1] + 0.55 * ied_t, pos_sd[0] - 0.55 * ied_t)
axsd.set_xlim(tw_s[0], tw_s[-1]); axsd.tick_params(labelsize=6, length=2)
axsd.set_xlabel("time in the plateau (s)", fontsize=6.5, labelpad=2)
axsd.set_title("single differential (SD) of that row", fontsize=6, pad=3, color=COL["sd"])
KN["hdemg_snippet"] = dict(level=0.35, t0_s=a / FS, t1_s=b / FS, row_shown=row_rep, sd_column_highlighted=col_sd,
                           mono_max_uV=float(ymono), sd_max_uV=float(ysd),
                           rms_map_uV=rms_map.tolist(), rms_map_uV_min=float(rms_map.min()), rms_map_uV_max=float(rms_map.max()),
                           rms_map_argmax_row_col=[int(x) for x in np.unravel_index(rms_map.argmax(), rms_map.shape)],
                           rms_row_shown_uV=rms_map[row_rep].tolist(),
                           rms_sd_row_shown_uV=np.sqrt((single_diff(grid[row_rep, :, pl], axis=0) ** 2).mean(-1)).tolist())
print(f"HD-EMG plateau RMS map: {rms_map.min():.1f}–{rms_map.max():.1f} µV (max at row/col "
      f"{KN['hdemg_snippet']['rms_map_argmax_row_col']}); row {row_rep} shown: {np.round(rms_map[row_rep], 1)} µV")

# (c) recruitment + rate coding ---------------------------------------------
gsc = top[2].subgridspec(2, 1, hspace=0.55)
axc1 = fig.add_subplot(gsc[0]); axc2 = fig.add_subplot(gsc[1])
axc1.plot(E_ax, n_act, color="k", lw=1.0)
for lv in C.LEVELS:
    axc1.plot(lv, np.sum(mn.rte < lv), "o", ms=3, color=COL["direct"])
axc1.set_xlabel("drive (fraction of max)"); axc1.set_ylabel("# active MUs")
axc1.set_xlim(0, 1); axc1.set_ylim(0, 105)
for u in rate_units:
    r = rates[u].copy(); r[r == 0] = np.nan
    axc2.plot(E_ax, r, lw=1.0, color=plt.cm.viridis(u / 99), label=f"MU {u}")
axc2.set_xlabel("drive (fraction of max)"); axc2.set_ylabel("discharge rate (Hz)")
axc2.set_xlim(0, 1); axc2.set_ylim(0, mn.pfr1 + 4)
axc2.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, handlelength=1.0,
            columnspacing=0.6, borderpad=0.2, labelspacing=0.2, handletextpad=0.3)   # below the axes, off the data
# panel letters a/b/c on one line at the top of the row (the equal-aspect map would move its own)
y_top = top[0].get_position(fig).y1 + 0.008
for s, cell, dx in (("a", top[0], 0.045), ("b", gsb_outer[0], 0.04), ("c", top[2], 0.045)):
    fig.text(cell.get_position(fig).x0 - dx, y_top, s, fontsize=9, fontweight="bold", va="bottom", ha="left")

# (d) trapezoid at drive 0.5 -------------------------------------------------
tr = trials[0.5]
t = np.arange(T) / FS
gsd = bot[0].subgridspec(4, 1, hspace=0.12, height_ratios=[0.7, 1.4, 1.0, 0.9])
axd = [fig.add_subplot(gsd[r]) for r in range(4)]
axd[0].plot(t, tr["drive"], color="k", lw=0.9); axd[0].set_ylabel("drive"); axd[0].set_ylim(0, 0.6)
# the common-drive noise (σ 0.015, < 2 Hz) is added over the whole record, lead-in included,
# and clipped at 0 — hence the non-zero start; kept as is so the panel matches dataset D3
axd[0].axvspan(0, C.TRAP["lead_s"], color="0.9", lw=0, zorder=0)
axd[0].annotate(f"lead-in {C.TRAP['lead_s']:.1f} s: common-drive noise only\n(σ = {C.COMMON_DRIVE['sigma']}, "
                f"< {C.COMMON_DRIVE['cutoff_hz']:.0f} Hz, clipped at 0), as in dataset D3",
                xy=(0.1, 0.08), xytext=(0.95, 0.2), textcoords="data", fontsize=5.5, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4", shrinkB=1))
for m, sp in enumerate(tr["spikes"]):
    if m % 5 == 0 and len(sp):
        axd[1].plot(sp / FS, np.full(len(sp), m), "|", color=plt.cm.viridis(m / 99), ms=2.2, mew=0.5)
axd[1].set_ylabel("MU index"); axd[1].set_ylim(-3, 102)
axd[2].plot(t, tr["emg_single"] * 1e6, color="0.2", lw=0.3)
axd[2].set_ylabel("EMG (µV)")
y_emg = max(np.abs(tr["emg_single"]).max(), np.abs(dyn["emg"]).max()) * 1e6 * 1.05   # shared with (e)
axd[2].set_ylim(-y_emg, y_emg)
KN["emg_ylim_uV"] = float(y_emg)
axd[3].plot(t, tr["force"] * 100, color=COL["fourier"], lw=1.0, label="force")
axd[3].plot(t, tr["drive"] * 100, color="0.6", lw=0.6, ls="--", label="drive × 100")
axd[3].set_ylabel("force (%MVC)"); axd[3].set_xlabel("time (s)")
axd[3].legend(fontsize=5.5, loc="upper right", handlelength=1.5, borderpad=0.2)
for ax in axd:
    ax.set_xlim(0, t[-1])
for ax in axd[:3]:
    ax.set_xticklabels([])
axd[1].text(0.01, 0.98, f"{nact[0.5]} MUs active · plateau RMS {rms[0.5]*1e6:.2f} µV · force {frc[0.5]['mean']:.0f} %MVC",
            transform=axd[1].transAxes, fontsize=6, ha="left", va="top")
letter(axd[0], "d", dx=-0.13)

# (e) dynamic trial ----------------------------------------------------------
t2 = np.arange(len(ang)) / FS
gse = bot[1].subgridspec(2, 1, hspace=0.12, height_ratios=[0.7, 1.6])
axe1 = fig.add_subplot(gse[0]); axe2 = fig.add_subplot(gse[1])
axe1.fill_between(t2, ang, color=COL["dd"], alpha=0.2, lw=0)
axe1.plot(t2, ang, color=COL["dd"], lw=1.0)
axe1.set_ylabel("angle\n(0 ext · 1 flex)"); axe1.set_ylim(0, 1.05); axe1.set_xticklabels([])
axe2.plot(t2, dyn["emg"] * 1e6, color="0.2", lw=0.3)
axe2.plot(t2, rms_envelope(dyn["emg"], int(0.15 * FS)) * 1e6, color=COL["fourier"], lw=1.2, label="RMS (0.15 s)")
axe2.set_ylabel("EMG (µV)"); axe2.set_xlabel("time (s)")
axe2.set_ylim(-y_emg, y_emg)                                       # same EMG scale as (d)
axe2.legend(fontsize=6, loc="upper right")
axe2.text(0.01, 0.97, f"RMS flexed / extended = {rms_flex/rms_ext:.2f}", transform=axe2.transAxes, fontsize=6, va="top")
for ax in (axe1, axe2):
    ax.set_xlim(0, t2[-1])
letter(axe1, "e", dx=-0.2)

save(fig, "fig_hdemg_activation")
KN["total_s"] = time.time() - T0
C.update_key_numbers("fig8_hdemg_activation", KN)
