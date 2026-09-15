"""Fig 8 — HD-EMG and the activation layer: one MU on the 5×5 grid (propagation down the
columns), an interference HD-EMG plateau snippet + single-differential column, recruitment
and onion-skin rate coding, a trapezoid contraction (drive, raster, EMG, force) and an
angle-modulated dynamic trial.

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


def column_cv(mu):
    grid = W[mu].reshape(M, M, -1); best = None
    for j in range(M):
        col = grid[:, j]
        if np.ptp(col) == 0:
            continue
        lags = np.array([0.0] + [_sub_lag(col[0], col[i]) for i in range(1, M)])
        A = np.vstack([np.arange(M), np.ones(M)]).T
        sol = np.linalg.lstsq(A, lags, rcond=None)[0]; slope = sol[0]; fit = A @ sol
        r2 = 1 - np.sum((lags - fit) ** 2) / max(np.sum((lags - lags.mean()) ** 2), 1e-9)
        if abs(slope) < 1e-6:
            continue
        cv = ied_z / (abs(slope) * 1000.0 / FS)
        energy = float(np.sqrt((col ** 2).mean()))
        cand = (r2 * energy, cv, r2, j)
        if best is None or cand[0] > best[0]:
            best = cand
    return best


cv_all = [(mu, *c) for mu in range(W.shape[0]) if (c := column_cv(mu)) is not None]
cv_ok = [c for c in cv_all if c[3] > 0.9]
cvs = np.array([c[2] for c in cv_ok])
mu_rep = max(cv_ok, key=lambda c: c[1])[0]
col_rep = [c for c in cv_ok if c[0] == mu_rep][0][4]
cv_rep = [c for c in cv_ok if c[0] == mu_rep][0][2]
grid_rms = np.sqrt((W[mu_rep].reshape(M, M, -1) ** 2).mean(-1))
KN["cv"] = dict(median_m_per_s=float(np.median(cvs)), min=float(cvs.min()), max=float(cvs.max()),
                n_mus_r2_above_0p9=int(len(cvs)), n_mus_total=int(W.shape[0]),
                corr_size_cv=float(np.corrcoef(sizes[[c[0] for c in cv_ok]], cvs)[0, 1]),
                set_cv=C.CV, mu_rep=int(mu_rep), mu_rep_size=int(sizes[mu_rep]), mu_rep_cv=float(cv_rep),
                mu_rep_column=int(col_rep),
                mu_rep_selectivity_peak_over_mean_rms=float(grid_rms.max() / grid_rms.mean()),
                mu_rep_p2p_uV_range=[float(np.ptp(W[mu_rep], 1).min() * 1e6), float(np.ptp(W[mu_rep], 1).max() * 1e6)])
print(f"CV from the grid: median {np.median(cvs):.2f} m/s ({cvs.min():.2f}–{cvs.max():.2f}, {len(cvs)}/{W.shape[0]} MUs with R²>0.9; "
      f"set {C.CV}); representative MU {mu_rep} ({sizes[mu_rep]} fibres, CV {cv_rep:.2f}, column {col_rep}); IED {ied_z:.1f}×{ied_t:.1f} mm")

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
fig = plt.figure(figsize=(W2, 7.4))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], left=0.07, right=0.985, top=0.965, bottom=0.06, hspace=0.32)
top = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.35, 0.75], wspace=0.3)
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


# (a) one MU on the grid --------------------------------------------------
gsa = top[0].subgridspec(M, M, wspace=0.08, hspace=0.08)
axa = small_multiples(gsa, M, M)
Wr = W[mu_rep].reshape(M, M, -1) * 1e6
ym = np.abs(Wr).max() * 1.05
tsel = (t_ms >= -5) & (t_ms <= 60)
for i in range(M):
    for j in range(M):
        ax = axa[i, j]
        ax.plot(t_ms[tsel], Wr[i, j, tsel], lw=0.6, color=COL["golden"] if j == col_rep else "0.25")
        ax.set_ylim(-ym, ym); ax.set_xlim(-5, 60)
axa[0, M // 2].set_title(f"MU {mu_rep} ({sizes[mu_rep]} fibres) · CV {cv_rep:.1f} m/s", fontsize=6, pad=3)
axa[M - 1, 0].plot([44, 44], [-ym * 0.95, -ym * 0.95 + ym], color="k", lw=1.0)
axa[M - 1, 0].text(36, -ym * 0.95 + ym + ym * 0.05, f"{ym:.0f} µV", fontsize=5, ha="center", va="bottom")
axa[M - 1, 0].plot([0, 20], [-ym * 0.95, -ym * 0.95], color="k", lw=1.0)
axa[M - 1, 0].text(10, -ym * 0.9, "20 ms", fontsize=5, ha="center", va="bottom")
axa[M - 1, M // 2].set_xlabel(f"→ around the arm ({ied_t:.0f} mm)", fontsize=6.5, labelpad=2)
axa[M // 2, 0].set_ylabel(f"↓ along the arm ({ied_z:.0f} mm)", fontsize=6.5, labelpad=2)
letter(axa[0, 0], "a", dx=-0.45, dy=1.25)

# (b) interference HD-EMG plateau snippet + SD column ------------------------
tr = trials[0.35]
grid = tr["emg_grid"].reshape(M, M, -1) * 1e6
a, b = int(1.4 * FS), int(2.0 * FS)
tw_s = np.arange(a, b) / FS
gsb = top[1].subgridspec(M, M + 2, wspace=0.08, hspace=0.08, width_ratios=[1] * M + [0.35, 1.0])
axb = small_multiples(gsb, M, M)
yb = np.abs(grid[:, :, a:b]).max() * 1.05
col_sd = int(np.argmax(np.sqrt((grid[:, :, a:b] ** 2).mean(axis=(0, 2)))))
for i in range(M):
    for j in range(M):
        axb[i, j].plot(tw_s, grid[i, j, a:b], lw=0.35, color=COL["mono"] if j == col_sd else "0.25")
        axb[i, j].set_ylim(-yb, yb); axb[i, j].set_xlim(tw_s[0], tw_s[-1])
jb = M - 1 if col_sd != M - 1 else 0                              # scale bars in a quiet corner cell
axb[M - 1, jb].plot([tw_s[0] + 0.45] * 2, [-yb * 0.95, -yb * 0.95 + yb], color="k", lw=1.0)
axb[M - 1, jb].text(tw_s[0] + 0.45, -yb * 0.95 + yb + yb * 0.05, f"{yb:.0f} µV", fontsize=5, ha="center", va="bottom")
axb[M - 1, jb].plot([tw_s[0] + 0.03, tw_s[0] + 0.23], [-yb * 0.95] * 2, color="k", lw=1.0)
axb[M - 1, jb].text(tw_s[0] + 0.13, -yb * 0.9, "0.2 s", fontsize=5, ha="center", va="bottom")
axb[0, M // 2].set_title(f"drive 0.35 · {nact[0.35]}/{len(W1)} MUs active"
                         + (f" ({n_grid} on grid)" if n_grid < len(W1) else ""), fontsize=6, pad=3)
axb[M - 1, M // 2].set_xlabel("→ around the arm", fontsize=6.5, labelpad=2)
sd = single_diff(grid[:, col_sd, a:b], axis=0)                     # (M-1, T)
ysd = np.abs(sd).max() * 1.05
axsd = small_multiples(gsb, 1, M - 1, r0=0, c0=M + 1)
for i in range(M - 1):
    axsd[i, 0].plot(tw_s, sd[i], lw=0.35, color=COL["sd"])
    axsd[i, 0].set_ylim(-ysd, ysd); axsd[i, 0].set_xlim(tw_s[0], tw_s[-1])
axsd[0, 0].set_title("SD", fontsize=6, pad=3, color=COL["sd"])
axsd[M - 2, 0].plot([tw_s[0] + 0.3] * 2, [-ysd * 0.95, -ysd * 0.95 + ysd], color="k", lw=1.0)
axsd[M - 2, 0].text(tw_s[0] + 0.3, -ysd * 0.95 + ysd + ysd * 0.05, f"{ysd:.0f} µV", fontsize=5, ha="center", va="bottom")
KN["hdemg_snippet"] = dict(level=0.35, t0_s=a / FS, t1_s=b / FS, mono_max_uV=float(yb / 1.05), sd_max_uV=float(ysd / 1.05),
                           sd_column=int(col_sd), grid_rms_uV_min=float(np.sqrt((grid[:, :, a:b] ** 2).mean(-1)).min()),
                           grid_rms_uV_max=float(np.sqrt((grid[:, :, a:b] ** 2).mean(-1)).max()))
letter(axb[0, 0], "b", dx=-0.45, dy=1.25)

# (c) recruitment + rate coding ---------------------------------------------
gsc = top[2].subgridspec(2, 1, hspace=0.55)
axc1 = fig.add_subplot(gsc[0]); axc2 = fig.add_subplot(gsc[1])
axc1.plot(E_ax, n_act, color="k", lw=1.0)
for lv in C.LEVELS:
    axc1.plot(lv, np.sum(mn.rte < lv), "o", ms=3, color=COL["golden"])
axc1.set_xlabel("drive (fraction of max)"); axc1.set_ylabel("# active MUs")
axc1.set_xlim(0, 1); axc1.set_ylim(0, 105)
for u in rate_units:
    r = rates[u].copy(); r[r == 0] = np.nan
    axc2.plot(E_ax, r, lw=1.0, color=plt.cm.viridis(u / 99), label=f"MU {u}")
axc2.set_xlabel("drive (fraction of max)"); axc2.set_ylabel("discharge rate (Hz)")
axc2.set_xlim(0, 1); axc2.set_ylim(0, mn.pfr1 + 4)
axc2.legend(fontsize=5.5, loc="lower right", ncol=1, handlelength=1.2, borderpad=0.2, labelspacing=0.2)
letter(axc1, "c", dx=-0.4)

# (d) trapezoid at drive 0.5 -------------------------------------------------
tr = trials[0.5]
t = np.arange(T) / FS
gsd = bot[0].subgridspec(4, 1, hspace=0.12, height_ratios=[0.7, 1.4, 1.0, 0.9])
axd = [fig.add_subplot(gsd[r]) for r in range(4)]
axd[0].plot(t, tr["drive"], color="k", lw=0.9); axd[0].set_ylabel("drive"); axd[0].set_ylim(0, 0.6)
for m, sp in enumerate(tr["spikes"]):
    if m % 5 == 0 and len(sp):
        axd[1].plot(sp / FS, np.full(len(sp), m), "|", color=plt.cm.viridis(m / 99), ms=2.2, mew=0.5)
axd[1].set_ylabel("MU index"); axd[1].set_ylim(-3, 102)
axd[2].plot(t, tr["emg_single"] * 1e6, color="0.2", lw=0.3)
axd[2].set_ylabel("EMG (µV)")
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
axe2.legend(fontsize=6, loc="upper right")
axe2.text(0.01, 0.97, f"RMS flexed / extended = {rms_flex/rms_ext:.2f}", transform=axe2.transAxes, fontsize=6, va="top")
for ax in (axe1, axe2):
    ax.set_xlim(0, t2[-1])
letter(axe1, "e", dx=-0.2)

save(fig, "fig_hdemg_activation")
KN["total_s"] = time.time() - T0
C.update_key_numbers("fig8_hdemg_activation", KN)
