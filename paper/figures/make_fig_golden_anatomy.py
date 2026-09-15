"""Figure 3 — anatomy of the golden SFAP recipe: (a) Rosenfalck IAP and its derivatives,
(b) the CSD(z, t) source matrix (boxcar and one-sided tendon windows), (c) the fibre-end
windows, (d) FEM φ raw vs monopole-denoised vs analytical, with φ'', (e) the resulting SFAPs.

Run from the repo root:  python paper/figures/make_fig_golden_anatomy.py   (~10 s)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "paper/figures")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt

from style import use, save, letter, COL, W1, W2               # noqa: F401
from f1_common import DZ, W, Z, ana_phi, dc_free, fem_phi, golden_cfg, normed, record, sfap
from emgforge.synthesis.iap import rosenfalck_vm
from emgforge.synthesis.engines.spatial import build_csd_matrix
from emgforge.synthesis.preprocessing import create_fiber_windows, denoise_field_n, upsample_cubic
from emgforge.synthesis.metrics import jaggedness

use()
NUM = {}
POSZ, LEN1, LEN2 = -20.0, 40.0, 80.0             # golden test fibre: NMJ 20 mm from the electrode

# ------------------------------------------------------------------ (a) IAP
xi = np.linspace(0, 15, 1501)
vm = rosenfalck_vm(xi) * 1e3                       # mV
dvm = np.gradient(vm, xi); ddvm = np.gradient(dvm, xi)
NUM["iap"] = dict(vm_peak_mV=float(vm.max()), vm_peak_at_mm=float(xi[np.argmax(vm)]),
                  d2vm_lobes_mm=[float(xi[np.argmax(ddvm)]), float(xi[np.argmin(ddvm)]),
                                 float(xi[np.argmax(np.where(xi > xi[np.argmin(ddvm)], ddvm, -1e9))])])

# ------------------------------------------------------------------ (b) CSD matrices
dz_e = DZ / 2.0                                    # the engine integrates on the 2× upsampled grid
z_e = (np.arange(2 * W) - W) * dz_e
t_img = np.arange(-2.0, 20.0001, 0.05)
csd = {w: build_csd_matrix(z_e, t_img, 0.0, 60.0, 60.0, dz_e, golden_cfg(fiber_window=w))
       for w in ("boxcar", "one_sided")}
prop = np.abs(csd["boxcar"][(t_img > 3) & (t_img < 8)][:, np.abs(z_e) < 45]).max()   # propagating tripole scale
NUM["csd"] = dict(propagating_max=float(prop),
                  boxcar_tendon_term_over_propagating=float(np.abs(csd["boxcar"]).max() / prop),
                  one_sided_tendon_term_over_propagating=float(np.abs(csd["one_sided"]).max() / prop),
                  net_current_max_over_abs={w: float(np.max(np.abs(c.sum(1))) / np.max(np.abs(c).sum(1)))
                                            for w, c in csd.items()})

# ------------------------------------------------------------------ (c) windows
n_pts = int(120.0 / dz_e)
z_w = -60.0 + np.arange(n_pts) * dz_e
wins = {}
for name, wt in (("boxcar", "boxcar"), ("tukey (α = 0.25)", "tukey"), ("one-sided (α = 0.25)", "one_sided")):
    wl, wr = create_fiber_windows(n_pts, 0.5, window_type=wt, tukey_alpha=0.25)
    wins[name] = wl + wr
NUM["windows"] = {k: dict(value_at_nmj=float(v[n_pts // 2]), value_at_nmj_minus1=float(v[n_pts // 2 - 1]),
                          taper_len_mm=float(dz_e * np.sum(v[: n_pts // 2] < 0.999))) for k, v in wins.items()}

# ------------------------------------------------------------------ (d) φ and φ''
pf_raw = fem_phi(30.0, 0.0)
pf_den = denoise_field_n(pf_raw, DZ, n=3)
pa = ana_phi(30.0)[0]
sc_f = np.abs(dc_free(pf_raw)).max(); sc_a = np.abs(dc_free(pa)).max()
phi_d = {"FEM φ, raw": dc_free(pf_raw) / sc_f, "FEM φ, monopole-denoised (n = 3)": dc_free(pf_den) / sc_f,
         "analytical φ": dc_free(pa) / sc_a}
def dd(p):                                         # the engine's route: 2× cubic upsample, then ∂²/∂z²
    u = upsample_cubic(p, 2)
    return np.gradient(np.gradient(u, dz_e), dz_e)
phi_dd = {k: dd(v) for k, v in phi_d.items()}
core = np.abs(z_e) <= 60
NUM["phi_r30"] = dict(
    r_phi_raw_vs_ana=float(np.corrcoef(phi_d["FEM φ, raw"][np.abs(Z) <= 60], phi_d["analytical φ"][np.abs(Z) <= 60])[0, 1]),
    r_phi_den_vs_ana=float(np.corrcoef(phi_d["FEM φ, monopole-denoised (n = 3)"][np.abs(Z) <= 60], phi_d["analytical φ"][np.abs(Z) <= 60])[0, 1]),
    r_phidd_raw_vs_ana=float(np.corrcoef(phi_dd["FEM φ, raw"][core], phi_dd["analytical φ"][core])[0, 1]),
    r_phidd_den_vs_ana=float(np.corrcoef(phi_dd["FEM φ, monopole-denoised (n = 3)"][core], phi_dd["analytical φ"][core])[0, 1]),
    phidd_ripple_rms_raw_over_den=float(np.std(np.diff(phi_dd["FEM φ, raw"][core])) / np.std(np.diff(phi_dd["FEM φ, monopole-denoised (n = 3)"][core]))),
    max_abs_dev_den_vs_raw_phi=float(np.abs(phi_d["FEM φ, raw"] - phi_d["FEM φ, monopole-denoised (n = 3)"]).max()),
)

# ------------------------------------------------------------------ (e) SFAPs
t, s_ana = sfap(pa, DZ, LEN1, LEN2, POSZ)
_, s_raw = sfap(pf_raw, DZ, LEN1, LEN2, POSZ, golden_cfg(denoise="none"))
_, s_den = sfap(pf_raw, DZ, LEN1, LEN2, POSZ)
NUM["sfap_r30_nmj-20"] = dict(r_raw_vs_ana=float(np.corrcoef(s_ana, s_raw)[0, 1]),
                             r_den_vs_ana=float(np.corrcoef(s_ana, s_den)[0, 1]),
                             jaggedness_raw=float(jaggedness(s_raw)), jaggedness_den=float(jaggedness(s_den)),
                             jaggedness_ana=float(jaggedness(s_ana)),
                             p2p_fem_over_ana_raw=float(np.ptp(s_raw) / np.ptp(s_ana)),
                             p2p_fem_over_ana_den=float(np.ptp(s_den) / np.ptp(s_ana)))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(W2, 4.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], width_ratios=[1, 1.15, 1.15], hspace=0.55, wspace=0.42,
                      left=0.07, right=0.98, top=0.95, bottom=0.09)

# (a) IAP
ax = fig.add_subplot(gs[0, 0])
ax.plot(xi, vm / vm.max(), color=COL["first"], lw=1.1, label="$V_m$")
ax.plot(xi, dvm / np.abs(dvm).max(), color=COL["analytical"], lw=0.9, ls="--", label="$V_m'$")
ax.plot(xi, ddvm / np.abs(ddvm).max(), color=COL["fourier"], lw=0.9, label="$V_m''$ ∝ $i_m$")
ax.axhline(0, color="0.85", lw=0.5, zorder=0)
for x_, s_, y_ in zip(NUM["iap"]["d2vm_lobes_mm"], ("+", "−", "+"), (1.12, -0.72, 0.26)):
    ax.text(x_, y_, s_, fontsize=8, ha="center", va="center", color=COL["fourier"])
ax.set_xlim(0, 12); ax.set_ylim(-1.25, 1.2)
ax.set_xlabel("ξ = v t − |z − z$_{NMJ}$| (mm)"); ax.set_ylabel("normalised")
# legend lower right (all three curves have decayed there) so the +/− lobe markers stay clear of it
ax.legend(loc="lower right", handlelength=1.5, borderaxespad=0.1, fontsize=6.5, labelspacing=0.3)
ax.text(0.98, 0.97, "$V_m$ = 96 ξ³e$^{-ξ}$ mV", transform=ax.transAxes, fontsize=6.5, ha="right", va="top")
letter(ax, "a", dx=-0.25)

# (b) CSD images
zsel = np.abs(z_e) <= 75
ims = []
for k, (w, tag) in enumerate((("boxcar", "boxcar tendon window"), ("one_sided", "one-sided tendon window"))):
    ax = fig.add_subplot(gs[0, 1 + k])
    im = ax.imshow(csd[w][:, zsel] / prop, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[z_e[zsel][0], z_e[zsel][-1], t_img[0], t_img[-1]], interpolation="nearest",
                   rasterized=True)
    ims.append(im)
    for zt in (-60, 60):
        ax.axvline(zt, color="0.4", lw=0.5, ls=(0, (2, 1.5)))
    ax.axhline(15.0, color="0.4", lw=0.5, ls=(0, (2, 1.5)))
    ax.text(0.97, 0.04, tag, transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom")
    if k == 0:
        ax.annotate("NMJ cusp", (0, 0.3), (16, 4.5), fontsize=6, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="0.2"))
        ax.annotate("tendon end", (60, 15.3), (10, 18.3), fontsize=6, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="0.2"))
        ax.text(-72, 15.4, "L/v", fontsize=6, ha="left", va="bottom", color="0.25")
        ax.set_ylabel("t (ms)")
    else:
        ax.tick_params(labelleft=False)
        ax.annotate("softened end", (55, 13.2), (5, 18.3), fontsize=6, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="0.2"))
    ax.set_xlabel("z (mm)"); ax.set_xticks([-60, -30, 0, 30, 60])
    letter(ax, "b" if k == 0 else "", dx=-0.16)
cb = fig.colorbar(ims[1], ax=ax, fraction=0.06, pad=0.03)
cb.set_label("$i_m$ / max propagating", fontsize=6.5, labelpad=2); cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)

# (c) windows
ax = fig.add_subplot(gs[1, 0])
zpad = np.r_[-70, -60 - 1e-6, z_w, 60, 70]
sty = {"boxcar": dict(color=COL["first"], lw=0.9), "tukey (α = 0.25)": dict(color=COL["raw"], lw=0.9, ls="--"),
       "one-sided (α = 0.25)": dict(color=COL["golden"], lw=1.3)}
for name, wv in wins.items():
    ax.plot(zpad, np.r_[0, 0, wv, 0, 0], label=name, **sty[name])
ax.axvline(0, color="0.85", lw=0.5, zorder=0)
ax.text(0, 1.06, "NMJ", fontsize=6, ha="center", va="bottom", color="0.3")
ax.set_xlim(-70, 70); ax.set_ylim(-0.05, 1.2); ax.set_xticks([-60, -30, 0, 30, 60])
ax.set_xlabel("z − z$_{NMJ}$ (mm)"); ax.set_ylabel("fibre-end window")
# the tukey window dips to 0 at the NMJ, so no in-axes spot is free of data: legend floats above the axes
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1, handlelength=1.6, borderaxespad=0.0, fontsize=6.2,
          labelspacing=0.25)
letter(ax, "c", dx=-0.25)

# (d) φ and φ''
sub = gs[1, 1].subgridspec(2, 1, hspace=0.12, height_ratios=[1, 1])
axd1 = fig.add_subplot(sub[0]); axd2 = fig.add_subplot(sub[1], sharex=axd1)
colr = {"FEM φ, raw": COL["raw"], "FEM φ, monopole-denoised (n = 3)": COL["fem"], "analytical φ": COL["analytical"]}
for name in ("FEM φ, raw", "analytical φ", "FEM φ, monopole-denoised (n = 3)"):
    lw = 1.4 if name.startswith("FEM φ, raw") else 0.9
    axd1.plot(Z, phi_d[name], color=colr[name], lw=lw, label=name.replace("FEM φ, ", "FEM ").replace("analytical φ", "analytical"))
    axd2.plot(z_e, phi_dd[name] / np.abs(phi_dd["analytical φ"][core]).max(), color=colr[name], lw=lw)
axd1.set_xlim(-60, 60); axd1.set_ylim(-0.1, 1.12)
axd1.set_ylabel("φ (norm.)"); axd1.tick_params(labelbottom=False)
# φ fills the whole panel (0.15 at the edges, 1 at the peak): legend floats above the axes, one column
# (a two-column version reaches into the gutter and collides with the panel letter of (e))
axd1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, handlelength=1.4, borderaxespad=0.0, fontsize=6,
            labelspacing=0.25, handletextpad=0.5)
axd2.axhline(0, color="0.85", lw=0.5, zorder=0)
axd2.set_ylim(-4.2, 3.2)
axd2.set_ylabel("φ'' (norm.)"); axd2.set_xlabel("z (mm)"); axd2.set_xticks([-60, -30, 0, 30, 60])
axd2.text(0.02, 0.05, "mesh ripple → φ''", transform=axd2.transAxes, fontsize=6, ha="left", va="bottom", color="0.3")
letter(axd1, "d", dx=-0.25)

# (e) SFAPs
ax = fig.add_subplot(gs[1, 2])
ax.plot(t, normed(s_raw), color=COL["raw"], lw=1.4, label=f"FEM φ, no denoise (r = {NUM['sfap_r30_nmj-20']['r_raw_vs_ana']:.3f})")
ax.plot(t, normed(s_ana), color=COL["analytical"], lw=0.9, label="analytical φ (reference)")
ax.plot(t, normed(s_den), color=COL["fem"], lw=0.9, ls="--", label=f"FEM φ, golden (r = {NUM['sfap_r30_nmj-20']['r_den_vs_ana']:.3f})")
ax.axhline(0, color="0.85", lw=0.5, zorder=0)
ax.set_xlim(-2, 30); ax.set_ylim(-1.6, 1.35); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel("t (ms), t = 0 at the NMJ"); ax.set_ylabel("SFAP (norm.)")
ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0.1, fontsize=6, labelspacing=0.2)
# text box in the head-room below the −1 trough so it never touches the traces
ax.text(0.02, 0.02, f"r = 30 mm, NMJ at −20 mm, fibre [−60, 60]\njaggedness {NUM['sfap_r30_nmj-20']['jaggedness_raw']:.3f} → "
        f"{NUM['sfap_r30_nmj-20']['jaggedness_den']:.3f}", transform=ax.transAxes, fontsize=6, ha="left", va="bottom", color="0.3")
letter(ax, "e", dx=-0.25)

save(fig, "fig_golden_anatomy")
record("fig3_golden_anatomy", NUM)
