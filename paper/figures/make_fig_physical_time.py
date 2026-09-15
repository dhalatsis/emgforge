"""Figure 5 — physical time: golden SFAPs on the FEM lead field along a 9-electrode
array. (a) monopolar, boxcar tendons: the propagating lobe walks at |z|/v, the
end-of-fibre potential is pinned at L/v; (b) single-differential montage with the
phase reversal at the innervation zone; (c) monopolar with the golden one-sided window.

Run from the repo root:  python paper/figures/make_fig_physical_time.py   (~5 s)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "paper/figures")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt

from style import use, save, letter, COL, W1, W2               # noqa: F401
from f1_common import DZ, FS, V, fem_phi, golden_cfg, record, sfap, shift_phi

use()
NUM = {}
pf = fem_phi(30.0, 0.0)
ZC = np.arange(-40.0, 40.1, 10.0)
L = 60.0
IED = 10.0


def one(zel, cfg, l1=L, l2=L):
    return sfap(shift_phi(pf, DZ, zel), DZ, l1, l2, 0.0, cfg)


cfg_box, cfg_os = golden_cfg(fiber_window="boxcar"), golden_cfg()
t = one(0.0, cfg_box)[0]
mono_box = np.array([one(z, cfg_box)[1] for z in ZC])
mono_os = np.array([one(z, cfg_os)[1] for z in ZC])
sd_box = np.array([one(z + IED / 2, cfg_box)[1] - one(z - IED / 2, cfg_box)[1] for z in ZC])

# numbers: propagating-lobe timing (negative peak before the EOF) → CV; EOF onset by subtraction
lobe_t = np.array([t[np.argmin(np.where(t < 13.5, m, np.inf))] for m in mono_box])
sel = np.abs(ZC) >= 10
slope = np.polyfit(np.abs(ZC[sel]), lobe_t[sel], 1)
NUM["propagating_lobe"] = dict(t_neg_peak_ms={f"{z:+.0f}": float(v) for z, v in zip(ZC, lobe_t)},
                               cv_from_lobe_fit_m_per_s=float(1.0 / slope[0]), lobe_delay_at_z0_ms=float(slope[1]))
long_ = np.array([one(z, cfg_box, 120.0, 120.0)[1] for z in ZC])
diff = mono_box - long_
onset = [float(t[np.where(np.abs(d) > 0.03 * np.abs(d).max())[0][0]]) for d in diff]
NUM["eof"] = dict(expected_L_over_v_ms=L / V, onset_ms={f"{z:+.0f}": o for z, o in zip(ZC, onset)},
                  onset_spread_ms=float(np.ptp(onset)),
                  eof_over_propagating_boxcar=float(np.abs(diff).max() / np.ptp(long_, axis=1).max()),
                  eof_over_propagating_one_sided=float(np.abs(mono_os - np.array([one(z, cfg_os, 120.0, 120.0)[1] for z in ZC])).max()
                                                       / np.ptp(long_, axis=1).max()))
sd_null = float(np.ptp(sd_box[4]) / max(np.ptp(s) for s in sd_box))
rev = [float(np.corrcoef(sd_box[4 + k], sd_box[4 - k])[0, 1]) for k in (1, 2, 3, 4)]
sym = [float(np.corrcoef(mono_box[4 + k], mono_box[4 - k])[0, 1]) for k in (1, 2, 3, 4)]
NUM["iz"] = dict(sd_at_iz_over_max=sd_null, sd_r_plus_minus=rev, mono_r_plus_minus=sym)

# figure --------------------------------------------------------------------
# (the lead-field / fibre parameters — FEM φ, fibre 10 mm deep, tendons ±60 mm, v = 4 m/s — are in the caption)
fig, axes = plt.subplots(1, 3, figsize=(W2, 3.1), sharex=True, gridspec_kw=dict(wspace=0.18, left=0.07, right=0.99, top=0.93, bottom=0.14))
STEP = 1.0
D0 = NUM["propagating_lobe"]["lobe_delay_at_z0_ms"]      # the negative lobe trails the wave front by ~1.05 ms (tripole)
guide = np.abs(ZC) / V + D0
panels = [(mono_box, "a", COL["mono"], "monopolar, boxcar tendons"),
          (sd_box, "b", COL["sd"], f"single differential (IED {IED:g} mm), boxcar"),
          (mono_os, "c", COL["golden"], "monopolar, one-sided tendon window")]
for ax, (stack, let, col, tag) in zip(axes, panels):
    scale = 0.55 * STEP / np.abs(stack).max()
    for k, (z, s) in enumerate(zip(ZC, stack)):
        ax.axhline(k * STEP, color="0.92", lw=0.5, zorder=0)
        ax.plot(t, s * scale + k * STEP, color=col, lw=0.9)
    # propagating guide: t = |z| / v + D0 (wave-front arrival + the lobe's lag behind the front)
    ax.plot(guide, np.arange(len(ZC)) * STEP, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axvline(L / V, color="0.35", lw=0.7, ls=(0, (1, 1.2)), zorder=1)
    ax.set_yticks(np.arange(len(ZC)) * STEP); ax.set_yticklabels([f"{z:+.0f}" for z in ZC])
    ax.set_xlim(-2, 24); ax.set_ylim(-0.8, len(ZC) * STEP - 0.05)
    ax.set_xlabel("t (ms), t = 0 at the NMJ")
    ax.text(0.02, 0.995, tag, transform=ax.transAxes, fontsize=6.5, ha="left", va="top", color="0.25")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    letter(ax, let, dx=-0.1 if ax is axes[0] else -0.06)
axes[0].set_ylabel("electrode z (mm)")
axes[0].text(L / V + 0.3, -0.72, "L/v", fontsize=6, ha="left", va="bottom", color="0.3")
axes[0].text(guide[0] - 1.2, -0.72, f"|z|/v + {D0:.1f} ms", fontsize=6, ha="right", va="bottom", color="0.3")
on30 = NUM["eof"]["onset_ms"]["+30"]                       # measured EOF onset on the +30 channel (≈ L/v)
axes[0].annotate("EOF onset", (on30, 7.0 + 0.06), (18.0, 7.55), fontsize=6, ha="left", va="bottom", color="0.3",
                 arrowprops=dict(arrowstyle="->", lw=0.4, color="0.4", shrinkB=1.5))
axes[1].annotate("IZ: null,\nphase reversal", (2.5, 4.0), (9.0, 3.3), fontsize=6, ha="left", va="center", color="0.3",
                 arrowprops=dict(arrowstyle="->", lw=0.4, color="0.4"))
axes[2].annotate("EOF softened", (L / V, 7.0 + 0.06), (17.5, 7.55), fontsize=6, ha="left", va="bottom", color="0.3",
                 arrowprops=dict(arrowstyle="->", lw=0.4, color="0.4", shrinkB=1.5))
save(fig, "fig_physical_time")
record("fig5_physical_time", NUM)
