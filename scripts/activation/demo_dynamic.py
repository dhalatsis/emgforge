"""Dynamic EMG: a wrist flexion–extension movement → a non-stationary interference signal.

The isometric chain gives one fixed MUAP per unit. Here a joint-angle track drives both the
excitation (agonist active during flexion) and the MUAPs themselves — amplitude (fibres bulge
toward the electrode as the muscle shortens) and a mild time-warp (apparent AP duration). The
result is EMG whose amplitude and shape are locked to the movement, not just to the drive.

Run: python scripts/activation/demo_dynamic.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.activation import MotoneuronPool, drive
from emgforge.activation.dynamic import (
    angle_track, drive_from_angle, amp_from_angle, warp_from_angle, dynamic_compound_emg,
)
from emgforge.activation.emg import compound_emg, rms_envelope

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/activation"; OUT.mkdir(parents=True, exist_ok=True)
FS = 2048

# --- MUAP pool (single channel) resampled to the run rate ------------------------------
d = np.load(ROOT / "_results/mu_pool/spatial/mu_pool.npz")
t_ms = d["t_ms"]; raw = d["muap_wave"]                       # (100, 256)
w = max(8, round((t_ms[-1] - t_ms[0]) / 1000 * FS))          # MUAP length at FS
muaps = np.array([np.interp(np.linspace(0, 1, w), np.linspace(0, 1, raw.shape[1]), m)
                  for m in raw])
N = len(muaps)

# --- movement: two flexion–extension cycles over 6 s -----------------------------------
DUR = 6.0
angle = angle_track(DUR, fs=FS, cycles=2)                    # θ(t) ∈ [0,1]
E = drive_from_angle(angle)                                  # agonist excitation
E = drive.add_common_drive(E, sigma=0.012, fs=FS, seed=1)
amp = amp_from_angle(angle, gain=0.8)                        # MUAP amplitude vs angle
warp = warp_from_angle(angle, gain=0.22)                     # MUAP time-warp vs angle

pool = MotoneuronPool(n_mu=N, fs=FS)
spikes = pool.spike_trains(E, seed=0)
n_samp = len(E)

dyn = dynamic_compound_emg(spikes, muaps, amp, warp, n_samples=n_samp)     # non-stationary
sta = compound_emg(spikes, muaps, n_samples=n_samp)                        # same spikes, fixed MUAP
t = np.arange(n_samp) / FS
n_active = sum(len(s) > 0 for s in spikes)
print(f"Dynamic demo: {N} MUs, {n_active} active · dyn {dyn.shape} · "
      f"amp {amp.min():.2f}–{amp.max():.2f} · warp {warp.min():.2f}–{warp.max():.2f}")

# --- figure ----------------------------------------------------------------------------
fig = plt.figure(figsize=(13.5, 9))
gs = fig.add_gridspec(4, 2, width_ratios=[3, 1], height_ratios=[1, 1, 1.2, 1.2],
                      hspace=0.42, wspace=0.22)

# (A) joint angle
axA = fig.add_subplot(gs[0, 0])
axA.fill_between(t, angle, color="tab:purple", alpha=0.18)
axA.plot(t, angle, color="tab:purple", lw=2)
axA.set_ylabel("joint angle\n0=ext · 1=flex"); axA.set_title("Wrist flexion–extension movement")
axA.set_xlim(0, DUR); axA.grid(alpha=0.3)

# (B) drive + spike raster
axB = fig.add_subplot(gs[1, 0], sharex=axA)
axB.plot(t, E * 100, "k--", lw=1.2, alpha=0.7, label="excitation")
axB.set_ylabel("% MVC"); axB.legend(fontsize=8, loc="upper right"); axB.grid(alpha=0.3)
axBr = axB.twinx()
for m, sp in enumerate(spikes):
    if len(sp):
        axBr.plot(sp / FS, np.full(len(sp), m), "|", color=plt.cm.viridis(m / N), ms=2.5, mew=.5)
axBr.set_ylabel("MU index", fontsize=8); axBr.set_ylim(-1, N)

# (C) dynamic vs stationary EMG + RMS
axC = fig.add_subplot(gs[2, 0], sharex=axA)
axC.plot(t, dyn * 1e6, lw=0.35, color="0.6")
axC.plot(t, rms_envelope(dyn, int(0.15 * FS)) * 1e6, "tab:red", lw=2, label="dynamic RMS")
axC.plot(t, rms_envelope(sta, int(0.15 * FS)) * 1e6, "tab:blue", lw=1.6, ls="--",
         label="stationary RMS (fixed MUAP)")
axC.set_ylabel("EMG (µV)")
axC.set_title("Interference EMG — movement modulates amplitude beyond the drive")
axC.legend(fontsize=8, loc="upper right"); axC.grid(alpha=0.3)
plt.setp(axC.get_xticklabels(), visible=False)

# (D) MUAP shape: extended phase vs flexed phase (one representative unit)
mu = int(np.argmax([np.ptp(m) for m in muaps]))
i_ext = int(np.argmin(angle)); i_flex = int(np.argmax(angle))
def placed(i):
    k_fac = warp[i]
    idx = np.arange(w); stretched = np.interp(idx, idx * k_fac, muaps[mu], right=0.0)
    return stretched * amp[i]
tm = np.arange(w) / FS * 1000
axD = fig.add_subplot(gs[2:, 1])
axD.plot(tm, placed(i_ext) * 1e6, color="tab:blue", lw=2, label=f"extended (θ={angle[i_ext]:.2f})")
axD.plot(tm, placed(i_flex) * 1e6, color="tab:red", lw=2, label=f"flexed (θ={angle[i_flex]:.2f})")
axD.set_xlabel("ms"); axD.set_ylabel("µV"); axD.set_title(f"MU {mu} — extended vs flexed")
axD.legend(fontsize=7); axD.grid(alpha=0.3)

# (E) spectrogram of the dynamic EMG — spectral non-stationarity
axE = fig.add_subplot(gs[3, 0], sharex=axA)
axE.specgram(dyn, NFFT=512, Fs=FS, noverlap=384, cmap="magma")
axE.set_ylim(0, 400); axE.set_ylabel("Hz"); axE.set_xlabel("time (s)")
axE.set_title("Spectrogram — spectral content tracks the movement")

# (top-right) amp & warp tracks
axT = fig.add_subplot(gs[:2, 1])
axT.plot(t, amp, color="tab:red", lw=1.8, label="amplitude ×")
axT.plot(t, warp, color="tab:orange", lw=1.8, label="warp ×")
axT.axhline(1, color="0.6", lw=0.8, ls=":")
axT.set_title("MUAP modulation\nvs joint angle", fontsize=10)
axT.legend(fontsize=7); axT.grid(alpha=0.3); axT.set_xlim(0, DUR)

fig.suptitle("emgforge — dynamic (non-stationary) EMG over a movement", fontsize=13, y=0.995)
fig.savefig(OUT / "dynamic_demo.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "dynamic_demo.png")
