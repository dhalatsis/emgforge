"""Sample surface EMG for a few contractions — the full chain end to end.

neural drive → MotoneuronPool spikes → × per-MU FEM MUAPs → interference EMG.
MUAPs are the 100 size-ordered per-MU waveforms saved by the WR-FCU pool run
(_results/mu_pool/spatial/mu_pool.npz), one skin electrode over FCU.

Run: python scripts/activation/demo_emg.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.activation import MotoneuronPool, compound_emg, rms_envelope, drive

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/activation"; OUT.mkdir(parents=True, exist_ok=True)
FS = 2048

d = np.load(ROOT / "_results/mu_pool/spatial/mu_pool.npz")
muap = d["muap_wave"]                                  # (100, 256), size-ordered, µV·1e-6·V
N = muap.shape[0]
pool = MotoneuronPool(n_mu=N, fs=FS)


def contraction(level, rise, hold, fall, seed=1, lead=0.4):
    E = drive.trapezoid(level, rise, hold, fall, fs=FS, lead_s=lead)
    E = drive.add_common_drive(E, sigma=0.015, cutoff_hz=1.5, fs=FS, seed=seed)
    sp = pool.spike_trains(E, seed=seed)
    emg = compound_emg(sp, muap, n_samples=len(E))
    return E, emg, sp


fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.22)

# (A) three sustained levels — weak / moderate / strong
levels = [0.10, 0.35, 0.65]
for k, lv in enumerate(levels):
    E, emg, sp = contraction(lv, 1.0, 2.5, 1.0)
    t = np.arange(len(emg)) / FS
    ax = fig.add_subplot(gs[k, 0])
    ax.plot(t, emg * 1e6, lw=0.4, color="tab:blue")
    ax.set_title(f"{int(lv*100)}% MVC sustained · {sum(len(s)>0 for s in sp)} MUs active",
                 fontsize=10)
    ax.set_ylabel("EMG (µV)"); ax.set_xlim(0, t[-1]); ax.grid(alpha=0.3)
    if k == 2:
        ax.set_xlabel("time (s)")

# (B) ramp — amplitude tracks recruitment
E, emg, sp = contraction(0.7, 6.0, 0.5, 0.5, lead=0.3)   # long rise = ramp
t = np.arange(len(emg)) / FS
axB = fig.add_subplot(gs[0:2, 1])
axB.plot(t, emg * 1e6, lw=0.3, color="0.5", label="EMG")
axB.plot(t, rms_envelope(emg, int(0.2 * FS)) * 1e6, "tab:red", lw=2, label="RMS (200 ms)")
axB2 = axB.twinx()
axB2.plot(t, E * 100, "k--", lw=1.5, alpha=0.6, label="drive")
axB2.set_ylabel("% MVC")
axB.set_title("Ramp contraction — EMG amplitude follows recruitment", fontsize=11)
axB.set_ylabel("EMG (µV)"); axB.set_xlabel("time (s)"); axB.legend(loc="upper left", fontsize=8)
axB.grid(alpha=0.3)

# (C) EMG–force relationship: RMS vs %MVC
mvc = np.linspace(0.05, 0.95, 12)
rms = []
for lv in mvc:
    E = drive.add_common_drive(drive.constant(lv, 2.0, FS), sigma=0.015, fs=FS, seed=2)
    emg = compound_emg(pool.spike_trains(E, seed=2), muap, n_samples=len(E))
    rms.append(rms_envelope(emg, int(0.25 * FS))[FS:].mean())
axC = fig.add_subplot(gs[2, 1])
axC.plot(mvc * 100, np.array(rms) * 1e6, "o-", color="tab:green")
axC.set_title("EMG amplitude vs contraction level", fontsize=11)
axC.set_xlabel("% MVC"); axC.set_ylabel("EMG RMS (µV)"); axC.grid(alpha=0.3)

fig.suptitle("Sample surface EMG — WR-FCU, drive → spikes → MUAPs → interference EMG", fontsize=13)
fig.savefig(OUT / "emg_contractions.png", dpi=130, bbox_inches="tight")
print(f"wrote {OUT/'emg_contractions.png'}")
print("RMS by %MVC:", {int(m*100): round(r*1e6, 1) for m, r in zip(mvc, rms)})
