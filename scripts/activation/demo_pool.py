"""Demo: a trapezoid excitation (with common drive) → 100-MU spike trains, showing
recruitment/de-recruitment and onion-skin rate coding — the standard motoneuron-pool
picture (NeuroMotion/Fuglevand phenomenology), reproduced from emgforge.activation.

Run: python scripts/activation/demo_pool.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.activation import MotoneuronPool, drive

OUT = Path(__file__).resolve().parents[2] / "_results/activation"
OUT.mkdir(parents=True, exist_ok=True)
FS, N = 2048, 100

E = drive.trapezoid(peak=0.6, rise_s=3, hold_s=2, fall_s=3, fs=FS, lead_s=0.5)
E = drive.add_common_drive(E, sigma=0.02, cutoff_hz=1.5, fs=FS, seed=0)
t = np.arange(E.shape[-1]) / FS

pool = MotoneuronPool(n_mu=N, fs=FS)
spikes = pool.spike_trains(E, seed=0)
n_active = sum(len(s) > 0 for s in spikes)
print(f"{n_active}/{N} MUs recruited at peak drive {E.max():.2f}")

fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                       gridspec_kw={"height_ratios": [1, 3, 2]})

# (A) drive
ax[0].plot(t, E, "k", lw=1.5)
ax[0].set_ylabel("excitation E(t)"); ax[0].set_title("Neural drive (trapezoid + common drive)")
ax[0].grid(alpha=0.3)

# (B) spike raster — MUs recruit in order as E rises, de-recruit as it falls
for m, sp in enumerate(spikes):
    if len(sp):
        ax[1].plot(sp / FS, np.full(len(sp), m), "|", color=plt.cm.viridis(m / N),
                   ms=3, mew=0.6)
ax[1].set_ylabel("MU index (small → large)")
ax[1].set_title("Spike raster — recruitment/de-recruitment"); ax[1].set_ylim(-1, N)

# (C) smoothed instantaneous discharge rate (onion-skin: later MUs fire slower)
for m in range(0, N, 6):
    sp = spikes[m]
    if len(sp) > 3:
        idr = FS / np.diff(sp)                       # instantaneous rate (Hz)
        tm = sp[1:] / FS
        ax[2].plot(tm, idr, color=plt.cm.viridis(m / N), lw=1.2, alpha=0.8)
ax[2].set_ylabel("discharge rate (Hz)"); ax[2].set_xlabel("time (s)")
ax[2].set_title("Onion-skin rate coding (earlier MUs discharge faster)")
ax[2].grid(alpha=0.3); ax[2].set_ylim(0, pool.pfr1 + 5)

fig.suptitle(f"emgforge.activation — {N}-MU motoneuron pool (NeuroMotion/Fuglevand model)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT / "pool_demo.png", dpi=130, bbox_inches="tight")
print(f"wrote {OUT/'pool_demo.png'}")
