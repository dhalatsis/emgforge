"""The Simulator in one call: a muscle + drive → a labelled HD-EMG recording.

    sim = Simulator.from_mri(muscle=8)      # FCU grid MUAPs (cached)
    rec = sim.run(drive)                    # → EMG + force + spikes

Run: python scripts/activation/demo_simulator.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.simulator import Simulator
from emgforge.activation import drive, rms_envelope

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/activation"; OUT.mkdir(parents=True, exist_ok=True)
FS = 2048

sim = Simulator.from_mri(muscle=8, m=5, root=ROOT)          # loads the cached FCU tensor
E = drive.trapezoid(0.35, 2.0, 2.5, 2.0, fs=FS, lead_s=0.4)
E = drive.add_common_drive(E, sigma=0.015, fs=FS, seed=0)
rec = sim.run(E, seed=0)                                    # one call → everything
grid = rec.grid_view()                                     # (M, M, T)
M = rec.grid[0]
print(f"Simulator: {sim.n_mu} MUs, {'multi' if rec.multichannel else 'single'}, "
      f"{rec.n_active} active · emg {rec.emg.shape} · force {rec.force.shape}")

fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.2)

# (A) drive + force
axA = fig.add_subplot(gs[0, 0])
axA.plot(rec.t, rec.drive * 100, "k--", lw=1.3, alpha=0.6, label="drive")
axA.plot(rec.t, rec.force * 100, "tab:red", lw=2, label="force")
axA.set_ylabel("% MVC"); axA.set_title("Drive → force"); axA.legend(fontsize=8); axA.grid(alpha=0.3)
axA.set_xlim(0, rec.t[-1])

# (B) spike raster — the ground-truth firings
axB = fig.add_subplot(gs[1, 0], sharex=axA)
for m, sp in enumerate(rec.spikes):
    if len(sp):
        axB.plot(sp / FS, np.full(len(sp), m), "|", color=plt.cm.viridis(m / sim.n_mu), ms=3, mew=.6)
axB.set_ylabel("MU index"); axB.set_title("Motor-unit spike trains (ground truth)")
axB.set_ylim(-1, sim.n_mu)

# (C) one EMG channel + RMS
ch = np.unravel_index(np.argmax(np.abs(grid).max(2)), grid.shape[:2])
axC = fig.add_subplot(gs[2, 0], sharex=axA)
axC.plot(rec.t, grid[ch] * 1e6, lw=0.4, color="0.5")
axC.plot(rec.t, rms_envelope(grid[ch], int(0.2 * FS)) * 1e6, "tab:blue", lw=1.8, label="RMS")
axC.set_ylabel("EMG (µV)"); axC.set_xlabel("time (s)")
axC.set_title(f"EMG channel {ch} + RMS"); axC.legend(fontsize=8); axC.grid(alpha=0.3)

# (D) grid RMS heatmap — the footprint
axD = fig.add_subplot(gs[:, 1])
rmsmap = np.sqrt((grid ** 2).mean(2)) * 1e6
im = axD.imshow(rmsmap, cmap="magma", aspect="auto")
axD.set_title("channel RMS (µV)\nover the array"); axD.set_xlabel("across arm"); axD.set_ylabel("along arm")
axD.set_xticks(range(M)); axD.set_yticks(range(M))
fig.colorbar(im, ax=axD, shrink=0.6)

fig.suptitle("emgforge.Simulator — one call: muscle + drive → HD-EMG + force + spikes", fontsize=13)
fig.savefig(OUT / "simulator_demo.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "simulator_demo.png")
