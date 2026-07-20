"""HD-EMG interference across the grid, from one shared drive.

Loads the precomputed (n_mu, n_elec, w) MUAP tensor, drives the pool through a
contraction, and sums onto every electrode → an electrode×time array. Shows the
interference grid, one unit's travelling wave down a column, and the single-
differential montage that removes the common mode.

Run (needs build_grid_tensor.py first): python scripts/activation/demo_hdemg.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.activation import MotoneuronPool, compound_emg_multi, single_diff, drive

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/activation"; OUT.mkdir(parents=True, exist_ok=True)
FS = 2048

d = np.load(ROOT / "_results/mu_pool/electrode_grid/muap_tensor_L8_M5.npz")
W, t_ms, M = d["W"], d["t_ms"], int(d["M"])          # W: (NMU, M*M, w)
NMU, E, w = W.shape

pool = MotoneuronPool(n_mu=100, fs=FS)
drv = drive.trapezoid(0.35, 2.0, 2.5, 2.0, fs=FS, lead_s=0.4)
drv = drive.add_common_drive(drv, sigma=0.015, fs=FS, seed=0)
spikes = pool.spike_trains(drv, seed=0)[:NMU]        # only the units we have MUAPs for
nact = sum(len(s) > 0 for s in spikes)
emg = compound_emg_multi(spikes, W, n_samples=len(drv))     # (E, T)
grid = emg.reshape(M, M, -1)                                # (M, M, T)
print(f"{nact}/{NMU} MUs active · HD-EMG {emg.shape}")

fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(M, 3, hspace=0.15, wspace=0.32, width_ratios=[2.0, 1.0, 1.0])

# (A) interference grid — a 1.2 s window at the plateau, shared scale
a, b = int(3.0 * FS), int(4.2 * FS)
tw = np.arange(a, b) / FS
ymax = np.abs(grid[:, :, a:b]).max() * 1e6
axg = fig.add_subplot(gs[:, 0])
for i in range(M):
    for j in range(M):
        axg.plot(tw, grid[i, j, a:b] * 1e6 * 0.9 / ymax * 0.5 + (M - 1 - i) + j * 0.0 + 0.5 * 0,
                 lw=0.5, color=plt.cm.viridis(j / M))
axg.set_yticks([]); axg.set_xlabel("time (s)")
axg.set_title(f"HD-EMG interference — {M}×{M} grid, {nact} active MUs @35% MVC", fontsize=11)
axg.set_ylabel("← along arm  ·  colour = across arm →")

# (B) one large unit's MUAP down a column → travelling wave (monopolar)
col = int(np.argmax(np.abs(W).sum(axis=(0, 2)).reshape(M, M).sum(0)))  # strongest column
mu = NMU - 1                                             # a large unit
axm = fig.add_subplot(gs[:, 1])
for i in range(M):
    axm.plot(t_ms, W[mu, i * M + col] * 1e6 + (M - 1 - i) * 8, lw=1.3, color="tab:blue")
axm.set_xlim(-5, 45); axm.set_yticks([]); axm.set_xlabel("t (ms)")
axm.set_title(f"MU{mu} down column {col}\n(monopolar — propagation)", fontsize=10)

# (C) single-differential of that column — common mode removed
sd = single_diff(W[mu].reshape(M, M, w), axis=0)        # (M-1, M, w)
axd = fig.add_subplot(gs[:, 2])
for i in range(M - 1):
    axd.plot(t_ms, sd[i, col] * 1e6 + (M - 2 - i) * 8, lw=1.3, color="tab:red")
axd.set_xlim(-5, 45); axd.set_yticks([]); axd.set_xlabel("t (ms)")
axd.set_title("single-differential\n(sharper, common mode gone)", fontsize=10)

fig.suptitle("emgforge.activation — HD-EMG multichannel EMG (one drive → electrode array)", fontsize=13)
fig.savefig(OUT / "hdemg_demo.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "hdemg_demo.png")
