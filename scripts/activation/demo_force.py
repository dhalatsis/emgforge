"""Force from the twitch model — twitch shapes, force tracking a contraction, and the
force–drive relation. Run: python scripts/activation/demo_force.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.activation import MotoneuronPool, TwitchPool, compound_emg, rms_envelope, drive

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/activation"; OUT.mkdir(parents=True, exist_ok=True)
FS = 2048
muap = np.load(ROOT / "_results/mu_pool/spatial/mu_pool.npz")["muap_wave"]
N = muap.shape[0]
pool = MotoneuronPool(n_mu=N, fs=FS)
tw = TwitchPool(pool, fs=FS)

fig = plt.figure(figsize=(14, 5.2))
gs = fig.add_gridspec(1, 3, wspace=0.28)

# (A) twitch shapes — small vs large MU
axA = fig.add_subplot(gs[0, 0])
for m, lab in [(2, "small MU"), (N // 2, "mid MU"), (N - 3, "large MU")]:
    t = np.arange(tw.L) / FS * 1000
    axA.plot(t, tw.tw[m], lw=1.8, label=f"{lab} (T={tw.T[m]/FS*1000:.0f}ms)")
axA.set_title("Motor-unit twitches", fontsize=11)
axA.set_xlabel("time (ms)"); axA.set_ylabel("twitch force (a.u.)"); axA.set_xlim(0, 220)
axA.legend(fontsize=8); axA.grid(alpha=0.3)

# (B) contraction: drive vs force vs EMG
E = drive.trapezoid(0.6, 2.5, 2.0, 2.5, fs=FS, lead_s=0.4)
E = drive.add_common_drive(E, sigma=0.015, fs=FS, seed=0)
sp = pool.spike_trains(E, seed=0)
force = tw.force(sp, n_samples=len(E))
emg = compound_emg(sp, muap, n_samples=len(E))
t = np.arange(len(E)) / FS
axB = fig.add_subplot(gs[0, 1])
axB.plot(t, E * 100, "k--", lw=1.4, alpha=0.6, label="drive")
axB.plot(t, force * 100, color="tab:red", lw=2.2, label="force")
axB.plot(t, rms_envelope(emg, int(0.2 * FS)) / rms_envelope(emg, int(0.2 * FS)).max() * 100 * 0.9,
         color="tab:blue", lw=1.2, alpha=0.7, label="EMG RMS (scaled)")
axB.set_title("Contraction — force tracks drive", fontsize=11)
axB.set_xlabel("time (s)"); axB.set_ylabel("% MVC"); axB.legend(fontsize=8); axB.grid(alpha=0.3)

# (C) force–drive relation
lv = np.linspace(0.05, 1.0, 12); fout = []
for x in lv:
    Ex = drive.constant(x, 3.0, FS)
    fout.append(tw.force(pool.spike_trains(Ex, seed=2), n_samples=len(Ex))[FS:].mean())
axC = fig.add_subplot(gs[0, 2])
axC.plot(lv * 100, np.array(fout) * 100, "o-", color="tab:red")
axC.plot([0, 100], [0, 100], "k:", lw=1, alpha=0.4)
axC.set_title("Force vs excitation", fontsize=11)
axC.set_xlabel("excitation (% max drive)"); axC.set_ylabel("force (% MVC)"); axC.grid(alpha=0.3)

fig.suptitle("emgforge.activation — twitch model → force (Fuglevand)", fontsize=13)
fig.savefig(OUT / "force_demo.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "force_demo.png")
