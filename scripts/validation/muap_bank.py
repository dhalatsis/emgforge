"""The MUAP bank the chain-level checks (tier C, tier S) run on.

Default: the legacy caches under ``_results/mu_pool/`` (single-channel bank + the 5×5
grid tensor of ``scripts/activation/build_grid_tensor.py``). Set
``EMGFORGE_MUAP_BANK=/path/to/forearm_fcu_mu_pool.npz`` (the released dataset D2, see
``paper/datasets/``) to run them on the regular 10 mm grid with all 100 units instead —
the legacy grid's electrodes are snapped to mesh vertices (irregular spacing, its centre
~17 mm off the muscle), which is where the "0.6 µV / 33 ms" MUAP scale came from.

``load_bank`` returns amplitudes in volts either way (D2 stores µV).
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _metrics(t_ms, muaps):
    from emgforge.synthesis.api import _compute_metrics
    p2p = np.zeros(len(muaps)); dur = np.zeros(len(muaps))
    for k, m in enumerate(muaps):
        met = _compute_metrics(np.asarray(t_ms, float), np.asarray(m, float))
        p2p[k] = met.get("peak_to_peak", float(np.ptp(m))); dur[k] = met.get("duration_ms", 0.0)
    return p2p, dur


def load_bank(root: Path) -> SimpleNamespace:
    src = os.environ.get("EMGFORGE_MUAP_BANK")
    if src:
        d = np.load(src)
        muaps = d["muap_single"].astype(float) * 1e-6                 # µV → V
        W = d["muap_grid"].astype(float) * 1e-6
        t_ms = d["t_ms"].astype(float)
        M = int(round(np.sqrt(W.shape[1])))
        e = d["elec_xyz"].reshape(M, M, 3)                             # rows along the arm
        ied_z = float(np.median(np.linalg.norm(np.diff(e, axis=0), axis=2)))
        p2p, dur = _metrics(t_ms, muaps)
        return SimpleNamespace(name=f"D2 regular grid ({src})", muaps=muaps, p2p=p2p, duration_ms=dur,
                               W=W, M=M, sizes=d["mu_sizes"].astype(int), ied_z=ied_z, t_ms=t_ms,
                               fs=float(d["fs"]))
    pn = np.load(root / "_results/mu_pool/spatial/mu_pool.npz")
    gt = np.load(root / "_results/mu_pool/electrode_grid/muap_tensor_L8_M5.npz")
    elec = np.load(root / "_results/mu_pool/electrode_grid/_phigrid_M5_dt15_z0.30-0.70_N637.npz",
                   allow_pickle=True)["elec_xyz"]
    return SimpleNamespace(name="legacy grid (_results/mu_pool)", muaps=pn["muap_wave"], p2p=pn["p2p"],
                           duration_ms=pn["duration_ms"], W=gt["W"], M=int(gt["M"]), sizes=gt["sizes"],
                           ied_z=float(np.median(np.linalg.norm(np.diff(elec, axis=0), axis=2))),
                           t_ms=gt["t_ms"], fs=2048.0)


__all__ = ["load_bank"]
