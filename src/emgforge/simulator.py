"""Simulator — the one-call facade over the whole chain.

Wires the motoneuron pool + twitch/force + the per-MU MUAPs into a single object so a
contraction becomes one line::

    sim = Simulator.from_tensor("…/muap_tensor_L8_M5.npz")     # per-electrode MUAPs
    rec = sim.run(drive.trapezoid(0.4, 2, 2, 2, fs=2048))      # → EMG + force + spikes

The MUAPs come from the FEM + ``emgforge.synthesis`` pipeline (single-channel from a pool
run, or the multichannel grid tensor). The Simulator owns the neural side and the summation;
it does not re-solve the FEM — build the tensor once (``scripts/activation/build_grid_tensor``)
and hand it in, or point ``from_mri`` at the standard cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emgforge.activation import (
    MotoneuronPool, TwitchPool, compound_emg, compound_emg_multi,
)


@dataclass
class Recording:
    """The output of one :meth:`Simulator.run` — a labelled synthetic recording."""

    t: np.ndarray                 # (T,) seconds
    drive: np.ndarray             # (T,) excitation
    emg: np.ndarray               # (T,) single-channel or (E, T) multichannel
    spikes: list                  # per-MU spike sample indices (ground truth)
    force: np.ndarray | None      # (T,) %MVC, or None
    fs: float
    grid: tuple | None            # (M, M) electrode layout, or None

    @property
    def multichannel(self) -> bool:
        return self.emg.ndim == 2

    @property
    def n_active(self) -> int:
        return int(sum(len(s) > 0 for s in self.spikes))

    def grid_view(self) -> np.ndarray:
        """Reshape a multichannel EMG to ``(M, M, T)``."""
        if self.grid is None:
            raise ValueError("recording has no grid layout")
        return self.emg.reshape(*self.grid, -1)


class Simulator:
    """Neural pool + twitch + MUAPs → EMG (+force) from a drive."""

    def __init__(self, muaps: np.ndarray, fs: float = 2048.0, grid: tuple | None = None,
                 twitch: bool = True, pool_kwargs: dict | None = None):
        self.muaps = np.asarray(muaps, dtype=float)        # (N, w) or (N, E, w)
        self.multichannel = self.muaps.ndim == 3
        self.n_mu = self.muaps.shape[0]
        self.fs = float(fs)
        self.grid = grid
        self.pool = MotoneuronPool(n_mu=self.n_mu, fs=self.fs, **(pool_kwargs or {}))
        self.twitch = TwitchPool(self.pool, fs=self.fs) if twitch else None

    def run(self, drive: np.ndarray, seed: int = 0) -> Recording:
        """Drive the pool and synthesise the recording."""
        drive = np.asarray(drive, dtype=float)
        T = drive.shape[-1]
        spikes = self.pool.spike_trains(drive, seed=seed)
        if self.multichannel:
            emg = compound_emg_multi(spikes, self.muaps, n_samples=T)
        else:
            emg = compound_emg(spikes, self.muaps, n_samples=T)
        force = self.twitch.force(spikes, n_samples=T) if self.twitch else None
        return Recording(np.arange(T) / self.fs, drive, emg, spikes, force, self.fs, self.grid)

    # ---- constructors from saved MUAPs ----
    @classmethod
    def from_tensor(cls, npz_path, fs: float = 2048.0, **kw) -> "Simulator":
        """Multichannel Simulator from a saved ``(N, E, w)`` grid MUAP tensor."""
        d = np.load(npz_path)
        M = int(d["M"]) if "M" in d.files else None
        return cls(d["W"], fs=fs, grid=(M, M) if M else None, **kw)

    @classmethod
    def from_single(cls, npz_path, key: str = "muap_wave", fs: float = 2048.0, **kw) -> "Simulator":
        """Single-channel Simulator from a saved pool run (``muap_wave`` = ``(N, w)``)."""
        return cls(np.load(npz_path)[key], fs=fs, **kw)

    @classmethod
    def from_mri(cls, muscle: int = 8, m: int = 5, root: str | Path | None = None,
                 fs: float = 2048.0, **kw) -> "Simulator":
        """Load the standard grid tensor for a muscle+grid (built by build_grid_tensor).

        Convenience for the anatomical path; the tensor is expensive to compute, so it is
        produced once by ``scripts/activation/build_grid_tensor.py`` and cached.
        """
        root = Path(root) if root else Path(__file__).resolve().parents[2]
        p = root / f"_results/mu_pool/electrode_grid/muap_tensor_L{muscle}_M{m}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"no MUAP tensor at {p} — run scripts/activation/build_grid_tensor.py first")
        return cls.from_tensor(p, fs=fs, **kw)


__all__ = ["Simulator", "Recording"]
