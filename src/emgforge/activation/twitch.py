"""Motor-unit twitch → force (Fuglevand 1993), clean-room.

The mechanical counterpart of the spike trains: each discharge triggers a twitch, and
the twitches sum — sub-linearly at low rates (isolated) and fusing toward tetanus at high
rates. Same phenomenological family as ``MotoneuronPool``; pairs with it to give EMG *and*
force from one drive. No vendored code.

Per motor unit ``i`` (0 = smallest, first-recruited):
  * peak twitch force ``P_i = exp(ln(rp)·i/N)`` — spans a force fold of ``rp``;
  * contraction time ``T_i = T_max·(1/P_i)^{1/c}`` — larger units twitch *bigger and faster*;
  * twitch shape ``f(t) = (P/T)·t·e^{1−t/T}`` (critically-damped, peaks at ``t=T`` with height ``P``).
Fusion: the twitch amplitude is scaled by a rate-dependent gain ``S/ifr`` with
``S = 1−e^{−2·ifr³}``, ``ifr = T/ISI`` — near 1 for well-separated spikes, rising as they fuse.
Force is normalised to the mean force at 100% excitation, so the output is in %MVC.
"""

from __future__ import annotations

import numpy as np


class TwitchPool:
    """Twitch/force layer for a :class:`MotoneuronPool`."""

    def __init__(self, pool, fs: float | None = None, rp: float = 100.0,
                 tmax_ms: float = 90.0, tr: float = 3.0):
        self.pool = pool
        self.fs = float(fs if fs is not None else pool.fs)
        N = pool.n_mu
        i = np.arange(1, N + 1)
        self.P = np.exp(np.log(rp) / N * i)                         # peak twitch force
        tmax = tmax_ms / 1000.0 * self.fs
        tcoeff = np.log(rp) / np.log(tr)
        self.T = tmax * (1.0 / self.P) ** (1.0 / tcoeff)           # contraction time (samples)
        self.L = int(np.ceil(5 * self.T.max()))                    # twitch support
        tl = np.arange(self.L)
        self.tw = (self.P[:, None] / self.T[:, None]) * tl[None, :] * np.exp(1.0 - tl[None, :] / self.T[:, None])
        self._fmax = 1.0
        self._fmax = self._mvc()                                   # normalise to %MVC

    def _gain(self, ifr: np.ndarray) -> np.ndarray:
        """Rate-dependent fusion gain; ~1 for isolated twitches, >1 as they fuse."""
        S = 1.0 - np.exp(-2.0 * ifr ** 3)
        base = (1.0 - np.exp(-2.0 * 0.4 ** 3)) / 0.4
        return np.where(ifr > 0.4, (S / np.maximum(ifr, 1e-9)) / base, 1.0)

    def force(self, spike_trains, n_samples: int | None = None) -> np.ndarray:
        """Sum twitches (rate-scaled) over all units → force trace, in %MVC."""
        last = max((int(s.max()) for s in spike_trains if len(s)), default=0)
        total = (last + self.L) if n_samples is None else int(n_samples) + self.L
        f = np.zeros(total)
        for m, sp in enumerate(spike_trains):
            prev = None
            tw = self.tw[m]
            for t in sp:
                if prev is None:
                    g = 1.0
                else:
                    g = float(self._gain(np.array([self.T[m] / max(t - prev, 1)]))[0])
                f[t:t + self.L] += g * tw
                prev = t
        out = f[:(last + self.L if n_samples is None else int(n_samples))]
        return out / self._fmax

    def _mvc(self) -> float:
        """Mean steady force at 100% excitation — the MVC used to normalise."""
        fs = int(self.fs)
        E = np.ones(4 * fs)
        sp = self.pool.spike_trains(E, seed=0)
        fmvc = self.force(sp, n_samples=len(E))
        m = float(fmvc[fs:].mean())
        return m if m > 0 else 1.0


__all__ = ["TwitchPool"]
