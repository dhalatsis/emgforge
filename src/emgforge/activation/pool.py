"""Motoneuron pool — neural drive → motor-unit spike trains.

A clean-room reimplementation of the phenomenological pool used by NeuroMotion
(Ma et al., PLOS Comp Biol 2024), which itself follows Fuglevand (1993) / the iEMG
simulator parametrisation. **No code is vendored** — this is written from the model
equations. It is the high-level (phenomenological) tier: recruitment thresholds +
onion-skin rate coding + a Gaussian-ISI renewal process. No membrane biophysics.

Given a scalar excitation ``E(t) ∈ [0, 1]`` (fraction of max drive / %MVC):

  * **Recruitment** — each MU ``i`` has an excitation threshold ``RTE_i`` spread
    exponentially over ``[·, rm]`` with range ``rr`` (small MUs recruited first).
  * **Rate coding** — once ``E ≥ RTE_i`` the discharge rate is
    ``FR_i = min(PFR_i, MFR_i + (E − RTE_i)·slope_i)``, with the peak rate ``PFR``
    *decreasing* from first to last MU (the "onion-skin").
  * **Spikes** — an inhomogeneous renewal process: inter-spike intervals drawn as
    ``ISI = fs/FR · (1 + 𝒩(0, 1/6))`` (discharge CV ≈ 1/6), re-evaluated at each firing.

The MU index ``i`` (0 = smallest/first-recruited) lines up with a size-sorted
``sample_henneman_pool`` bed, so the spike trains drive the FEM-derived MUAPs directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# NeuroMotion / iEMG-simulator defaults (mn_params.mn_default_settings).
DEFAULTS = dict(
    rr=50.0,     # recruitment range: RTE_last / RTE_first
    rm=0.75,     # excitation at which the last MU is recruited (fraction)
    pfr1=40.0,   # peak firing rate of the first MU (Hz)
    pfrd=10.0,   # peak-rate drop from first to last MU (Hz) → onion-skin
    mfr1=10.0,   # minimum firing rate of the first MU (Hz)
    mfrd=5.0,    # minimum-rate drop from first to last MU (Hz)
    frs1=50.0,   # drive→rate slope of the first MU (Hz per unit E)
    frsd=20.0,   # slope drop from first to last MU (Hz)
)
ISI_CV = 1.0 / 6.0     # NeuroMotion hardcodes std = ipi/6 in generate_spike_trains


@dataclass
class MotoneuronPool:
    """A phenomenological motoneuron pool: excitation → per-MU spike trains."""

    n_mu: int
    fs: float = 2048.0
    rr: float = DEFAULTS["rr"]
    rm: float = DEFAULTS["rm"]
    pfr1: float = DEFAULTS["pfr1"]
    pfrd: float = DEFAULTS["pfrd"]
    mfr1: float = DEFAULTS["mfr1"]
    mfrd: float = DEFAULTS["mfrd"]
    frs1: float = DEFAULTS["frs1"]
    frsd: float = DEFAULTS["frsd"]
    recruit: str = "ls2n"      # "ls2n" (NeuroMotion default) or "fuglevand"

    def __post_init__(self) -> None:
        N = int(self.n_mu)
        if N < 1:
            raise ValueError("n_mu must be >= 1")
        # ---- recruitment threshold excitation RTE_i ----
        i = np.arange(N)
        if self.recruit == "ls2n":
            rt = (self.rm / self.rr) * np.exp(i * np.log(self.rr) / (N - 1)) if N > 1 else np.array([self.rm])
            rte = (self.rm / self.rr) * (np.exp(i * np.log(self.rr + 1.0) / N) - 1.0)
            rte = rte * (rt.max() / rte.max()) if rte.max() > 0 else rt
        elif self.recruit == "fuglevand":
            rte = np.exp(np.arange(1, N + 1) * np.log(self.rr) / N)
            rte = rte / rte.max() * self.rm          # scale so the last MU recruits at rm
        else:
            raise ValueError(f"recruit must be 'ls2n' or 'fuglevand', got {self.recruit!r}")
        self.rte = rte                                # (N,)
        # ---- rate-coding parameters (exp mode: linear in the threshold) ----
        f = rte / rte.max()
        self.min_fr = self.mfr1 - self.mfrd * f       # (N,) min discharge rate
        self.peak_fr = self.pfr1 - self.pfrd * f      # (N,) peak (onion-skin: decreasing)
        self.slope = self.frs1 - self.frsd * f        # (N,) drive→rate slope

    # ------------------------------------------------------------------
    def firing_rate(self, E: np.ndarray) -> np.ndarray:
        """Instantaneous discharge rate ``(N, T)`` for excitation ``E`` ``(T,)`` — 0 below threshold."""
        E = np.atleast_1d(np.asarray(E, dtype=float))
        rate = self.min_fr[:, None] + (E[None, :] - self.rte[:, None]) * self.slope[:, None]
        rate = np.minimum(self.peak_fr[:, None], rate)
        rate[E[None, :] < self.rte[:, None]] = 0.0
        return rate

    def spike_trains(self, E: np.ndarray, seed: int = 0) -> list[np.ndarray]:
        """Per-MU spike sample-indices for excitation ``E`` ``(T,)``.

        Inhomogeneous renewal process: at each firing the next ISI is
        ``fs/FR·(1 + 𝒩(0, 1/6))``; when ``E`` drops below the MU's threshold the MU
        de-recruits (its pending firing is cancelled) and re-recruits when ``E`` rises.
        """
        E = np.atleast_1d(np.asarray(E, dtype=float))
        T = E.shape[-1]
        fr = self.firing_rate(E)
        rng = np.random.default_rng(int(seed))
        trains: list[np.ndarray] = []
        for m in range(self.n_mu):
            thr = self.rte[m]
            spikes: list[int] = []
            next_fire = -1
            for t in range(T):
                if E[t] > thr and fr[m, t] > 0:
                    if next_fire < 0:                      # (re)recruited → schedule first
                        ipi = self.fs / fr[m, t] * (1.0 + rng.standard_normal() * ISI_CV)
                        next_fire = t + int(ipi)
                    elif t == next_fire:                   # fire + reschedule
                        spikes.append(t)
                        ipi = self.fs / fr[m, t] * (1.0 + rng.standard_normal() * ISI_CV)
                        next_fire = t + int(ipi)
                else:
                    next_fire = -1                         # de-recruited
            trains.append(np.asarray(spikes, dtype=int))
        return trains

    def n_active(self, E: np.ndarray) -> int:
        """How many MUs are recruited at the peak of ``E``."""
        return int(np.sum(self.rte < np.max(E)))


__all__ = ["MotoneuronPool", "DEFAULTS", "ISI_CV"]
