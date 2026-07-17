"""Excitation / neural-drive profiles E(t) ∈ [0, 1] to feed a ``MotoneuronPool``.

Deterministic force targets (constant / ramp / trapezoid / sinusoid) plus
``add_common_drive`` — a band-limited fluctuation *shared* across MUs. The common
drive is the physiologically important part: because every MU sees the same E(t)
wobble, their spike trains become correlated (the basis of decomposition and of
force steadiness), which the plain deterministic profile alone does not produce.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _t(duration_s: float, fs: float) -> np.ndarray:
    return np.arange(int(round(duration_s * fs))) / fs


def constant(level: float, duration_s: float, fs: float = 2048.0) -> np.ndarray:
    return np.full(int(round(duration_s * fs)), float(level))


def ramp(peak: float, duration_s: float, fs: float = 2048.0,
         rise_frac: float = 1.0) -> np.ndarray:
    """Linear ramp 0→``peak`` over ``rise_frac`` of the record, then hold."""
    n = int(round(duration_s * fs))
    nr = max(int(round(rise_frac * n)), 1)
    up = np.linspace(0.0, peak, nr)
    return np.concatenate([up, np.full(n - nr, peak)])[:n]


def trapezoid(peak: float, rise_s: float, hold_s: float, fall_s: float,
              fs: float = 2048.0, lead_s: float = 0.0) -> np.ndarray:
    """0 → up-ramp → plateau at ``peak`` → down-ramp → 0."""
    segs = [np.zeros(int(round(lead_s * fs))),
            np.linspace(0.0, peak, max(int(round(rise_s * fs)), 1)),
            np.full(int(round(hold_s * fs)), peak),
            np.linspace(peak, 0.0, max(int(round(fall_s * fs)), 1))]
    return np.concatenate(segs)


def sinusoid(mean: float, amp: float, freq_hz: float, duration_s: float,
             fs: float = 2048.0) -> np.ndarray:
    """Sinusoidal drive ``mean + amp·sin(2π f t)``, clipped to [0, 1]."""
    t = _t(duration_s, fs)
    return np.clip(mean + amp * np.sin(2 * np.pi * freq_hz * t), 0.0, 1.0)


def add_common_drive(E: np.ndarray, sigma: float = 0.02, cutoff_hz: float = 2.0,
                     fs: float = 2048.0, seed: int = 0) -> np.ndarray:
    """Add a shared low-pass fluctuation to ``E`` (correlated common drive).

    White noise → 2nd-order Butterworth low-pass at ``cutoff_hz`` → scaled to unit std
    → ``sigma``. The same realisation is added to the scalar E(t), so every MU inherits
    it. Clipped to [0, 1].
    """
    E = np.asarray(E, dtype=float)
    rng = np.random.default_rng(int(seed))
    w = rng.standard_normal(E.shape[-1])
    b, a = signal.butter(2, cutoff_hz / (fs / 2), btype="low")
    lp = signal.filtfilt(b, a, w)
    lp = lp / (lp.std() + 1e-12) * sigma
    return np.clip(E + lp, 0.0, 1.0)


__all__ = ["constant", "ramp", "trapezoid", "sinusoid", "add_common_drive"]
