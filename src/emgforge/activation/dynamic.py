"""Dynamic (non-stationary) EMG — a movement modulates the drive and the MUAPs.

The isometric chain gives a stationary MUAP per unit. During a movement the joint angle
changes the muscle length, which shifts the fibres relative to the electrode (amplitude)
and rescales the action-potential timing (a mild time-warp). This is a **phenomenological**
dynamic layer — angle → drive + MUAP modulation — not a musculoskeletal model; full
kinematics (OpenSim-style) is a later phase.

``dynamic_compound_emg`` places each spike's MUAP scaled by the amplitude track and warped
by the warp track at that instant, so the same unit's contribution changes across the
movement — a non-stationary interference signal.
"""

from __future__ import annotations

import numpy as np


def angle_track(duration_s: float, fs: float = 2048.0, cycles: float = 2.0,
                phase: float = -np.pi / 2) -> np.ndarray:
    """Normalised joint angle ``θ(t) ∈ [0, 1]`` (0 = extended, 1 = flexed)."""
    t = np.arange(int(round(duration_s * fs))) / fs
    return 0.5 * (1.0 + np.sin(2 * np.pi * cycles / duration_s * t + phase))


def drive_from_angle(angle: np.ndarray, base: float = 0.08, gain: float = 0.42) -> np.ndarray:
    """Agonist excitation from the angle — active while the muscle shortens (flexion)."""
    return np.clip(base + gain * np.asarray(angle, float), 0.0, 1.0)


def amp_from_angle(angle: np.ndarray, gain: float = 0.7) -> np.ndarray:
    """MUAP amplitude vs angle — shortening brings fibres toward the electrode → larger."""
    return 1.0 + gain * (np.asarray(angle, float) - 0.5)


def warp_from_angle(angle: np.ndarray, gain: float = 0.18) -> np.ndarray:
    """MUAP time-warp vs angle — the apparent AP duration shifts as geometry/CV change."""
    return 1.0 + gain * (np.asarray(angle, float) - 0.5)


def _warp_bank(muaps: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Precompute time-warped MUAPs at each factor (>1 stretches within the same window)."""
    w = muaps.shape[-1]
    idx = np.arange(w)
    bank = np.zeros((len(factors),) + muaps.shape)
    for k, fac in enumerate(factors):
        src = idx * max(fac, 1e-6)          # value m[j] sits at j·fac → fac>1 stretches
        bank[k] = np.apply_along_axis(lambda m: np.interp(idx, src, m, right=0.0), -1, muaps)
    return bank


def dynamic_compound_emg(spike_trains, muaps: np.ndarray, amp_track: np.ndarray,
                         warp_track: np.ndarray | None = None, warp_levels: int = 7,
                         n_samples: int | None = None) -> np.ndarray:
    """Interference EMG with per-spike amplitude + time-warp modulation.

    ``muaps`` is ``(N, w)`` (single channel) or ``(N, E, w)`` (grid). ``amp_track`` and
    ``warp_track`` are length-``T`` and evaluated at each spike time.
    """
    muaps = np.asarray(muaps, dtype=float)
    w = muaps.shape[-1]
    multi = muaps.ndim == 3
    E = muaps.shape[1] if multi else 1
    amp_track = np.asarray(amp_track, float)
    T = amp_track.shape[-1]
    last = max((int(s.max()) for s in spike_trains if len(s)), default=0)
    total = (last + w) if n_samples is None else int(n_samples) + w
    emg = np.zeros((E, total)) if multi else np.zeros(total)

    factors = None
    if warp_track is not None:
        wt = np.asarray(warp_track, float)
        lo, hi = float(wt.min()), float(wt.max())
        factors = np.linspace(lo, hi, warp_levels) if hi > lo else np.array([lo])
        bank = _warp_bank(muaps, factors)                 # (K, N, [E,] w)

    for i, sp in enumerate(spike_trains):
        for t in sp:
            ti = min(int(t), T - 1)
            a = amp_track[ti]
            if factors is not None:
                k = int(np.argmin(np.abs(factors - wt[ti])))
                m = bank[k, i] * a
            else:
                m = muaps[i] * a
            if multi:
                emg[:, t:t + w] += m
            else:
                emg[t:t + w] += m

    end = (last + w) if n_samples is None else int(n_samples)
    return emg[..., :end]


__all__ = ["angle_track", "drive_from_angle", "amp_from_angle", "warp_from_angle",
           "dynamic_compound_emg"]
