"""Spike trains × MUAPs → interference EMG.

The last stage of the chain: each motor unit's spike train (from ``MotoneuronPool``)
is convolved with that unit's MUAP (from the FEM + ``emgforge.synthesis`` pipeline) and
summed. This is the physiological interference pattern — the same operation as the old
``simulate_compound_emg``, but driven by proper recruitment/rate-coded spikes instead of a
fixed-rate Poisson, and reading real per-MU MUAPs instead of a peak-aligned template.

MUAP row ``i`` must correspond to spike train ``i`` (both size/recruitment ordered), and
both must share the sampling rate the MUAPs were generated at (2048 Hz for the pool).
"""

from __future__ import annotations

import numpy as np


def compound_emg(spike_trains, muaps: np.ndarray, n_samples: int | None = None) -> np.ndarray:
    """Sum per-MU MUAP trains into one interference-EMG channel.

    Parameters
    ----------
    spike_trains : list of (k_i,) int arrays — spike sample indices per MU.
    muaps : (N, w) array — one MUAP waveform per MU (same order as ``spike_trains``).
    n_samples : output length; defaults to the last spike + one MUAP window.
    """
    muaps = np.asarray(muaps, dtype=float)
    N, w = muaps.shape
    if len(spike_trains) != N:
        raise ValueError(f"{len(spike_trains)} spike trains vs {N} MUAPs")
    last = max((int(s.max()) for s in spike_trains if len(s)), default=0)
    total = (last + w) if n_samples is None else int(n_samples) + w
    emg = np.zeros(total)
    for i, sp in enumerate(spike_trains):
        m = muaps[i]
        for t in sp:
            emg[t:t + w] += m
    return emg[:(last + w if n_samples is None else int(n_samples))]


def compound_emg_multi(spike_trains, muaps: np.ndarray, n_samples: int | None = None) -> np.ndarray:
    """Multichannel interference EMG — one shared drive summed onto every electrode.

    Parameters
    ----------
    spike_trains : list of (k_i,) int arrays — the same trains drive all channels.
    muaps : (N, E, w) — each MU's MUAP at each of E electrodes (order matches spike_trains).
    Returns ``(E, T)`` — an electrode × time HD-EMG array.
    """
    muaps = np.asarray(muaps, dtype=float)
    N, E, w = muaps.shape
    if len(spike_trains) != N:
        raise ValueError(f"{len(spike_trains)} spike trains vs {N} MUAPs")
    last = max((int(s.max()) for s in spike_trains if len(s)), default=0)
    total = (last + w) if n_samples is None else int(n_samples) + w
    emg = np.zeros((E, total))
    for i, sp in enumerate(spike_trains):
        m = muaps[i]                                   # (E, w)
        for t in sp:
            emg[:, t:t + w] += m
    return emg[:, :(last + w if n_samples is None else int(n_samples))]


def single_diff(grid: np.ndarray, axis: int = 0) -> np.ndarray:
    """Single-differential montage — subtract adjacent channels along ``axis``.

    ``grid`` is an ``(M, M, T)`` electrode array; differencing along the fibre axis
    removes the common far-field component and sharpens the travelling wave.
    """
    return np.diff(np.asarray(grid, dtype=float), axis=axis)


def double_diff(grid: np.ndarray, axis: int = 0) -> np.ndarray:
    """Double-differential montage (Laplacian along ``axis``) — even tighter localisation."""
    return np.diff(np.asarray(grid, dtype=float), n=2, axis=axis)


def rms_envelope(emg: np.ndarray, win: int) -> np.ndarray:
    """Moving-RMS envelope of an EMG channel (window in samples)."""
    win = max(int(win), 1)
    p = np.convolve(emg ** 2, np.ones(win) / win, mode="same")
    return np.sqrt(p)


__all__ = ["compound_emg", "compound_emg_multi", "single_diff", "double_diff", "rms_envelope"]
