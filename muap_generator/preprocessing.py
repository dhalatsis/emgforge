"""
Shared preprocessing utilities: smoothing, upsampling, windowing, fiber sampling.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy import interpolate, ndimage, signal


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_butterworth(
    phi: np.ndarray,
    cutoff_norm: float = 0.1,
    order: int = 4,
) -> np.ndarray:
    """Butterworth zero-phase lowpass filter.

    Parameters
    ----------
    phi : (N,) or (Nfib, Nz) array
    cutoff_norm : normalised cutoff frequency (0, 1)
    order : filter order
    """
    cutoff_norm = max(0.01, min(0.99, cutoff_norm))
    b, a = signal.butter(order, cutoff_norm, btype="low")

    def _filt(x: np.ndarray) -> np.ndarray:
        padlen = min(len(x) - 1, 3 * max(len(a), len(b)))
        return signal.filtfilt(b, a, x, padlen=padlen)

    if phi.ndim == 1:
        return _filt(phi)
    return np.stack([_filt(phi[u]) for u in range(phi.shape[0])], axis=0)


def smooth_savgol(
    phi: np.ndarray,
    window_length: int = 21,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay filter."""
    def _filt(x: np.ndarray) -> np.ndarray:
        n = len(x)
        wl = window_length
        if wl >= n:
            wl = n - 1 if n % 2 == 0 else n - 2
        if wl < polyorder + 2:
            return x.copy()
        if wl % 2 == 0:
            wl += 1
        return signal.savgol_filter(x, wl, polyorder)

    if phi.ndim == 1:
        return _filt(phi)
    return np.stack([_filt(phi[u]) for u in range(phi.shape[0])], axis=0)


def smooth_gaussian(phi: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian smoothing."""
    if phi.ndim == 1:
        return ndimage.gaussian_filter1d(phi, sigma=sigma)
    return np.stack(
        [ndimage.gaussian_filter1d(phi[u], sigma=sigma) for u in range(phi.shape[0])],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Edge tapering
# ---------------------------------------------------------------------------

def taper_edges(phi: np.ndarray, n_taper: int = 10) -> np.ndarray:
    """Cosine-taper the first and last *n_taper* samples to zero.

    Prevents Gibbs ringing when a lead field is truncated (non-zero at
    its boundaries) and subsequently zero-padded during resampling.

    Parameters
    ----------
    phi : (N,) or (Nfib, Nz) array
    n_taper : number of edge samples to taper (0 = no-op)
    """
    if n_taper <= 0:
        return phi
    out = phi.copy()
    taper = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))

    if out.ndim == 1:
        n_taper = min(n_taper, len(out) // 2)
        t = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        out[:n_taper] *= t
        out[-n_taper:] *= t[::-1]
    else:
        n_taper = min(n_taper, out.shape[1] // 2)
        t = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        out[:, :n_taper] *= t[np.newaxis, :]
        out[:, -n_taper:] *= t[::-1][np.newaxis, :]
    return out


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------

def upsample_cubic(y: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a 1-D signal using cubic spline interpolation."""
    if factor <= 1:
        return y
    x_old = np.arange(len(y))
    x_new = np.linspace(0, len(y) - 1, len(y) * factor)
    cs = interpolate.CubicSpline(x_old, y)
    return cs(x_new)


def upsample_matrix(phi_mat: np.ndarray, factor: int) -> np.ndarray:
    """Upsample all rows (fibers) of a matrix."""
    if factor <= 1:
        return phi_mat
    return np.stack(
        [upsample_cubic(phi_mat[u], factor) for u in range(phi_mat.shape[0])], axis=0
    )


# ---------------------------------------------------------------------------
# Windowing (numerical pipeline)
# ---------------------------------------------------------------------------

def create_fiber_windows(
    n_points: int,
    nmj_ratio: float,
    window_type: Literal["tukey", "boxcar", "hann", "none"] = "tukey",
    tukey_alpha: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create fibre-end windows for the two semi-fibres.

    Returns
    -------
    window_left, window_right : arrays of length *n_points*
    """
    n_left = int(n_points * nmj_ratio)
    n_right = n_points - n_left

    def _win(n: int) -> np.ndarray:
        n = max(n, 1)
        if window_type == "tukey":
            return signal.windows.tukey(n, alpha=tukey_alpha)
        if window_type == "hann":
            return np.hanning(n)
        return np.ones(n)  # boxcar / none

    wl, wr = _win(n_left), _win(n_right)

    window_left = np.zeros(n_points)
    window_right = np.zeros(n_points)
    window_left[:n_left] = wl
    window_right[n_left:] = wr
    return window_left, window_right


# ---------------------------------------------------------------------------
# Spatial resampling
# ---------------------------------------------------------------------------

def resample_centered_line(
    phi_z: np.ndarray,
    *,
    delta_s_mm: float,
    w_out: int,
    delta_s_out_mm: float,
    pad_mode: str = "edge",
) -> np.ndarray:
    """Resample a centred φ(z) line from *(w_in, delta_s_mm)* to *(w_out, delta_s_out_mm)*.

    pad_mode controls how the output is filled outside the input z-range:
      "edge" : pad with phi_z[0] / phi_z[-1] (default).
      "zero" : pad with 0. Historical default — creates a step discontinuity
               at the edges of the original phi when phi_z[0] or phi_z[-1] are
               non-zero, which radiates as a spectral-leakage spike through
               the downstream FFT-based SFAP pipeline.
    """
    phi_z = np.asarray(phi_z, dtype=float)
    w_in = phi_z.shape[0]
    z_in = (np.arange(w_in) - w_in // 2) * delta_s_mm
    z_out = (np.arange(w_out) - w_out // 2) * delta_s_out_mm
    if pad_mode == "edge":
        left, right = float(phi_z[0]), float(phi_z[-1])
    elif pad_mode == "zero":
        left, right = 0.0, 0.0
    else:
        raise ValueError(f"pad_mode must be 'edge' or 'zero', got {pad_mode!r}")
    return np.interp(z_out, z_in, phi_z, left=left, right=right)
