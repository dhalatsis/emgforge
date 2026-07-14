"""
Shared preprocessing utilities: smoothing, upsampling, windowing, fiber sampling.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy import interpolate, optimize, signal


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

def _tendon_ramp(n: int, alpha: float) -> np.ndarray:
    """A half-window that is flat at the NMJ end and cosine-tapered at the tendon end."""
    n = max(n, 1)
    k = max(int(alpha * n), 1)
    r = np.ones(n)
    r[:k] = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    return r


def create_fiber_windows(
    n_points: int,
    nmj_ratio: float,
    window_type: Literal["tukey", "boxcar", "hann", "one_sided"] = "tukey",
    tukey_alpha: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create fibre-end windows for the two semi-fibres.

    ``one_sided`` tapers ONLY the outer (tendon) end of each semi-fibre and stays
    flat through the NMJ. A symmetric ``tukey`` tapers both ends of each half and
    so notches the junction, where the travelling wave is born; the one-sided form
    softens the end-of-fibre effect without that artifact. It is the settled choice
    for the MRI/PM pipeline (alpha = 0.25).

    Returns
    -------
    window_left, window_right : arrays of length *n_points*
    """
    n_left = int(n_points * nmj_ratio)
    n_right = n_points - n_left

    if window_type == "one_sided":
        wl = _tendon_ramp(n_left, tukey_alpha)
        wr = _tendon_ramp(n_right, tukey_alpha)[::-1]
    else:
        def _win(n: int) -> np.ndarray:
            n = max(n, 1)
            if window_type == "tukey":
                return signal.windows.tukey(n, alpha=tukey_alpha)
            if window_type == "hann":
                return np.hanning(n)
            if window_type == "boxcar":
                return np.ones(n)  # hard rectangular tendon cut
            raise ValueError(
                f"unknown window_type {window_type!r}; "
                "expected 'tukey', 'boxcar', 'hann', or 'one_sided'")

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


# ---------------------------------------------------------------------------
# Monopole denoising
# ---------------------------------------------------------------------------

def _n_monopoles(z: np.ndarray, *p) -> np.ndarray:
    """N free-position monopoles plus a constant offset.

    φ(z) = Σ_i  A_i / √(d_i² + (z − z_i)²)  +  c
    """
    n = (len(p) - 1) // 3
    out = np.full_like(z, p[-1], dtype=float)
    for i in range(n):
        out = out + p[3 * i] / np.sqrt(p[3 * i + 1] ** 2 + (z - p[3 * i + 2]) ** 2)
    return out


def denoise_field_n(
    phi: np.ndarray,
    dz_mm: float,
    n: int = 3,
    maxfev: int | None = 600,
    ftol: float = 2e-3,
) -> np.ndarray:
    """Replace a noisy lead field φ(z) with a free-position N-monopole fit.

    A FEM-sampled φ(z) carries mesh-scale ripple that the CSD's second derivative
    amplifies. Rather than lowpass-filtering it (which also rounds the physical
    peak), fit the analytic form the field actually has -- a sum of monopoles --
    and keep the fit. Poles are fitted greedily, 1 → n, each solution seeding the
    next; free position, depth and amplitude per pole.

    n=3 is the settled default. On a clean analytical cylinder φ this is close to
    a no-op (r ≈ 0.99); it earns its keep on FEM fields.

    Falls back to a light Butterworth if the fit does not converge.
    """
    phi = np.asarray(phi, float)
    z = (np.arange(len(phi)) - len(phi) // 2) * float(dz_mm)

    z0 = z[int(np.argmax(np.abs(phi)))]
    c0 = float(np.median(np.r_[phi[:8], phi[-8:]]))
    excursion = float(phi.max() - c0)
    z_lo, z_hi = float(z.min()), float(z.max())

    p0 = [excursion * 10.0, 10.0, z0, c0]
    best = None
    for k in range(1, n + 1):
        lo = [-np.inf, 2.0, z_lo] * k + [-np.inf]
        hi = [np.inf, 400.0, z_hi] * k + [np.inf]
        try:
            popt, _ = optimize.curve_fit(
                _n_monopoles, z, phi, p0=p0, bounds=(lo, hi),
                maxfev=(maxfev if maxfev is not None else 20000 * k),
                ftol=ftol, xtol=ftol,
            )
        except Exception:
            break
        best = popt
        # seed the next pole: a weaker, broader copy at the global peak
        p0 = list(popt[:-1]) + [0.3 * popt[0], float(popt[1]) * 1.8, z0, popt[-1]]

    if best is None:
        return smooth_butterworth(phi, 0.10, 2)
    return _n_monopoles(z, *best)
