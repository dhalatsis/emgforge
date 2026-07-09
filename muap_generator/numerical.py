"""
Numerical (time-domain) SFAP generation — **experimental**.

This approach directly convolves the Rosenfalck IAP derivative with the
reciprocal field φ(z) in the time domain, applying Tukey windows for
fibre-end effects.

It is simpler but **less accurate** than the Fourier pipeline because fibre-end
effects are modelled as 1-D windows rather than the full 2-D (kt, kz) ``pare``
function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import numpy as np
from scipy import signal

from muap_generator.preprocessing import (
    create_fiber_windows,
    smooth_butterworth,
    upsample_cubic,
)


# ---------------------------------------------------------------------------
# IAP model (Rosenfalck)
# ---------------------------------------------------------------------------

def rosenfalck_dvm_dz(z_mm: np.ndarray) -> np.ndarray:
    """First spatial derivative of the Rosenfalck IAP: dVm/dz = 96(3z² − z³)e^{−z}."""
    z = np.asarray(z_mm)
    # Rosenfalck mV→V conversion (audit 2026-06-10)
    return np.where(z >= 0, 96e-3 * np.exp(-z) * (3 * z**2 - z**3), 0.0)


def rosenfalck_d2vm_dz2(z_mm: np.ndarray) -> np.ndarray:
    """Second derivative (CSD): d²Vm/dz² = 96 e^{−z} z (6 − 6z + z²)."""
    z = np.asarray(z_mm)
    # Rosenfalck mV→V conversion (audit 2026-06-10)
    return np.where(z >= 0, 96e-3 * np.exp(-z) * z * (6 - 6 * z + z**2), 0.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NumericalConfig:
    """Configuration for the numerical SFAP pipeline."""

    # Preprocessing
    smoothing: bool = True
    butterworth_cutoff: float = 0.1
    butterworth_order: int = 4
    upsample_factor: int = 2

    # Windowing
    phi_window: Literal["hann", "tukey", "none"] = "hann"
    fiber_window: Literal["tukey", "boxcar", "hann", "none"] = "tukey"
    tukey_alpha: float = 0.25

    # Physical parameters
    v: float = 4.0  # conduction velocity (m/s ≡ mm/ms)
    sigma_in: float = 1.0
    fiber_radius_mm: float = 0.05

    # Output
    fsamp: float = 4096.0
    w: int = 256


# ---------------------------------------------------------------------------
# Numerical SFAP
# ---------------------------------------------------------------------------

def compute_sfap_numerical(
    phi_z: np.ndarray,
    dz_mm: float,
    len1_mm: float = 60.0,
    len2_mm: float = 60.0,
    posz_mm: float = 0.0,
    config: NumericalConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute an SFAP via direct time-domain convolution.

    Parameters
    ----------
    phi_z : (Nz,) array — reciprocal field along z (centred at electrode).
    dz_mm : spatial sampling step (mm).
    len1_mm, len2_mm : semi-fibre lengths (mm).
    posz_mm : NMJ offset from electrode (mm).
    config : pipeline settings.

    Returns
    -------
    t_ms, sfap, debug
    """
    if config is None:
        config = NumericalConfig()

    phi = np.asarray(phi_z, dtype=float).flatten().copy()

    # --- preprocessing ---
    if config.smoothing:
        phi = smooth_butterworth(phi, config.butterworth_cutoff, config.butterworth_order)

    if config.upsample_factor > 1:
        phi = upsample_cubic(phi, config.upsample_factor)
        dz_mm = dz_mm / config.upsample_factor

    Nz = len(phi)

    if config.phi_window == "hann":
        phi_win = np.hanning(Nz)
    elif config.phi_window == "tukey":
        phi_win = signal.windows.tukey(Nz, alpha=config.tukey_alpha)
    else:
        phi_win = np.ones(Nz)

    phi_windowed = phi * phi_win

    # --- spatial / time grids ---
    z = (np.arange(Nz) - Nz // 2) * dz_mm
    t_ms = np.arange(config.w) / config.fsamp * 1000.0
    v = config.v

    # --- fibre geometry ---
    fiber_length = len1_mm + len2_mm
    nmj_ratio = len1_mm / fiber_length
    n_fiber_points = int(fiber_length / dz_mm)
    win_left, win_right = create_fiber_windows(
        n_fiber_points, nmj_ratio,
        window_type=config.fiber_window,
        tukey_alpha=config.tukey_alpha,
    )

    # --- time-domain loop ---
    sfap = np.zeros(config.w)
    scale = config.sigma_in * np.pi * config.fiber_radius_mm**2

    for i, t in enumerate(t_ms):
        z_offset = v * t
        contrib = 0.0

        for j, zj in enumerate(z):
            if not (posz_mm - len1_mm <= zj <= posz_mm + len2_mm):
                continue

            z_in_fiber = zj - (posz_mm - len1_mm)
            fiber_idx = max(0, min(n_fiber_points - 1, int(z_in_fiber / dz_mm)))

            # wave towards +z
            if zj >= posz_mm:
                xi1 = z_offset - (zj - posz_mm)
                if xi1 >= 0:
                    csd1 = rosenfalck_dvm_dz(xi1)
                    if fiber_idx < len(win_right):
                        csd1 *= win_right[fiber_idx]
                    contrib += phi_windowed[j] * csd1

            # wave towards −z
            if zj <= posz_mm:
                xi2 = z_offset - (posz_mm - zj)
                if xi2 >= 0:
                    csd2 = rosenfalck_dvm_dz(xi2)
                    if fiber_idx < len(win_left):
                        csd2 *= win_left[fiber_idx]
                    contrib -= phi_windowed[j] * csd2

        sfap[i] = contrib * dz_mm * scale / v

    debug: Dict[str, Any] = {
        "phi_orig": phi_z,
        "phi_smoothed": phi if config.smoothing else phi_z,
        "phi_windowed": phi_windowed,
        "phi_win": phi_win,
        "z": z,
        "dz_mm": dz_mm,
        "win_left": win_left,
        "win_right": win_right,
    }

    return t_ms, sfap, debug
