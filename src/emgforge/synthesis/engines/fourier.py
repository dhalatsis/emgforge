"""
Fourier-domain SFAP / MUAP generation from reciprocal-field data.

This is the **production-ready** pipeline.  Given a spatial reciprocal field
φ(z) (e.g. from FEM), the steps are:

1. (Optional) resample φ(z) onto the grid expected by the coupled (kt, kz) grids.
2. Apply a Hann window and compute C(kz) = dz · FFT_c{φ(z)}.
3. Build the 2-D fibre-end function ``pare(kα, kβ)`` that encodes finite-length
   effects — this is where the Fourier approach excels over time-domain methods.
4. Multiply by the IAP spectrum ``spe2(kz)`` and the derivative factor ``j·kz``.
5. Radon-section back to the time domain.

References
----------
Farina et al., IEEE Trans. Biomed. Eng., 2004.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from emgforge.synthesis.iap import ROSENFALCK_AMPLITUDE_V


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------

def fftc(x: np.ndarray) -> np.ndarray:
    """Centred FFT (ifftshift → fft → fftshift)."""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def ifftc(X: np.ndarray) -> np.ndarray:
    """Centred IFFT."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X)))


# ---------------------------------------------------------------------------
# Radon section
# ---------------------------------------------------------------------------

def radon_section(
    fx: np.ndarray,
    fy: np.ndarray,
    H: np.ndarray,
    section: float,
) -> np.ndarray:
    """Radon section: collapse 2-D spectrum *H* at detector position *section* (mm)."""
    F = H * np.exp(1j * 2 * np.pi * fy * section)
    sig = (
        np.real(
            np.fft.fftshift(np.fft.ifft(np.fft.fftshift(np.sum(F, axis=0))))
        )
        / len(H)
    ) * (fy[1, 0] - fy[0, 0])
    return sig


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def _linfreq_bins(w: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0 - 2.0 / w, w)


def build_fourier_grids(*, w: int, fsamp: float, v: float) -> Dict[str, np.ndarray]:
    """Build coupled (kt, kz) frequency grids."""
    fm = fsamp / (2.0 * v * 1000.0)
    bins = _linfreq_bins(w)
    kz_axis = (2.0 * np.pi * fm) * bins
    kt_axis = (2.0 * np.pi * fm * v) * bins
    kz_t, kt = np.meshgrid(kz_axis, kt_axis)
    kalpha = kz_t + kt / v
    kbeta = kz_t - kt / v
    return {
        "fm": np.array(fm),
        "kz_t": kz_t,
        "kt": kt,
        "kalpha": kalpha,
        "kbeta": kbeta,
        "kz_axis": kz_axis,
        "kt_axis": kt_axis,
    }


def build_time_vector_ms(*, w: int, fsamp: float) -> np.ndarray:
    """Time vector in ms."""
    return np.arange(0, w / fsamp * 1000.0, 1.0 / fsamp * 1000.0)


def build_spe2_iap_spectrum(
    *, w: int, fsamp: float, v: float, iap_flip: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Rosenfalck intracellular action-potential spectrum ``spe2(kz)``.

    ``iap_flip`` is convention C1 (``V2 = -np.flip(V2)``); default True
    reproduces the validated Farina behaviour.
    """
    fm = fsamp / (2.0 * v * 1000.0)
    kzz = 2 * np.pi * np.arange(-2, 2, (2 * fm) / w)
    z = np.arange(0, 15.25, 0.25)
    # dVm/dz of the Rosenfalck IAP; amplitude shared with the spatial engine via
    # emgforge.synthesis.iap (mV→V; audit 2026-06-10, fixes ~10³× amplitude mismatch).
    V2 = ROSENFALCK_AMPLITUDE_V * (np.exp(-z) * (3 * z**2 - z**3))
    V2 = np.concatenate([V2, np.zeros(len(kzz) - len(z))])
    if iap_flip:  # convention C1
        V2 = -np.flip(V2)
    spe2_full = np.fft.fftshift(np.fft.fft(V2))
    spe2 = spe2_full[len(spe2_full) // 2 - w // 2 : len(spe2_full) // 2 + w // 2]
    return spe2, V2


# ---------------------------------------------------------------------------
# φ(z) → C(kz)
# ---------------------------------------------------------------------------

def compute_C_from_phi_z(
    phi_z: np.ndarray,
    *,
    delta_s_mm: float,
    apply_z_window: bool = True,
    window: str = "hann",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert spatial samples φ(z) → C(kz) via centred FFT."""
    phi_z = np.asarray(phi_z, dtype=float)
    w = phi_z.shape[0]

    if apply_z_window:
        if window == "hann":
            win = np.hanning(w)
        else:
            raise ValueError(f"Unsupported window: {window}")
        phi_w = phi_z * win
    else:
        win = np.ones(w)
        phi_w = phi_z

    C_kz = delta_s_mm * fftc(phi_w)
    dbg = {
        "phi_z": phi_z,
        "phi_windowed": phi_w,
        "win_z": win,
        "delta_s_mm": float(delta_s_mm),
    }
    return C_kz, dbg


# ---------------------------------------------------------------------------
# Shared SFAP/MUAP assembly  (single source of truth for the Farina engine)
#
# `build_pare` → `fiber_field_contribution` → `section_from_field_spectrum` is the
# canonical pare → E1 → radon assembly. `api._compute_muap_core` is its only
# in-package caller. `build_pare` carries the C2 sign convention (both
# semi-lengths positive — Farina MATLAB convention).
#
# NOTE: two scripts still hand-roll this block instead of calling these helpers —
# `scripts/synthesis/build_golden.py` and
# `scripts/synthesis/compare_spatial_vs_fourier.py`. The first generates the cylindrical
# reference set the spatial engine is gated against, so a drift there is invisible to CI.
# ---------------------------------------------------------------------------

def build_pare(
    L1_mm: float, L2_mm: float, kalpha: np.ndarray, kbeta: np.ndarray
) -> np.ndarray:
    """Farina fibre-end (tendon-termination) operator `pare`.

    Convention C2: both semi-fibre lengths enter POSITIVE (Farina 2004 MATLAB).
    L1 (proximal) on the kα axis, L2 (distal) on the kβ axis.
    """
    return (
        np.exp(-1j * L1_mm / 2 * kalpha) * L1_mm * np.sinc(L1_mm / 2 * kalpha / np.pi)
        - np.exp(1j * L2_mm / 2 * kbeta) * L2_mm * np.sinc(L2_mm / 2 * kbeta / np.pi)
    )


def fiber_field_contribution(
    C_kz: np.ndarray,
    L1_mm: float,
    L2_mm: float,
    posz_mm: float,
    grids: Dict[str, np.ndarray],
    *,
    swap_ends: bool = False,
) -> np.ndarray:
    """One fibre's (w, w) field-spectrum contribution.

    `pare(L1,L2) · exp(j·kz·posz) · (1 ⊗ C(kz))`. The `exp(j·kz·posz)` factor
    is convention C3 (axial NMJ offset). `swap_ends` is convention C2 (swap
    L1↔L2); default False reproduces the validated behaviour. Sum these over
    fibres to form E_mu.
    """
    w = C_kz.shape[0]
    if swap_ends:  # convention C2
        L1_mm, L2_mm = L2_mm, L1_mm
    pare = build_pare(L1_mm, L2_mm, grids["kalpha"], grids["kbeta"])
    return (
        pare
        * np.exp(1j * grids["kz_t"] * posz_mm)
        * (np.ones((w, 1)) @ C_kz.reshape(1, -1))
    )


def section_from_field_spectrum(
    E_field: np.ndarray,
    spe2: np.ndarray,
    v: float,
    grids: Dict[str, np.ndarray],
    z_det_mm,
    *,
    output_flip: bool = True,
    polarity: int = 1,
) -> np.ndarray:
    """Field spectrum (summed fibre contributions) → time-domain section(s).

    Applies the IAP spectrum + (1/v)·(j·kz) factor, then the radon section at
    each detector position. `z_det_mm` scalar → (w,) trace (legacy 1-D shape);
    a length>1 array → (n_electrodes, w) waterfall. `output_flip` is
    convention C5 (default True) and `polarity` is the overall output sign
    (default +1) — both reproduce the validated behaviour at their defaults.
    """
    w = E_field.shape[0]
    kz_t, kt = grids["kz_t"], grids["kt"]
    E1 = (1.0 / v) * (np.ones((w, 1)) @ spe2.reshape(1, -1)).T * E_field * (1j * kz_t)
    fx = kt.T / (2 * np.pi)
    fy = kz_t.T / (2 * np.pi)
    H = E1.T

    def _section(z: float) -> np.ndarray:
        s = radon_section(fx, fy, H, float(z))
        return np.flip(s) if output_flip else s  # convention C5

    z_arr = np.atleast_1d(np.asarray(z_det_mm, dtype=float))
    if z_arr.shape[0] == 1:
        return polarity * np.real(_section(z_arr[0]))
    return polarity * np.real(np.stack([_section(z) for z in z_arr], axis=0))


