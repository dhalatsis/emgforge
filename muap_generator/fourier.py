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

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from muap_generator.conventions import FARINA_DEFAULT, Conventions
from muap_generator.preprocessing import resample_centered_line, taper_edges


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
    # Rosenfalck coefficient 96 is in mV (per Rosenfalck 1969); convert to V
    # for SI consistency. Audit 2026-06-10 — fixes ~10³× MUAP-amplitude
    # mismatch vs Neurodec ground truth.
    V2 = 96e-3 * (np.exp(-z) * (3 * z**2 - z**3))
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
# The pare → E1 → radon block was previously copy-pasted across
# compute_sfap_from_phi_z, compute_muap_from_phi_matrix, and the inline engine
# in api._compute_muap_core. These three helpers are now the canonical
# implementation that all callers share. `build_pare` carries the C2 sign
# convention (both semi-lengths positive — Farina MATLAB convention).
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


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SFAPParams:
    """Parameters for a single-fibre action potential.

    Attributes
    ----------
    v : conduction velocity (m/s)
    fsamp : sampling frequency (Hz)
    w : number of output samples. If ``None``, the pipeline picks ``w``
        adaptively via ``muap_generator.adaptive_w.choose_w`` using the
        input φ tail-fit + L1+L2 + dz. Pass an integer to lock the window
        (legacy behaviour).
    w_min, w_max : floor/ceiling on adaptive ``w`` (only used when w=None).
    len1_mm, len2_mm : semi-fibre lengths (mm)
    posz_mm : axial NMJ offset from detector (mm)
    z_det_mm : detector axial location for Radon section (mm)
    apply_z_window / window : taper applied to φ(z) before FFT
    enforce_delta_s_match : if True, resample φ(z) to ``v·1000/fsamp``
    edge_taper : number of samples to cosine-taper at each end of φ(z)
        before resampling.  Prevents Gibbs ringing when the lead field
        is truncated (non-zero at its boundaries).  Set to 0 to disable.
        Recommended: 10 for FEM/MRI data where the measurement extent
        is shorter than the lead-field support.
    """

    v: float = 4.0
    fsamp: float = 4096.0
    w: "int | None" = 256
    w_min: int = 256
    w_max: int = 1024

    len1_mm: float = 60.0
    len2_mm: float = 60.0
    posz_mm: float = 0.0
    z_det_mm: float = 0.0

    apply_z_window: bool = True
    window: str = "hann"

    enforce_delta_s_match: bool = True
    delta_s_tol: float = 1e-6

    edge_taper: int = 0

    # Sign/timing conventions (C1/C2/C5/C6 + polarity). Default reproduces the
    # validated Farina behaviour byte-for-byte; pass FEM_NEURODEC for the
    # FEM-vs-Neurodec convention (polarity=−1).
    conventions: Conventions = FARINA_DEFAULT

    def __post_init__(self):
        # Tolerate a plain dict for `conventions` (e.g. asdict round-trip).
        if isinstance(self.conventions, dict):
            self.conventions = Conventions(**self.conventions)


# ---------------------------------------------------------------------------
# Single-fibre AP
# ---------------------------------------------------------------------------

def compute_sfap_from_phi_z(
    phi_z: np.ndarray,
    *,
    delta_s_mm: float,
    params: SFAPParams,
    return_debug: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute a single-fibre AP from a reciprocal-field line φ(z).

    Returns
    -------
    t_ms : (w,) array
    sfap : (w,) array
    debug : dict   (empty when *return_debug* is False)
    """
    v, fsamp = float(params.v), float(params.fsamp)
    zstep_expected = v * 1000.0 / fsamp

    phi_z = np.asarray(phi_z, dtype=float).reshape(-1)

    if params.w is None:
        from muap_generator.adaptive_w import choose_w
        w = choose_w(
            L_fibre_mm=float(params.len1_mm + params.len2_mm),
            phi_z=phi_z,
            dz_mm=float(delta_s_mm),
            w_min=int(params.w_min),
            w_max=int(params.w_max),
            verbose=False,
        )
    else:
        w = int(params.w)

    # Edge taper: prevent Gibbs ringing from truncated lead fields
    if params.edge_taper > 0:
        phi_z = taper_edges(phi_z, n_taper=params.edge_taper)

    phi_used = phi_z
    delta_used = float(delta_s_mm)

    if params.enforce_delta_s_match and abs(delta_s_mm - zstep_expected) > params.delta_s_tol:
        phi_used = resample_centered_line(
            phi_z, delta_s_mm=float(delta_s_mm), w_out=w, delta_s_out_mm=zstep_expected,
        )
        delta_used = zstep_expected
    elif phi_used.shape[0] != w:
        phi_used = resample_centered_line(
            phi_used, delta_s_mm=float(delta_s_mm), w_out=w, delta_s_out_mm=float(delta_s_mm),
        )

    C_kz, dbg_C = compute_C_from_phi_z(
        phi_used, delta_s_mm=delta_used,
        apply_z_window=params.apply_z_window, window=params.window,
    )

    grids = build_fourier_grids(w=w, fsamp=fsamp, v=v)
    kz_t, kt = grids["kz_t"], grids["kt"]
    kalpha, kbeta = grids["kalpha"], grids["kbeta"]

    cv = params.conventions
    spe2, V2 = build_spe2_iap_spectrum(w=w, fsamp=fsamp, v=v, iap_flip=cv.iap_flip)

    # Fibre-end function + assembly via the shared canonical helpers.
    # len1_mm/len2_mm are positive semi-fibre lengths (Farina MATLAB / C2).
    E = fiber_field_contribution(
        C_kz, params.len1_mm, params.len2_mm, params.posz_mm, grids,
        swap_ends=cv.swap_ends,
    )
    sfap = section_from_field_spectrum(
        E, spe2, v, grids, params.z_det_mm,
        output_flip=cv.output_flip, polarity=cv.polarity,
    )
    t_ms = build_time_vector_ms(w=w, fsamp=fsamp)
    if cv.center_time:  # convention C6
        t_ms = t_ms - (w / fsamp) * 1000.0 / 2

    debug: Dict[str, Any] = {}
    if return_debug:
        pare = build_pare(params.len1_mm, params.len2_mm, kalpha, kbeta)
        E1 = (1.0 / v) * (np.ones((w, 1)) @ spe2.reshape(1, -1)).T * E * (1j * kz_t)
        debug.update({
            "phi_z_in": phi_z, "phi_z_used": phi_used,
            "delta_s_in_mm": float(delta_s_mm), "delta_s_used_mm": float(delta_used),
            "zstep_expected_mm": float(zstep_expected),
            "C_kz": C_kz, "E": E, "E1": E1, "pare": pare,
            "spe2": spe2, "V2": V2,
            "kz_t": kz_t, "kt": kt, "kalpha": kalpha, "kbeta": kbeta,
            **dbg_C,
        })

    return t_ms, sfap, debug


# ---------------------------------------------------------------------------
# Multi-fibre MUAP
# ---------------------------------------------------------------------------

def compute_muap_from_phi_matrix(
    phi_mat: np.ndarray,
    *,
    delta_s_mm: float,
    params: SFAPParams,
    len1_mm_arr: np.ndarray,
    len2_mm_arr: np.ndarray,
    posz_mm_arr: np.ndarray,
    return_debug: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Sum many single-fibre APs into a MUAP.

    Parameters
    ----------
    phi_mat : (Nfib, Nz) array — one φ(z) line per fibre.
    len1_mm_arr, len2_mm_arr, posz_mm_arr : (Nfib,) per-fibre geometry.
    """
    v, fsamp = float(params.v), float(params.fsamp)
    zstep_expected = v * 1000.0 / fsamp
    delta_used = float(delta_s_mm)

    phi_mat = np.asarray(phi_mat, dtype=float)

    if params.w is None:
        from muap_generator.adaptive_w import choose_w
        L_fibre = float(len1_mm_arr[0] + len2_mm_arr[0])
        w = choose_w(
            L_fibre_mm=L_fibre,
            phi_z=phi_mat[0] if phi_mat.ndim == 2 else phi_mat,
            dz_mm=float(delta_s_mm),
            w_min=int(params.w_min),
            w_max=int(params.w_max),
            verbose=False,
        )
    else:
        w = int(params.w)
    if phi_mat.ndim != 2:
        raise ValueError("phi_mat must be 2-D (Nfib, Nz)")
    if phi_mat.shape[0] != len(posz_mm_arr) and phi_mat.shape[1] == len(posz_mm_arr):
        phi_mat = phi_mat.T

    Nfib = phi_mat.shape[0]
    if not (len(len1_mm_arr) == len(len2_mm_arr) == len(posz_mm_arr) == Nfib):
        raise ValueError("len1/len2/posz arrays must match number of fibres")

    grids = build_fourier_grids(w=w, fsamp=fsamp, v=v)
    kz_t, kt = grids["kz_t"], grids["kt"]
    kalpha, kbeta = grids["kalpha"], grids["kbeta"]

    cv = params.conventions
    spe2, V2 = build_spe2_iap_spectrum(w=w, fsamp=fsamp, v=v, iap_flip=cv.iap_flip)

    E_mu = np.zeros((w, w), dtype=complex)
    C_list = []

    for u in range(Nfib):
        phi_u = phi_mat[u, :]

        # Edge taper: prevent Gibbs ringing from truncated lead fields
        if params.edge_taper > 0:
            phi_u = taper_edges(phi_u, n_taper=params.edge_taper)

        if params.enforce_delta_s_match and abs(delta_s_mm - zstep_expected) > params.delta_s_tol:
            phi_u = resample_centered_line(
                phi_u, delta_s_mm=float(delta_s_mm), w_out=w, delta_s_out_mm=zstep_expected,
            )
            delta_used = zstep_expected
        elif phi_u.shape[0] != w:
            phi_u = resample_centered_line(
                phi_u, delta_s_mm=float(delta_s_mm), w_out=w, delta_s_out_mm=float(delta_s_mm),
            )

        C_u, _ = compute_C_from_phi_z(
            phi_u, delta_s_mm=delta_used,
            apply_z_window=params.apply_z_window, window=params.window,
        )
        C_list.append(C_u)

        # Per-fibre field contribution via the shared canonical helper
        # (pare = C2, posz phase = C3).
        E_mu += fiber_field_contribution(
            C_u, len1_mm_arr[u], len2_mm_arr[u], posz_mm_arr[u], grids,
            swap_ends=cv.swap_ends,
        )

    muap = section_from_field_spectrum(
        E_mu, spe2, v, grids, params.z_det_mm,
        output_flip=cv.output_flip, polarity=cv.polarity,
    )
    t_ms = build_time_vector_ms(w=w, fsamp=fsamp)
    if cv.center_time:  # convention C6
        t_ms = t_ms - (w / fsamp) * 1000.0 / 2

    debug: Dict[str, Any] = {}
    if return_debug:
        E1_mu = (1.0 / v) * (np.ones((w, 1)) @ spe2.reshape(1, -1)).T * E_mu * (1j * kz_t)
        debug.update({
            "C_list": C_list, "E_mu": E_mu, "E1_mu": E1_mu,
            "spe2": spe2, "V2": V2,
            "kz_t": kz_t, "kt": kt, "kalpha": kalpha, "kbeta": kbeta,
            "zstep_expected_mm": float(zstep_expected), "delta_s_used_mm": float(delta_used),
        })

    return t_ms, muap, debug
