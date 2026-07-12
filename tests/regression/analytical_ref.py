"""Analytical 4-layer cylindrical reference for the regression bench.

Wraps the Farina 2004 analytical model — the vendored ``emgforge.analytical``
package (multilayer cylindrical volume conductor). Provides two things the
bench needs:

1. ``analytical_muap(...)``        — the reference MUAP waveform.
2. ``analytical_phi_along_fibre`` — the analytical lead field φ(z) along
   the fibre, extracted from the SignalGenerator internals via the
   `compute_C_from_phi_z` inverse identity (φ(z) = ifftc(C) / dz).

Both share one geometry/conductivity dict (`ANAL_GEOMETRY`,
`ANAL_CONDUCTIVITIES`) so that comparisons are like-for-like.

Sign convention: pass ``L1=+60, L2=+60`` (MATLAB convention) to match the
production pipeline's positive-sign `pare` formula. See
`muap_smoothness/16_sign_convention_sort.py`.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from emgforge.analytical import (
    CylindricalVolumeConductor,
    MotorUnit,
    DetectionSystem,
    SignalGenerator,
)


ANAL_GEOMETRY = {
    "r_bone": 10.0,
    "r_muscle": 35.0,
    "r_fat": 38.0,
    "r_skin": 40.0,
}

ANAL_CONDUCTIVITIES = {
    "bone":   {"r": 0.02, "theta": 0.02, "z": 0.02},
    "muscle": {"r": 0.1,  "theta": 0.1,  "z": 0.5},
    "fat":    {"r": 0.04, "theta": 0.04, "z": 0.04},
    "skin":   {"r": 1.0,  "theta": 1.0,  "z": 1.0},
}


@dataclass
class AnalyticalCase:
    """Geometry / fibre / sampling parameters for one analytical run."""

    fiber_depth_mm: float
    L1_mm: float = 60.0
    L2_mm: float = 60.0
    v_m_per_s: float = 4.0
    fsamp_hz: float = 4096.0
    w: int = 256
    distfib_deg: float = 0.0       # angular separation of MU from detector
    electrode_dim1_mm: float = 5.0  # circular electrode radius (analytical det)


def _build(case: AnalyticalCase):
    """Construct the analytical (vc, mu, det, sg) tuple."""
    y0 = (
        ANAL_GEOMETRY["r_skin"]
        - ANAL_GEOMETRY["r_fat"]
        - (ANAL_GEOMETRY["r_skin"] - case.fiber_depth_mm)
    )
    np.random.seed(42)  # reproducible MU random fiber placement
    vc = CylindricalVolumeConductor(
        r_bone=ANAL_GEOMETRY["r_bone"],
        r_muscle=ANAL_GEOMETRY["r_muscle"],
        r_fat=ANAL_GEOMETRY["r_fat"],
        r_skin=ANAL_GEOMETRY["r_skin"],
        fiber_depth=case.fiber_depth_mm,
        conductivities=ANAL_CONDUCTIVITIES,
        v=case.v_m_per_s,
    )
    mu = MotorUnit(
        n_fibers=1,
        radius=0.1,
        y0=y0,
        innervation_spread=0.1,
        zi=0.0,
        L1=case.L1_mm,
        L2=case.L2_mm,
        Ten1=0.1,
        Ten2=0.1,
        distfib=case.distfib_deg,
        r=ANAL_GEOMETRY["r_skin"],
        h=ANAL_GEOMETRY["r_skin"] - ANAL_GEOMETRY["r_fat"],
        d=ANAL_GEOMETRY["r_fat"] - ANAL_GEOMETRY["r_muscle"],
    )
    det = DetectionSystem(
        channels=1,
        dint=10.0,
        center=0.0,
        alpha=0.0,
        det_type=1,
        dintsf=10.0,
        electrode_type="circ",
        dim1=case.electrode_dim1_mm,
        dim2=case.electrode_dim1_mm,
        r=ANAL_GEOMETRY["r_skin"],
    )
    sg = SignalGenerator(
        vc, mu, det,
        v=case.v_m_per_s,
        fsamp=case.fsamp_hz,
        w=case.w,
    )
    return vc, mu, det, sg


def analytical_muap(case: AnalyticalCase) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_ms, muap) from the analytical 4-layer cylinder."""
    _, _, _, sg = _build(case)
    with contextlib.redirect_stdout(io.StringIO()):
        t, sig_arr, _ = sg.generate_muap()
    return t, np.asarray(sig_arr[0]).flatten()


def analytical_phi_along_fibre(
    case: AnalyticalCase,
    n_fibers_average: int = 1,
) -> Tuple[np.ndarray, float]:
    """Reconstruct analytical φ(z) at the fibre line from SignalGenerator innards.

    The analytical pipeline computes per-channel ``C[ch][u]`` = sum over k_theta
    of ``Htissue * H * exp(j (fiber_pos_th + th_center) k_theta)``. C is exactly
    what the production pipeline calls ``C_kz``. So ``φ(z) = ifftc(C) / dz``
    recovers the spatial lead field — perfect for operator-consistency tests.

    Parameters
    ----------
    case : AnalyticalCase
    n_fibers_average : int
        Number of fibres in the MU to average over. Default 1 (single fibre,
        matches the bench's main path).

    Returns
    -------
    phi_z : (w,) array
        Spatial lead field along the fibre axis.
    dz_mm : float
        Sample spacing in mm — equals ``v · 1000 / fsamp``.
    """
    vc, mu, det, sg = _build(case)

    with contextlib.redirect_stdout(io.StringIO()):
        Htissue = vc.compute_transfer_function(w=case.w, fsamp=case.fsamp_hz).T

    fiber_info = mu.get_fiber_info()
    fiber_pos_th = fiber_info["fiber_pos_th"]
    th_center, _ = det.get_electrode_centers()

    # Spatial frequency grids — match SignalGenerator
    ktheta, kz = np.meshgrid(
        np.arange(
            -1 / (2 * sg.tstep) * 2 * np.pi,
            1 / (2 * sg.tstep) * 2 * np.pi,
            1 / (2 * np.pi) * 2 * np.pi,
        ),
        np.arange(
            -1 / (2 * sg.zstep) * 2 * np.pi,
            (1 / (2 * sg.zstep) - 1 / (case.w / (2 * sg.fm))) * 2 * np.pi
            + 2 * np.pi / (case.w / (2 * sg.fm)),
            1 / (case.w / (2 * sg.fm)) * 2 * np.pi,
        ),
    )

    H = det.compute_complete_filter(ktheta, kz, case.w)

    # The k_theta integration that gives C(kz)
    Phi_mat = Htissue * H
    Phi_mat_true = Phi_mat * np.exp(1j * (fiber_pos_th + th_center[0]) * ktheta)
    C_kz = np.sum(Phi_mat_true.T, axis=0)   # length w

    # Inverse of the production pipeline's `compute_C_from_phi_z` (no window):
    #   C_kz = dz · fftc(phi_z)   ⇒   phi_z = ifftc(C_kz) / dz
    dz_mm = case.v_m_per_s * 1000.0 / case.fsamp_hz
    phi_z = np.real(
        np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(C_kz)))
    ) / dz_mm

    return phi_z, dz_mm


__all__ = [
    "ANAL_GEOMETRY",
    "ANAL_CONDUCTIVITIES",
    "AnalyticalCase",
    "analytical_muap",
    "analytical_phi_along_fibre",
]
