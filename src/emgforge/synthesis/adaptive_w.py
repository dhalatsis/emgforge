"""Adaptive pipeline-window selection for the FEM → MUAP pipeline.

The pipeline traditionally fixed ``w=256`` and the corresponding spatial
window ``W = w · v / fs``. That window must satisfy two independent
constraints:

  A) **Field-decay constraint** — `|φ(±W/2)| ≪ |φ(0)|`.
     Fails for deep electrodes (wide spatial field).

  B) **Fibre-fit constraint** — `L1 + L2 + IAP_envelope (~30 mm) ≤ W`.
     Fails for long fibres.

This module exposes:

  * ``required_window_mm(L_fibre_mm, field_decay_mm)`` — required spatial
    window in mm.
  * ``estimate_field_decay_from_phi(phi_z, dz_mm)`` — fits ``exp(-|z|/λ)``
    on the outer tails of φ to estimate the lead-field decay length.
  * ``fallback_decay_from_depth(depth_mm)`` — cheap proxy when φ isn't
    available yet.
  * ``choose_w(...)`` — top-level entry: picks the smallest power-of-2 ``w``
    in ``[w_min, w_max]`` that satisfies both constraints.

Skeleton lifted from `field_to_muap_study/deliverables/fem_worker_integration
/skeletons/adaptive_w.py`. Integration-tested against the regression bench in
`tests/regression/`.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


# Rosenfalck IAP spatial envelope. Slightly conservative; the IAP itself is
# ~20 mm but with derivative ringing and integration support ~30 mm captures
# the meaningful range.
IAP_ENVELOPE_MM = 30.0


def next_power_of_2(n: int) -> int:
    """Smallest power of 2 ≥ n. next_power_of_2(257) = 512."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def required_window_mm(L_fibre_mm: float, field_decay_mm: float) -> float:
    """Smallest spatial window length W (mm) that satisfies both constraints.

    Parameters
    ----------
    L_fibre_mm : total fibre length ``L1 + L2``. For default L1=L2=60, pass 120.
    field_decay_mm : characteristic decay length λ of the lead field along
        the fibre. For ``exp(-|z|/λ)`` we want ``|φ(W/2)| ≤ 5%·|φ(0)|`` →
        ``W ≥ 6λ``.

    Returns
    -------
    W_mm : minimum spatial window length in mm. Caller converts to ``w``
        samples via ``w = W / dz``.
    """
    W_field = 6.0 * field_decay_mm
    W_fibre = L_fibre_mm + IAP_ENVELOPE_MM
    return max(W_field, W_fibre)


def estimate_field_decay_from_phi(
    phi_z: np.ndarray,
    dz_mm: float,
    min_distance_mm: float = 5.0,
) -> Optional[float]:
    """Estimate lead-field decay length λ from sampled φ(z).

    Fits the parametric model ``φ(z) ≈ A · exp(-|z − z_peak|/λ) + B`` via
    nonlinear least squares, separately on each tail (left / right of the
    peak), then returns the average λ.

    Joint estimation of (A, λ, B) handles both:
    - **No baseline** (synthetic exp-decay): converges to B ≈ 0 → λ correct.
    - **Real DC offset** (analytical / FEM lead fields, which have residual
      mean from the Neumann nullspace removal): converges to B at the
      observed baseline → λ correct.

    Returns ``None`` on any failure (insufficient samples, non-decaying,
    NaN, optimizer divergence).

    Parameters
    ----------
    phi_z : (N,) array
    dz_mm : sample spacing in mm
    min_distance_mm : skip samples within this many mm of the peak
    """
    from scipy.optimize import curve_fit

    phi = np.asarray(phi_z, dtype=float)
    n = phi.shape[0]
    if n < 20:
        return None

    abs_phi = np.abs(phi)
    z_idx_peak = int(np.argmax(abs_phi))
    peak_abs = abs_phi[z_idx_peak]
    if peak_abs <= 0 or not np.isfinite(peak_abs):
        return None
    # Sign of the peak determines the sign convention for the fit's A.
    sign = float(np.sign(phi[z_idx_peak]))

    z_mm = (np.arange(n) - z_idx_peak) * dz_mm
    z_abs = np.abs(z_mm)

    def model(z_, A_, lam_, B_):
        return A_ * np.exp(-z_ / lam_) + B_

    lambdas = []
    # Two-sided tails: peak-to-left and peak-to-right.
    slices = [
        (z_idx_peak - np.arange(max(0, z_idx_peak - 1), -1, -1) - 1)[::-1] if False else None,
        None,
    ]
    # Simpler indexing — explicit slices.
    left_idx = np.arange(0, max(0, z_idx_peak))
    right_idx = np.arange(min(n, z_idx_peak + 1), n)
    for idx_arr in (left_idx, right_idx):
        if idx_arr.size < 6:
            continue
        z_sub = z_abs[idx_arr]
        mask = z_sub > min_distance_mm
        if mask.sum() < 6:
            continue
        z_use = z_sub[mask]
        phi_use = sign * phi[idx_arr][mask]
        # Initial guesses: A = peak_signed - baseline; lam = z_range/3; B = baseline_estimate
        baseline_init = float(np.median(phi_use[-max(3, len(phi_use)//5):]))
        A_init = float(peak_abs - abs(baseline_init))
        lam_init = float((z_use[-1] - z_use[0]) / 3.0)
        if lam_init <= 0 or not np.isfinite(lam_init):
            continue
        try:
            popt, _ = curve_fit(
                model, z_use, phi_use,
                p0=(A_init, lam_init, baseline_init),
                bounds=([1e-30, 1e-3, -np.inf], [np.inf, 1e4, np.inf]),
                maxfev=400,
            )
        except (RuntimeError, ValueError):
            continue
        lam_fit = float(popt[1])
        if lam_fit > 0 and np.isfinite(lam_fit) and lam_fit < 1e4:
            lambdas.append(lam_fit)

    if not lambdas:
        return None
    return float(np.mean(lambdas))


def fallback_decay_from_depth(depth_mm: float) -> float:
    """Cheap estimate of field decay length from electrode depth.

    Calibrated against the cylinder dataset: depth=15 → λ≈43, depth=30 → λ≈144.
    A simple ``depth · 3 + depth² · 0.05`` (capped at 30 mm) reproduces both
    points and stays sensible for the very-shallow regime.
    """
    if depth_mm <= 0:
        return 30.0
    return max(depth_mm * 3.0 + depth_mm * depth_mm * 0.05, 30.0)


def choose_w(
    L_fibre_mm: float,
    phi_z: Optional[np.ndarray] = None,
    depth_mm: Optional[float] = None,
    dz_mm: float = 1.0,
    w_min: int = 256,
    w_max: int = 1024,
    cap_to_input_length: bool = True,
    edge_decay_threshold: float = 0.05,
    verbose: bool = False,
) -> int:
    """Top-level entry: choose the pipeline window ``w`` adaptively.

    Provide one of (``phi_z``) for accurate field-decay fitting, (``depth_mm``)
    for a cheap estimate, or both (prefers ``phi_z``, falls back to depth).

    Parameters
    ----------
    L_fibre_mm : total fibre length (L1 + L2)
    phi_z : optional, sampled φ along the fibre — used for fit-based λ AND
        as the upper bound on the chosen window (we never adapt above what
        the input data physically supports — see ``cap_to_input_length``).
    depth_mm : optional electrode-to-fibre distance for fallback λ
    dz_mm : sample spacing — ``v · 1000 / fsamp`` at the coupled-grid mode
    w_min : floor on chosen ``w`` (preserves backward compat by default)
    w_max : ceiling on chosen ``w`` (avoid runaway memory)
    cap_to_input_length : when True (default), never adapt above
        ``next_power_of_2(len(phi_z)) // 2`` IF the input φ is not decayed
        to ``edge_decay_threshold`` of its peak at the boundary. The reason:
        going wider would zero-pad outside the input span and re-introduce
        the truncation artifact (FEM_AND_MESH_GUIDE.md §11). When the input
        is already well-decayed, growing w just adds a clean zero-pad of
        already-zero values and is safe.
    edge_decay_threshold : the |φ(edge)|/|φ(peak)| ratio below which we
        consider the input "decayed enough" that extending w is safe.
    verbose : emit a one-line log

    Returns
    -------
    w : int, power-of-2 clamped to [w_min, w_max]
    """
    lam = None
    if phi_z is not None:
        lam = estimate_field_decay_from_phi(phi_z, dz_mm)
    if lam is None and depth_mm is not None:
        lam = fallback_decay_from_depth(depth_mm)
    if lam is None:
        lam = 60.0
        if verbose:
            warnings.warn(
                "choose_w: no phi_z or depth_mm provided; "
                "defaulting to lambda=60 mm (conservative).",
                stacklevel=2,
            )

    W_mm = required_window_mm(L_fibre_mm, lam)
    w_required = int(np.ceil(W_mm / dz_mm))
    w_p2 = next_power_of_2(w_required)
    w_chosen = int(np.clip(w_p2, w_min, w_max))

    cap_note = ""
    if cap_to_input_length and phi_z is not None:
        n_in = int(np.asarray(phi_z).shape[-1])
        peak = float(np.max(np.abs(phi_z)))
        edge = max(float(abs(phi_z[0])), float(abs(phi_z[-1])))
        decayed = peak > 0 and (edge / peak) <= edge_decay_threshold
        if not decayed:
            # Input not zero at the boundary → going wider would zero-pad
            # over a real-valued tail and create an edge step. Cap.
            w_in_p2 = next_power_of_2(n_in)
            # If n_in is exactly a power of 2 we keep it; if not, take the
            # power-of-2 floor (n_in_p2 // 2) since rounding up would over-
            # interpolate.
            if w_in_p2 > n_in:
                w_in_p2 = w_in_p2 // 2
            new_w = int(np.clip(w_in_p2, w_min, w_chosen))
            if new_w != w_chosen:
                cap_note = f" [capped {w_chosen}->{new_w} by input-length safety]"
                w_chosen = new_w

    if verbose:
        actual_W = w_chosen * dz_mm
        clamp = "" if w_p2 == w_chosen and not cap_note else (cap_note or f" [clamped from {w_p2}]")
        print(
            f"[adaptive_w] L_fibre={L_fibre_mm:.0f} mm, lambda={lam:.1f} mm "
            f"-> W_req={W_mm:.0f} mm, w_req={w_required} -> w={w_chosen}"
            f"{clamp} (W_actual={actual_W:.0f} mm)"
        )
    return w_chosen


__all__ = [
    "IAP_ENVELOPE_MM",
    "next_power_of_2",
    "required_window_mm",
    "estimate_field_decay_from_phi",
    "fallback_decay_from_depth",
    "choose_w",
]
