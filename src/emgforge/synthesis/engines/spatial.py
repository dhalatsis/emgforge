"""Spatial (time-domain) SFAP / MUAP engine — the *spatial refactor*.

This is the revived 2024/2025 spatial method (origin: ``FEM/motor_units.py``,
``mpi_test.py:create_muaps_for_mu``; cleaned descendant:
``emgforge.synthesis/numerical.py``). It computes the single-fibre action
potential as the **spatial line-source integral**

    SFAP(t) = (σ_in · π · a²) / v · ∫ φ(z) · CSD(z, t) dz

where ``φ(z)`` is the lead field sampled along the fibre (reciprocity: the
potential field produced by a source at the *electrode*) and ``CSD(z, t)`` is
the current-source density of the travelling intracellular action potential:
two counter-propagating Rosenfalck IAP-derivative waves launched from the
neuromuscular junction (NMJ) and clipped at the tendons by fibre-end windows.

Discretised, the integral is a single matrix–vector product::

    sfap = (CSD @ φ) · dz · scale / v          # CSD is (n_time, n_z)

There is **no FFT, no ``pare``, no ``radon_section``, no ``np.flip``** — the
travelling-wave sign and the fibre-end termination are explicit. That is the
entire point of the refactor: it sidesteps the Fourier sign/timing convention
minefield that the production ``fourier.py`` pipeline suffers from.

The volume conductor enters *only* through φ(z). Feed it an analytical
4-layer-cylinder φ (cylindrical tier) or an FEM φ sampled along a real fibre
path (MRI tier) — the engine does not care where φ comes from.

See ``README.md`` in this package for the method, the spatial-vs-Fourier
comparison, and the golden-set verification workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np

from emgforge.synthesis.config import SynthesisConfig
from emgforge.synthesis.preprocessing import (
    create_fiber_windows,
    denoise_field_n,
    smooth_butterworth,
    upsample_cubic,
)


# ---------------------------------------------------------------------------
# Rosenfalck IAP (mV→V corrected, audit 2026-06-10)
# ---------------------------------------------------------------------------

def rosenfalck_vm(z_mm: np.ndarray) -> np.ndarray:
    """Rosenfalck intracellular action potential (depolarisation above baseline):
    Vm(z) = 96·z³·e^{−z} [mV→V via 96e-3], zero for z < 0 (wave not yet arrived).
    Onset is smooth (V, V', V'' all → 0 as z→0⁺), so no spurious wavefront source.
    """
    z = np.asarray(z_mm, dtype=float)
    return np.where(z >= 0.0, 96e-3 * z**3 * np.exp(-z), 0.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpatialConfig(SynthesisConfig):
    """Settings for the spatial SFAP/MUAP engine.

    Defaults mirror the canonical PM-dataset recipe so results are directly
    comparable to the Fourier pipeline: a light edge taper + Butterworth on
    φ(z), no upsample, Tukey fibre-end windows.
    """

    # φ(z) preprocessing (applied per fibre before the integral)
    butterworth_cutoff: float = 0.03      # canonical (muap_smoothness study)
    butterworth_order: int = 2
    edge_taper_left: int = 5              # cosine ramp on φ endpoints (Gibbs guard)
    edge_taper_right: int = 10
    # The 2nd-derivative CSD amplifies high-frequency content, so the integration
    # grid must be finer than the raw φ sampling or the SFAP develops a Nyquist
    # zigzag. Upsampling φ 2× before the integral removes it (jaggedness 0.08→0.01)
    # and lifts r vs Neurodec from 0.52 to 0.78. 2 is the sweet spot (4 adds cost,
    # no gain). Set 1 only with already-fine φ (e.g. the analytical cylinder tier).
    upsample_factor: int = 2

    # φ(z) denoising. "butterworth" is a zero-phase lowpass applied AFTER the edge
    # taper. "monopole" replaces φ with a free-position N-monopole fit — the analytic
    # form the field actually has — applied to the RAW field BEFORE the taper (a taper
    # would corrupt the tails it fits against); the settled choice for FEM lead fields,
    # where mesh ripple is otherwise amplified by the CSD's 2nd derivative, and near a
    # no-op on analytical φ. "none" passes through. These are mutually exclusive; the
    # two positions are a property of each method, not a composition knob.
    denoise: Literal["none", "butterworth", "monopole"] = "butterworth"
    denoise_n_poles: int = 3

    # fibre-end windows (tendon termination of the travelling wave). "one_sided"
    # tapers only the tendon end of each semi-fibre and stays flat through the NMJ;
    # a symmetric "tukey" notches the junction. "none" is an alias for "boxcar".
    fiber_window: Literal["tukey", "boxcar", "hann", "none", "one_sided"] = "tukey"
    tukey_alpha: float = 0.25

    # physical
    v: float = 4.0                        # conduction velocity (m/s ≡ mm/ms)
    sigma_in: float = 1.0
    fiber_radius_mm: float = 0.05
    polarity: int = 1                     # +1 = raw physics; -1 to flip sign

    # The current-source density that drives the SFAP is the SECOND spatial
    # derivative of Vm (transmembrane current i_m ∝ ∂²Vm/∂z², triphasic +/−/+).
    # The original 2024 motor_units.py obtained this via a discrete ∂/∂z of the
    # bidirectional dVm/dz wave; the packaged numerical.py dropped that extra
    # derivative and used dVm/dz (biphasic), which kills the trailing EOF lobe.
    # csd_derivative=2 is physically correct; =1 reproduces the old behaviour.
    csd_derivative: Literal[1, 2] = 2

    # time axis. The spatial method is naturally in PHYSICAL time: t=0 is when
    # the NMJ fires, the proximal EOF lands at len1/v, the distal at len2/v.
    # So we do NOT centre (unlike the Fourier radon section). Set center_time
    # True only to compare against a centred Fourier output.
    fsamp: float = 2048.0
    w: int = 256
    center_time: bool = False
    # Pre-roll: shift the window start to t_start_ms (negative = baseline before
    # the NMJ fire at t=0). Needed when the NMJ sits at/near the electrode (e.g.
    # the symmetric cylinder, posz≈0): the complex would otherwise be pinned
    # against t=0 with no lead-in. The signal is genuinely 0 for t<0, so this
    # only prepends baseline — it never changes the waveform. Default 0 (the MRI
    # case detects far from the NMJ, so its action already lands inside [0, w/fs]).
    t_start_ms: float = 0.0


# ---------------------------------------------------------------------------
# φ(z) preprocessing
# ---------------------------------------------------------------------------

def _preprocess_phi(phi_z: np.ndarray, dz_mm: float, cfg: SpatialConfig
                    ) -> Tuple[np.ndarray, float]:
    phi = np.asarray(phi_z, dtype=float).flatten().copy()
    # The monopole fit must see the RAW field: it models φ as a sum of analytic
    # monopoles, and an edge taper would corrupt the tails it fits against.
    if cfg.denoise == "monopole":
        phi = denoise_field_n(phi, float(dz_mm), n=cfg.denoise_n_poles)
    lN, rN = cfg.edge_taper_left, cfg.edge_taper_right
    if rN > 0:
        phi[-rN:] *= 0.5 * (1 + np.cos(np.pi * np.arange(rN) / rN))
    if lN > 0:
        phi[:lN] *= (0.5 * (1 + np.cos(np.pi * np.arange(lN) / lN)))[::-1]
    if cfg.denoise == "butterworth":
        phi = smooth_butterworth(phi, cfg.butterworth_cutoff, cfg.butterworth_order)
    if cfg.upsample_factor > 1:
        phi = upsample_cubic(phi, cfg.upsample_factor)
        dz_mm = dz_mm / cfg.upsample_factor
    return phi, dz_mm


# ---------------------------------------------------------------------------
# CSD matrix (vectorised — the prize from FEM/motor_units.py:csd_matrix)
# ---------------------------------------------------------------------------

def build_csd_matrix(
    z: np.ndarray,
    t_ms: np.ndarray,
    posz_mm: float,
    len1_mm: float,
    len2_mm: float,
    dz_mm: float,
    cfg: SpatialConfig,
) -> np.ndarray:
    """Build the (n_time, n_z) current-source-density matrix.

    Correct construction (validated to r≈0.99 vs the analytical/Fourier reference,
    2026-06-19): build the **full bidirectional Vm field** and take its numerical
    spatial derivative(s). The IAP propagates outward from the NMJ at ``posz_mm``
    in both directions, so

        Vm(z, t) = Vm_iap(v·t − |z − posz|)         (windowed to the fibre extent)

    and the CSD is ``∂ⁿVm/∂zⁿ`` (n = ``cfg.csd_derivative``; n=2 is the physical
    current-source density). Taking the derivative *numerically over the whole
    field* — rather than evaluating an analytic kernel per half-fibre — is what
    captures the **source terms at the NMJ junction** (the |·| cusp) and the
    **tendon ends** (the window edges). An earlier per-half / opposite-sign
    construction missed both and scored only r≈0.05–0.2 on the golden set.
    """
    v = cfg.v
    # fibre-end window over the z-grid, split at the NMJ; boxcar = sharp tendon
    # (matches the Fourier `pare` box), tukey = softened (tendon scatter).
    n_fiber_points = max(int((len1_mm + len2_mm) / dz_mm), 2)
    nmj_ratio = len1_mm / (len1_mm + len2_mm)
    win_left, win_right = create_fiber_windows(
        n_fiber_points, nmj_ratio,
        window_type=cfg.fiber_window, tukey_alpha=cfg.tukey_alpha,
    )
    z_in_fiber = z - (posz_mm - len1_mm)
    idx = np.clip((z_in_fiber / dz_mm).astype(int), 0, n_fiber_points - 1)
    win = np.where(z < posz_mm, win_left[idx], win_right[idx])
    win = win * ((z >= posz_mm - len1_mm) & (z <= posz_mm + len2_mm))

    # full bidirectional IAP field (NMJ at posz_mm, wave fronts at posz ± v·t)
    Z = z[None, :]                       # (1, Nz)
    T = t_ms[:, None]                    # (w, 1)
    Vfield = rosenfalck_vm(v * T - np.abs(Z - posz_mm)) * win[None, :]   # (w, Nz)

    # CSD = ∂ⁿVm/∂zⁿ by numerical difference along z (captures junction + ends)
    csd = Vfield
    for _ in range(int(cfg.csd_derivative)):
        csd = np.gradient(csd, dz_mm, axis=1)
    csd *= cfg.sigma_in * np.pi * cfg.fiber_radius_mm**2
    return csd


# ---------------------------------------------------------------------------
# Single-fibre SFAP
# ---------------------------------------------------------------------------

def compute_sfap_spatial(
    phi_z: np.ndarray,
    dz_mm: float,
    len1_mm: float = 60.0,
    len2_mm: float = 60.0,
    posz_mm: float = 0.0,
    config: SpatialConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Single-fibre SFAP via the spatial integral ``(CSD @ φ)``.

    Parameters
    ----------
    phi_z : (Nz,) lead field along the fibre, sampled at ``dz_mm``.
        Assumed centred so the electrode-nearest point is at the array
        centre (z=0); ``posz_mm`` then locates the NMJ relative to z=0.
    dz_mm : spatial step (mm).
    len1_mm, len2_mm : semi-fibre lengths NMJ→proximal / NMJ→distal tendon.
    posz_mm : NMJ position relative to the electrode (z=0). For an FEM φ that
        already encodes the NMJ in its sampling, leave at 0.

    Returns ``(t_ms, sfap, debug)``.
    """
    cfg = config or SpatialConfig()
    phi, dz = _preprocess_phi(phi_z, dz_mm, cfg)
    Nz = len(phi)
    z = (np.arange(Nz) - Nz // 2) * dz
    t_ms = np.arange(cfg.w) / cfg.fsamp * 1000.0 + cfg.t_start_ms

    csd = build_csd_matrix(z, t_ms, posz_mm, len1_mm, len2_mm, dz, cfg)
    sfap = (csd @ phi) * dz * cfg.polarity / cfg.v

    t_out = t_ms - (cfg.w / cfg.fsamp * 1000.0 / 2.0 if cfg.center_time else 0.0)
    debug = {"phi": phi, "dz_mm": dz, "z": z, "csd": csd}
    return t_out, sfap, debug


# ---------------------------------------------------------------------------
# Multi-fibre MUAP (the MotorUnit summation layer)
# ---------------------------------------------------------------------------

@dataclass
class Fibre:
    """One fibre's inputs for the spatial MUAP sum."""
    phi_z: np.ndarray
    dz_mm: float
    len1_mm: float
    len2_mm: float
    posz_mm: float = 0.0
    v: float | None = None               # per-fibre velocity override


def compute_muap_spatial(
    fibres: Sequence[Fibre],
    config: SpatialConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Sum per-fibre SFAPs into a MUAP.

    Returns ``(t_ms, muap, sfaps)`` where ``sfaps`` is the per-fibre stack.
    """
    cfg = config or SpatialConfig()
    sfaps: List[np.ndarray] = []
    t_ms = None
    for fb in fibres:
        c = cfg
        if fb.v is not None and fb.v != cfg.v:
            c = SpatialConfig(**{**cfg.__dict__, "v": fb.v})
        t_ms, s, _ = compute_sfap_spatial(
            fb.phi_z, fb.dz_mm, fb.len1_mm, fb.len2_mm, fb.posz_mm, c)
        sfaps.append(s)
    muap = np.sum(sfaps, axis=0)
    return t_ms, muap, sfaps
