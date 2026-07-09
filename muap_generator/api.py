"""
High-level API for MUAP generation from reciprocal-field data.

Wraps the Fourier pipeline with optimal preprocessing defaults determined
from parameter-sweep experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from muap_generator.conventions import FARINA_DEFAULT, Conventions
from muap_generator.fourier import (
    build_fourier_grids,
    build_spe2_iap_spectrum,
    build_time_vector_ms,
    fftc,
    fiber_field_contribution,
    section_from_field_spectrum,
)
from muap_generator.preprocessing import (
    extract_phi_lines,
    resample_centered_line,
    sample_fibers_in_annulus,
    smooth_butterworth,
    smooth_gaussian,
    smooth_savgol,
    taper_edges,
    upsample_matrix,
)


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------

@dataclass
class MUAPConfig:
    """Full configuration for MUAP generation."""

    # Smoothing
    smoothing_method: Literal["butterworth", "savgol", "gaussian", "none", "auto"] = "butterworth"
    butterworth_cutoff: float = 0.03
    butterworth_order: int = 2
    savgol_window: int = 21
    savgol_polyorder: int = 3
    gaussian_sigma: float = 3.0
    # When smoothing_method=="auto", apply Butterworth iff the input phi's
    # high-frequency fraction exceeds this threshold. Analytical phi has HF
    # near machine epsilon (~1e-30); FEM phi has HF in the 1e-6 .. 1e-4 range.
    # 1e-10 cleanly separates them.
    auto_smoothing_hf_threshold: float = 1e-10
    auto_smoothing_hf_freq: float = 0.3  # normalised frequency cutoff for HF

    # Upsampling
    upsample_factor: int = 1

    # Pipeline
    v: float = 4.0       # conduction velocity (m/s)
    fsamp: float = 4096.0  # sampling frequency (Hz)
    w: Optional[int] = 256
    # w==None triggers adaptive selection via muap_generator.adaptive_w.choose_w
    # using the input phi(z) tail-fit + L_fibre + dz. Pass an integer to lock the
    # window for backward compatibility.
    apply_z_window: bool = False  # Hanning window on phi(z) before FFT

    # Adaptive-w bounds (only used when w=None).
    w_min: int = 256
    w_max: int = 1024

    # Fibre geometry
    len1_mm: float = 60.0
    len2_mm: float = 60.0

    # Edge taper (for truncated lead fields — FEM/MRI data).
    # Either an int (taper that many samples, 0=off) or "auto" — apply
    # ``auto_edge_taper_n`` samples iff the input φ has not decayed at its
    # boundary (``_edge_over_peak(phi) >= auto_edge_taper_threshold``).
    # Default 0 preserves backward compatibility.
    edge_taper: "int | str" = 0
    # When edge_taper=="auto", taper this many samples if the truncation
    # signal exceeds auto_edge_taper_threshold. Defaults are picked from
    # the verification spot-check: analytical φ has edge/peak < 0.01,
    # FEM offgrid_complex φ has edge/peak ≈ 1.0 — five orders of
    # magnitude separation, like the auto-smoothing threshold.
    auto_edge_taper_threshold: float = 0.3
    auto_edge_taper_n: int = 15

    # Fibre sampling (for NPZ workflow)
    n_fibers: int = 50
    r_min: int = 10
    r_max: int = 35
    seed: int = 42

    # Per-fibre physiological jitter (added after Neurodec comparison —
    # zero scatter gives 4× too-short MUAP durations and 2× too-high f_dom).
    # σ=0 → no jitter, identical to legacy behaviour (regression-safe).
    nmj_sigma_mm: float = 0.0           # per-fibre NMJ-position scatter (typical 5-15 mm)
    cv_sigma_m_per_s: float = 0.0       # per-fibre conduction-velocity scatter (typical 0.2-0.5 m/s)
    tendon_sigma_mm: float = 0.0        # per-fibre ASYMMETRIC tendon-length scatter; L1 and L2 drawn independently
    fiber_length_sigma_mm: float = 0.0  # per-fibre SYMMETRIC fibre-length scatter; same Δ added to both L1 and L2 (preserves L1:L2 ratio, varies the fibre-to-tendon ratio)
    jitter_seed: int = 42

    # Electrode position along the muscle's z-axis (mm), with the IZ at z=0.
    # Scalar → one electrode → one MUAP (default, backward-compatible).
    # 1-D array → multi-electrode waterfall → MUAPResult.muap has shape
    # (n_electrodes, n_time).
    #
    # WHAT IT ACTUALLY DOES (verified empirically 2026-05-29):
    # `radon_section`'s third arg applies a Fourier phase factor
    # `exp(j·k_z·z_det)` to the lead-field spectrum before summing over
    # k_z. Empirically this produces:
    #   - AMPLITUDE modulation (electrode at z=20 mm sees ~0.84× the
    #     IZ-overlying value on a 14 mm-deep PL fibre; ratio depends on
    #     fibre depth and L1/L2 envelopes).
    #   - small SHAPE perturbation (Pearson r ≈ 0.985 between an
    #     amplitude-scaled z=0 MUAP and the true z=20 mm MUAP).
    #
    # WHAT IT DOES *NOT* DO:
    #   - No time-of-flight delay. The radon centring forces every
    #     electrode's MUAP to sit at t = w/(2·fsamp). Tested with
    #     SignalGenerator at z={-10, 0, +10}: peak index is identical
    #     across channels — only amplitude differs.
    #   - So no propagating-wave diagonal in a waterfall directly from
    #     this knob. To visualise propagation, the caller must post-hoc
    #     time-shift each channel by `(z_det − z_IZ) / v`.
    #
    # ANALYTICAL vs FEM:
    #   - Analytical infinite cylinder: z_det_mm sweeps lead-field
    #     amplitude correctly (the source-electrode pair is translation-
    #     invariant by symmetry). One phi(z) covers all electrodes.
    #   - FEM finite mesh: the translation-equivalence breaks near the
    #     end caps and is only approximate in the bulk. For accurate
    #     multi-electrode FEM you should solve FEM PER ELECTRODE; using
    #     z_det_mm on a single-electrode FEM phi is a rough surrogate.
    z_det_mm: "float | np.ndarray" = 0.0

    # Sign/timing conventions (C1/C2/C5/C6 + polarity). Default
    # FARINA_DEFAULT reproduces the validated, regression-frozen behaviour
    # byte-for-byte. Pass muap_generator.conventions.FEM_NEURODEC for the
    # FEM-vs-Neurodec convention (polarity=−1; pair with posz=0 on the FEM
    # path). See muap_generator/conventions.py.
    conventions: Conventions = FARINA_DEFAULT

    def __post_init__(self):
        # Tolerate a plain dict for `conventions` (e.g. from a
        # dataclasses.asdict round-trip or a JSON/YAML config) by coercing it
        # back into a Conventions instance.
        if isinstance(self.conventions, dict):
            self.conventions = Conventions(**self.conventions)


@dataclass
class MUAPResult:
    """Container returned by the generation functions."""

    t_ms: np.ndarray
    muap: np.ndarray
    fiber_positions: Tuple[np.ndarray, np.ndarray]
    config: MUAPConfig
    metrics: Dict[str, float] = field(default_factory=dict)

    def plot(self, ax=None, **kwargs):
        """Quick plot of the MUAP waveform."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.t_ms, self.muap, **kwargs)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title("MUAP Waveform")
        ax.grid(True, alpha=0.3)
        return ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _edge_over_peak(phi_z: np.ndarray) -> float:
    """Truncation signal — large when |φ| has not decayed at its boundary.

    The original Phase 3 sign-flip in `field_to_muap_study` was localised to
    electrodes whose lead field never decays within the sampling window;
    `j·kz` then rings on the zero-padded step and produces a spurious
    inverted MUAP. `edge_taper` removes the step. This helper exposes the
    signal so we can detect when to apply the taper automatically.

    Returns ``max(|φ[0]|, |φ[-1]|) / max(|φ|)``. Analytical / well-decayed
    fields return ~1e-3; truncated fields return ~1.0 — five orders of
    magnitude separation makes the 0.3 default threshold comfortable.
    """
    p = np.abs(np.asarray(phi_z, dtype=float))
    peak = float(p.max())
    if peak <= 0:
        return 0.0
    if p.ndim > 1:
        edge = float(max(p[..., 0].max(), p[..., -1].max()))
    else:
        edge = float(max(p[0], p[-1]))
    return edge / peak


def _hf_fraction(phi_z: np.ndarray, hf_freq: float = 0.3) -> float:
    """HF energy fraction above ``hf_freq`` (normalised, 0..0.5).

    A clean analytical lead field returns ~1e-30; a Gaussian-source FEM
    lead field returns ~1e-6 .. 1e-4. The two are easy to distinguish.
    """
    x = np.asarray(phi_z, dtype=float)
    x = x - x.mean()
    n = x.shape[-1]
    if n < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(x, axis=-1))
    f = np.fft.rfftfreq(n)
    total = float((spec ** 2).sum())
    if total <= 0:
        return 0.0
    hf = float((spec[..., f > hf_freq] ** 2).sum())
    return hf / total


def _apply_smoothing(phi_mat: np.ndarray, config: MUAPConfig) -> np.ndarray:
    if config.smoothing_method == "butterworth":
        return smooth_butterworth(phi_mat, config.butterworth_cutoff, config.butterworth_order)
    if config.smoothing_method == "savgol":
        return smooth_savgol(phi_mat, config.savgol_window, config.savgol_polyorder)
    if config.smoothing_method == "gaussian":
        return smooth_gaussian(phi_mat, config.gaussian_sigma)
    if config.smoothing_method == "auto":
        # Detect HF on the first fibre as representative; all rows share dz.
        first = phi_mat[0] if phi_mat.ndim == 2 else phi_mat
        hf = _hf_fraction(first, config.auto_smoothing_hf_freq)
        if hf >= config.auto_smoothing_hf_threshold:
            return smooth_butterworth(
                phi_mat, config.butterworth_cutoff, config.butterworth_order
            )
        return phi_mat.copy()
    return phi_mat.copy()


def _compute_muap_core(
    phi_mat: np.ndarray,
    dz_mm: float,
    config: MUAPConfig,
    len1_mm_arr: np.ndarray,
    len2_mm_arr: np.ndarray,
    posz_mm_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core Fourier MUAP from an already-smoothed phi matrix."""
    v, fsamp = config.v, config.fsamp
    if config.w is None:
        from muap_generator.adaptive_w import choose_w
        L_fibre = float(len1_mm_arr[0] + len2_mm_arr[0])
        # Use the first fibre's phi as representative for decay estimation.
        # All rows share the same dz, so this is a reasonable proxy.
        w = choose_w(
            L_fibre_mm=L_fibre,
            phi_z=phi_mat[0],
            dz_mm=dz_mm,
            w_min=config.w_min,
            w_max=config.w_max,
            verbose=False,
        )
    else:
        w = int(config.w)
    Nfib = phi_mat.shape[0]

    # Resolve edge_taper. Like auto-smoothing, the decision is made once on
    # the first fibre's φ — all rows share dz and the truncation property
    # is geometry-driven, not fibre-specific.
    if isinstance(config.edge_taper, str) and config.edge_taper == "auto":
        first = phi_mat[0] if phi_mat.ndim == 2 else phi_mat
        eop = _edge_over_peak(first)
        n_taper = int(config.auto_edge_taper_n) if eop >= config.auto_edge_taper_threshold else 0
    elif isinstance(config.edge_taper, str):
        raise ValueError(
            f"edge_taper must be an int or 'auto', got {config.edge_taper!r}"
        )
    else:
        n_taper = int(config.edge_taper)

    if config.upsample_factor > 1:
        phi_mat = upsample_matrix(phi_mat, config.upsample_factor)
        dz_mm = dz_mm / config.upsample_factor

    cv = config.conventions
    zstep_expected = v * 1000.0 / fsamp
    grids = build_fourier_grids(w=w, fsamp=fsamp, v=v)
    spe2, _ = build_spe2_iap_spectrum(w=w, fsamp=fsamp, v=v, iap_flip=cv.iap_flip)

    E_mu = np.zeros((w, w), dtype=complex)

    for u in range(Nfib):
        phi_u = phi_mat[u, :]

        # Edge taper: prevent Gibbs ringing from truncated lead fields.
        # n_taper was resolved above (handles config.edge_taper="auto").
        if n_taper > 0:
            phi_u = taper_edges(phi_u, n_taper=n_taper)

        if abs(dz_mm - zstep_expected) > 1e-9:
            phi_u = resample_centered_line(
                phi_u, delta_s_mm=dz_mm, w_out=w, delta_s_out_mm=zstep_expected,
            )
            delta_used = zstep_expected
        else:
            if phi_u.shape[0] != w:
                phi_u = resample_centered_line(
                    phi_u, delta_s_mm=dz_mm, w_out=w, delta_s_out_mm=dz_mm,
                )
            delta_used = dz_mm

        if config.apply_z_window:
            win = np.hanning(len(phi_u))
            C_u = delta_used * fftc(phi_u * win)
        else:
            C_u = delta_used * fftc(phi_u)

        # Per-fibre field contribution via the shared canonical helper
        # (pare = C2 sign convention, posz phase = C3). config.z_det_mm may be
        # a scalar (single electrode, legacy 1-D output) or a 1-D array of
        # electrode z-positions (multi-electrode waterfall) — handled below.
        E_mu += fiber_field_contribution(
            C_u, len1_mm_arr[u], len2_mm_arr[u], posz_mm_arr[u], grids,
            swap_ends=cv.swap_ends,
        )

    muap = section_from_field_spectrum(
        E_mu, spe2, v, grids, config.z_det_mm,
        output_flip=cv.output_flip, polarity=cv.polarity,
    )
    t_ms = build_time_vector_ms(w=w, fsamp=fsamp)
    if cv.center_time:  # convention C6
        t_ms = t_ms - (w / fsamp) * 1000.0 / 2
    return t_ms, muap


def _compute_muap_per_fiber_summation(
    phi_mat: np.ndarray,
    dz_mm: float,
    config: MUAPConfig,
    vs_per_fiber: np.ndarray,
    len1_per_fiber: np.ndarray,
    len2_per_fiber: np.ndarray,
    posz_per_fiber: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slow path: loop over fibres when per-fibre `v` differs.

    Required when ``cv_sigma_m_per_s > 0``: each fibre needs its own
    kt/kalpha/kbeta grids and SPE2 spectrum (all depend on v). Per-fibre
    NMJ + tendon scatter alone don't need this path (they're consumed
    inside the existing `_compute_muap_core` per-fibre loop).
    """
    from dataclasses import replace

    n_fib = phi_mat.shape[0]
    t_common = None
    muap_sum = None

    for fi in range(n_fib):
        cfg_fi = replace(
            config,
            v=float(vs_per_fiber[fi]),
            len1_mm=float(len1_per_fiber[fi]),
            len2_mm=float(len2_per_fiber[fi]),
        )
        phi_fi = phi_mat[fi:fi + 1]
        phi_smooth = _apply_smoothing(phi_fi, cfg_fi)
        t_ms, m_fi = _compute_muap_core(
            phi_smooth, dz_mm, cfg_fi,
            np.array([len1_per_fiber[fi]]),
            np.array([len2_per_fiber[fi]]),
            np.array([posz_per_fiber[fi]]),
        )
        if t_common is None:
            t_common = t_ms
            muap_sum = m_fi.copy()
        else:
            muap_sum += m_fi

    return t_common, muap_sum


def _resolve_per_fiber_jitter(
    config: MUAPConfig,
    Nfib: int,
    posz_mm_arr: Optional[np.ndarray],
    vs_per_fiber: Optional[np.ndarray],
    len1_per_fiber: Optional[np.ndarray],
    len2_per_fiber: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, bool]:
    """Resolve per-fibre arrays from explicit args + config-level σ.

    Explicit per-fibre arrays win; otherwise fall back to N(mean, σ) draws
    using the σ values in config. Returns (posz, vs_or_None, len1, len2,
    needs_per_fiber_path) — `needs_per_fiber_path` is True iff per-fibre
    `v` is in play.
    """
    rng = np.random.default_rng(int(config.jitter_seed))

    if posz_mm_arr is None:
        if config.nmj_sigma_mm > 0.0:
            posz_mm_arr = rng.normal(0.0, float(config.nmj_sigma_mm), Nfib)
        else:
            posz_mm_arr = np.zeros(Nfib)
    else:
        posz_mm_arr = np.asarray(posz_mm_arr, dtype=float)
        if posz_mm_arr.shape[0] != Nfib:
            raise ValueError(
                f"posz_mm_arr length {posz_mm_arr.shape[0]} != n_fibres {Nfib}"
            )

    if vs_per_fiber is None and config.cv_sigma_m_per_s > 0.0:
        vs_per_fiber = rng.normal(
            float(config.v), float(config.cv_sigma_m_per_s), Nfib
        )
        # Clip to physiological range (1.5-7 m/s) to avoid pathological draws.
        vs_per_fiber = np.clip(vs_per_fiber, 1.5, 7.0)
    elif vs_per_fiber is not None:
        vs_per_fiber = np.asarray(vs_per_fiber, dtype=float)
        if vs_per_fiber.shape[0] != Nfib:
            raise ValueError(
                f"vs_per_fiber length {vs_per_fiber.shape[0]} != n_fibres {Nfib}"
            )

    # Per-fibre lengths: compose asymmetric (tendon) + symmetric (fibre-length)
    # jitters. Both σ default to 0 — legacy behaviour preserved.
    if len1_per_fiber is None or len2_per_fiber is None:
        # Symmetric fibre-length component (shared draw for both sides)
        if config.fiber_length_sigma_mm > 0.0:
            d_sym = rng.normal(0.0, float(config.fiber_length_sigma_mm), Nfib)
        else:
            d_sym = np.zeros(Nfib)
        # Asymmetric tendon-end component (independent draws per side)
        if config.tendon_sigma_mm > 0.0:
            d_asym_1 = rng.normal(0.0, float(config.tendon_sigma_mm), Nfib)
            d_asym_2 = rng.normal(0.0, float(config.tendon_sigma_mm), Nfib)
        else:
            d_asym_1 = np.zeros(Nfib)
            d_asym_2 = np.zeros(Nfib)

    if len1_per_fiber is None:
        len1_per_fiber = np.full(Nfib, float(config.len1_mm)) + d_sym + d_asym_1
        len1_per_fiber = np.maximum(len1_per_fiber, 10.0)
    else:
        len1_per_fiber = np.asarray(len1_per_fiber, dtype=float)

    if len2_per_fiber is None:
        len2_per_fiber = np.full(Nfib, float(config.len2_mm)) + d_sym + d_asym_2
        len2_per_fiber = np.maximum(len2_per_fiber, 10.0)
    else:
        len2_per_fiber = np.asarray(len2_per_fiber, dtype=float)

    needs_per_fiber_path = vs_per_fiber is not None
    return posz_mm_arr, vs_per_fiber, len1_per_fiber, len2_per_fiber, needs_per_fiber_path


def _compute_metrics(t_ms: np.ndarray, muap: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # Skip per-MUAP metrics for multi-electrode output — caller can compute
    # per-channel metrics themselves from the (n_elec, n_time) array.
    if muap.ndim != 1:
        metrics["n_channels"] = int(muap.shape[0])
        metrics["n_time"]     = int(muap.shape[-1])
        metrics["peak_to_peak_max"] = float(np.max(muap) - np.min(muap))
        return metrics
    dt_ms = t_ms[1] - t_ms[0] if len(t_ms) > 1 else 1.0
    metrics["peak_to_peak"] = float(np.max(muap) - np.min(muap))
    metrics["rms_amplitude"] = float(np.sqrt(np.mean(muap**2)))

    if len(muap) > 2:
        d2 = (muap[2:] - 2 * muap[1:-1] + muap[:-2]) / (dt_ms**2)
        metrics["roughness"] = float(np.mean(d2**2))

    fft_muap = np.fft.rfft(muap)
    freqs = np.fft.rfftfreq(len(muap), d=dt_ms / 1000)
    power = np.abs(fft_muap) ** 2
    total_power = np.sum(power)
    if total_power > 0:
        metrics["hf_ratio"] = float(np.sum(power[freqs > 500]) / total_power)

    threshold = 0.1 * metrics["peak_to_peak"]
    above = np.abs(muap) > threshold
    if np.any(above):
        first = int(np.argmax(above))
        last = len(above) - 1 - int(np.argmax(above[::-1]))
        metrics["duration_ms"] = float((last - first) * dt_ms)

    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_muap_from_phi(
    phi_mat: np.ndarray,
    dz_mm: float,
    config: Optional[MUAPConfig] = None,
    posz_mm_arr: Optional[np.ndarray] = None,
    vs_per_fiber: Optional[np.ndarray] = None,
    len1_per_fiber: Optional[np.ndarray] = None,
    len2_per_fiber: Optional[np.ndarray] = None,
) -> MUAPResult:
    """Generate a MUAP from a matrix of φ(z) lines (one per fibre).

    Parameters
    ----------
    phi_mat : (Nfib, Nz) array
    dz_mm : spatial step (mm)
    config : optional configuration
    posz_mm_arr : optional per-fibre NMJ offsets (mm). If None and
        ``config.nmj_sigma_mm > 0``, drawn from N(0, σ).
    vs_per_fiber : optional per-fibre conduction velocity (m/s). If None
        and ``config.cv_sigma_m_per_s > 0``, drawn from N(config.v, σ).
        Triggers the slow per-fibre summation path.
    len1_per_fiber, len2_per_fiber : optional per-fibre proximal/distal
        tendon lengths (mm). If None and ``config.tendon_sigma_mm > 0``,
        drawn from N(config.len{1,2}_mm, σ).

    Notes
    -----
    With all four args None and config σ values at 0 (default), behaviour
    is byte-identical to the pre-jitter API — regression bench safe.

    When `vs_per_fiber` is provided (or implied by `cv_sigma_m_per_s > 0`),
    the slow per-fibre summation path runs (one FFT per fibre instead of
    one shared FFT). Roughly N× the runtime for an N-fibre MU.
    """
    if config is None:
        config = MUAPConfig()

    Nfib = phi_mat.shape[0]

    # Resolve per-fibre arrays (drawing jitter if needed).
    posz, vs, len1, len2, needs_per_fiber_path = _resolve_per_fiber_jitter(
        config, Nfib,
        posz_mm_arr, vs_per_fiber, len1_per_fiber, len2_per_fiber,
    )

    # If adaptive `w` is requested, decide BEFORE smoothing so the decay-length
    # fit uses the raw lead field. Smoothing slightly inflates the apparent λ
    # which can flip the chosen w one step higher than intended.
    if config.w is None:
        from muap_generator.adaptive_w import choose_w
        w_chosen = choose_w(
            L_fibre_mm=float(config.len1_mm + config.len2_mm),
            phi_z=phi_mat[0],
            dz_mm=float(dz_mm),
            w_min=int(config.w_min),
            w_max=int(config.w_max),
            verbose=False,
        )
        # Pin the chosen w into a local config copy for the rest of the call.
        from dataclasses import replace
        config = replace(config, w=w_chosen)

    if needs_per_fiber_path:
        # Slow path: rebuild kt/kalpha/kbeta + SPE2 per fibre because v varies.
        t_ms, muap = _compute_muap_per_fiber_summation(
            phi_mat, dz_mm, config, vs, len1, len2, posz,
        )
    else:
        # Fast path: shared grids, vectorised over fibres.
        phi_smooth = _apply_smoothing(phi_mat, config)
        t_ms, muap = _compute_muap_core(phi_smooth, dz_mm, config, len1, len2, posz)

    metrics = _compute_metrics(t_ms, muap)

    return MUAPResult(
        t_ms=t_ms,
        muap=muap,
        fiber_positions=(np.array([]), np.array([])),
        config=config,
        metrics=metrics,
    )


def generate_muap_from_npz(
    npz_path: str,
    n_fibers: int = 50,
    config: Optional[MUAPConfig] = None,
    use_prediction: bool = False,
) -> MUAPResult:
    """Generate a MUAP from an NPZ file with ``target`` / ``prediction`` fields.

    Parameters
    ----------
    npz_path : path to ``.npz`` file.
    n_fibers : number of fibres to sample.
    config : optional configuration.
    use_prediction : if True use ``prediction`` key, else ``target``.
    """
    if config is None:
        config = MUAPConfig()

    d = np.load(npz_path)
    field_key = "prediction" if use_prediction else "target"
    field = d[field_key][0, 0]  # (Z, Y, X)
    spacing = d["spacing"][0]
    dz_mm = float(spacing[2])

    xs, ys = sample_fibers_in_annulus(
        n_fibers, field.shape,
        r_min=config.r_min, r_max=config.r_max, seed=config.seed,
    )
    phi_mat = extract_phi_lines(field, xs, ys)

    result = generate_muap_from_phi(phi_mat, dz_mm, config)
    result.fiber_positions = (xs, ys)
    return result


def generate_muaps_batch(
    npz_paths: List[str],
    n_fibers: int = 50,
    config: Optional[MUAPConfig] = None,
    use_prediction: bool = False,
) -> List[MUAPResult]:
    """Generate MUAPs from multiple NPZ files."""
    results = []
    for path in npz_paths:
        try:
            results.append(generate_muap_from_npz(path, n_fibers, config, use_prediction))
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return results


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def get_optimal_config() -> MUAPConfig:
    """Optimal settings from parameter-sweep experiments (muap_smoothness study).

    Butterworth c=0.03 o=2 achieves mean r=0.990 against the Farina 2004
    analytical model across 21 depths (vs r=0.894 for the old c=0.10 o=4).
    No upsampling needed — it has negligible effect with proper smoothing.
    """
    return MUAPConfig(smoothing_method="butterworth", butterworth_cutoff=0.03, butterworth_order=2, upsample_factor=1)


def get_fast_config() -> MUAPConfig:
    """Fast settings (Savitzky-Golay, no upsampling)."""
    return MUAPConfig(smoothing_method="savgol", savgol_window=21, upsample_factor=1)


def get_high_quality_config() -> MUAPConfig:
    """High-quality settings (same as optimal — upsampling has no benefit)."""
    return MUAPConfig(smoothing_method="butterworth", butterworth_cutoff=0.03, butterworth_order=2, upsample_factor=1)


def get_adaptive_config(w_min: int = 256, w_max: int = 1024) -> MUAPConfig:
    """All-auto preset — recommended for new code.

    Two adaptive behaviours combine, each detecting a property of the
    input φ and conditionally applying its treatment:

    1. **Window `w`** (Workstream A): picked via
       ``muap_generator.adaptive_w.choose_w`` from the input lead field's
       decay length and fibre length, bounded by ``[w_min, w_max]``. A
       defensive cap prevents adapting above the input φ's natural extent
       (FEM_AND_MESH_GUIDE.md §11).

    2. **Smoothing** (Workstream A.2): ``smoothing_method="auto"`` applies
       Butterworth (c=0.03, o=2) iff the input φ has measurable HF
       content above ``auto_smoothing_hf_threshold`` (default 1e-10).
       Analytical inputs (HF ≈ 1e-30) skip smoothing → operator
       consistency r → 1.0. FEM inputs (HF ≈ 1e-5) get smoothed.

    **Note on edge_taper (Round 3, May 2026)**: ``edge_taper="auto"``
    was *removed* from this preset after the discovery that the
    spectral-leakage signature it was treating actually came from a
    numerical bug in ``resample_centered_line`` (zero-pad → step
    discontinuity at the spatial edge of phi). The fix
    (``pad_mode="edge"`` in ``resample_centered_line``) eliminates the
    leakage at its source. With the fix in place, ``edge_taper="auto"``
    is redundant — and on some MRI cases it actively distorts the
    signal by subtracting real lead-field information at the edges.
    Both the 200-case cylinder regression bench AND the Phase 3
    cross-geometry sign-flip bench now pass at 0% flips with
    ``edge_taper=0``. See ``progress_reports/adhoc/05`` and ``10``.

    Backwards-compatible with the legacy ``MUAPConfig()`` default
    (w=256, butterworth, edge_taper=0) — opting in is a deliberate
    per-caller choice.
    """
    return MUAPConfig(
        smoothing_method="auto",
        butterworth_cutoff=0.03, butterworth_order=2,
        upsample_factor=1,
        w=None, w_min=w_min, w_max=w_max,
        edge_taper=0,
    )


def get_mri_config() -> MUAPConfig:
    """Recommended preset for MRI volume-conductor inputs.

    MRI lead fields phi(z) along anatomically-constrained muscle fibres
    routinely have **edge/peak = 1.0** — the fibre is much shorter than
    the lead field's natural decay length, so phi is at maximum
    magnitude at the fibre endpoints. The downstream FFT-based SFAP
    pipeline then suffers from spectral leakage at the periodic-wrap
    (phi[-1] → phi[0]) even after the ``pad_mode="edge"`` fix to
    ``resample_centered_line``.

    `edge_taper="auto"` ramps the first/last 15 samples to zero
    *whenever* edge/peak ≥ 0.3 (the auto threshold). On MRI cases this
    removes ~85% of the spurious HF content above 500 Hz without
    distorting the biphasic SFAP shape.

    Cylinder cases (the 200-case regression bench and Phase 3
    cross-geometry bench) have edge/peak < 0.1 → the auto-taper does
    not fire → this preset is byte-equivalent to
    ``get_adaptive_config()`` on those inputs. The branch separation
    is conceptually motivated: MRI users opt in via
    ``get_mri_config()``; cylinder / analytical users get the cleaner
    ``get_adaptive_config()``.

    See ``progress_reports/adhoc/10`` for the round-3 HF investigation
    that motivated this preset.
    """
    return MUAPConfig(
        smoothing_method="butterworth",
        butterworth_cutoff=0.03, butterworth_order=2,
        upsample_factor=1, w=256,
        edge_taper="auto",
        auto_edge_taper_threshold=0.3, auto_edge_taper_n=15,
    )


def get_realistic_config(
    nmj_sigma_mm: float = 10.0,
    cv_sigma_m_per_s: float = 0.3,
    tendon_sigma_mm: float = 5.0,
    jitter_seed: int = 42,
) -> MUAPConfig:
    """Preset enabling physiological per-fibre scatter.

    Use this when you want the MUAP shape to reflect real motor-unit
    biology rather than a synchronous-fibre idealisation. Defaults:

    - **σ_NMJ = 10 mm**: per-fibre innervation-zone position scatter
      (Stålberg & Trontelj, typical range 5–15 mm).
    - **σ_CV = 0.3 m/s**: per-fibre conduction-velocity scatter (around
      a 4 m/s mean). Empirical range 0.2–0.5 m/s for healthy motor units.
    - **σ_tendon = 5 mm**: per-fibre proximal/distal tendon-length
      scatter (≈10% of a 50 mm half-fibre). Tendons converge over a
      finite distance, not at a single z.

    Motivation: zero scatter (legacy default) gives MUAPs 4× too short
    and 2× too high f_dom relative to Neurodec's MRI-FEM reference. With
    these defaults, the per-fibre summation broadens the SFAP envelope to
    physiological 25–40 ms durations.

    The slow per-fibre summation path is required because cv_sigma > 0
    forces per-fibre kt/kalpha/kbeta grids. Expected runtime: ~N × the
    fast path, where N is the fibre count.

    Returns the same Butterworth c=0.03 o=2 smoothing as
    ``get_optimal_config()`` so accuracy on analytical inputs is
    preserved.
    """
    return MUAPConfig(
        smoothing_method="butterworth",
        butterworth_cutoff=0.03, butterworth_order=2,
        upsample_factor=1,
        nmj_sigma_mm=nmj_sigma_mm,
        cv_sigma_m_per_s=cv_sigma_m_per_s,
        tendon_sigma_mm=tendon_sigma_mm,
        jitter_seed=jitter_seed,
    )


def get_truncated_input_config() -> MUAPConfig:
    """DEPRECATED — use ``get_adaptive_config()`` instead.

    Was: fixed-w pipeline with ``edge_taper=15`` for callers who knew
    their input was truncated. As of Round 2, ``get_adaptive_config()``
    detects truncation automatically and applies the taper exactly where
    this preset would have, so this preset is obsolete.

    Kept for backwards compatibility. Will be removed in a future
    release.
    """
    import warnings
    warnings.warn(
        "get_truncated_input_config() is deprecated; "
        "get_adaptive_config() now auto-detects truncation. "
        "Switch your callers to get_adaptive_config().",
        DeprecationWarning, stacklevel=2,
    )
    return MUAPConfig(
        smoothing_method="butterworth",
        butterworth_cutoff=0.03, butterworth_order=2,
        upsample_factor=1,
        w=256,
        edge_taper=15,
    )
