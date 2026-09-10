"""Waveform-comparison metrics for MUAP / SFAP outputs.

Salvaged from ``scripts/synthesis/compare_spatial_vs_fourier.py`` (A-07): the
shape / alignment / end-of-fibre metrics used to score one waveform against
another — engine vs engine, or an engine vs the Neurodec ground truth. They take
a ``(t, waveform)`` pair each, so they are agnostic to which engine produced it.

``lobe_metrics``' *pos-after-trough* is the only quantitative **EOF** (end-of-fibre,
the trailing lobe) metric in the repo — Phase C's MUAP-shape questions need it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def normalize_peak(x: np.ndarray) -> np.ndarray:
    """``x`` scaled so ``max|x| = 1`` (returned unchanged if ``x`` is all-zero)."""
    m = np.abs(x).max()
    return x / m if m > 0 else x


def align_score(t_ref, ref, t_b, b, max_lag_ms: float = 12.0, dt: float = 0.05):
    """Best peak-aligned Pearson r of ``b`` onto ``ref``. Returns ``(shift_ms, r)``.

    Both waveforms are peak-normalised; the shift is seeded at the extremum offset
    and then refined by scanning ``±max_lag_ms`` at ``dt`` resolution for the lag
    that maximises r. ``r`` is signed — a polarity flip reads as a negative score,
    not a pass.
    """
    rn, bn = normalize_peak(ref), normalize_peak(b)
    coarse = t_ref[np.argmax(np.abs(rn))] - t_b[np.argmax(np.abs(bn))]
    grid = np.arange(t_ref.min(), t_ref.max(), dt)
    rg = np.interp(grid, t_ref, rn, left=0, right=0)
    best_r, best_s = -2.0, coarse
    for d in np.arange(-max_lag_ms, max_lag_ms + dt, dt):
        sg = np.interp(grid, t_b + coarse + d, bn, left=0, right=0)
        if sg.std() < 1e-12:
            continue
        r = pearsonr(rg, sg)[0]
        if r > best_r:
            best_r, best_s = r, coarse + d
    return best_s, best_r


def nrmse_aligned(t_ref, ref, t_b, b) -> float:
    """RMSE of the peak-aligned, peak-normalised ``b`` against ``ref``."""
    shift, _ = align_score(t_ref, ref, t_b, b)
    rn = normalize_peak(ref)
    bi = np.interp(t_ref, t_b + shift, normalize_peak(b), left=0, right=0)
    return float(np.sqrt(np.mean((rn - bi) ** 2)))


def raw_r_vs(t, m, t_g, g) -> float:
    """Pearson r on the overlapping time window — no alignment, no normalisation."""
    lo, hi = max(t.min(), t_g.min()), min(t.max(), t_g.max())
    msk = (t >= lo) & (t <= hi)
    gi = np.interp(t[msk], t_g, g)
    return float(pearsonr(m[msk], gi)[0])


def lobe_metrics(t, m, win_ms: float = 12.0):
    """``(trough_ms, pos_before, pos_after)`` around the global min of peak-normalised ``m``.

    ``pos_after`` is the largest positive excursion in the ``win_ms`` window *after*
    the trough — the trailing end-of-fibre lobe, the repo's only quantitative EOF
    measure. ``pos_before`` is the same for the window before the trough.
    """
    mn = normalize_peak(m)
    ti = int(np.argmin(mn))
    dt = t[1] - t[0]
    win = max(int(win_ms / dt), 1)
    before = mn[max(0, ti - win):ti].max() if ti > 0 else 0.0
    after = mn[ti:ti + win].max()
    return float(t[ti]), float(before), float(after)


def jaggedness(m) -> float:
    """Mean ``|2nd difference|`` normalised by peak-to-peak — a smoothness proxy.

    High when a waveform carries Nyquist-scale zigzag (e.g. an under-resolved CSD
    integral); ``0`` for a straight line. Returns ``0`` for a flat waveform.
    """
    span = float(np.ptp(m))
    return float(np.mean(np.abs(np.diff(m, 2))) / span) if span > 0 else 0.0


def waveform_features(
    t_ms,
    waveform,
    *,
    active_threshold: float = 0.05,
    tail_fraction: float = 0.1,
    hf_cutoff_hz: float = 500.0,
) -> dict[str, float | int]:
    """Return morphology features for one SFAP or MUAP waveform.

    The features deliberately describe the waveform instead of deciding whether it is
    physiological. Absolute amplitude, duration and phase-count limits depend strongly
    on electrode montage, filtering, fibre depth and motor-unit anatomy. Callers can
    compare these values across controlled parameter sweeps or against an experimental
    reference population without baking one recording protocol into the library.

    Parameters
    ----------
    t_ms, waveform
        Matching one-dimensional arrays. ``t_ms`` must be finite, uniformly sampled
        and strictly increasing; ``waveform`` must be finite.
    active_threshold
        Fraction of ``max(abs(waveform))`` used to measure active duration and phases.
    tail_fraction
        Fraction at each end of the record used for the boundary-energy diagnostic.
    hf_cutoff_hz
        Frequency above which ``hf_ratio`` is computed.

    Notes
    -----
    ``dc_area_ratio = |integral(x)| / integral(|x|)`` is close to zero for a balanced
    multi-phase transient and one for a one-signed pulse. It is invariant to amplitude
    and polarity. ``tail_energy_ratio`` exposes clipped or periodically wrapped MUAPs.
    Neither quantity is a universal pass/fail threshold.
    """
    t = np.asarray(t_ms, dtype=float)
    x = np.asarray(waveform, dtype=float)
    if t.ndim != 1 or x.ndim != 1 or t.shape != x.shape:
        raise ValueError("t_ms and waveform must be matching one-dimensional arrays")
    if t.size < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError("t_ms and waveform must contain only finite values")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("t_ms must be strictly increasing")
    dt_ms = float(np.median(dt))
    if not np.allclose(dt, dt_ms, rtol=1e-6, atol=1e-12):
        raise ValueError("t_ms must be uniformly sampled")
    if not 0.0 < active_threshold < 1.0:
        raise ValueError("active_threshold must lie in (0, 1)")
    if not 0.0 < tail_fraction <= 0.5:
        raise ValueError("tail_fraction must lie in (0, 0.5]")
    if not np.isfinite(hf_cutoff_hz) or hf_cutoff_hz <= 0.0:
        raise ValueError("hf_cutoff_hz must be finite and positive")

    dt_s = dt_ms / 1000.0
    peak_abs = float(np.max(np.abs(x)))
    peak_to_peak = float(np.ptp(x))
    rms = float(np.sqrt(np.mean(x**2)))

    # Trapezoidal integration without np.trapz / np.trapezoid keeps compatibility
    # across NumPy 1.x and 2.x.
    signed_area = float(np.sum(0.5 * (x[:-1] + x[1:]) * dt))
    absolute_area = float(np.sum(0.5 * (np.abs(x[:-1]) + np.abs(x[1:])) * dt))
    dc_area_ratio = abs(signed_area) / absolute_area if absolute_area > 0 else 0.0

    energy = float(np.sum(x**2))
    n_tail = max(1, int(np.ceil(tail_fraction * x.size)))
    tail_mask = np.zeros(x.size, dtype=bool)
    tail_mask[:n_tail] = True
    tail_mask[-n_tail:] = True
    tail_energy = float(np.sum(x[tail_mask] ** 2))
    tail_energy_ratio = tail_energy / energy if energy > 0 else 0.0

    active = (
        np.flatnonzero(np.abs(x) >= active_threshold * peak_abs)
        if peak_abs > 0
        else np.array([], dtype=int)
    )
    if active.size:
        duration_ms = float((active[-1] - active[0]) * dt_ms)
        segment = x[active[0] : active[-1] + 1]
        signs = np.sign(segment)
        signs[np.abs(segment) < active_threshold * peak_abs] = 0
        signs = signs[signs != 0]
        n_phases = int(1 + np.count_nonzero(signs[1:] != signs[:-1])) if signs.size else 0
    else:
        duration_ms = 0.0
        n_phases = 0

    centred = x - np.mean(x)
    spectrum = np.fft.rfft(centred)
    freqs = np.fft.rfftfreq(x.size, d=dt_s)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    if total_power > 0:
        cumulative = np.cumsum(power)
        median_frequency = float(freqs[np.searchsorted(cumulative, 0.5 * total_power)])
        mean_frequency = float(np.sum(freqs * power) / total_power)
        dominant_frequency = float(freqs[int(np.argmax(power))])
        hf_ratio = float(np.sum(power[freqs > hf_cutoff_hz]) / total_power)
    else:
        median_frequency = mean_frequency = dominant_frequency = hf_ratio = 0.0

    roughness = 0.0
    if x.size > 2:
        d2 = np.diff(x, n=2) / (dt_ms**2)
        roughness = float(np.mean(d2**2))

    return {
        "peak_abs": peak_abs,
        "peak_to_peak": peak_to_peak,
        "rms_amplitude": rms,
        "duration_ms": duration_ms,
        "signed_area": signed_area,
        "absolute_area": absolute_area,
        "dc_area_ratio": float(dc_area_ratio),
        "tail_energy_ratio": float(tail_energy_ratio),
        "median_frequency_hz": median_frequency,
        "mean_frequency_hz": mean_frequency,
        "dominant_frequency_hz": dominant_frequency,
        "hf_ratio": hf_ratio,
        "roughness": roughness,
        "n_phases": n_phases,
    }


__all__ = [
    "normalize_peak",
    "align_score",
    "nrmse_aligned",
    "raw_r_vs",
    "lobe_metrics",
    "jaggedness",
    "waveform_features",
]
