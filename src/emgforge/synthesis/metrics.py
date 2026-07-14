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
    return float(np.mean(np.abs(np.diff(m, 2))) / m.ptp()) if m.ptp() > 0 else 0.0


__all__ = [
    "normalize_peak",
    "align_score",
    "nrmse_aligned",
    "raw_r_vs",
    "lobe_metrics",
    "jaggedness",
]
