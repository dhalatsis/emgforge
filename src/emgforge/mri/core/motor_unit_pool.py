"""Motor-unit pool sampling per Henneman size principle + MU activation.

A muscle has many MUs (~30-500 depending on muscle). Their sizes follow
an approximately exponential distribution — many small MUs, few large.
Recruitment proceeds smallest → largest as force increases.

This module turns a ``FiberBed`` into a list of ``MotorUnit``s by:
  1. Drawing N_mu sizes from an exponential distribution (small to large).
  2. For each MU, picking a centre at random within the muscle.
  3. Inflating the MU's territory radius until it contains the target size
     number of fibres (or saturating at the maximum bed coverage).
  4. Optionally: assigning a recruitment threshold + firing rate per MU.

Then ``simulate_compound_emg`` sums the per-MU MUAPs into a single
time-series, with a Poisson-like firing pattern per MU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MotorUnit:
    """One motor unit drawn from a fiber bed."""
    idx: int
    centre_xy: np.ndarray         # (2,) physical at z-mid
    territory_radius_mm: float
    fiber_idxs: np.ndarray        # indices into FiberBed (variable length)
    size: int                     # = len(fiber_idxs)
    recruitment_threshold: float  # arbitrary units (lower = recruited first)
    firing_rate_hz: float         # tonic firing rate when active


def sample_henneman_pool(
    bed,
    n_mu=30,
    size_min=5,
    size_max=400,
    distribution="exponential",
    overlap_ok=True,
    seed=0,
):
    """Sample a pool of ``n_mu`` motor units from a fibre bed.

    Parameters
    ----------
    bed : FiberBed
        Pre-built muscle fibre bed.
    n_mu : int
        Number of motor units to create.
    size_min, size_max : int
        Range of MU sizes (number of fibres per MU).
    distribution : {"exponential", "lognormal", "powerlaw"}
        Shape of the MU size distribution:
        - "exponential" (default): size ∝ exp(α·u), u ∈ [0, 1]
                                   — exponentially more small MUs
        - "lognormal":   size ∝ exp(σ·z), z ~ N(0, 1)
        - "powerlaw":    size = a·u^(-β), u ∈ [eps, 1]
    overlap_ok : bool
        Real MU territories overlap (multiple MUs interleave in the same
        region). If False, fibre indices are partitioned so each fibre
        belongs to exactly one MU.
    seed : int

    Returns
    -------
    list[MotorUnit] sorted by ascending size (Henneman: small recruited first).
    """
    rng = np.random.default_rng(seed)

    # 1. Sample MU sizes
    if distribution == "exponential":
        # Inverse-CDF of an exponential mapped to [size_min, size_max]
        # using log-spaced — gives "many small, few large"
        u = rng.uniform(0, 1, n_mu)
        sizes = size_min * (size_max / size_min) ** u
    elif distribution == "lognormal":
        log_min, log_max = np.log(size_min), np.log(size_max)
        sigma = (log_max - log_min) / 4.0
        sizes = np.exp(rng.normal(0.5 * (log_min + log_max), sigma, n_mu))
        sizes = np.clip(sizes, size_min, size_max)
    elif distribution == "powerlaw":
        beta = 1.5
        u = rng.uniform(1 / n_mu, 1, n_mu)
        sizes = size_min * u ** (-beta / 2)
        sizes = np.clip(sizes, size_min, size_max)
    else:
        raise ValueError(distribution)

    sizes = np.sort(np.round(sizes).astype(int))

    # 2. For each MU, pick a centre + grow territory until size is hit
    mus = []
    used = np.zeros(len(bed.r_norms), dtype=bool) if not overlap_ok else None
    cent = bed.centroid_xy
    R_max = float(np.max(np.linalg.norm(bed.xy_mid - cent, axis=1))) * 1.05
    for i, size in enumerate(sizes):
        # Random centre within the muscle: pick a random bed fibre as anchor
        if not overlap_ok:
            available = np.where(~used)[0]
            if len(available) == 0:
                break
            anchor = available[rng.integers(0, len(available))]
        else:
            anchor = rng.integers(0, len(bed.r_norms))
        cxy = bed.xy_mid[anchor]
        # Distances from this centre to all fibres
        dxy = bed.xy_mid - cxy
        d = np.linalg.norm(dxy, axis=1)
        # Order by distance, take the closest `size` available fibres
        order = np.argsort(d)
        if not overlap_ok:
            order = order[~used[order]]
        chosen = order[: int(size)]
        if not overlap_ok:
            used[chosen] = True
        actual_size = len(chosen)
        territory_R = float(d[chosen].max()) if actual_size else 0.0

        # Recruitment threshold ~ proportional to size^(1.5) (matches
        # Fuglevand: threshold force ~ size with steeper-than-linear)
        thr = float(actual_size ** 1.5)
        # Firing rate decays with threshold (large MUs fire slower)
        fr = float(40 - 25 * (i / max(n_mu - 1, 1)))  # 40 → 15 Hz

        mus.append(MotorUnit(
            idx=i,
            centre_xy=cxy.copy(),
            territory_radius_mm=territory_R,
            fiber_idxs=chosen,
            size=actual_size,
            recruitment_threshold=thr,
            firing_rate_hz=fr,
        ))

    # Sort by size ascending (Henneman: smallest first to recruit)
    mus.sort(key=lambda m: m.size)
    # Re-index
    for new_idx, mu in enumerate(mus):
        mu.idx = new_idx
        mu.recruitment_threshold = float(mu.size ** 1.5)
    return mus


def simulate_compound_emg(
    pool,
    muaps,
    duration_s=1.0,
    fsamp=4096.0,
    activation_level=0.5,
    seed=0,
):
    """Sum MU MUAPs into a continuous EMG signal.

    Parameters
    ----------
    pool : list[MotorUnit]
        Sorted by ascending size; each has firing_rate_hz +
        recruitment_threshold.
    muaps : dict[mu.idx] = (t_ms_axis, muap_signal)
        Pre-computed MUAP per MU. The signal is replicated at each
        firing time.
    duration_s : float
        Total EMG duration.
    activation_level : float in [0, 1]
        Fraction of the MU pool's recruitment range that's active.
        Henneman: only MUs with threshold ≤ activation·max_threshold fire.
    seed : int

    Returns
    -------
    t_s : (N_samp,) array
    emg : (N_samp,) array
    fired_mus : list of (mu_idx, list of firing times in s)
    """
    rng = np.random.default_rng(seed)
    n_samp = int(duration_s * fsamp)
    t_s = np.arange(n_samp) / fsamp
    emg = np.zeros(n_samp)

    if not pool:
        return t_s, emg, []

    max_thr = max(mu.recruitment_threshold for mu in pool)
    cutoff = activation_level * max_thr

    fired = []
    for mu in pool:
        if mu.recruitment_threshold > cutoff:
            continue
        if mu.idx not in muaps:
            continue
        t_muap_ms, muap_sig = muaps[mu.idx]
        # Resample to common sampling rate
        dt_muap = (t_muap_ms[1] - t_muap_ms[0]) / 1000.0
        # Plant firings as a Poisson process at the MU's firing rate
        rate = mu.firing_rate_hz
        # Mean inter-spike interval
        if rate <= 0:
            continue
        firings = []
        t_now = rng.exponential(1.0 / rate)
        while t_now < duration_s:
            firings.append(t_now)
            t_now += rng.exponential(1.0 / rate) + 1e-3  # 1ms refractory min
        fired.append((mu.idx, firings))

        # Convolve / superpose: shift the MUAP to each firing time
        # Sample MUAP onto our fsamp grid
        muap_t_s = t_muap_ms / 1000.0
        # MUAP envelope is short (< 100ms), use direct indexing
        muap_n = len(muap_sig)
        n_muap = int(np.ceil((muap_t_s[-1] - muap_t_s[0]) * fsamp))
        # Resample muap_sig to fsamp grid
        muap_resamp = np.interp(
            np.linspace(muap_t_s[0], muap_t_s[-1], n_muap),
            muap_t_s, muap_sig,
        )
        # Centre MUAP at t=0 (peak-aligned)
        pk_idx = int(np.argmax(np.abs(muap_resamp - muap_resamp.mean())))
        for t_fire in firings:
            # Sample index where firing peak lands
            i_peak = int(t_fire * fsamp)
            i_start = i_peak - pk_idx
            i_end = i_start + n_muap
            # Clip to EMG window
            mu_s_start = max(0, -i_start)
            mu_s_end = n_muap - max(0, i_end - n_samp)
            i_start_clip = max(0, i_start)
            i_end_clip = min(n_samp, i_end)
            if i_end_clip > i_start_clip:
                emg[i_start_clip:i_end_clip] += muap_resamp[mu_s_start:mu_s_end]

    return t_s, emg, fired
