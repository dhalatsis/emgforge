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
    # --- innervation-zone geometry (harmonic beds only; None otherwise) ---
    iz_fraction: Optional[float] = None
    # ^ single innervation-zone band centre this MU sits in (longitudinal
    #   fraction). Literature: one MU = ONE compact endplate cluster, not
    #   endplates smeared across the whole muscle. See sample_henneman_pool.
    half1_mm: Optional[np.ndarray] = None   # (size,) NMJ→prox tendon, jittered
    half2_mm: Optional[np.ndarray] = None   # (size,) NMJ→dist tendon, jittered
    nmj_posz_mm: Optional[np.ndarray] = None  # (size,) NMJ axial offset, jittered


def sample_henneman_pool(
    bed,
    n_mu=30,
    size_min=5,
    size_max=400,
    distribution="exponential",
    overlap_ok=True,
    seed=0,
    single_iz=True,
    iz_band_tol=0.03,
    nmj_jitter_mm=3.0,
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
    single_iz : bool
        LITERATURE-ALIGNMENT (default True). A motor unit's endplates form a
        single, compact innervation zone — one NMJ band per MU — not endplates
        scattered across every band of the muscle (Masuda & Sadoyama 1991;
        Beck 2012; Merletti & Farina). On a ``method="harmonic"`` bed the fibres
        carry a per-fibre ``iz_fractions`` (their NMJ band). Selecting a MU by
        (x, y) proximity ALONE — the previous behaviour — pulls fibres from all
        bands, giving the physiologically-wrong "multiple NMJs per MU". With
        ``single_iz`` the candidate pool for each MU is first restricted to
        fibres whose IZ band is within ``iz_band_tol`` of the anchor's band,
        THEN the closest ``size`` in (x, y) are taken. No effect on uniform/
        poisson/hex beds (all full-length mid-belly NMJ → nothing to constrain).
    iz_band_tol : float
        Half-width (in longitudinal fraction) of the IZ band a single MU may
        occupy. Bands in a harmonic bed are discrete (e.g. 0.17, 0.62 for a
        bipennate FCU), so the default 0.03 (≈6 mm over a 210 mm fibre) snaps
        each MU to exactly one band while tolerating float noise.
    nmj_jitter_mm : float
        Physiological endplate scatter WITHIN the MU's single IZ. Real endplates
        are not perfectly aligned; the innervation zone has a Gaussian axial
        spread of σ≈3–8 mm (Roeleveld 1997; Merletti). Applied as a per-fibre
        N(0, σ) shift of the NMJ, which co-adjusts the two semi-lengths
        (half1 -= δ, half2 += δ). Set 0.0 to disable. Stored on each MotorUnit
        as ``half1_mm``/``half2_mm``/``nmj_posz_mm`` for the MUAP path — see
        :func:`mu_fiber_geometry`. This REPLACES the old habit of a single global
        ``posz=0`` synchronous NMJ (which made MUAPs 4× too sharp) with a
        controlled, single-cluster scatter.

    Returns
    -------
    list[MotorUnit] sorted by ascending size (Henneman: small recruited first).
    """
    rng = np.random.default_rng(seed)

    # Per-fibre IZ band (longitudinal fraction of each NMJ) — only harmonic beds
    # carry this. When present we constrain each MU to one band (single_iz).
    iz_fr = getattr(bed, "iz_fractions", None)
    use_iz = bool(single_iz) and iz_fr is not None
    if use_iz:
        iz_fr = np.asarray(iz_fr, dtype=float)

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
        # SINGLE-IZ CONSTRAINT: keep only candidates in the anchor's endplate
        # band, so the MU has ONE innervation zone (not one per band). Done
        # before the closest-`size` cut so proximity is judged within the band.
        mu_iz = None
        if use_iz:
            mu_iz = float(iz_fr[anchor])
            order = order[np.abs(iz_fr[order] - mu_iz) <= iz_band_tol]
        chosen = order[: int(size)]
        if not overlap_ok:
            used[chosen] = True
        actual_size = len(chosen)
        territory_R = float(d[chosen].max()) if actual_size else 0.0

        # Per-fibre NMJ geometry with physiological endplate scatter within the
        # single IZ (harmonic beds only). δ ~ N(0, σ) shifts the NMJ along the
        # fibre and co-adjusts the two semi-lengths.
        h1 = h2 = posz = None
        if use_iz and actual_size and getattr(bed, "half1_mm", None) is not None:
            h1 = np.asarray(bed.half1_mm[chosen], dtype=float).copy()
            h2 = np.asarray(bed.half2_mm[chosen], dtype=float).copy()
            posz = np.asarray(bed.posz_mm[chosen], dtype=float).copy()
            if nmj_jitter_mm > 0:
                delta = rng.normal(0.0, nmj_jitter_mm, actual_size)
                # keep each semi-length physically positive (>=1 mm). The NMJ
                # slides by +delta: h1 -= delta, h2 += delta. So h1>=1 bounds
                # delta above (delta <= h1-1) and h2>=1 bounds it below
                # (delta >= -(h2-1)). (Rails were swapped before 2026-07-24.)
                delta = np.clip(delta, -(h2 - 1.0), (h1 - 1.0))
                h1 = h1 - delta
                h2 = h2 + delta
                posz = posz + delta

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
            iz_fraction=mu_iz,
            half1_mm=h1,
            half2_mm=h2,
            nmj_posz_mm=posz,
        ))

    # Sort by size ascending (Henneman: smallest first to recruit)
    mus.sort(key=lambda m: m.size)
    # Re-index
    for new_idx, mu in enumerate(mus):
        mu.idx = new_idx
        mu.recruitment_threshold = float(mu.size ** 1.5)
    return mus


def mu_fiber_geometry(mu, bed):
    """Per-fibre NMJ geometry for a motor unit's MUAP, aligned to ``mu.fiber_idxs``.

    Returns ``(half1_mm, half2_mm, posz_mm)`` for the fibres of ``mu``. Uses the
    MU's single-IZ + endplate-jittered values when the pool was sampled with
    ``single_iz`` on a harmonic bed; otherwise falls back to the bed's stored
    per-fibre semi-lengths (or ``None`` for non-harmonic beds). Feed the result
    straight into :func:`realistic_muap.compute_muap_from_bed` overrides.
    """
    if mu.half1_mm is not None:
        return mu.half1_mm, mu.half2_mm, mu.nmj_posz_mm
    idx = np.asarray(mu.fiber_idxs)
    if getattr(bed, "half1_mm", None) is None:
        return None, None, None
    return bed.half1_mm[idx], bed.half2_mm[idx], bed.posz_mm[idx]


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
