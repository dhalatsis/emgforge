"""Unit tests for the salvaged waveform metrics (A-07).

Pure NumPy on synthetic bumps: identity cases, a known time shift, a polarity
flip, and a trough+trailing-lobe waveform pin the behaviour the comparator relied
on before it was deleted.
"""
import numpy as np
import pytest

from emgforge.synthesis.metrics import (
    align_score,
    jaggedness,
    lobe_metrics,
    normalize_peak,
    nrmse_aligned,
    raw_r_vs,
    waveform_features,
)

T = np.linspace(0.0, 30.0, 601)      # dt = 0.05 ms


def bump(center, width=1.5, amp=1.0):
    return amp * np.exp(-((T - center) ** 2) / (2 * width ** 2))


def test_normalize_peak_scales_to_unit_and_passes_zero():
    assert np.isclose(np.abs(normalize_peak(np.array([0.0, -4.0, 2.0]))).max(), 1.0)
    z = np.zeros(5)
    assert np.array_equal(normalize_peak(z), z)     # all-zero untouched (no /0)


def test_align_score_identical_is_zero_shift_r_one():
    s, r = align_score(T, bump(15), T, bump(15))
    assert abs(s) < 0.1 and r > 0.999


def test_align_score_recovers_a_known_shift():
    # b peaks 3 ms later than ref → shift to add to t_b is -3.
    s, r = align_score(T, bump(15), T, bump(18))
    assert abs(s - (-3.0)) < 0.15 and r > 0.999


def test_align_score_discriminates_a_polarity_flip():
    """A flip must not score like a true match. (Not necessarily <0: align_score
    takes the max r over lags, and two disjoint opposite-sign bumps over a zero
    baseline give a small spurious positive r at large lag — so we assert the flip
    scores far below the same-sign match, which is the property that matters.)"""
    _, r_match = align_score(T, bump(15), T, bump(18))     # same sign, shifted
    _, r_flip = align_score(T, bump(15), T, -bump(18))     # flipped
    assert r_match > 0.99
    assert r_flip < r_match - 0.5


def test_nrmse_aligned_zero_for_identical():
    assert nrmse_aligned(T, bump(15), T, bump(15)) < 1e-9


def test_raw_r_vs_one_for_identical():
    assert raw_r_vs(T, bump(15), T, bump(15)) > 0.999


def test_lobe_metrics_finds_trough_and_trailing_lobe():
    # global min at 15 (amp -1), a positive EOF lobe at 18 (amp +0.5).
    m = -bump(15, 1.0) + bump(18, 1.0, amp=0.5)
    trough, before, after = lobe_metrics(T, m, win_ms=6.0)
    assert abs(trough - 15.0) < 0.2
    assert after > 0.3           # the trailing lobe is picked up
    assert after > before        # and it is AFTER the trough, not before


def test_jaggedness_smooth_below_noisy_and_flat_is_zero():
    rng = np.random.default_rng(0)
    smooth = bump(15, 2.0)
    noisy = smooth + 0.05 * rng.standard_normal(T.size)
    assert jaggedness(smooth) < jaggedness(noisy)
    assert jaggedness(np.ones(100)) == 0.0


def test_waveform_features_capture_balance_spectrum_and_tails():
    t = np.arange(0.0, 1000.0, 1.0)
    balanced = bump(12.0, 1.0) - bump(18.0, 1.0)
    # Use the module's short T axis for morphology, and a separate exact sinusoid
    # for the spectral check.
    f = waveform_features(T, balanced)
    assert f["peak_to_peak"] > 0
    assert f["duration_ms"] > 0
    assert f["n_phases"] == 2
    assert f["dc_area_ratio"] < 0.02
    assert f["tail_energy_ratio"] < 1e-4

    sine = np.sin(2 * np.pi * 120.0 * t / 1000.0)
    sf = waveform_features(t, sine)
    assert sf["dominant_frequency_hz"] == pytest.approx(120.0, abs=1.0)
    assert sf["median_frequency_hz"] == pytest.approx(120.0, abs=1.0)


def test_waveform_features_reject_bad_axes_and_nonfinite_values():
    with pytest.raises(ValueError):
        waveform_features([0.0, 1.0], [1.0])
    with pytest.raises(ValueError):
        waveform_features([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        waveform_features([0.0, 1.0], [1.0, np.nan])
    with pytest.raises(ValueError, match="uniformly sampled"):
        waveform_features([0.0, 1.0, 2.1], [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="positive"):
        waveform_features([0.0, 1.0], [0.0, 1.0], hf_cutoff_hz=0.0)


def test_waveform_features_tail_energy_does_not_double_count_overlap():
    features = waveform_features(
        np.arange(3, dtype=float), np.ones(3), tail_fraction=0.5
    )
    assert features["tail_energy_ratio"] == pytest.approx(1.0)
