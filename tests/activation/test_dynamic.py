"""Dynamic (non-stationary) EMG — amplitude/warp tracking and movement modulation."""
import numpy as np
import pytest

from emgforge.activation.dynamic import (
    amp_from_angle, angle_track, drive_from_angle, dynamic_compound_emg, warp_from_angle,
)


def test_angle_track_bounds_and_length():
    a = angle_track(2.0, fs=1000, cycles=1)
    assert a.shape == (2000,)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_drive_from_angle_tracks_and_clips():
    a = np.array([0.0, 0.5, 1.0])
    d = drive_from_angle(a, base=0.1, gain=0.5)
    assert d[0] < d[1] < d[2]              # monotone with flexion
    assert np.all((d >= 0) & (d <= 1))     # stays a valid excitation


def test_amp_from_angle_centered_at_unity():
    assert amp_from_angle(np.array([0.5]))[0] == pytest.approx(1.0)
    assert amp_from_angle(np.array([1.0]), gain=0.6)[0] > 1.0


def test_amplitude_modulation_scales_emg():
    # one unit, one spike, flat vs rising amplitude track
    muaps = np.zeros((1, 8)); muaps[0, 3] = 1.0
    spikes = [np.array([100])]
    T = 200
    flat = dynamic_compound_emg(spikes, muaps, np.ones(T), n_samples=T)
    hot = dynamic_compound_emg(spikes, muaps, np.full(T, 2.0), n_samples=T)
    assert np.abs(hot).max() == pytest.approx(2.0 * np.abs(flat).max())


def test_amplitude_uses_track_at_spike_time():
    muaps = np.zeros((1, 4)); muaps[0, 0] = 1.0
    amp = np.ones(300); amp[:150] = 0.5      # first half quiet, second half loud
    spikes = [np.array([50, 250])]
    emg = dynamic_compound_emg(spikes, muaps, amp, n_samples=300)
    assert emg[50] == pytest.approx(0.5)     # early spike scaled by 0.5
    assert emg[250] == pytest.approx(1.0)    # late spike scaled by 1.0


def test_warp_stretches_muap_support():
    # a compact bump at the window start; factor > 1 must widen its non-zero support
    muaps = np.zeros((1, 64)); muaps[0, :10] = np.hanning(10)
    spikes = [np.array([0])]
    T = 128
    narrow = dynamic_compound_emg(spikes, muaps, np.ones(T), np.full(T, 0.6), n_samples=T)
    wide = dynamic_compound_emg(spikes, muaps, np.ones(T), np.full(T, 1.6), n_samples=T)
    assert (np.abs(wide) > 1e-6).sum() > (np.abs(narrow) > 1e-6).sum()


def test_multichannel_dynamic_shape_and_modulation():
    muaps = np.zeros((2, 3, 6)); muaps[:, :, 2] = 1.0     # (N, E, w)
    spikes = [np.array([10]), np.array([20])]
    T = 100
    ramp = np.linspace(1.0, 3.0, T)
    emg = dynamic_compound_emg(spikes, muaps, ramp, n_samples=T)
    assert emg.shape == (3, T)
    # later spike sees a larger amplitude than the earlier one
    assert np.abs(emg[:, 20:26]).max() > np.abs(emg[:, 10:16]).max()


def test_nonstationary_over_movement():
    # amplitude track that swells mid-record → RMS is higher in the middle than the ends
    rng = np.random.default_rng(0)
    muaps = np.tile(np.hanning(20), (5, 1))
    T = 3000
    spikes = [np.sort(rng.integers(0, T, 60)) for _ in range(5)]
    amp = 0.2 + np.hanning(T)                     # single-humped, peaks mid-record
    emg = dynamic_compound_emg(spikes, muaps, amp, n_samples=T)
    thirds = np.array_split(emg, 3)
    rms = [np.sqrt((x ** 2).mean()) for x in thirds]
    assert rms[1] > rms[0] and rms[1] > rms[2]   # swell in the middle
