"""Multichannel interference EMG + differential montages."""
import numpy as np
import pytest

from emgforge.activation import compound_emg, compound_emg_multi, single_diff, double_diff


def _muap(w=64):
    t = np.linspace(0, 1, w)
    return -np.exp(-((t - 0.4) ** 2) / 0.004) + 0.4 * np.exp(-((t - 0.6) ** 2) / 0.01)


def test_multi_reduces_to_single_channel_when_electrodes_identical():
    N, E, w = 5, 4, 64
    base = _muap(w)
    muaps = np.tile(base, (N, E, 1)) * np.linspace(0.5, 2, N)[:, None, None]
    rng = np.random.default_rng(0)
    spikes = [np.sort(rng.integers(0, 1000, 8)) for _ in range(N)]
    multi = compound_emg_multi(spikes, muaps, n_samples=1200)      # (E, T)
    single = compound_emg(spikes, muaps[:, 0, :], n_samples=1200)  # (T,)
    assert multi.shape == (E, 1200)
    for e in range(E):
        assert np.allclose(multi[e], single)                       # every channel identical


def test_single_diff_removes_common_mode():
    # grid (3,3,T): a per-channel signal + a common far-field added to all
    T = 500
    rng = np.random.default_rng(1)
    local = rng.standard_normal((3, 3, T)) * 0.1
    common = rng.standard_normal(T)                                # shared across all electrodes
    grid = local + common[None, None, :]
    sd = single_diff(grid, axis=0)                                 # (2,3,T)
    assert sd.shape == (2, 3, T)
    # the common mode cancels: differencing local-only vs local+common gives the same result
    assert np.allclose(sd, single_diff(local, axis=0))


def test_double_diff_shape_and_length_mismatch():
    grid = np.zeros((4, 4, 100))
    assert double_diff(grid, axis=1).shape == (4, 2, 100)
    with pytest.raises(ValueError):
        compound_emg_multi([np.array([1])], np.zeros((2, 3, 10)))   # 1 train vs 2 MUAPs
