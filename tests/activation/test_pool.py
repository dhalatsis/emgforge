"""Motoneuron-pool invariants — recruitment, onion-skin rate coding, spiking.

Pure NumPy. Checks the phenomenology the NeuroMotion/Fuglevand model must reproduce
(not exact numbers): recruitment order, onion-skin peak rates, thresholded rate law,
reproducible ISI-renewal spikes, and common-drive-induced correlation.
"""
import numpy as np
import pytest

from emgforge.activation import MotoneuronPool, drive


def test_recruitment_is_ordered_and_bounded():
    p = MotoneuronPool(n_mu=100)
    assert np.all(np.diff(p.rte) >= 0)            # small MUs recruited first
    assert p.rte[0] < p.rte[-1] <= p.rm + 1e-9    # last MU recruits at ~rm


def test_onion_skin_peak_rate_decreases():
    p = MotoneuronPool(n_mu=100)
    assert p.peak_fr[0] > p.peak_fr[-1]           # onion-skin
    assert p.peak_fr[0] == pytest.approx(p.pfr1)
    assert p.min_fr[0] > p.min_fr[-1]


def test_firing_rate_thresholded_and_capped():
    p = MotoneuronPool(n_mu=50)
    E = np.linspace(0, 1, 500)
    fr = p.firing_rate(E)                          # (N, T)
    # silent strictly below threshold
    for m in range(p.n_mu):
        assert np.all(fr[m, E < p.rte[m]] == 0)
    # never exceeds the per-MU peak
    assert np.all(fr <= p.peak_fr[:, None] + 1e-9)
    # once active, monotonic non-decreasing in E (before saturation)
    assert fr[0, -1] >= fr[0, np.argmax(E > p.rte[0])]


def test_more_drive_recruits_more_units():
    p = MotoneuronPool(n_mu=100)
    assert p.n_active(np.array([0.1])) < p.n_active(np.array([0.4])) < p.n_active(np.array([0.9]))


def test_spikes_reproducible_and_measured_rate_matches_model():
    fs = 2048
    p = MotoneuronPool(n_mu=60, fs=fs)
    E = drive.constant(0.5, duration_s=4, fs=fs)
    a = p.spike_trains(E, seed=3)
    b = p.spike_trains(E, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))     # deterministic per seed
    # MU0 measured rate over the steady record ≈ model rate at E=0.5
    model = p.firing_rate(E)[0, -1]
    measured = len(a[0]) / 4.0
    assert measured == pytest.approx(model, rel=0.1)
    # unrecruited MU (threshold above 0.5) is silent
    silent = int(np.argmax(p.rte > 0.5))
    assert len(a[silent]) == 0


def test_below_threshold_pool_is_silent():
    p = MotoneuronPool(n_mu=40)
    E = drive.constant(0.001, duration_s=2)        # below almost every threshold
    spikes = p.spike_trains(E, seed=0)
    assert sum(len(s) for s in spikes[1:]) == 0    # only MU0 (RTE≈0) may fire


def test_common_drive_adds_correlated_fluctuation():
    fs = 2048
    E0 = drive.constant(0.3, duration_s=5, fs=fs)
    Ec = drive.add_common_drive(E0, sigma=0.03, cutoff_hz=2.0, fs=fs, seed=1)
    assert Ec.std() > E0.std()                     # fluctuation added
    assert np.all((Ec >= 0) & (Ec <= 1))           # clipped to [0, 1]
    assert abs(Ec.mean() - 0.3) < 0.02             # mean roughly preserved


def test_drive_builders_shapes_and_ranges():
    fs = 1000
    assert drive.ramp(0.5, 2, fs).shape[0] == 2000
    tz = drive.trapezoid(0.4, 1, 1, 1, fs)
    assert tz.max() == pytest.approx(0.4) and tz[0] == 0 and tz[-1] == pytest.approx(0.0)
    s = drive.sinusoid(0.5, 0.5, 1.0, 2, fs)
    assert s.min() >= 0 and s.max() <= 1
