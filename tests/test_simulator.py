"""Simulator facade — one drive → EMG (+force) recording."""
import numpy as np
import pytest

from emgforge.simulator import Simulator, Recording
from emgforge.activation import drive


def _tensor(N=8, M=3, w=64):
    t = np.linspace(0, 1, w)
    base = -np.exp(-((t - 0.4) ** 2) / 0.004) + 0.4 * np.exp(-((t - 0.6) ** 2) / 0.01)
    amp = np.linspace(0.4, 2.0, N)[:, None, None]
    return base[None, None, :] * amp * np.ones((N, M * M, 1)), M


def test_multichannel_run_shapes_and_force():
    W, M = _tensor(N=10, M=3)
    sim = Simulator(W, fs=2048, grid=(M, M))
    rec = sim.run(drive.trapezoid(0.5, 1, 1, 1, fs=2048), seed=0)
    assert isinstance(rec, Recording) and rec.multichannel
    assert rec.emg.shape == (M * M, len(rec.drive))
    assert rec.force is not None and rec.force.shape == rec.drive.shape
    assert rec.grid_view().shape == (M, M, len(rec.drive))
    assert 0 < rec.n_active <= 10


def test_single_channel_run():
    t = np.linspace(0, 1, 64)
    muaps = (-np.exp(-((t - 0.5) ** 2) / 0.01))[None, :] * np.linspace(0.5, 2, 6)[:, None]
    sim = Simulator(muaps, fs=2048)
    rec = sim.run(drive.constant(0.6, 2, 2048), seed=1)
    assert not rec.multichannel and rec.emg.ndim == 1
    assert rec.emg.shape == rec.drive.shape


def test_run_is_deterministic_per_seed():
    W, M = _tensor()
    sim = Simulator(W, grid=(M, M))
    E = drive.constant(0.5, 2, 2048)
    a, b = sim.run(E, seed=4), sim.run(E, seed=4)
    assert np.array_equal(a.emg, b.emg) and np.array_equal(a.force, b.force)


def test_twitch_off_gives_no_force():
    W, M = _tensor()
    rec = Simulator(W, grid=(M, M), twitch=False).run(drive.constant(0.4, 1, 2048))
    assert rec.force is None


def test_from_mri_missing_tensor_raises():
    with pytest.raises(FileNotFoundError):
        Simulator.from_mri(muscle=999, m=5)
