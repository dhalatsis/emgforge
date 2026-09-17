"""Simulator facade — one drive → EMG (+force) recording."""
import warnings

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


def test_from_mri_missing_tensor_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_pipeline.py"):
        Simulator.from_mri(muscle=999, m=5, root=tmp_path)


# ---- from_mri resolution order: run_pipeline.py output > legacy electrode_grid cache ----

def _write_pipeline_output(path, N=6, M=3, w=64, fs=2048.0):
    """A minimal scripts/run_pipeline.py pipeline_output.npz (muap_grid in µV)."""
    W, _ = _tensor(N=N, M=M, w=w)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, muap_grid=(W * 1e6).astype(np.float32), grid_shape=np.array([M, M], dtype=np.int32),
             fs=np.float32(fs), t_ms=np.arange(w) / fs * 1e3 - 10.0)
    return W


def _write_legacy_tensor(path, N=4, M=3, w=64):
    W, _ = _tensor(N=N, M=M, w=w)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=W, M=M)
    return W


def test_from_mri_prefers_run_pipeline_output_and_is_silent(tmp_path):
    W = _write_pipeline_output(tmp_path / "_results/pipeline/L8_3x3_ied10_mu6/pipeline_output.npz")
    _write_legacy_tensor(tmp_path / "_results/mu_pool/electrode_grid/muap_tensor_L8_M3.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # any warning here is a failure
        sim = Simulator.from_mri(muscle=8, m=3, root=tmp_path)
    assert sim.n_mu == 6 and sim.grid == (3, 3) and sim.fs == 2048.0
    np.testing.assert_allclose(sim.muaps, W, rtol=1e-6, atol=1e-12)   # µV → V round trip (float32)


def test_from_mri_prefers_the_largest_pool_then_the_newest(tmp_path):
    import os
    big = tmp_path / "_results/pipeline/L8_3x3_ied10_mu6/pipeline_output.npz"
    small = tmp_path / "_results/pipeline/L8_3x3_ied10_mu4/pipeline_output.npz"
    _write_pipeline_output(big, N=6)
    _write_pipeline_output(small, N=4)
    os.utime(big, (1_000_000, 1_000_000))                        # the 6-unit run is the stale one
    assert Simulator.from_mri(muscle=8, m=3, root=tmp_path).n_mu == 6   # largest pool wins over recency
    # same size: the newest run wins
    other = tmp_path / "_results/pipeline_other"
    a = tmp_path / "_results/pipeline/L9_3x3_ied10_mu6/pipeline_output.npz"
    b = tmp_path / "_results/pipeline/L9_3x3_ied10_mu6_rerun/pipeline_output.npz"
    _write_pipeline_output(a, N=6)
    _write_pipeline_output(b, N=6)
    os.utime(a, (1_000_000, 1_000_000))
    assert Simulator.from_mri(muscle=9, m=3, root=tmp_path).n_mu == 6


def test_from_mri_legacy_fallback_warns_and_names_the_artefact(tmp_path):
    legacy = tmp_path / "_results/mu_pool/electrode_grid/muap_tensor_L8_M3.npz"
    _write_legacy_tensor(legacy)
    with pytest.warns(UserWarning, match="LEGACY") as rec:
        sim = Simulator.from_mri(muscle=8, m=3, root=tmp_path)
    msg = str(rec[0].message)
    assert str(legacy) in msg and "build_grid_tensor.py" in msg and "run_pipeline.py" in msg
    assert sim.n_mu == 4 and sim.grid == (3, 3)


def test_from_mri_explicit_path_takes_either_format(tmp_path):
    pipe = tmp_path / "some/where/pipeline_output.npz"
    tensor = tmp_path / "released/muap_tensor_L8_M3.npz"
    Wp = _write_pipeline_output(pipe, N=5)
    Wt = _write_legacy_tensor(tensor, N=7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = Simulator.from_mri(path=pipe)
        b = Simulator.from_mri(path=tensor, fs=4096.0)
    np.testing.assert_allclose(a.muaps, Wp, rtol=1e-6, atol=1e-12)
    assert np.array_equal(b.muaps, Wt) and b.fs == 4096.0
    with pytest.raises(ValueError, match="fs="):                # a pipeline output carries its own fs
        Simulator.from_mri(path=pipe, fs=4096.0)


def test_simulator_is_importable_from_the_package_root():
    import emgforge
    assert emgforge.Simulator is Simulator and emgforge.Recording is Recording
