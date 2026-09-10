"""Metamorphic sanity checks derived from the forward model's physics.

These tests avoid a brittle claim that every healthy surface MUAP has one fixed
shape. They vary one controlled cause and assert the response that should follow:
linearity, volume-conductor low-pass filtering, velocity scaling, propagation,
and desynchronisation from a distributed innervation zone.
"""
from __future__ import annotations

import numpy as np
import pytest

from emgforge.synthesis import (
    FibreBed,
    MUAPConfig,
    SpatialConfig,
    field_to_muap,
    waveform_features,
)
from emgforge.synthesis.metrics import align_score

FS = 4096.0
V = 4.0
DZ = V * 1000.0 / FS
Z = (np.arange(256) - 128) * DZ


def monopole(depth_mm: float, z0_mm: float = 0.0) -> np.ndarray:
    return 1.0 / np.sqrt(depth_mm**2 + (Z - z0_mm) ** 2)


def assert_waveforms_close(actual: np.ndarray, expected: np.ndarray) -> None:
    """Machine-precision comparison with an amplitude-relative zero tolerance."""
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=scale * 1e-12)


FOURIER = MUAPConfig(denoise="none", edge_taper=0, fsamp=FS, w=256, v=V)
SPATIAL = SpatialConfig(
    denoise="none",
    fiber_window="one_sided",
    upsample_factor=1,
    fsamp=FS,
    w=256,
    v=V,
    t_start_ms=-10.0,
)


@pytest.mark.parametrize("config", [FOURIER, SPATIAL], ids=["fourier", "spatial"])
def test_field_amplitude_and_polarity_are_linear(config):
    bed = FibreBed.uniform(1, DZ, v=V)
    base = field_to_muap(monopole(10.0), bed, config).muap
    scaled = field_to_muap(-2.75 * monopole(10.0), bed, config).muap
    assert_waveforms_close(scaled, -2.75 * base)


@pytest.mark.parametrize("config", [FOURIER, SPATIAL], ids=["fourier", "spatial"])
def test_motor_unit_is_the_superposition_of_its_fibres(config):
    fields = np.stack([monopole(8.0), monopole(12.0, 3.0), monopole(18.0, -5.0)])
    bed = FibreBed.uniform(3, DZ, v=V)
    together = field_to_muap(fields, bed, config).muap
    separate = sum(
        field_to_muap(field, FibreBed.uniform(1, DZ, v=V), config).muap
        for field in fields
    )
    assert_waveforms_close(together, separate)


def test_source_distance_attenuates_and_low_pass_filters_surface_muap():
    """A broader, more distant monopole field has less amplitude and bandwidth."""
    bed = FibreBed.uniform(1, DZ, v=V)
    near = field_to_muap(monopole(5.0), bed, FOURIER)
    far = field_to_muap(monopole(20.0), bed, FOURIER)
    near_f = waveform_features(near.t_ms, near.muap)
    far_f = waveform_features(far.t_ms, far.muap)

    assert near_f["peak_to_peak"] > 10.0 * far_f["peak_to_peak"]
    assert near_f["median_frequency_hz"] > far_f["median_frequency_hz"]


def test_conduction_velocity_compresses_time_and_raises_frequency():
    slow_v, fast_v = 3.0, 5.0
    slow = field_to_muap(
        monopole(10.0),
        FibreBed.uniform(1, DZ, v=slow_v),
        MUAPConfig(denoise="none", fsamp=FS, w=256, v=slow_v),
    )
    fast = field_to_muap(
        monopole(10.0),
        FibreBed.uniform(1, DZ, v=fast_v),
        MUAPConfig(denoise="none", fsamp=FS, w=256, v=fast_v),
    )
    slow_f = waveform_features(slow.t_ms, slow.muap)
    fast_f = waveform_features(fast.t_ms, fast.muap)

    assert fast_f["duration_ms"] < slow_f["duration_ms"]
    assert fast_f["median_frequency_hz"] > slow_f["median_frequency_hz"]


def test_spatial_engine_recovers_propagation_delay():
    """Moving a detector 20 mm along a 4 mm/ms fibre delays the MUAP by 5 ms."""
    bed = FibreBed.uniform(1, DZ, v=V)
    proximal = field_to_muap(monopole(8.0, 10.0), bed, SPATIAL)
    distal = field_to_muap(monopole(8.0, 30.0), bed, SPATIAL)

    shift_ms, signed_r = align_score(
        proximal.t_ms,
        proximal.muap,
        distal.t_ms,
        distal.muap,
        max_lag_ms=8.0,
        dt=0.01,
    )
    assert signed_r > 0.95
    assert shift_ms == pytest.approx(-(30.0 - 10.0) / V, abs=0.15)


def test_innervation_zone_scatter_broadens_and_desynchronises_muap():
    fields = np.repeat(monopole(10.0)[None, :], 9, axis=0)
    aligned_bed = FibreBed.from_arrays(DZ, 60.0, 60.0, np.zeros(9), V)
    spread_bed = FibreBed.from_arrays(DZ, 60.0, 60.0, np.linspace(-12.0, 12.0, 9), V)
    aligned = field_to_muap(fields, aligned_bed, SPATIAL)
    spread = field_to_muap(fields, spread_bed, SPATIAL)
    aligned_f = waveform_features(aligned.t_ms, aligned.muap)
    spread_f = waveform_features(spread.t_ms, spread.muap)

    assert spread_f["duration_ms"] > aligned_f["duration_ms"]
    assert spread_f["peak_to_peak"] < 0.6 * aligned_f["peak_to_peak"]
