"""Engine-level invariants: determinism, window behaviour, and the in-engine RNG."""
from __future__ import annotations

import numpy as np
import pytest

from emgforge.synthesis import generate_muap_from_phi, get_optimal_config
from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial

DZ = 4.0 * 1000.0 / 4096.0
Z = (np.arange(256) - 128) * DZ
PHI = np.exp(-(Z ** 2) / (2 * 12.0 ** 2))


def _fourier(phi, **cfg_kw):
    cfg = get_optimal_config()
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    r = generate_muap_from_phi(np.atleast_2d(phi), DZ, config=cfg)
    return np.real(np.asarray(r.muap)).ravel()


def test_fourier_is_deterministic():
    assert np.array_equal(_fourier(PHI), _fourier(PHI))


def test_spatial_is_deterministic():
    a = compute_sfap_spatial(PHI, DZ, 60.0, 60.0, 0.0, SpatialConfig())[1]
    b = compute_sfap_spatial(PHI, DZ, 60.0, 60.0, 0.0, SpatialConfig())[1]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("window", ["tukey", "boxcar", "hann", "one_sided"])
def test_spatial_window_produces_finite_signal(window):
    t, s, _ = compute_sfap_spatial(PHI, DZ, 60.0, 60.0, 0.0,
                                   SpatialConfig(fiber_window=window))
    assert np.all(np.isfinite(s))
    assert np.abs(s).max() > 0
    assert t.shape == s.shape


def _spatial(window):
    return compute_sfap_spatial(PHI, DZ, 60.0, 60.0, 0.0,
                                SpatialConfig(fiber_window=window))[1]


def test_spatial_windows_are_distinct():
    """A window that made no difference would mean the fibre-end model is inert."""
    out = {w: _spatial(w) for w in ("tukey", "boxcar", "hann")}
    names = list(out)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.array_equal(out[a], out[b]), f"{a} == {b}"


def test_window_none_is_rejected():
    """The redundant "none" enum value (a silent alias for "boxcar") was collapsed.

    It used to fall through to `np.ones(n)`, so "none" did NOT mean "no window" --
    it meant a hard rectangular tendon cut, identical to "boxcar". Now the enum has
    one spelling for that case ("boxcar") and a stale "none" raises instead of
    silently doing something other than its name.
    """
    with pytest.raises(ValueError):
        _spatial("none")


# --------------------------------------------------------------------------
# The Fourier engine draws fibre geometry from Gaussians *inside* the engine
# (api.py: `rng = np.random.default_rng(config.jitter_seed)`), so it is a pure
# function of its arguments only because `jitter_seed` happens to be pinned at
# 42. These tests characterise that. Moving the RNG out into a FibreBed value
# is tracked in the backlog; when it lands, these should be rewritten, not
# deleted -- they are the proof that the move preserved behaviour.
# --------------------------------------------------------------------------

JITTER = dict(nmj_sigma_mm=8.0, cv_sigma_m_per_s=0.3, tendon_sigma_mm=6.0)


def test_jitter_is_reproducible_for_a_fixed_seed():
    phi = np.repeat(PHI.reshape(1, -1), 5, 0)
    a = _fourier(phi, jitter_seed=7, **JITTER)
    b = _fourier(phi, jitter_seed=7, **JITTER)
    assert np.array_equal(a, b)


def test_jitter_seed_changes_the_fibre_bed():
    """The engine's output depends on `jitter_seed` -- i.e. on state the caller
    never sees and cannot inspect, save, or hand to the other engine."""
    phi = np.repeat(PHI.reshape(1, -1), 5, 0)
    a = _fourier(phi, jitter_seed=7, **JITTER)
    b = _fourier(phi, jitter_seed=8, **JITTER)
    assert not np.array_equal(a, b)


def test_no_jitter_is_the_default():
    """With all sigmas at 0 the engine must not touch its RNG at all."""
    phi = np.repeat(PHI.reshape(1, -1), 5, 0)
    a = _fourier(phi, jitter_seed=7)
    b = _fourier(phi, jitter_seed=999)
    assert np.array_equal(a, b)
