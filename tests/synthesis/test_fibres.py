"""Unit tests for the FibreBed value type (A-05, increment 1).

Pure NumPy. The load-bearing test is ``test_jittered_matches_legacy_draw``: it pins
that ``FibreBed.jittered`` reproduces the engine's inline RNG draw byte-for-byte, so
migrating the jitter out of ``MUAPConfig`` into the bed is a no-op.
"""
import numpy as np
import pytest

from emgforge.synthesis.fibres import Fibre, FibreBed, NonUniformDz


# ────────────────────────── Fibre validation ──────────────────────────

def test_fibre_rejects_bad_values():
    Fibre(dz_mm=1.0, len1_mm=60.0, len2_mm=60.0)          # ok
    with pytest.raises(ValueError):
        Fibre(dz_mm=0.0, len1_mm=60.0, len2_mm=60.0)      # dz <= 0
    with pytest.raises(ValueError):
        Fibre(dz_mm=1.0, len1_mm=60.0, len2_mm=60.0, v=0.0)   # v <= 0
    with pytest.raises(ValueError):
        Fibre(dz_mm=1.0, len1_mm=0.0, len2_mm=0.0)        # len1+len2 <= 0


# ────────────────────────── FibreBed basics ──────────────────────────

def test_bed_requires_a_fibre():
    with pytest.raises(ValueError):
        FibreBed(())


def test_bed_container_protocol_and_array_views():
    bed = FibreBed((Fibre(1.0, 50, 60, 2.0, 4.0), Fibre(1.5, 55, 65, -1.0, 4.5)))
    assert len(bed) == 2
    assert list(bed)[0].len1_mm == 50
    assert bed[1].v == 4.5
    assert np.allclose(bed.dz_mm, [1.0, 1.5])
    assert np.allclose(bed.len1_mm, [50, 55])
    assert np.allclose(bed.len2_mm, [60, 65])
    assert np.allclose(bed.posz_mm, [2.0, -1.0])
    assert np.allclose(bed.v, [4.0, 4.5])


def test_uniform_dz_returns_or_raises():
    assert FibreBed.uniform(4, dz_mm=0.9766).uniform_dz == pytest.approx(0.9766)
    mixed = FibreBed((Fibre(1.0, 60, 60), Fibre(1.2, 60, 60)))
    with pytest.raises(NonUniformDz):
        _ = mixed.uniform_dz


# ────────────────────────── factories ──────────────────────────

def test_from_arrays_broadcasts_scalars():
    bed = FibreBed.from_arrays(dz_mm=1.0, len1_mm=[50, 55, 60], len2_mm=[60, 65, 70],
                               posz_mm=0.0, v=4.0)
    assert len(bed) == 3
    assert np.allclose(bed.dz_mm, [1.0, 1.0, 1.0])       # scalar broadcast
    assert np.allclose(bed.len1_mm, [50, 55, 60])


def test_from_arrays_length_mismatch_raises():
    with pytest.raises(ValueError):
        FibreBed.from_arrays(dz_mm=[1.0, 1.0], len1_mm=[50, 55, 60], len2_mm=60.0)


def test_uniform_and_jittered_zero_sigma_are_identical():
    a = FibreBed.uniform(10, dz_mm=1.0, len1_mm=55, len2_mm=65, v=4.2)
    b = FibreBed.jittered(10, dz_mm=1.0, len1_mm=55, len2_mm=65, v=4.2)   # all σ = 0
    assert np.allclose(a.len1_mm, b.len1_mm)
    assert np.allclose(a.posz_mm, b.posz_mm)
    assert np.allclose(a.v, b.v)


def test_jittered_matches_legacy_draw():
    """FibreBed.jittered == the engine's inline RNG draw (byte-identical migration)."""
    from emgforge.synthesis.api import MUAPConfig, _resolve_per_fiber_jitter
    N = 64
    cfg = MUAPConfig(len1_mm=55.0, len2_mm=65.0, v=4.2, nmj_sigma_mm=8.0,
                     cv_sigma_m_per_s=0.3, tendon_sigma_mm=4.0,
                     fiber_length_sigma_mm=6.0, jitter_seed=7)
    posz, vs, l1, l2, _ = _resolve_per_fiber_jitter(cfg, N, None, None, None, None)
    bed = FibreBed.jittered(N, dz_mm=1.0, len1_mm=55.0, len2_mm=65.0, v=4.2,
                            nmj_sigma_mm=8.0, cv_sigma=0.3, tendon_sigma_mm=4.0,
                            fibre_length_sigma_mm=6.0, seed=7)
    assert np.allclose(bed.posz_mm, posz)
    assert np.allclose(bed.v, vs)
    assert np.allclose(bed.len1_mm, l1)
    assert np.allclose(bed.len2_mm, l2)
