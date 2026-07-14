"""``field_to_muap(field, bed, config)`` — the unified, bed-first synthesis entry.

The load-bearing tests are the byte-identity ones: given the fibre bed a call
already produced (``result.bed``), ``field_to_muap`` reproduces
``generate_muap_from_phi`` *exactly* on the Fourier side, and a manual
``compute_muap_spatial`` sum exactly on the spatial side. So the new entry is a
re-spelling of the existing computation, not a reimplementation — the migration
is verified, not asserted.

φ(z) inputs are the cylindrical golden cases (real Fourier reference fields),
tiled across a multi-fibre bed.
"""
from __future__ import annotations

import numpy as np
import pytest

from emgforge.synthesis import (
    FibreBed,
    MUAPConfig,
    SpatialConfig,
    SynthesisConfig,
    field_to_muap,
    generate_muap_from_phi,
    get_adaptive_config,
)
from emgforge.synthesis.fibres import NonUniformDz

from .conftest import CASE_NAMES


def _phi_mat(case, n: int) -> np.ndarray:
    """``case.phi`` tiled to an (n, Nz) fibre matrix."""
    return np.broadcast_to(case.phi, (n, case.phi.shape[0])).copy()


# ───────────────────────── Fourier byte-identity ─────────────────────────

@pytest.mark.parametrize("name", CASE_NAMES)
def test_fourier_matches_generate_on_every_golden_case(golden, name):
    """field_to_muap(phi, result.bed, cfg) == generate_muap_from_phi(phi, dz, cfg)."""
    c = golden[name]
    phi = _phi_mat(c, 6)
    cfg = MUAPConfig(len1_mm=c.L1_mm, len2_mm=c.L2_mm, v=c.v)

    ref = generate_muap_from_phi(phi, c.dz_mm, cfg)
    got = field_to_muap(phi, ref.bed, cfg)

    assert np.array_equal(got.muap, ref.muap)
    assert np.array_equal(got.t_ms, ref.t_ms)
    assert got.time_convention == "window_centred"


def test_fourier_jittered_uniform_v_matches(golden):
    """NMJ + tendon + length jitter (but cv_sigma=0 → uniform v) → shared fast path."""
    c = golden[CASE_NAMES[0]]
    phi = _phi_mat(c, 24)
    cfg = MUAPConfig(len1_mm=c.L1_mm, len2_mm=c.L2_mm, v=c.v,
                     nmj_sigma_mm=8.0, tendon_sigma_mm=4.0,
                     fiber_length_sigma_mm=6.0, jitter_seed=11)

    ref = generate_muap_from_phi(phi, c.dz_mm, cfg)
    got = field_to_muap(phi, ref.bed, cfg)

    assert np.allclose(ref.bed.v, ref.bed.v[0])       # v really is uniform here
    assert np.array_equal(got.muap, ref.muap)


def test_fourier_jittered_nonuniform_v_matches(golden):
    """cv_sigma>0 → non-uniform v → both take the per-fibre summation (slow) path."""
    c = golden[CASE_NAMES[0]]
    phi = _phi_mat(c, 16)
    cfg = MUAPConfig(len1_mm=c.L1_mm, len2_mm=c.L2_mm, v=c.v,
                     cv_sigma_m_per_s=0.3, nmj_sigma_mm=8.0, jitter_seed=3)

    ref = generate_muap_from_phi(phi, c.dz_mm, cfg)
    got = field_to_muap(phi, ref.bed, cfg)

    assert not np.allclose(ref.bed.v, ref.bed.v[0])   # v genuinely varies
    assert np.array_equal(got.muap, ref.muap)


def test_fourier_adaptive_w_matches(golden):
    """w=None: field_to_muap resolves the window from φ exactly as generate does."""
    c = golden[CASE_NAMES[0]]
    phi = _phi_mat(c, 4)
    cfg = get_adaptive_config()
    cfg = MUAPConfig(**{**cfg.__dict__, "len1_mm": c.L1_mm, "len2_mm": c.L2_mm, "v": c.v})

    ref = generate_muap_from_phi(phi, c.dz_mm, cfg)
    got = field_to_muap(phi, ref.bed, cfg)          # cfg still has w=None here

    assert ref.config.w is not None                  # generate pinned a concrete w
    assert got.config.w == ref.config.w              # field_to_muap pinned the same one
    assert np.array_equal(got.muap, ref.muap)


# ───────────────────────── Spatial byte-identity ─────────────────────────

def test_spatial_matches_manual_compute_muap_spatial(golden):
    """The spatial branch == a hand-built ``compute_muap_spatial`` over the same bed."""
    from emgforge.synthesis.engines.spatial import (
        Fibre as SpatialFibre,
        compute_muap_spatial,
    )
    c = golden[CASE_NAMES[0]]
    N = 5
    rng = np.random.default_rng(0)
    dz = c.dz_mm
    len1 = c.L1_mm + rng.normal(0, 3, N)
    len2 = c.L2_mm + rng.normal(0, 3, N)
    posz = rng.normal(0, 5, N)
    v = np.full(N, c.v)
    v[2] = c.v + 0.5                                  # one fibre with a CV override
    bed = FibreBed.from_arrays(dz, len1, len2, posz, v)
    phi = _phi_mat(c, N)

    cfg = SpatialConfig(v=c.v, fsamp=2048.0, w=256)
    got = field_to_muap(phi, bed, cfg)

    fibres = [SpatialFibre(phi[i], dz, float(len1[i]), float(len2[i]),
                           float(posz[i]), float(v[i])) for i in range(N)]
    t_ref, muap_ref, _ = compute_muap_spatial(fibres, cfg)

    assert np.array_equal(got.muap, muap_ref)
    assert np.array_equal(got.t_ms, t_ref)
    assert got.time_convention == "physical"
    assert isinstance(got.config, SpatialConfig)


# ───────────────────────── field coercion / dispatch ─────────────────────────

def test_1d_field_broadcasts_over_the_bed(golden):
    """A single φ(z) is applied to every fibre — same as tiling it by hand."""
    c = golden[CASE_NAMES[0]]
    bed = FibreBed.uniform(4, dz_mm=c.dz_mm, len1_mm=c.L1_mm, len2_mm=c.L2_mm, v=c.v)
    cfg = MUAPConfig(len1_mm=c.L1_mm, len2_mm=c.L2_mm, v=c.v)

    one = field_to_muap(c.phi, bed, cfg)                       # 1-D φ, broadcast
    many = field_to_muap(_phi_mat(c, 4), bed, cfg)             # explicit (4, Nz)
    assert np.array_equal(one.muap, many.muap)


def test_field_row_mismatch_raises(golden):
    c = golden[CASE_NAMES[0]]
    bed = FibreBed.uniform(4, dz_mm=c.dz_mm)
    with pytest.raises(ValueError):
        field_to_muap(_phi_mat(c, 3), bed, MUAPConfig())       # 3 rows, 4 fibres


def test_nonuniform_dz_bed_rejected_by_fourier(golden):
    """The Fourier engine needs one scalar dz; a ragged bed must raise, not pick dz[0]."""
    c = golden[CASE_NAMES[0]]
    from emgforge.synthesis.fibres import Fibre
    bed = FibreBed((Fibre(c.dz_mm, c.L1_mm, c.L2_mm), Fibre(c.dz_mm * 1.1, c.L1_mm, c.L2_mm)))
    with pytest.raises(NonUniformDz):
        field_to_muap(_phi_mat(c, 2), bed, MUAPConfig())


def test_bad_config_type_raises(golden):
    c = golden[CASE_NAMES[0]]
    bed = FibreBed.uniform(2, dz_mm=c.dz_mm)
    with pytest.raises(TypeError):
        field_to_muap(_phi_mat(c, 2), bed, SynthesisConfig())  # bare marker, no engine


def test_none_config_defaults_to_fourier(golden):
    """config=None → plain MUAPConfig() (Fourier), the historical default."""
    c = golden[CASE_NAMES[0]]
    bed = FibreBed.uniform(3, dz_mm=c.dz_mm, len1_mm=c.L1_mm, len2_mm=c.L2_mm, v=c.v)
    got = field_to_muap(_phi_mat(c, 3), bed, None)
    assert got.time_convention == "window_centred"
    assert isinstance(got.config, MUAPConfig)
    assert got.metrics                                          # metrics were computed
