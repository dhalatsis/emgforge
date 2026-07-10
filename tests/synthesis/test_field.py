"""The monopole denoiser, and its placement inside the spatial engine.

A FEM-sampled φ(z) carries mesh-scale ripple that the CSD's second derivative
amplifies. ``denoise_field_n`` fits the analytic form the field actually has -- a
sum of free-position monopoles -- rather than lowpass-filtering it.
"""
from __future__ import annotations

import numpy as np
import pytest

from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial
from emgforge.synthesis.preprocessing import denoise_field_n

DZ = 0.9766
Z = (np.arange(200) - 100) * DZ


def monopole(depth, amp=1.0, z0=0.0, offset=0.0):
    return offset + amp / np.sqrt(depth ** 2 + (Z - z0) ** 2)


def _rel_err(out, phi):
    return float(np.max(np.abs(out - phi)) / np.abs(phi).max())


def test_recovers_an_exact_single_monopole():
    """A field that IS a monopole must survive the fit untouched.

    This is the in-domain case: one electrode sees one dominant source along the
    fibre, so φ(z) has a single peak.
    """
    phi = monopole(depth=10.0)
    assert _rel_err(denoise_field_n(phi, DZ, n=3), phi) < 1e-6


def test_recovers_a_dominant_source_plus_a_weak_secondary():
    """The realistic shape of an FEM lead field: one peak, plus curvature in the tail."""
    phi = monopole(10.0, 1.0, 0.0) + monopole(25.0, 0.15, 40.0)
    assert _rel_err(denoise_field_n(phi, DZ, n=3), phi) < 1e-6


def test_does_not_resolve_two_equal_well_separated_sources():
    """A domain limit, pinned deliberately.

    The greedy fit seeds every new pole at the GLOBAL PEAK of φ (`z0 = z[argmax|φ|]`,
    fixed across all k). Given two equal sources it will never place a pole at the
    second one, and the residual stays ~56% of peak even at n=3. So this is a
    single-dominant-source denoiser, not a general N-monopole solver.

    That is fine for its purpose -- a lead field sampled along a fibre from one
    electrode has one peak -- but it is not what the name suggests, and n=3 must
    not be read as "fits any 3-pole field".
    """
    phi = monopole(8.0, 1.0, -40.0) + monopole(8.0, 1.0, 40.0)
    assert _rel_err(denoise_field_n(phi, DZ, n=3), phi) > 0.1


def test_greedy_fit_keeps_the_last_pole_count_not_the_best():
    """`best` is overwritten on every k, so denoise_field_n returns the n-pole fit
    even when a lower k fitted better. Harmless in-domain (both are ~1e-9), but it
    means `n` is a ceiling, not a search bound.

    If this ever fails, the function has learned to keep the best fit -- good;
    delete this test and tighten the ones above.
    """
    phi = monopole(10.0, 1.0, 0.0) + monopole(25.0, 0.15, 40.0)
    err2 = _rel_err(denoise_field_n(phi, DZ, n=2), phi)
    err3 = _rel_err(denoise_field_n(phi, DZ, n=3), phi)
    assert err2 < 1e-6 and err3 < 1e-6      # both fine in absolute terms
    assert err3 >= err2                      # but the extra pole did not help


def test_suppresses_ripple_on_a_noisy_field():
    rng = np.random.default_rng(0)
    clean = monopole(depth=10.0)
    noisy = clean * (1 + 5e-3 * rng.standard_normal(Z.size))
    out = denoise_field_n(noisy, DZ, n=3)

    err_before = np.abs(noisy - clean).max()
    err_after = np.abs(out - clean).max()
    assert err_after < err_before / 2, f"{err_after:.2e} vs {err_before:.2e}"


def test_is_deterministic():
    rng = np.random.default_rng(1)
    phi = monopole(10.0) * (1 + 3e-3 * rng.standard_normal(Z.size))
    assert np.array_equal(denoise_field_n(phi, DZ), denoise_field_n(phi, DZ))


def test_is_insensitive_to_dz_precision():
    """float32-rounded dz gives the same fit -- so the engine may own this stage
    even though it receives a float32 dz from callers that need one."""
    phi = monopole(depth=10.0)
    a = denoise_field_n(phi, float(DZ), n=3)
    b = denoise_field_n(phi, float(np.float32(DZ)), n=3)
    assert np.max(np.abs(a - b)) / np.abs(a).max() < 1e-9


# --------------------------------------------------------------------------
# Placement inside the engine
# --------------------------------------------------------------------------

BASE = dict(fsamp=2048.0, w=256, csd_derivative=2, upsample_factor=2,
            fiber_window="one_sided", tukey_alpha=0.25, smoothing=False,
            edge_taper_left=5, edge_taper_right=10, center_time=False,
            t_start_ms=-10.0)


def test_engine_denoise_equals_denoising_beforehand():
    """`denoise="monopole"` must be exactly equivalent to calling denoise_field_n
    on the raw field first -- i.e. it runs BEFORE the edge taper, not after.

    This pins the stage ORDER. The monopole fit models φ's analytic tails; an edge
    taper applied first would corrupt exactly the samples it fits against.
    """
    rng = np.random.default_rng(2)
    phi = monopole(depth=12.0) * (1 + 3e-3 * rng.standard_normal(Z.size))

    _, inside, _ = compute_sfap_spatial(
        phi, DZ, 60.0, 60.0, 0.0, SpatialConfig(denoise="monopole", **BASE))
    _, outside, _ = compute_sfap_spatial(
        denoise_field_n(phi, DZ, n=3), DZ, 60.0, 60.0, 0.0,
        SpatialConfig(denoise="none", **BASE))

    assert np.array_equal(inside, outside)


def test_denoise_defaults_to_off():
    """Adding the stage must not change any existing caller's numbers."""
    phi = monopole(depth=12.0)
    a = compute_sfap_spatial(phi, DZ, 60.0, 60.0, 0.0, SpatialConfig(**BASE))[1]
    b = compute_sfap_spatial(phi, DZ, 60.0, 60.0, 0.0,
                             SpatialConfig(denoise="none", **BASE))[1]
    assert np.array_equal(a, b)


def test_denoise_changes_the_sfap_on_a_noisy_field():
    """Guard against the stage silently becoming a no-op."""
    rng = np.random.default_rng(3)
    phi = monopole(depth=12.0) * (1 + 1e-2 * rng.standard_normal(Z.size))
    a = compute_sfap_spatial(phi, DZ, 60.0, 60.0, 0.0, SpatialConfig(**BASE))[1]
    b = compute_sfap_spatial(phi, DZ, 60.0, 60.0, 0.0,
                             SpatialConfig(denoise="monopole", **BASE))[1]
    assert not np.array_equal(a, b)


def test_denoise_is_near_a_noop_on_a_clean_analytic_field():
    """On analytical φ the fit reproduces the input, so the SFAP barely moves.
    It earns its keep on FEM fields, not on these."""
    phi = monopole(depth=12.0)
    a = compute_sfap_spatial(phi, DZ, 60.0, 60.0, 0.0, SpatialConfig(**BASE))[1]
    b = compute_sfap_spatial(phi, DZ, 60.0, 60.0, 0.0,
                             SpatialConfig(denoise="monopole", **BASE))[1]
    r = np.corrcoef(a, b)[0, 1]
    assert r > 0.99, f"r = {r:.4f}"
