"""Unit tests for ``emgforge.synthesis.adaptive_w``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


from emgforge.synthesis.adaptive_w import (
    choose_w,
    estimate_field_decay_from_phi,
    fallback_decay_from_depth,
    next_power_of_2,
    required_window_mm,
)


# ────────────────────────── unit pieces ──────────────────────────

def test_next_power_of_2_edge_cases():
    assert next_power_of_2(0) == 1
    assert next_power_of_2(1) == 1
    assert next_power_of_2(2) == 2
    assert next_power_of_2(256) == 256
    assert next_power_of_2(257) == 512
    assert next_power_of_2(1000) == 1024
    assert next_power_of_2(1024) == 1024


def test_required_window_fibre_dominated():
    # short field, long fibre — fibre constraint wins
    W = required_window_mm(L_fibre_mm=300.0, field_decay_mm=15.0)
    assert W == 300.0 + 30.0   # 330 (fibre + IAP envelope)


def test_required_window_field_dominated():
    # short fibre, wide field — field constraint wins
    W = required_window_mm(L_fibre_mm=120.0, field_decay_mm=80.0)
    assert W == 6.0 * 80.0      # 480


def test_fallback_decay_monotonic_in_depth():
    assert fallback_decay_from_depth(0) == 30.0
    assert fallback_decay_from_depth(5) >= 30.0
    assert fallback_decay_from_depth(30) > fallback_decay_from_depth(15)
    assert fallback_decay_from_depth(60) > fallback_decay_from_depth(30)


def test_estimate_field_decay_recovers_known_lambda():
    """Synthetic exp(-|z|/λ) — fit should recover λ within a few %."""
    z = np.linspace(-128, 128, 256)
    for lam_true in (20.0, 30.0, 50.0, 80.0):
        phi = np.exp(-np.abs(z) / lam_true)
        lam_est = estimate_field_decay_from_phi(phi, dz_mm=1.0)
        assert lam_est is not None
        rel = abs(lam_est - lam_true) / lam_true
        assert rel < 0.10, f"lam_true={lam_true}, lam_est={lam_est}, rel={rel:.3f}"


def test_estimate_field_decay_returns_none_on_garbage():
    # All zeros → no peak, fit fails
    phi = np.zeros(256)
    assert estimate_field_decay_from_phi(phi, dz_mm=1.0) is None
    # Too few samples
    phi = np.exp(-np.arange(10))
    assert estimate_field_decay_from_phi(phi, dz_mm=1.0) is None


# ────────────────────────── choose_w regimes ──────────────────────────

def _make_phi(lam_mm, n=256, dz_mm=1.0):
    """Default test phi — pre-decayed at boundary so input-length cap doesn't trigger."""
    # Center the exponential, then guarantee the boundary is below the
    # 5% edge-decay threshold by extending support if needed.
    z = (np.arange(n) - n // 2) * dz_mm
    return np.exp(-np.abs(z) / lam_mm)


def _make_phi_decayed(lam_mm, n=2048, dz_mm=1.0):
    """A phi sampled over enough extent that the boundary is well below 5%."""
    # For lam=300 mm and 5% threshold, |z| at edge must be >= 3·lam = 900 mm.
    # n=2048, dz=1 → span 2048 mm → half-span 1024 mm. Always safe.
    z = (np.arange(n) - n // 2) * dz_mm
    return np.exp(-np.abs(z) / lam_mm)


def test_choose_w_short_fibre_narrow_field():
    """L=120mm, λ=15 → W=max(90,150)=150 → w=256 (clamped at floor)."""
    w = choose_w(L_fibre_mm=120, phi_z=_make_phi(15))
    assert w == 256


def test_choose_w_short_fibre_wide_field():
    """λ=80, well-decayed phi → 6λ=480 → w=512."""
    w = choose_w(L_fibre_mm=120, phi_z=_make_phi_decayed(80))
    assert w == 512


def test_choose_w_long_fibre_narrow_field():
    """L=300, +30 IAP envelope = 330 → next_pow2=512."""
    w = choose_w(L_fibre_mm=300, phi_z=_make_phi_decayed(15))
    assert w == 512


def test_choose_w_both_stressors():
    """L=300 (→ fibre 330), λ=150 (→ 6λ=900) → 1024 with well-decayed input."""
    w = choose_w(L_fibre_mm=300, phi_z=_make_phi_decayed(150))
    assert w == 1024


def test_choose_w_respects_w_min():
    """Even with a tiny λ and short fibre, w_min holds the floor."""
    w = choose_w(L_fibre_mm=60, phi_z=_make_phi(5), w_min=256)
    assert w == 256


def test_choose_w_respects_w_max():
    """Pathologically wide field clamped to w_max."""
    w = choose_w(L_fibre_mm=400, phi_z=_make_phi_decayed(300), w_min=256, w_max=1024)
    assert w == 1024


def test_choose_w_input_length_cap_when_not_decayed():
    """Heuristic wants 512 but input is only 256 samples and not decayed at the edge.

    The safety cap should hold w at 256 to prevent the wider-window zero-pad
    re-introducing a boundary step (FEM_AND_MESH_GUIDE.md §11).
    """
    # phi at lam=80, 256 samples, edge-decay = exp(-128/80) ≈ 0.20 → not decayed
    phi = _make_phi(80, n=256, dz_mm=1.0)
    w = choose_w(L_fibre_mm=120, phi_z=phi)
    assert w == 256, f"expected cap to 256, got {w}"


def test_choose_w_input_length_cap_disabled():
    """When cap_to_input_length=False, behave like the un-safed heuristic."""
    phi = _make_phi(80, n=256, dz_mm=1.0)
    w = choose_w(L_fibre_mm=120, phi_z=phi, cap_to_input_length=False)
    assert w == 512


def test_choose_w_depth_fallback():
    """Without phi_z, use depth_mm as proxy."""
    w_shallow = choose_w(L_fibre_mm=120, depth_mm=15)
    w_deep = choose_w(L_fibre_mm=120, depth_mm=30)
    assert w_shallow >= 256
    assert w_deep >= w_shallow


def test_choose_w_no_input_uses_conservative_default():
    """No phi_z, no depth_mm → still returns a valid w (with warning)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = choose_w(L_fibre_mm=120, verbose=False)
    assert w >= 256


# ────────────────────────── pipeline integration ──────────────────────────

def test_pipeline_adaptive_w_via_api():
    """When MUAPConfig.w=None, the production API should select adaptively.

    Use a well-decayed wide-field phi (2048 samples covering 2 m) so the
    safety cap doesn't trigger. Heuristic should grow w to 512.
    """
    from emgforge.synthesis.api import MUAPConfig, generate_muap_from_phi

    dz = 4.0 * 1000.0 / 4096.0
    z = (np.arange(2048) - 1024) * dz
    phi_wide = np.exp(-np.abs(z) / 80.0).reshape(1, -1)

    cfg = MUAPConfig(w=None)
    res = generate_muap_from_phi(phi_wide, dz_mm=dz, config=cfg)
    assert res.t_ms.shape == (512,)
    assert res.muap.shape == (512,)


def test_pipeline_explicit_w_unchanged():
    """When MUAPConfig.w is explicit, no adaptation."""
    from emgforge.synthesis.api import MUAPConfig, generate_muap_from_phi

    z = (np.arange(256) - 128) * (4.0 * 1000.0 / 4096.0)
    phi_wide = np.exp(-np.abs(z) / 80.0).reshape(1, -1)

    cfg = MUAPConfig(w=256)
    res = generate_muap_from_phi(phi_wide, dz_mm=4.0 * 1000.0 / 4096.0, config=cfg)
    assert res.t_ms.shape == (256,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
