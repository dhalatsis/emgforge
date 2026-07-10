"""Unit tests for ``edge_taper="auto"`` HF-aware edge-taper detection.

Mirrors `tests/test_auto_smoothing.py`. The auto-taper feature
addresses the Round 1 verification gap (Phase 3 cross-geometry sign-flip
still reproducible with `get_adaptive_config()` because the legacy
`edge_taper=0` default left truncated FEM input unprotected).

Pattern:
  - `_edge_over_peak()` is the detector (analytical: ~1e-3; truncated FEM: ~1.0)
  - `MUAPConfig.edge_taper` accepts int (legacy) OR "auto" (new)
  - "auto" triggers an `auto_edge_taper_n`-sample taper iff
    `_edge_over_peak(phi) >= auto_edge_taper_threshold` (default 0.3).

See: field_to_muap_study/deliverables/fem_worker_integration/CHECKLIST_ROUND2.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


from emgforge.synthesis.api import (
    MUAPConfig,
    _edge_over_peak,
    _compute_muap_core,  # noqa: F401 (imported for completeness — not used directly)
    generate_muap_from_phi,
    get_adaptive_config,
)


def _decayed_phi(n=256, lam_mm=15.0):
    """Sharp Gaussian-decayed phi — analytical-like; edge/peak ≈ 0."""
    z = (np.arange(n) - n // 2) * 1.0
    return np.exp(-(z ** 2) / (2.0 * lam_mm ** 2))


def _truncated_phi(n=256, value=1.0):
    """Flat phi with no decay — edge/peak ≈ 1.0, the Phase 3 failure pattern."""
    rng = np.random.default_rng(0)
    return np.full(n, value) + 0.01 * rng.standard_normal(n)


# ────────────────────────── detector ──────────────────────────

def test_edge_over_peak_analytical_is_tiny():
    """A sharply-decayed Gaussian phi has edge/peak ≈ 0 (well below 0.3)."""
    phi = _decayed_phi(256, lam_mm=15.0)
    eop = _edge_over_peak(phi)
    assert eop < 0.01, f"expected edge/peak < 0.01, got {eop:.4f}"


def test_edge_over_peak_truncated_is_large():
    """A flat phi (no decay) has edge/peak ≈ 1 (well above 0.3)."""
    phi = _truncated_phi(256)
    eop = _edge_over_peak(phi)
    assert eop > 0.5, f"expected edge/peak > 0.5, got {eop:.4f}"


def test_edge_over_peak_2d_uses_worst_edge():
    """For 2-D input (Nfib, Nz) the helper takes the worst edge across rows."""
    phi_dec = _decayed_phi(256, lam_mm=15.0)
    phi_tru = _truncated_phi(256)
    mat = np.stack([phi_dec, phi_tru])
    eop = _edge_over_peak(mat)
    # The worst (truncated) row dominates
    assert eop > 0.5, f"2-D detector should pick up worst row, got {eop:.4f}"


# ────────────────────────── conditional application ──────────────────────────

def test_auto_edge_taper_skips_decayed():
    """Decayed input: auto-taper should be a no-op (taper=0 path)."""
    phi = _decayed_phi(256).reshape(1, -1)
    cfg_auto = MUAPConfig(denoise="none", edge_taper="auto")
    cfg_none = MUAPConfig(denoise="none", edge_taper=0)
    res_auto = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_auto)
    res_none = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_none)
    assert np.allclose(res_auto.muap, res_none.muap, atol=1e-12)


def test_auto_edge_taper_applies_to_truncated():
    """Truncated input: auto-taper should match explicit edge_taper=15."""
    phi = _truncated_phi(256).reshape(1, -1)
    cfg_auto    = MUAPConfig(denoise="none", edge_taper="auto")
    cfg_explicit = MUAPConfig(denoise="none", edge_taper=15)
    cfg_none    = MUAPConfig(denoise="none", edge_taper=0)
    res_auto = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_auto)
    res_explicit = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_explicit)
    res_none = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_none)
    # auto should match explicit
    assert np.allclose(res_auto.muap, res_explicit.muap, atol=1e-12), \
        "auto edge_taper on truncated input should match explicit=15"
    # And differ from no-taper
    assert not np.allclose(res_auto.muap, res_none.muap, atol=1e-6), \
        "auto edge_taper on truncated input should NOT match edge_taper=0"


def test_auto_edge_taper_threshold_override():
    """Bump the threshold above 1.0 — even truncated input shouldn't trigger."""
    phi = _truncated_phi(256).reshape(1, -1)
    cfg = MUAPConfig(denoise="none", edge_taper="auto",
                      auto_edge_taper_threshold=2.0)
    cfg_none = MUAPConfig(denoise="none", edge_taper=0)
    res_a = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg)
    res_n = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_none)
    assert np.allclose(res_a.muap, res_n.muap, atol=1e-12)


def test_auto_edge_taper_n_override():
    """The auto-taper sample count should be configurable."""
    phi = _truncated_phi(256).reshape(1, -1)
    cfg_n5  = MUAPConfig(denoise="none", edge_taper="auto",
                          auto_edge_taper_n=5)
    cfg_e5  = MUAPConfig(denoise="none", edge_taper=5)
    res_n5 = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_n5)
    res_e5 = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_e5)
    assert np.allclose(res_n5.muap, res_e5.muap, atol=1e-12), \
        "auto_edge_taper_n should control the taper width when triggered"


def test_invalid_edge_taper_string_raises():
    """Anything other than 'auto' or int should raise ValueError."""
    phi = _decayed_phi(256).reshape(1, -1)
    cfg = MUAPConfig(denoise="none", edge_taper="banana")
    with pytest.raises(ValueError):
        generate_muap_from_phi(phi, dz_mm=1.0, config=cfg)


# ────────────────────────── default preset wiring ──────────────────────────

def test_adaptive_config_uses_auto_taper():
    """get_adaptive_config() should ship with edge_taper='auto'."""
    cfg = get_adaptive_config()
    assert cfg.edge_taper == "auto"
    # And the other adaptive defaults still in place
    assert cfg.denoise == "auto"
    assert cfg.w is None


def test_legacy_default_keeps_edge_taper_zero():
    """MUAPConfig() must still default to edge_taper=0 for back compat."""
    cfg = MUAPConfig()
    assert cfg.edge_taper == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
