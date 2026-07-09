"""Unit tests for the ``smoothing_method='auto'`` HF-aware smoothing path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muap_generator.api import (
    MUAPConfig,
    _apply_smoothing,
    _hf_fraction,
    get_adaptive_config,
)


def _clean_phi(n=256):
    """Smooth (Gaussian) analytical-like phi — HF content near machine eps.

    A Gaussian with σ ~ window/8 is smooth enough that its rfft has essentially
    zero energy beyond ~0.1 normalised frequency.
    """
    z = (np.arange(n) - n // 2) * 1.0
    return np.exp(-(z ** 2) / (2.0 * (n / 8.0) ** 2))


def _noisy_phi(n=256, noise_scale=1e-3, seed=0):
    """Clean phi plus white noise — simulates FEM HF noise."""
    rng = np.random.default_rng(seed)
    return _clean_phi(n) + noise_scale * rng.standard_normal(n)


def test_hf_fraction_clean_is_tiny():
    """A smooth analytical phi should have HF well below the 1e-10 default threshold."""
    phi = _clean_phi(256)
    hf = _hf_fraction(phi)
    assert hf < 1e-10, f"smooth phi HF should be below auto threshold, got {hf:.2e}"


def test_hf_fraction_noisy_is_large():
    phi = _noisy_phi(256, noise_scale=1e-3)
    hf = _hf_fraction(phi)
    assert hf > 1e-6, f"noisy phi HF should be measurable, got {hf:.2e}"


def test_auto_smoothing_skips_clean():
    """Clean input: auto should pass through unchanged (no smoothing)."""
    phi = _clean_phi(256).reshape(1, -1)
    cfg = MUAPConfig(smoothing_method="auto")
    out = _apply_smoothing(phi, cfg)
    assert np.allclose(out, phi)


def test_auto_smoothing_filters_noisy():
    """Noisy input: auto should apply Butterworth (output differs from input)."""
    phi = _noisy_phi(256, noise_scale=1e-3).reshape(1, -1)
    cfg = MUAPConfig(smoothing_method="auto")
    out = _apply_smoothing(phi, cfg)
    assert not np.allclose(out, phi)
    # Output should be smoother than input — HF fraction drops
    hf_in = _hf_fraction(phi[0])
    hf_out = _hf_fraction(out[0])
    assert hf_out < hf_in / 10, f"smoothing should reduce HF, in={hf_in:.2e}, out={hf_out:.2e}"


def test_auto_smoothing_threshold_override():
    """High threshold disables auto-smoothing even on noisy input."""
    phi = _noisy_phi(256, noise_scale=1e-3).reshape(1, -1)
    cfg = MUAPConfig(smoothing_method="auto", auto_smoothing_hf_threshold=1.0)
    out = _apply_smoothing(phi, cfg)
    assert np.allclose(out, phi)


def test_adaptive_config_uses_auto():
    """The recommended preset should ship with auto smoothing."""
    cfg = get_adaptive_config()
    assert cfg.smoothing_method == "auto"
    assert cfg.w is None  # adaptive window too


def test_auto_smoothing_via_api_clean_input():
    """End-to-end: clean analytical-style input → no smoothing → output close
    to the unsmoothed pipeline output."""
    from muap_generator.api import generate_muap_from_phi

    phi = _clean_phi(256).reshape(1, -1)
    cfg_auto = MUAPConfig(smoothing_method="auto", w=256)
    cfg_none = MUAPConfig(smoothing_method="none", w=256)
    res_auto = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_auto)
    res_none = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_none)
    assert np.allclose(res_auto.muap, res_none.muap, atol=1e-10)


def test_auto_smoothing_via_api_noisy_input():
    """End-to-end: noisy input → smoothing applied → output close to butterworth path."""
    from muap_generator.api import generate_muap_from_phi

    phi = _noisy_phi(256, noise_scale=1e-3).reshape(1, -1)
    cfg_auto = MUAPConfig(smoothing_method="auto", w=256)
    cfg_bw = MUAPConfig(smoothing_method="butterworth", w=256)
    res_auto = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_auto)
    res_bw = generate_muap_from_phi(phi, dz_mm=1.0, config=cfg_bw)
    assert np.allclose(res_auto.muap, res_bw.muap, atol=1e-10)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
