"""The validated route is the default, and nobody can silently use the other one.

Four guards, in the order a reader would want them:

1. ``production_config()`` is the frozen direct line-source recipe
   (``synthesis/DIRECT_LINE_SOURCE.md`` §0), field by field, and every alias of it
   (``emgforge.mri.pipeline.production_config``, ``SpatialConfig.production``,
   ``scripts/validation/harness.golden_cfg``) is that same value.
2. ``field_to_muap(field, bed)`` with no config IS that recipe, bit for bit, in
   physical time from −10 ms.
3. The spatial engine reproduces the closed-form line-source SFAP of an infinite
   anisotropic medium (Rosenfalck 1969; Andreassen & Rosenfalck 1981): shape, timing,
   and an amplitude that is flat in the conduction velocity (no 1/v), from a
   monopole-free source. This is validation check A0 of ``scripts/validation``,
   pinned in CI.
4. Selecting the Fourier engine (``MUAPConfig``) warns that it is comparison-only.

Pure NumPy; runs in a few seconds.
"""
from __future__ import annotations

import importlib.util
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import emgforge.synthesis.api as api
from emgforge.synthesis import (
    FOURIER_ROUTE_WARNING,
    FibreBed,
    MUAPConfig,
    SpatialConfig,
    field_to_muap,
    production_config,
)
from emgforge.synthesis.engines.spatial import build_csd_matrix, compute_sfap_spatial
from emgforge.synthesis.iap import rosenfalck_vm

ROOT = Path(__file__).resolve().parents[2]

# ───────────────────────── 1. the frozen recipe ─────────────────────────

# DIRECT_LINE_SOURCE.md §0, verbatim. Changing any of these is a deliberate act that
# must come with a refreshed reference set (scripts/synthesis/build_muap_reference.py).
RECIPE = dict(
    denoise="monopole", denoise_n_poles=3,
    fiber_window="one_sided", tukey_alpha=0.25,
    csd_derivative=2, upsample_factor=2,
    edge_taper_left=5, edge_taper_right=10,
    center_time=False, t_start_ms=-10.0,
    sigma_in=1.0, fiber_radius_mm=0.05, polarity=1,
)
REGIME = dict(v=4.0, fsamp=2048.0, w=256)          # the only free parameters


def test_production_config_is_the_frozen_recipe():
    cfg = production_config()
    assert isinstance(cfg, SpatialConfig)
    for k, v in {**RECIPE, **REGIME}.items():
        assert getattr(cfg, k) == v, f"production_config().{k} = {getattr(cfg, k)!r}, recipe says {v!r}"
    # the regime knobs are the only ones that move
    cyl = production_config(fs=4096.0, v=3.0, w=512)
    assert (cyl.fsamp, cyl.v, cyl.w) == (4096.0, 3.0, 512)
    assert replace(cyl, **REGIME) == cfg


def test_every_alias_is_the_same_value():
    from emgforge.mri import pipeline as P
    assert P.production_config() == production_config()
    assert P.production_config(fs=4096.0, v=3.3, w=128) == production_config(fs=4096.0, v=3.3, w=128)
    assert SpatialConfig.production() == production_config()
    harness = _load_harness()
    if harness is not None:
        assert harness.golden_cfg() == production_config(fs=harness.FS, v=harness.V, w=harness.W)


def test_bare_spatial_config_is_not_the_recipe():
    """The bare defaults are frozen for the reference sets; they must not drift INTO the
    recipe silently either (a reader would then think one implies the other)."""
    assert SpatialConfig() != production_config()
    assert "not the production recipe" in SpatialConfig.__doc__


# ───────────────────────── 2. the default route ─────────────────────────

FS, V, W = 4096.0, 4.0, 256                         # cylinder regime
DZ = V * 1000.0 / FS                                # 0.977 mm
Z = (np.arange(W) - W // 2) * DZ


def _phi_line(rho_mm: float = 10.0) -> np.ndarray:
    return _phi_inf(Z, rho_mm)


def test_none_config_is_production_config_bit_for_bit():
    bed = FibreBed.from_arrays(DZ, [60.0, 40.0, 30.0], [60.0, 80.0, 90.0], [0.0, -20.0, -30.0], V)
    phi = np.vstack([_phi_line(10.0), _phi_line(12.0), _phi_line(15.0)])
    got = field_to_muap(phi, bed)
    ref = field_to_muap(phi, bed, production_config())
    assert np.array_equal(got.muap, ref.muap)
    assert np.array_equal(got.t_ms, ref.t_ms)
    assert got.config == production_config() and isinstance(got.config, SpatialConfig)
    assert got.time_convention == "physical"
    assert got.t_ms[0] == -10.0                       # physical time, −10 ms pre-roll
    assert np.isclose(got.t_ms[1] - got.t_ms[0], 1000.0 / 2048.0)


def test_single_fibre_primitive_defaults_to_the_recipe_too():
    t0, s0, _ = compute_sfap_spatial(_phi_line(), DZ, 60.0, 60.0, 0.0)
    t1, s1, _ = compute_sfap_spatial(_phi_line(), DZ, 60.0, 60.0, 0.0, production_config())
    assert np.array_equal(s0, s1) and np.array_equal(t0, t1)


# ───────────────────────── 3. first principles ─────────────────────────
# Line-source SFAP in an INFINITE ANISOTROPIC medium. φ has a closed form, so the SFAP
# is evaluated in the dual form  ∫ Vm(z,t)·win(z)·φ''(z) dz  on a fine grid with NO
# numerical derivative of the tendon step (integration by parts; exact for a boxcar
# fibre end). This mirrors scripts/validation/harness.sfap_first_principles; the
# harness copy is checked against this one below so the two cannot drift apart.

S_R, S_Z = 0.1, 0.5                 # muscle σ (S/m): radial / along the fibre
SIGMA_IN, A_FIB = 1.0, 0.05         # engine defaults (σ_in, fibre radius mm)


def _phi_inf(z, rho=10.0, s_r=S_R, s_z=S_Z):
    K = 1.0 / (4 * np.pi * s_r * np.sqrt(s_z))
    return K / np.sqrt(rho ** 2 / s_r + z ** 2 / s_z)


def _phi_inf_dd(z, rho=10.0, s_r=S_R, s_z=S_Z):
    K = 1.0 / (4 * np.pi * s_r * np.sqrt(s_z))
    A = rho ** 2 / s_r
    u = A + z ** 2 / s_z
    return K * (3 * z ** 2 / s_z ** 2 * u ** -2.5 - (1 / s_z) * u ** -1.5)


def _sfap_first_principles(posz=0.0, len1=60.0, len2=60.0, v=V, fs=FS, w=W, t0=-10.0,
                           rho=10.0, dz_fine=0.02):
    t = np.arange(w) / fs * 1000.0 + t0
    z = np.arange(posz - len1, posz + len2 + dz_fine / 2, dz_fine)
    Vm = rosenfalck_vm(v * t[:, None] - np.abs(z[None, :] - posz))
    return t, (Vm @ _phi_inf_dd(z, rho)) * dz_fine * (SIGMA_IN * np.pi * A_FIB ** 2)


def _best_r(t_ref, ref, t_b, b, max_lag_ms=6.0, dt=0.05):
    """Sign-aware, lag-bounded agreement of peak-normalised waveforms:
    (signed r at the best lag, that lag in ms, amplitude ratio b/ref)."""
    rn, bn = ref / np.abs(ref).max(), b / np.abs(b).max()
    best = (0.0, 0.0)
    for lag in np.arange(-max_lag_ms, max_lag_ms + dt, dt):
        bi = np.interp(t_ref, t_b + lag, bn, left=0, right=0)
        if bi.std() < 1e-12:
            continue
        r = float(np.corrcoef(rn, bi)[0, 1])
        if abs(r) > abs(best[0]):
            best = (r, float(lag))
    return best[0], best[1], float(np.abs(b).max() / np.abs(ref).max())


def _oracle_cfg(**over) -> SpatialConfig:
    """The recipe against a closed-form φ: no denoising (there is no mesh ripple to
    remove) and the oracle's boxcar tendon; everything else is production."""
    return replace(production_config(fs=FS, v=V, w=W), fiber_window="boxcar", denoise="none", **over)


def _load_harness():
    """scripts/validation/harness.py, if its dependencies are present (it is not a package)."""
    path = ROOT / "scripts/validation/harness.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_route_audit_harness", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:          # matplotlib / analytical stack missing → skip the drift check
        return None


def test_in_test_oracle_matches_the_validation_harness():
    harness = _load_harness()
    if harness is None:
        pytest.skip("scripts/validation/harness.py not importable here")
    t_h, s_h = harness.sfap_first_principles(-20.0, 40.0, 80.0, v=V, fs=FS, w=W)
    t_t, s_t = _sfap_first_principles(-20.0, 40.0, 80.0)
    assert np.array_equal(t_h, t_t) and np.array_equal(s_h, s_t)
    assert np.array_equal(harness.phi_inf(Z), _phi_inf(Z))


@pytest.mark.parametrize("posz,len1,len2", [(0.0, 60.0, 60.0), (-20.0, 40.0, 80.0), (-30.0, 30.0, 90.0)],
                         ids=["nmj-under-electrode", "nmj-20mm", "nmj-30mm"])
def test_spatial_engine_reproduces_the_line_source_integral(posz, len1, len2):
    """A0.1: same integral, so only discretisation can differ — r ≥ 0.999, |lag| ≤ 0.1 ms."""
    t_r, s_r = _sfap_first_principles(posz, len1, len2)
    t_s, s_s, _ = compute_sfap_spatial(_phi_line(), DZ, len1, len2, posz, _oracle_cfg())
    r, lag, amp = _best_r(t_r, s_r, t_s, s_s)
    assert r >= 0.999, f"signed r = {r:.5f}"            # signed: a polarity flip must fail
    assert abs(lag) <= 0.1, f"lag = {lag:+.3f} ms"
    assert 0.98 <= amp <= 1.01, f"amplitude ratio = {amp:.4f}"


@pytest.mark.parametrize("v", [2.0, 4.0])
def test_spatial_engine_amplitude_is_flat_in_cv(v):
    """A0.2: for a spatially defined IAP the extracellular potential does not depend on
    CV. The CSD is already σ_in·π·a²·∂²Vm/∂z², so there is no 1/v prefactor — with one,
    the ratio would read 0.50 at v = 2 and 0.25 at v = 4 (the bug removed 2026-09-15)."""
    t_r, s_r = _sfap_first_principles(0.0, 60.0, 60.0, v=v)
    t_s, s_s, _ = compute_sfap_spatial(_phi_line(), DZ, 60.0, 60.0, 0.0, _oracle_cfg(v=v))
    r, lag, amp = _best_r(t_r, s_r, t_s, s_s)
    assert r >= 0.999 and abs(lag) <= 0.1
    assert 0.98 <= amp <= 1.01, f"amplitude ratio {amp:.4f} at v = {v}"


@pytest.mark.parametrize("window", ["one_sided", "boxcar"])
def test_source_is_monopole_free(window):
    """A0.5: a fibre injects no net current — ∫ i_m(z,t) dz = 0 at every instant, so the
    generation and end-of-fibre terms exactly balance the propagating tripoles."""
    cfg = replace(production_config(fs=FS, v=V, w=W), fiber_window=window)
    t = np.arange(W) / FS * 1000.0 + cfg.t_start_ms
    csd = build_csd_matrix(Z, t, -20.0, 40.0, 80.0, DZ, cfg)
    net = float(np.max(np.abs(csd.sum(axis=1))) / np.max(np.abs(csd).sum(axis=1)))
    assert net < 1e-12, f"max_t |Σ_z i_m| / Σ_z |i_m| = {net:.1e}"


# ───────────────────────── 4. the other route warns ─────────────────────────

def test_fourier_route_warns_once_per_process():
    bed = FibreBed.uniform(2, dz_mm=DZ, len1_mm=60.0, len2_mm=60.0, v=V)
    fourier = MUAPConfig(v=V, fsamp=FS, w=W)
    api._FOURIER_WARNED = False                       # re-arm: earlier tests may have fired it
    with pytest.warns(UserWarning, match="comparison only") as rec:
        field_to_muap(_phi_line(), bed, fourier)
    assert [str(w.message) for w in rec if issubclass(w.category, UserWarning)] == [FOURIER_ROUTE_WARNING]
    assert "DIRECT_LINE_SOURCE.md" in FOURIER_ROUTE_WARNING
    with warnings.catch_warnings():                   # second call in the same process: silent
        warnings.simplefilter("error")
        field_to_muap(_phi_line(), bed, fourier)


def test_production_route_is_silent():
    bed = FibreBed.uniform(2, dz_mm=DZ, len1_mm=60.0, len2_mm=60.0, v=V)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        field_to_muap(_phi_line(), bed)
        field_to_muap(_phi_line(), bed, production_config(fs=FS))
