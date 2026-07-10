"""The 20 MUAP regression cases — one source of truth, shared by the builder and the test.

Each case is a fully-specified (φ, config) input that runs through one engine to a
MUAP. φ is built deterministically from analytic monopoles (with fixed-seed noise on
the FEM-like cases), so the whole suite is pure NumPy/SciPy — no FEM, no external data.

The cases are chosen to exercise every knob that a synthesis refactor could disturb,
with priority on the **production recipe** (spatial + monopole denoise + one-sided
taper), which until now had no self-regression anywhere.

`scripts/synthesis/build_muap_reference.py` runs these and writes the expected outputs
to `data/muap_reference.npz`. `test_muap_reference.py` re-runs them and compares. When a
change to the outputs is *intended*, rerun the builder to refresh the reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict

import numpy as np

from emgforge.synthesis import get_mri_config, get_optimal_config
from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial
from emgforge.synthesis import generate_muap_from_phi

# --- two sampling regimes ---------------------------------------------------
# cylinder: fsamp 4096, v 4.0 ; pm/mri: fsamp 2048, v 3.3
CYL_DZ = 4.0 * 1000.0 / 4096.0     # 0.9766 mm
PM_DZ = 3.3 * 1000.0 / 2048.0      # 1.6113 mm
NZ = 256
NZ_PM = 200


def _z(n, dz):
    return (np.arange(n) - n // 2) * dz


def mono(depth, *, z0=0.0, amp=1.0, offset=0.0, n=NZ, dz=CYL_DZ):
    """A single analytic monopole lead field along a fibre: A/√(d²+(z−z0)²) + c."""
    return offset + amp / np.sqrt(depth ** 2 + (_z(n, dz) - z0) ** 2)


def mono_noisy(depth, *, seed, eps=5e-3, **kw):
    """A monopole with multiplicative ripple — stands in for a raw FEM field.

    Noise is baked in here with a fixed seed, so the resulting φ is a constant that
    both the builder and the test see identically; the test itself draws no randomness.
    """
    rng = np.random.default_rng(seed)
    base = mono(depth, **kw)
    return base * (1.0 + eps * rng.standard_normal(base.shape))


# --- case definition --------------------------------------------------------

@dataclass(frozen=True)
class MuapCase:
    name: str
    engine: str                       # "spatial" | "fourier"
    phi: np.ndarray                   # (Nz,) single fibre, or (Nfib, Nz)
    dz: float
    config: Any                       # SpatialConfig | MUAPConfig
    call_kwargs: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


def run_case(c: MuapCase):
    """Run one case to (t_ms, muap). Multi-fibre spatial sums per-fibre SFAPs."""
    if c.engine == "spatial":
        phi = np.atleast_2d(c.phi)
        t = None
        m = None
        for row in phi:
            t, s, _ = compute_sfap_spatial(row, c.dz, config=c.config, **c.call_kwargs)
            m = s if m is None else m + s
        return np.asarray(t, float), np.asarray(m, float)

    if c.engine == "fourier":
        r = generate_muap_from_phi(np.atleast_2d(c.phi), c.dz, config=c.config, **c.call_kwargs)
        return np.asarray(r.t_ms, float), np.real(np.asarray(r.muap)).astype(float).ravel()

    raise ValueError(f"unknown engine {c.engine!r}")


# --- config shorthands ------------------------------------------------------

def _spatial(**kw) -> SpatialConfig:
    base = dict(v=4.0, fsamp=4096.0, w=256, csd_derivative=2, upsample_factor=2,
                denoise="none", edge_taper_left=5, edge_taper_right=10,
                center_time=False, t_start_ms=-10.0)
    base.update(kw)
    return SpatialConfig(**base)


def _fourier(**kw):
    base = dict(v=4.0, fsamp=4096.0, w=256, len1_mm=60.0, len2_mm=60.0)
    base.update(kw)
    return replace(get_optimal_config(), **base)


# --- the 20 cases -----------------------------------------------------------

def _build_cases():
    C = []

    # ---- spatial engine (13) ----
    C += [
        MuapCase("spat_tukey_shallow", "spatial", mono(8.0), CYL_DZ,
                 _spatial(fiber_window="tukey"),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0), "shallow fibre, symmetric tukey"),
        MuapCase("spat_tukey_deep", "spatial", mono(25.0), CYL_DZ,
                 _spatial(fiber_window="tukey"),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0), "deep fibre"),
        MuapCase("spat_boxcar", "spatial", mono(12.0), CYL_DZ,
                 _spatial(fiber_window="boxcar"),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0), "hard tendon cut — EOF visible"),
        MuapCase("spat_onesided", "spatial", mono(12.0), CYL_DZ,
                 _spatial(fiber_window="one_sided"),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0), "production window"),
        MuapCase("spat_hann", "spatial", mono(12.0), CYL_DZ,
                 _spatial(fiber_window="hann"),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0), "hann window"),
        MuapCase("spat_prod_clean", "spatial", mono(12.0), CYL_DZ,
                 _spatial(fiber_window="one_sided", denoise="monopole", denoise_n_poles=3),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0),
                 "FULL production recipe on clean φ (denoise near no-op)"),
        MuapCase("spat_prod_noisy", "spatial", mono_noisy(12.0, seed=101), CYL_DZ,
                 _spatial(fiber_window="one_sided", denoise="monopole", denoise_n_poles=3),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0),
                 "FULL production recipe on FEM-like noisy φ (the real use case)"),
        MuapCase("spat_csd1_biphasic", "spatial", mono(12.0), CYL_DZ,
                 _spatial(fiber_window="one_sided", csd_derivative=1),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0), "1st-derivative CSD (old biphasic)"),
        MuapCase("spat_pm_velocity", "spatial", mono(12.0, n=NZ_PM, dz=PM_DZ), PM_DZ,
                 _spatial(v=3.3, fsamp=2048.0, w=256, fiber_window="one_sided"),
                 dict(len1_mm=55.0, len2_mm=65.0, posz_mm=0.0), "PM regime v=3.3, fsamp=2048"),
        MuapCase("spat_asym_lengths", "spatial", mono(12.0), CYL_DZ,
                 _spatial(fiber_window="one_sided"),
                 dict(len1_mm=40.0, len2_mm=100.0, posz_mm=0.0), "asymmetric semi-fibres"),
        MuapCase("spat_posz_offset", "spatial", mono(12.0, z0=15.0), CYL_DZ,
                 _spatial(fiber_window="one_sided"),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=15.0), "NMJ off the electrode"),
        MuapCase("spat_mu_sum5", "spatial",
                 np.stack([mono_noisy(d, seed=200 + k, z0=zz)
                           for k, (d, zz) in enumerate([(10, -4), (11, -1), (12, 0), (13, 2), (14, 5)])]),
                 CYL_DZ,
                 _spatial(fiber_window="one_sided", denoise="monopole", denoise_n_poles=3),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0),
                 "5-fibre MU, production recipe, summed"),
        MuapCase("spat_polarity_neg", "spatial", mono(12.0), CYL_DZ,
                 _spatial(fiber_window="one_sided", polarity=-1),
                 dict(len1_mm=60.0, len2_mm=60.0, posz_mm=0.0), "flipped output polarity"),
    ]

    # ---- fourier engine (7) ----
    C += [
        MuapCase("four_optimal", "fourier", mono(10.0), CYL_DZ, _fourier(),
                 dict(posz_mm_arr=np.array([0.0])), "optimal preset, single fibre"),
        MuapCase("four_mri_shortfibre", "fourier", mono(10.0, n=NZ, dz=CYL_DZ), CYL_DZ,
                 replace(get_mri_config(), v=4.0, fsamp=4096.0, w=256, len1_mm=30.0, len2_mm=30.0),
                 dict(posz_mm_arr=np.array([0.0])), "MRI preset, short fibre (auto edge-taper fires)"),
        MuapCase("four_multi_posz", "fourier",
                 np.stack([mono(10.0), mono(11.0), mono(12.0)]), CYL_DZ, _fourier(),
                 dict(posz_mm_arr=np.array([-5.0, 0.0, 5.0])), "3 fibres, per-fibre NMJ offset"),
        MuapCase("four_multi_lengths", "fourier",
                 np.stack([mono(10.0), mono(11.0), mono(12.0)]), CYL_DZ, _fourier(),
                 dict(len1_per_fiber=np.array([50.0, 60.0, 70.0]),
                      len2_per_fiber=np.array([70.0, 60.0, 50.0])), "3 fibres, per-fibre lengths"),
        MuapCase("four_jitter", "fourier",
                 np.stack([mono(10.0 + 0.5 * k) for k in range(5)]), CYL_DZ,
                 _fourier(nmj_sigma_mm=8.0, cv_sigma_m_per_s=0.3, tendon_sigma_mm=6.0, jitter_seed=42),
                 dict(), "in-engine RNG jitter path (seed 42)"),
        MuapCase("four_asym_lengths", "fourier", mono(10.0), CYL_DZ,
                 _fourier(len1_mm=40.0, len2_mm=100.0),
                 dict(posz_mm_arr=np.array([0.0])), "asymmetric semi-fibres"),
        MuapCase("four_deep", "fourier", mono(25.0), CYL_DZ, _fourier(),
                 dict(posz_mm_arr=np.array([0.0])), "deep fibre"),
    ]
    return C


CASES = _build_cases()
CASE_NAMES = [c.name for c in CASES]
CASE_BY_NAME = {c.name: c for c in CASES}

assert len(CASES) == 20, f"expected 20 cases, have {len(CASES)}"
assert len(CASE_NAMES) == len(set(CASE_NAMES)), "duplicate case name"
