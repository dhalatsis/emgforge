"""Shared harness for the validation suite: the check registry, the golden recipe,
the analytical (Farina 2004) oracle wrappers, and small signal helpers.

Every tier script imports this, registers its checks with ``check(...)`` and ends
with ``finish(tier)``. ``run_all.py`` concatenates the per-tier JSON records into
``_results/validation/VALIDATION_REPORT.md``.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")           # noqa: E702
import matplotlib.pyplot as plt                     # noqa: E402,F401  (tiers use plt)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                       # tests.regression.analytical_ref
OUT = ROOT / "_results/validation"
OUT.mkdir(parents=True, exist_ok=True)

from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial  # noqa: E402
from emgforge.synthesis.metrics import align_score                                   # noqa: E402
from tests.regression.analytical_ref import (                                         # noqa: E402
    ANAL_CONDUCTIVITIES, ANAL_GEOMETRY, AnalyticalCase, analytical_phi_along_fibre)

# ----------------------------------------------------------------------------- recipe
V, FS, W = 4.0, 4096.0, 256          # cylinder-tier regime (dz = v/fs = 0.977 mm)


def golden_cfg(**over) -> SpatialConfig:
    """The production spatial recipe (GOLDEN_METHOD.md §0), cylinder regime."""
    base = SpatialConfig(denoise="monopole", denoise_n_poles=3, csd_derivative=2,
                         upsample_factor=2, fiber_window="one_sided",
                         edge_taper_left=5, edge_taper_right=10, center_time=False,
                         t_start_ms=-10.0, v=V, fsamp=FS, w=W, polarity=1)
    return replace(base, **over) if over else base


def sfap(phi, dz, len1=60.0, len2=60.0, posz=0.0, cfg=None):
    """Golden-recipe SFAP → (t_ms physical, sfap)."""
    t, s, _ = compute_sfap_spatial(np.asarray(phi, float), float(dz), len1, len2, posz,
                                   cfg or golden_cfg())
    return np.asarray(t, float), np.asarray(s, float)


# ----------------------------------------------------------------------------- registry
RESULTS: list[dict] = []
_T0 = time.time()


def check(tier: str, name: str, passed, measured: str, expect: str,
          principle: str = "", refs: str = "", known: bool = False):
    """Record one check. ``known=True`` flags a documented limitation: it is reported
    but does not fail the tier (mirrors scripts/sanity/simulator_sanity.py)."""
    passed = bool(passed)
    rec = dict(tier=tier, name=name, passed=passed, known=known, measured=str(measured),
               expect=str(expect), principle=principle, refs=refs)
    RESULTS.append(rec)
    tag = "KNOWN" if (known and not passed) else ("PASS" if passed else "FAIL")
    mark = {"PASS": "✓", "FAIL": "✗", "KNOWN": "⚠"}[tag]
    print(f"  [{mark} {tag:5s}] {tier} · {name}")
    print(f"            measured : {measured}")
    print(f"            expect   : {expect}")
    if refs:
        print(f"            refs     : {refs}")
    print()
    return passed


def finish(tier: str) -> int:
    """Summarise: pass / total counts EVERY check (a passing 'known' check is a pass);
    'known' = documented limitations that failed and do not fail the tier."""
    recs = [r for r in RESULTS if r["tier"] == tier]
    n_pass = sum(r["passed"] for r in recs)
    n_known = sum(1 for r in recs if r["known"] and not r["passed"])
    n_fail = sum(1 for r in recs if not r["known"] and not r["passed"])
    print("=" * 74)
    print(f"  {tier}: {n_pass}/{len(recs)} checks passed"
          + (f"  ·  {n_known} known limitation(s) flagged" if n_known else "")
          + (f"  ·  {n_fail} FAILED" if n_fail else "")
          + f"  ·  {time.time() - _T0:.0f}s")
    print("=" * 74)
    (OUT / f"{tier}.json").write_text(json.dumps(RESULTS, indent=1))
    return 0 if n_fail == 0 else 1


# ----------------------------------------------------------------------------- analytical oracle
G = dict(ANAL_GEOMETRY)              # r_bone 10, r_muscle 35, r_fat 38, r_skin 40
COND = {k: dict(v) for k, v in ANAL_CONDUCTIVITIES.items()}


def ana_phi(depth=30.0, distfib=0.0, v=V, fs=FS, w=W, dim1=5.0):
    """Analytical lead field φ(z) along a fibre at radial position ``depth`` (mm from the
    axis), electrode at θ=``distfib``. Centred on index n//2. Returns (phi, dz)."""
    return analytical_phi_along_fibre(AnalyticalCase(
        fiber_depth_mm=depth, v_m_per_s=v, fsamp_hz=fs, w=w, distfib_deg=distfib,
        electrode_dim1_mm=dim1))


def farina(depth=30.0, zi=0.0, L1=60.0, L2=60.0, det_type=1, channels=1, dint=10.0,
           center=0.0, distfib=0.0, dim1=5.0, dim2=None, electrode_type=None,
           v=V, fs=FS, w=W, n_fibers=1, radius=0.1, inn=0.1, ten=0.1, seed=42,
           geo=None, cond=None, physical=True):
    """Farina-2004 analytical MUAP(s) — the oracle.

    Fibre spans z ∈ [−L2, +L1] with the NMJ at ``zi`` (MATLAB convention: L1/L2 are
    tendon *positions*). ``det_type`` 1 mono / 2 SD / 3 DD; ``channels`` along z at
    ``dint`` spacing about ``center``. ``physical=True`` converts the window-centred,
    time-reversed raw output to physical time (NMJ fires at t=0).
    Returns (t_ms, sig (channels, w), z_center (channels,)).
    """
    from emgforge.analytical import (CylindricalVolumeConductor, DetectionSystem,
                                     MotorUnit, SignalGenerator)
    g = {**G, **(geo or {})}
    c = cond or COND
    np.random.seed(seed)
    vc = CylindricalVolumeConductor(r_bone=g["r_bone"], r_muscle=g["r_muscle"],
                                    r_fat=g["r_fat"], r_skin=g["r_skin"],
                                    fiber_depth=depth, conductivities=c, v=v)
    h, d = g["r_skin"] - g["r_fat"], g["r_fat"] - g["r_muscle"]
    y0 = g["r_skin"] - g["r_fat"] - (g["r_skin"] - depth)
    mu = MotorUnit(n_fibers=n_fibers, radius=radius, y0=y0, innervation_spread=inn,
                   zi=zi, L1=L1, L2=L2, Ten1=ten, Ten2=ten, distfib=distfib,
                   r=g["r_skin"], h=h, d=d)
    et = electrode_type or ("circ" if det_type == 1 else "rect")
    det = DetectionSystem(channels=channels, dint=dint, center=center, alpha=0.0,
                          det_type=det_type, dintsf=dint, electrode_type=et,
                          dim1=dim1, dim2=(dim1 if dim2 is None else dim2), r=g["r_skin"])
    with contextlib.redirect_stdout(io.StringIO()):
        t, sig, _ = SignalGenerator(vc, mu, det, v=v, fsamp=fs, w=w).generate_muap()
    t, sig = np.asarray(t, float), np.asarray(sig, float)
    if physical:                                 # flip to causal, NMJ at t=0
        T = w / fs * 1000.0
        return t - T / 2.0, sig[:, ::-1], det.z_center
    return t, sig, det.z_center


def shift_phi(phi, dz, shift_mm):
    """Translate φ so its feature moves to +shift_mm (z-invariant conductor). Cubic-spline
    resampling: linear interpolation smooths fractional-sample shifts unequally, which
    corrupts differences between electrodes (SD/DD montages)."""
    from scipy.interpolate import CubicSpline
    n = len(phi)
    z = (np.arange(n) - n // 2) * dz
    zq = np.clip(z - shift_mm, z[0], z[-1])
    return CubicSpline(z, phi)(zq)


# ----------------------------------------------------------------------------- signal helpers
def p2p(x):
    return float(np.ptp(np.asarray(x)))


def sub_lag(a, b):
    """Cross-correlation lag (samples) of b relative to a, parabolic sub-sample refinement."""
    cc = np.correlate(b, a, "full")
    k = int(np.argmax(cc)); delta = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return (k + delta) - (len(a) - 1)


def cv_from_array(sig, ied_mm, fs):
    """Conduction velocity (m/s) + R² from a channel stack (n_ch, n_t) along the fibre,
    by linear fit of the per-channel cross-correlation lag vs channel index."""
    n = sig.shape[0]
    lags = np.array([0.0] + [sub_lag(sig[0], sig[i]) for i in range(1, n)])
    A = np.vstack([np.arange(n), np.ones(n)]).T
    coef = np.linalg.lstsq(A, lags, rcond=None)[0]
    fit = A @ coef
    r2 = 1 - np.sum((lags - fit) ** 2) / max(np.sum((lags - lags.mean()) ** 2), 1e-12)
    cv = ied_mm / (abs(coef[0]) * 1000.0 / fs)               # mm / ms = m/s
    return float(cv), float(r2), lags


def mnf(x, fs):
    """Mean (power-weighted) frequency, Hz."""
    X = np.abs(np.fft.rfft(np.asarray(x) - np.mean(x))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float((f * X).sum() / max(X.sum(), 1e-30))


def duration_ms(t, x, frac=0.05):
    """Width of the interval where |x| exceeds ``frac``·max|x| (ms)."""
    a = np.abs(x); idx = np.where(a >= frac * a.max())[0]
    return float(t[idx[-1]] - t[idx[0]]) if len(idx) else 0.0


def fwhm(xgrid, y):
    """Full width at half maximum of |y| over ``xgrid`` (linear interpolation)."""
    a = np.abs(np.asarray(y)); k = int(np.argmax(a)); half = a[k] / 2
    lo = k
    while lo > 0 and a[lo] > half:
        lo -= 1
    hi = k
    while hi < len(a) - 1 and a[hi] > half:
        hi += 1
    xl = np.interp(half, [a[lo], a[lo + 1]], [xgrid[lo], xgrid[lo + 1]]) if lo < k else xgrid[lo]
    xr = np.interp(half, [a[hi], a[hi - 1]], [xgrid[hi], xgrid[hi - 1]]) if hi > k else xgrid[hi]
    return float(xr - xl)


def power_law(d, a):
    """Fit a ∝ d^−n on log-log; returns (n, r²)."""
    x, y = np.log(np.asarray(d, float)), np.log(np.asarray(a, float))
    coef = np.polyfit(x, y, 1)
    fit = np.polyval(coef, x)
    r2 = 1 - np.sum((y - fit) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return float(-coef[0]), float(r2)


def shape_r(t_ref, ref, t_b, b, max_lag_ms=12.0):
    """(|r|, signed r, shift_ms) of b vs ref after peak alignment — the repo's metric."""
    s, r = align_score(t_ref, ref, t_b, b, max_lag_ms=max_lag_ms)
    return abs(float(r)), float(r), float(s)


# ----------------------------------------------------------------------------- first-principles oracle
# Line-source SFAP in an INFINITE ANISOTROPIC medium (Rosenfalck 1969; Andreassen &
# Rosenfalck 1981; Dimitrov & Dimitrova 1998). φ has a closed form, so the SFAP can be
# evaluated in the dual form  ∫ Vm(z,t)·win(z)·φ''(z) dz  on a fine grid with NO numerical
# derivative of the tendon step (integration by parts; exact for a boxcar fibre end).
S_R, S_Z = 0.1, 0.5                 # muscle σ (S/m): radial / along the fibre (ratio 5)
SIGMA_IN, A_FIB = 1.0, 0.05         # engine defaults (σ_in, fibre radius mm) — same constant


def phi_inf(z, rho=10.0, s_r=S_R, s_z=S_Z):
    """Closed-form lead field of a point electrode at distance ``rho`` from a fibre line."""
    K = 1.0 / (4 * np.pi * s_r * np.sqrt(s_z))
    return K / np.sqrt(rho ** 2 / s_r + z ** 2 / s_z)


def phi_inf_dd(z, rho=10.0, s_r=S_R, s_z=S_Z):
    """Analytic second z-derivative of ``phi_inf``."""
    K = 1.0 / (4 * np.pi * s_r * np.sqrt(s_z))
    A = rho ** 2 / s_r; u = A + z ** 2 / s_z
    return K * (3 * z ** 2 / s_z ** 2 * u ** -2.5 - (1 / s_z) * u ** -1.5)


def sfap_first_principles(posz=0.0, len1=60.0, len2=60.0, v=V, fs=FS, w=W, t0=-10.0,
                          rho=10.0, dz_fine=0.02, mirror=False):
    """Reference SFAP (t_ms, sfap): boxcar fibre [posz−len1, posz+len2], NMJ at posz.
    ``mirror=True`` propagates a spatially mirrored IAP (slow rise / fast fall) — the
    diagnostic for the Fourier engine's orientation."""
    from emgforge.synthesis.iap import rosenfalck_vm
    t = np.arange(w) / fs * 1000.0 + t0
    z = np.arange(posz - len1, posz + len2 + dz_fine / 2, dz_fine)
    arg = v * t[:, None] - np.abs(z[None, :] - posz)
    Vm = rosenfalck_vm(15.0 - arg) if mirror else rosenfalck_vm(arg)
    return t, (Vm @ phi_inf_dd(z, rho)) * dz_fine * (SIGMA_IN * np.pi * A_FIB ** 2)


def best_r(t_ref, ref, t_b, b, max_lag_ms=6.0, dt=0.05):
    """Sign-agnostic, lag-bounded agreement: max |r| over lags in ±max_lag_ms of the
    peak-normalised waveforms. Returns (signed r, lag_ms, amplitude ratio b/ref)."""
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


def fourier_physical(phi, dz, len1=60.0, len2=60.0, posz=0.0, v=V, fs=FS, w=W):
    """The Fourier engine (production preset) converted to physical time the way the
    sanity suite does (flip to causal, window centre → 0). Returns (t_ms, muap)."""
    from emgforge.synthesis.api import generate_muap_from_phi, get_optimal_config
    cfg = replace(get_optimal_config(), v=v, fsamp=fs, w=w, len1_mm=len1, len2_mm=len2)
    res = generate_muap_from_phi(np.asarray(phi, float).reshape(1, -1), dz, config=cfg,
                                 posz_mm_arr=np.array([float(posz)]))
    m = np.real(res.muap).astype(float).ravel()[::-1]
    return np.asarray(res.t_ms, float) - w / fs * 1000.0 / 2.0, m


def eof_time(t, x, expect_ms, half_win_ms=3.0):
    """Time of the sharpest slope (|dx/dt| max) within ±half_win of the expected
    end-of-fibre instant — a robust EOF locator for sharp-tendon waveforms."""
    m = (t > expect_ms - half_win_ms) & (t < expect_ms + half_win_ms)
    d = np.abs(np.gradient(x, t))
    idx = np.where(m)[0]
    return float(t[idx[np.argmax(d[idx])]])


__all__ = [n for n in dir() if not n.startswith("_")]
