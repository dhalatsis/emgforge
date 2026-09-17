"""Fixtures for the synthesis tests.

Everything here is pure NumPy: the synthesis stack takes φ(z) as an array, so no
FEM stack, mesh file, or conda environment is needed. The suite runs in ~1 s on a
bare ``pip install -e .``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent / "data"
GOLDEN_NPZ = DATA / "golden_cylindrical.npz"


@dataclass(frozen=True)
class GoldenCase:
    """One cylindrical reference case: the φ(z) input and the Fourier SFAP it produced."""
    name: str
    phi: np.ndarray
    dz_mm: float
    L1_mm: float
    L2_mm: float
    v: float
    posz_mm: float
    t_ms: np.ndarray
    sfap: np.ndarray


def _load() -> dict[str, GoldenCase]:
    d = np.load(GOLDEN_NPZ, allow_pickle=True)
    out = {}
    for n in (str(x) for x in d["case_names"]):
        out[n] = GoldenCase(
            name=n,
            phi=d[f"{n}__phi"].astype(float),
            dz_mm=float(d[f"{n}__dz_mm"]),
            L1_mm=float(d[f"{n}__L1"]),
            L2_mm=float(d[f"{n}__L2"]),
            v=float(d[f"{n}__v"]),
            posz_mm=float(d[f"{n}__posz"]),
            t_ms=d[f"{n}__golden_t_ms"].astype(float),
            sfap=d[f"{n}__golden_sfap"].astype(float),
        )
    return out


# Read once at import so tests can parametrise over the case names at collection time.
GOLDEN_CASES = _load()
CASE_NAMES = sorted(GOLDEN_CASES)


@pytest.fixture(scope="session")
def golden() -> dict[str, GoldenCase]:
    return GOLDEN_CASES


# The config under which the two engines are expected to agree: a hard boxcar
# tendon cut (matching Fourier's sharp `pare` box), no φ smoothing, no edge taper.
# This is NOT the production recipe.
MATCHED = dict(
    fsamp=2048.0, w=256, csd_derivative=2, upsample_factor=2,
    fiber_window="boxcar", denoise="none",
    edge_taper_left=0, edge_taper_right=0, t_start_ms=-40.0,
)


@dataclass(frozen=True)
class Score:
    r: float          # signed, time-aligned Pearson
    lag_ms: float
    peak_ratio: float  # peak(spatial) / peak(fourier reference)


@pytest.fixture(scope="session")
def scores() -> dict[str, Score]:
    """Run the spatial engine once per reference case and score it. Session-scoped:
    the lag scan is the expensive part and the result never changes."""
    from emgforge.synthesis.engines.spatial import SpatialConfig, compute_sfap_spatial

    out = {}
    for name, c in GOLDEN_CASES.items():
        t, s, _ = compute_sfap_spatial(
            c.phi, c.dz_mm, len1_mm=c.L1_mm, len2_mm=c.L2_mm,
            posz_mm=c.posz_mm, config=SpatialConfig(v=c.v, **MATCHED),
        )
        r, lag = aligned_pearson(c.t_ms, c.sfap, t, s)
        out[name] = Score(r, lag, float(np.abs(s).max() / np.abs(c.sfap).max()))
    return out


def _corr(a, b):
    """Pearson r without scipy's per-call overhead."""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(a @ b / (na * nb))


def aligned_pearson(t_ref, ref, t, sig, max_lag_ms=20.0, dt=0.02):
    """Best-|r| time alignment of *sig* onto *ref*. Returns the SIGNED r and its lag (ms).

    Sign is preserved deliberately: a polarity flip must not look like a pass.
    Coarse-to-fine: |r(lag)| is smooth and unimodal near its peak, so a 0.2 ms sweep
    locates the basin and a 0.02 ms sweep refines it -- ~12x fewer evaluations than
    scanning the whole range at full resolution.
    """
    grid = np.arange(t_ref.min(), t_ref.max(), dt)
    ref_g = np.interp(grid, t_ref, ref, left=0.0, right=0.0)
    coarse = t_ref[np.argmax(np.abs(ref))] - t[np.argmax(np.abs(sig))]

    def scan(lags):
        best_r, best_lag, best_abs = 0.0, 0.0, -1.0
        for d in lags:
            sig_g = np.interp(grid, t + coarse + d, sig, left=0.0, right=0.0)
            r = _corr(ref_g, sig_g)
            if abs(r) > best_abs:
                best_r, best_lag, best_abs = r, float(d), abs(r)
        return best_r, best_lag

    _, d0 = scan(np.arange(-max_lag_ms, max_lag_ms + 0.2, 0.2))
    best_r, d1 = scan(np.arange(d0 - 0.25, d0 + 0.25 + dt, dt))
    return best_r, float(coarse + d1)
