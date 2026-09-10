"""MUAP regression gate: 21 reference MUAPs, compared numerically and semantically.

This is the fast self-regression net for the synthesis engines. Unlike test_golden.py
(which compares the spatial engine to the Fourier engine), this pins each engine to ITS
OWN previously-recorded output, across both engines and every config knob — most
importantly the production recipe (spatial + monopole denoise + one-sided taper).

Two tiers per case:

* ``test_numerically_stable`` — exact time axis and tight peak-relative waveform
  tolerance. This detects numerical drift while remaining portable across supported
  NumPy/SciPy/BLAS combinations.

* ``test_correlation_and_amplitude`` — signed correlation ≥ 1−1e-9 and peak-amplitude
  ratio within 5e-6. This semantic gate fails on meaningful shape or scale changes.

Exact algebraic behavior is covered separately by the analytical-cylinder oracle. A
committed floating-point snapshot cannot provide a portable bitwise contract unless the
entire numerical stack and CPU instruction path are pinned.

To refresh after an *intended* change:  python scripts/synthesis/build_muap_reference.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .muap_cases import CASE_BY_NAME, CASE_NAMES, run_case

REFERENCE = Path(__file__).resolve().parent / "data" / "muap_reference.npz"


@pytest.fixture(scope="session")
def reference():
    d = np.load(REFERENCE, allow_pickle=True)
    names = [str(n) for n in d["case_names"]]
    return {n: (d[f"{n}__t"], d[f"{n}__muap"]) for n in names}


@pytest.fixture(scope="session")
def produced():
    """Run all 21 cases once; share the outputs across every test."""
    return {name: run_case(CASE_BY_NAME[name]) for name in CASE_NAMES}


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 if na < 1e-30 and nb < 1e-30 else float(a @ b / (na * nb))


@pytest.mark.parametrize("name", CASE_NAMES)
def test_numerically_stable(reference, produced, name):
    """Waveform stays within a tight cross-environment numerical tolerance."""
    ref_t, ref_m = reference[name]
    t, m = produced[name]
    assert t.shape == ref_t.shape and m.shape == ref_m.shape, (
        f"{name}: shape {t.shape},{m.shape} vs ref {ref_t.shape},{ref_m.shape}")
    assert np.array_equal(t, ref_t), f"{name}: time axis drifted"
    peak = float(np.max(np.abs(ref_m)))
    np.testing.assert_allclose(m, ref_m, rtol=1e-9, atol=5e-6 * peak,
                               err_msg=f"{name}: waveform drifted")


@pytest.mark.parametrize("name", CASE_NAMES)
def test_correlation_and_amplitude(reference, produced, name):
    """Shape and scale reproduce the reference (survives benign FP noise, not real change)."""
    _, ref_m = reference[name]
    _, m = produced[name]
    r = _corr(m, ref_m)
    assert r >= 1.0 - 1e-9, f"{name}: correlation {r:.9f} vs reference"
    ratio = np.abs(m).max() / (np.abs(ref_m).max() + 1e-300)
    assert abs(ratio - 1.0) < 5e-6, f"{name}: peak ratio {ratio:.9f}"


def test_reference_covers_exactly_the_case_set(reference):
    """The committed reference and the case module must not drift apart."""
    assert sorted(reference) == sorted(CASE_NAMES), (
        "reference file and muap_cases.py disagree — rerun build_muap_reference.py")


def test_reference_waveforms_are_distinct(reference):
    """Guard against a case silently collapsing onto another (e.g. a window becoming a
    no-op). Every reference waveform must differ from every other; a sign flip counts
    as distinct (polarity cases), a scale change does not (all are peak-normalised here).
    """
    names = sorted(reference)
    normed = {}
    for n in names:
        m = reference[n][1]
        pk = np.abs(m).max()
        normed[n] = m / pk if pk > 0 else m
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if normed[a].shape != normed[b].shape:
                continue
            assert not np.array_equal(normed[a], normed[b]), f"{a} and {b} are identical"


def test_no_reference_waveform_is_trivial(reference):
    """Every case produced a real signal — no all-zero or non-finite MUAP slipped in."""
    for n, (_, m) in reference.items():
        assert np.all(np.isfinite(m)), f"{n}: non-finite"
        assert np.abs(m).max() > 0, f"{n}: all-zero"
