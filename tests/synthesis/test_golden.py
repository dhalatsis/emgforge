"""The spatial engine against the cylindrical reference set.

The reference SFAPs were produced by the Fourier engine (itself scored r=0.997 against
the Farina MATLAB reference). These tests therefore assert **agreement between the
two engines**, not correctness of either. An independent oracle -- Fourier scored
against ``emgforge.analytical`` -- is tracked separately in the backlog.

Three properties are asserted where the old ``verify_spatial.py`` script asserted
one. It scored ``abs(pearsonr(...))`` on peak-normalised traces, so a polarity flip
and a 1000x amplitude error both passed. Both are now pinned.
"""
from __future__ import annotations

import numpy as np

from .conftest import CASE_NAMES

import pytest

SHAPE_MIN = 0.95          # |r| floor
LAG_MS = (-3.0, -1.0)     # spatial is physical-time, Fourier window-centred

# peak(spatial) / peak(fourier). Re-pinned 2026-09-15 when the spurious `/ v` was
# removed from ``compute_sfap_spatial`` (validation check A0.2: the engine was exactly
# 1/v of the closed-form line-source oracle). Every ratio scaled by exactly its case's v
# (the set spans v = 3.0 / 3.3 / 4.0 / 5.0), from 0.19..0.32 to the values now measured:
#   dog_biphasic_asym 0.841   dog_biphasic_sym 0.962   gauss_s12_sym 0.968
#   gauss_s18_deep    0.977   gauss_s5_sym     0.913   gauss_s8_asym40-100 0.962
#   gauss_s8_offset+20 1.013  gauss_s8_offset-30 0.998 gauss_s8_pm_asym (v3.3) 0.939
#   gauss_s8_sym      0.962   gauss_s8_v3 (v3) 0.951   gauss_s8_v5 (v5) 0.963
# i.e. a ~constant ratio ~0.95 x (0.88..1.06), independent of v. The band is set just
# outside the measured extremes; a reintroduced 1/v (or x v) prefactor lands every case
# outside it.
AMPLITUDE = (0.80, 1.05)
AMPLITUDE_SPREAD_MAX = 1.3   # max/min over the 12 cases; measured 1.204
V_INDEPENDENCE = 0.10        # v != 4 cases within +-10 % of the v = 4 median; measured <= 2.4 %


@pytest.mark.parametrize("name", CASE_NAMES)
def test_shape_matches_golden(scores, name):
    """Waveform shape agrees to |r| >= 0.95 after time alignment."""
    r = scores[name].r
    assert abs(r) >= SHAPE_MIN, f"{name}: |r| = {abs(r):.3f}"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_polarity_is_inverted_vs_fourier(scores, name):
    """The spatial engine is ANTI-PHASE with the Fourier reference at polarity=+1.

    A characterisation test, not an endorsement. Measured: the best *positive*
    correlation at any lag is only +0.38..+0.65, while the best negative is -0.99
    at a consistent -2 ms -- so this is a true sign inversion, not a half-period
    alignment artifact.

    Which engine carries the correct sign is an open physics question. Until it is
    settled the inversion is pinned here, so that changing it is a deliberate act
    with a failing test attached rather than a silent drift.
    """
    r = scores[name].r
    assert r <= -SHAPE_MIN, f"{name}: signed r = {r:+.3f} (expected <= -0.95)"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_lag_vs_golden(scores, name):
    """Residual alignment lag is a stable ~-2 ms, not drifting."""
    lag = scores[name].lag_ms
    assert LAG_MS[0] <= lag <= LAG_MS[1], f"{name}: lag = {lag:+.2f} ms"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_amplitude_vs_golden(scores, name):
    """Peak amplitude ratio is pinned at ~1 (0.80..1.05).

    Until 2026-09-15 the spatial engine divided the line-source integral by v once
    too often, so this band sat at 0.15..0.35 and the ratio *looked* non-constant
    (1.65x spread) -- that spread was the 1/v factor across the set's v = 3/3.3/4/5.
    With the constant fixed the two engines agree on absolute amplitude to within
    ~16 % (0.84..1.01); the residual is the window / phi-handling difference between
    the methods, not a scale error. The old gate peak-normalised both traces and would
    have missed all of this.
    """
    ratio = scores[name].peak_ratio
    assert AMPLITUDE[0] <= ratio <= AMPLITUDE[1], f"{name}: peak ratio = {ratio:.3f}"


def test_amplitude_ratio_is_near_constant(scores):
    """The engines now differ by (close to) a single scale factor ~0.95.

    Guards the claim in ``test_amplitude_vs_golden``: the max/min spread over the 12
    cases is 1.20x (measured), well under the 1.65x the spurious 1/v produced. If this
    fails, a geometry- or v-dependent factor has crept back into one of the engines.
    """
    ratios = [s.peak_ratio for s in scores.values()]
    spread = max(ratios) / min(ratios)
    assert spread < AMPLITUDE_SPREAD_MAX, f"amplitude ratio spread {spread:.3f}x"


def test_amplitude_ratio_is_v_independent(scores, golden):
    """A CV prefactor cannot be reintroduced silently.

    The IAP is defined in space, so the spatial SFAP amplitude must not depend on v.
    The v = 3.0 / 3.3 / 5.0 cases sit within 2.4 % of the v = 4 median (0.951 / 0.939 /
    0.963 vs 0.962); a stray 1/v or x v would move them by 25..33 %.
    """
    by_v4 = [scores[n].peak_ratio for n in CASE_NAMES if golden[n].v == 4.0]
    ref = float(np.median(by_v4))
    for n in CASE_NAMES:
        if golden[n].v == 4.0:
            continue
        rel = scores[n].peak_ratio / ref
        assert abs(rel - 1.0) < V_INDEPENDENCE, (
            f"{n} (v={golden[n].v}): ratio {scores[n].peak_ratio:.3f} vs v=4 median {ref:.3f} "
            f"(rel {rel:.3f})")
