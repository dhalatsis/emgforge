"""The spatial engine against the golden cylindrical reference set.

The golden SFAPs were produced by the Fourier engine (itself scored r=0.997 against
the Farina MATLAB reference). These tests therefore assert **agreement between the
two engines**, not correctness of either. An independent oracle -- Fourier scored
against ``emgop.analytical`` -- is tracked separately in the backlog.

Three properties are asserted where the old ``verify_spatial.py`` script asserted
one. It scored ``abs(pearsonr(...))`` on peak-normalised traces, so a polarity flip
and a 1000x amplitude error both passed. Both are now pinned.
"""
from __future__ import annotations

from .conftest import CASE_NAMES

import pytest

SHAPE_MIN = 0.95          # |r| floor
LAG_MS = (-3.0, -1.0)     # spatial is physical-time, Fourier window-centred
AMPLITUDE = (0.15, 0.35)  # peak(spatial) / peak(fourier)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_shape_matches_golden(scores, name):
    """Waveform shape agrees to |r| >= 0.95 after time alignment."""
    r = scores[name].r
    assert abs(r) >= SHAPE_MIN, f"{name}: |r| = {abs(r):.3f}"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_polarity_is_inverted_vs_fourier(scores, name):
    """The spatial engine is ANTI-PHASE with the Fourier golden at polarity=+1.

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
    """Peak amplitude ratio is pinned.

    The engines disagree on absolute amplitude by 3-5x, and NOT by a constant
    factor: the ratio spans 0.19..0.32 across the 12 cases (a 1.65x spread), so no
    single scale constant reconciles them. The old gate peak-normalised both traces
    and never saw this. Pinned pending an amplitude audit.
    """
    ratio = scores[name].peak_ratio
    assert AMPLITUDE[0] <= ratio <= AMPLITUDE[1], f"{name}: peak ratio = {ratio:.3f}"


def test_amplitude_ratio_is_not_a_constant(scores):
    """Guard the claim made above: if this ever fails, a single scale factor DOES
    reconcile the engines and the amplitude discrepancy has a one-line fix."""
    ratios = [s.peak_ratio for s in scores.values()]
    spread = max(ratios) / min(ratios)
    assert spread > 1.2, f"ratios collapsed to a constant (spread {spread:.3f}x)"
