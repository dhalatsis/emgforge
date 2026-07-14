"""Fibre-end windows, and why ``one_sided`` exists.

The fibre-end window models the tendon termination of the travelling wave. It is
applied to each semi-fibre (NMJ→proximal, NMJ→distal), so a *symmetric* window
tapers both ends of each half -- including the end at the NMJ, where the wave is
born. That notch is an artifact. ``one_sided`` tapers only the tendon end.

Losing this window costs r 0.79..0.92 and 0.31x amplitude against the settled
pipeline, so it is the load-bearing preprocessing stage, not a cosmetic one.
"""
from __future__ import annotations

import numpy as np
import pytest

from emgforge.synthesis.preprocessing import create_fiber_windows

N = 200
RATIO = 0.4          # NMJ at 40% along the fibre
N_LEFT = int(N * RATIO)
ALPHA = 0.25


def test_one_sided_is_flat_through_the_nmj():
    """Both semi-fibres reach exactly 1.0 at the junction."""
    wl, wr = create_fiber_windows(N, RATIO, "one_sided", ALPHA)
    assert wl[N_LEFT - 1] == pytest.approx(1.0)   # proximal half, NMJ end
    assert wr[N_LEFT] == pytest.approx(1.0)       # distal half, NMJ end


def test_one_sided_tapers_to_zero_at_the_tendons():
    wl, wr = create_fiber_windows(N, RATIO, "one_sided", ALPHA)
    assert wl[0] == pytest.approx(0.0)            # proximal tendon
    assert wr[-1] == pytest.approx(0.0)           # distal tendon


def test_symmetric_tukey_notches_the_nmj():
    """The defect one_sided was introduced to remove.

    A tukey window on each semi-fibre ramps down at BOTH ends of that half, so the
    junction -- the source of the travelling wave -- gets attenuated to zero.
    """
    wl, wr = create_fiber_windows(N, RATIO, "tukey", ALPHA)
    assert wl[N_LEFT - 1] < 0.05, "tukey should notch the NMJ (that is the bug)"
    assert wr[N_LEFT] < 0.05

    ol, _ = create_fiber_windows(N, RATIO, "one_sided", ALPHA)
    assert ol[N_LEFT - 1] > 0.99, "one_sided must not"


def test_one_sided_support_matches_the_semi_fibres():
    """Each half-window is zero outside its own semi-fibre."""
    wl, wr = create_fiber_windows(N, RATIO, "one_sided", ALPHA)
    assert np.all(wl[N_LEFT:] == 0.0)
    assert np.all(wr[:N_LEFT] == 0.0)


@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5])
def test_alpha_sets_the_taper_length(alpha):
    """The ramp occupies alpha of each semi-fibre."""
    wl, _ = create_fiber_windows(N, RATIO, "one_sided", alpha)
    ramp = wl[:N_LEFT]
    n_below_one = int(np.sum(ramp < 0.999))
    assert n_below_one == pytest.approx(int(alpha * N_LEFT), abs=1)


def test_one_sided_is_monotonic_on_each_half():
    wl, wr = create_fiber_windows(N, RATIO, "one_sided", ALPHA)
    assert np.all(np.diff(wl[:N_LEFT]) >= -1e-12)     # rises tendon -> NMJ
    assert np.all(np.diff(wr[N_LEFT:]) <= 1e-12)      # falls NMJ -> tendon


@pytest.mark.parametrize("window", ["tukey", "boxcar", "hann", "one_sided"])
def test_windows_are_bounded_and_finite(window):
    wl, wr = create_fiber_windows(N, RATIO, window, ALPHA)
    for w in (wl, wr):
        assert w.shape == (N,)
        assert np.all(np.isfinite(w))
        assert w.min() >= 0.0 and w.max() <= 1.0 + 1e-12
