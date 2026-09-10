"""Independent analytical-cylinder oracle for the production synthesis path.

The analytical ``SignalGenerator`` directly assembles the Farina four-layer
cylinder solution in the two-dimensional frequency domain. The production path
instead receives the reciprocal field phi(z), transforms it back to C(kz), and
then synthesizes the waveform. Agreement therefore exercises the public
field-to-MUAP boundary, FFT normalization, fibre-end operator, action-potential
scale, polarity, time convention, and SI unit conversion.
"""
from __future__ import annotations

import numpy as np
import pytest

from emgforge.synthesis import FibreBed, MUAPConfig, field_to_muap
from tests.regression.analytical_ref import (
    AnalyticalCase,
    analytical_muap,
    analytical_phi_along_fibre,
)


CASES = (
    pytest.param(AnalyticalCase(10.0, w=128), id="central-symmetric"),
    pytest.param(
        AnalyticalCase(20.0, L1_mm=40.0, L2_mm=90.0, distfib_deg=20.0, w=128),
        id="off-axis-asymmetric",
    ),
    pytest.param(
        AnalyticalCase(28.0, v_m_per_s=3.0, electrode_dim1_mm=2.0, w=128),
        id="superficial-slow-small-electrode",
    ),
)


@pytest.mark.parametrize("case", CASES)
def test_analytical_cylinder_reproduces_through_public_pipeline(case):
    """Analytical phi -> production pipeline equals direct analytical MUAP."""
    t_ref, muap_ref = analytical_muap(case)
    phi, dz_mm = analytical_phi_along_fibre(case)

    bed = FibreBed.uniform(
        1,
        dz_mm=dz_mm,
        len1_mm=case.L1_mm,
        len2_mm=case.L2_mm,
        posz_mm=0.0,
        v=case.v_m_per_s,
    )
    config = MUAPConfig(
        denoise="none",
        apply_z_window=False,
        edge_taper=0,
        w=case.w,
        fsamp=case.fsamp_hz,
        v=case.v_m_per_s,
        len1_mm=case.L1_mm,
        len2_mm=case.L2_mm,
    )
    got = field_to_muap(phi, bed, config)

    assert np.array_equal(got.t_ms, t_ref)
    peak = float(np.max(np.abs(muap_ref)))
    np.testing.assert_allclose(got.muap, muap_ref, rtol=2e-11, atol=peak * 1e-12)

    signed_r = float(np.corrcoef(got.muap, muap_ref)[0, 1])
    peak_ratio = float(np.max(np.abs(got.muap)) / np.max(np.abs(muap_ref)))
    assert signed_r >= 1.0 - 1e-13
    assert peak_ratio == pytest.approx(1.0, rel=2e-12)


@pytest.mark.parametrize("case", CASES)
def test_analytical_reciprocal_field_round_trip_is_lossless(case):
    """The extracted real phi retains the complete analytical C(kz) spectrum."""
    from emgforge.synthesis.engines.fourier import compute_C_from_phi_z
    from tests.regression.analytical_ref import _build

    _, _, _, generator = _build(case)
    debug = generator.generate_muap_debug()
    expected_c = np.asarray(debug["C"][0][0])
    phi, dz_mm = analytical_phi_along_fibre(case)
    recovered_c, _ = compute_C_from_phi_z(
        phi, delta_s_mm=dz_mm, apply_z_window=False
    )

    relative_error = float(
        np.linalg.norm(recovered_c - expected_c) / np.linalg.norm(expected_c)
    )
    assert relative_error < 1e-13
