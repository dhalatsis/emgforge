"""The extracted reciprocity pieces: KSPConfig + GaussianSource (L1)."""
import numpy as np

from emgop.fem.leadfield import GaussianSource, KSPConfig


def test_ksp_config_defaults_are_the_p3_values():
    k = KSPConfig()
    assert k.solver_type == "gmres" and k.pc_type == "ilu"
    assert k.rtol == 1e-8 and k.atol == 1e-10 and k.max_it == 5000


def test_gaussian_source_is_zero_mean(solved):
    """The Gaussian RHS must integrate to zero (required for the pure-Neumann solve)."""
    from dolfinx import fem
    from ufl import dx

    m, _ = solved
    m.assign_source_to_point(np.array([10.0, 0.0, 20.0]))   # populates m.volume
    sf = fem.Function(m.V_pol)
    GaussianSource(3.0).assign(sf, np.array([10.0, 0.0, 20.0]), m.volume)
    integral = float(fem.assemble_scalar(fem.form(sf * dx)))
    assert abs(integral) < 1e-9 * m.volume


def test_leadfield_reproduces_femmodel(solved, tiny_geometry):
    """The standalone LeadField solve + evaluate reproduce FEMModel byte-for-byte."""
    from emgop.fem.leadfield import GaussianSource, LeadField

    m, uh = solved
    g = tiny_geometry
    lf = LeadField(m.mesh, m.sigma_anisotropic)
    uh_lf = lf.solve(g.electrode_on_skin(0.0, 20.0),
                     GaussianSource(float(m.options["source_sigma"])))
    pts, _ = g.fibre_points(*g.fibre_xy_radial(4.0, 0.0), 20.0, nz=64, dz=0.5)
    assert np.array_equal(m.evaluate_solution_at_points(pts, uh=uh), lf.phi(pts, uh_lf))


def test_gaussian_source_mean_after_assemble_is_close_not_identical(solved):
    """The two zero-mean forms agree to rounding but differ in the last ULPs, which is
    why each backend keeps its own (MRI = mean_after_assemble=True)."""
    from dolfinx import fem

    m, _ = solved
    pt = np.array([10.0, 0.0, 20.0])
    m.assign_source_to_point(pt)   # populates m.volume
    a = fem.Function(m.V_pol); GaussianSource(3.0, mean_after_assemble=False).assign(a, pt, m.volume)
    b = fem.Function(m.V_pol); GaussianSource(3.0, mean_after_assemble=True).assign(b, pt, m.volume)
    av, bv = np.asarray(a.vector.array), np.asarray(b.vector.array)
    assert np.allclose(av, bv, rtol=1e-12)


def test_gaussian_source_matches_the_engine_source(solved):
    """GaussianSource reproduces FEMModel's in-engine source field exactly."""
    from dolfinx import fem

    m, _ = solved
    point = np.array([10.0, 0.0, 20.0])
    # the engine's own path
    m.assign_source_to_point(point)
    engine = np.asarray(m.source_function.vector.array).copy()
    # the standalone object, same inputs
    sf = fem.Function(m.V_pol)
    GaussianSource(float(m.options["source_sigma"])).assign(sf, point, m.volume)
    assert np.array_equal(np.asarray(sf.vector.array), engine)
