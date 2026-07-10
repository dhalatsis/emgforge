"""FEM solve smoke tests — the reciprocity path P5 will refactor.

Exercises the real end-to-end solve on a small in-process mesh: the conductivity
parameter (P2), the σ-map, the constrained Neumann solve + convergence guard (P3),
and evaluate_solution_at_points.
"""
import numpy as np
import pytest


# ---- P2: conductivity as a parameter -------------------------------------

def test_conductivity_param_sets_the_instance_table(tiny_mesh):
    from emgop.fem import FEMModel
    from emgop.fem.conductivity import TissueTable

    m = FEMModel(tiny_mesh, conductivity=TissueTable.analytical(), source_sigma=3.0)
    assert np.array_equal(m.conductivity["Muscle"], np.diag([0.10, 0.10, 0.50]))


def test_conductivity_param_does_not_mutate_module_defaults(tiny_mesh):
    import emgop.fem.constants as C
    from emgop.fem import FEMModel
    from emgop.fem.conductivity import TissueTable

    FEMModel(tiny_mesh, conductivity=TissueTable.analytical(), source_sigma=3.0)
    assert np.array_equal(C.CONDUCTIVITY["Muscle"], np.diag([0.2455, 0.2455, 1.2275]))


def test_conductivity_param_changes_the_sigma_map(tiny_mesh):
    from emgop.fem import FEMModel
    from emgop.fem.conductivity import TissueTable

    a = FEMModel(tiny_mesh, conductivity=TissueTable.analytical(), source_sigma=3.0)
    e = FEMModel(tiny_mesh, conductivity=TissueTable.emgop(), source_sigma=3.0)
    sa = np.asarray(a.sigma_anisotropic.vector.array)
    se = np.asarray(e.sigma_anisotropic.vector.array)
    assert sa.shape == se.shape
    assert not np.allclose(sa, se)   # different muscle σ → different conductivity field


# ---- L2: LayeredSigma builder --------------------------------------------

def test_layered_sigma_reproduces_the_engine_field(solved):
    """The injectable σ-builder produces exactly the field FEMModel builds."""
    from emgop.fem.sigma import LayeredSigma

    m, _ = solved
    sig = LayeredSigma(m.conductivity)(m.mesh, m.cell_markers)
    assert np.array_equal(np.asarray(sig.vector.array),
                          np.asarray(m.sigma_anisotropic.vector.array))


def test_sigma_map_is_anisotropic_muscle_isotropic_elsewhere(solved):
    m, _ = solved
    cells = np.asarray(m.sigma_anisotropic.vector.array).reshape(-1, 9)  # 3x3 per cell
    xx, zz = cells[:, 0], cells[:, 8]
    # analytical muscle: σ = diag(0.10, 0.10, 0.50) → anisotropic (xx != zz)
    assert np.any(np.isclose(xx, 0.10) & np.isclose(zz, 0.50))
    # every other tissue is isotropic (xx == zz)
    assert np.any(np.isclose(xx, zz))


# ---- the solve -----------------------------------------------------------

def test_solve_produces_a_finite_monopole_like_field(solved, tiny_geometry):
    m, uh = solved
    g = tiny_geometry
    # fibre at radial depth 4 (inside muscle), θ=0, centred at the electrode's z=20
    pts, _ = g.fibre_points(*g.fibre_xy_radial(4.0, 0.0), 20.0, nz=64, dz=0.5)
    phi = m.evaluate_solution_at_points(pts, uh=uh)

    assert phi.shape == (64,)
    assert np.all(np.isfinite(phi))
    assert np.abs(phi).max() > 0
    # closest approach to the electrode (z=20) is the window centre → peak there
    assert abs(int(np.argmax(np.abs(phi))) - 32) < 12
    # and the field decays away from the peak
    assert np.abs(phi[32]) > np.abs(phi[0])
    assert np.abs(phi[32]) > np.abs(phi[-1])


def test_evaluate_returns_one_value_per_point(solved):
    m, uh = solved
    pts = np.array([[0.0, 0.0, 20.0], [1.0, 0.0, 20.0], [2.0, 0.0, 20.0]])
    v = m.evaluate_solution_at_points(pts, uh=uh)
    assert v.shape == (3,)
    assert np.all(np.isfinite(v))


def test_solve_is_deterministic(tiny_mesh, tiny_geometry):
    from emgop.fem import FEMModel
    from emgop.fem.conductivity import TissueTable

    el = tiny_geometry.electrode_on_skin(0.0, 20.0)
    pts, _ = tiny_geometry.fibre_points(4.0, 0.0, 20.0, nz=32, dz=1.0)
    out = []
    for _ in range(2):
        m = FEMModel(tiny_mesh, conductivity=TissueTable.analytical(), source_sigma=3.0)
        out.append(m.evaluate_solution_at_points(pts, uh=m.solve_for_point(el)))
    assert np.array_equal(out[0], out[1])


# ---- P3: explicit KSP + convergence guard --------------------------------

def test_convergence_guard_raises_on_nonconvergence(solved):
    """ConstrainedLinearProblem.solve raises rather than returning a bad solution
    when the KSP does not converge (here: capped at a single iteration)."""
    import ufl
    from dolfinx import fem

    from emgop.fem import ConstrainedLinearProblem
    from emgop.fem.leadfield import KSPConfig

    m, _ = solved
    V = m.V_scalar
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    f = fem.Function(V)
    f.interpolate(lambda x: x[0] - float(np.mean(x[0])))   # spatially-varying, ~zero-mean
    L = f * v * ufl.dx

    p = ConstrainedLinearProblem(a, L, V)
    with pytest.raises(RuntimeError, match="converge"):
        p.solve(ksp=KSPConfig(max_it=1))
