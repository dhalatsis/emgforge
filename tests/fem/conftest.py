"""Fixtures for the FEM / lead-field tests.

These require the dolfinx/fenicsx stack (conda-only). If it is not importable the
whole directory skips cleanly, so `pytest` stays green on a pip-only install. In
`fenicsx-env` they run: ~3 s to build a small mesh once, reused across the suite.
"""
import pytest

pytest.importorskip("dolfinx", reason="FEM tests require the dolfinx/fenicsx stack")

from emgforge.fem.geometry import ParametricGeometry  # noqa: E402


@pytest.fixture(scope="session")
def tiny_geometry():
    """A small layered cylinder that meshes reliably at the validated char_length.
    Same class the cylinder/ellipse sanity tiers use, just a small domain."""
    return ParametricGeometry(r_bone=4.0, r_muscle=8.0, r_fat=9.0, r_skin=10.0,
                              length=40.0, r_canc=3.0)


@pytest.fixture(scope="session")
def tiny_mesh(tiny_geometry, tmp_path_factory):
    """Build the small mesh once (~1.5 s). Returns the .msh path."""
    d = tmp_path_factory.mktemp("fem_mesh")
    msh, _ = tiny_geometry.build(d / "tiny.msh", d / "tiny.json", char_length=0.3)
    return str(msh)


@pytest.fixture(scope="session")
def solved(tiny_geometry, tiny_mesh):
    """A solved FEMModel (analytical conductivities), reused across tests."""
    from emgforge.fem import FEMModel
    from emgforge.fem.conductivity import TissueTable

    m = FEMModel(tiny_mesh, conductivity=TissueTable.analytical(), source_sigma=3.0)
    uh = m.solve_for_point(tiny_geometry.electrode_on_skin(0.0, 20.0))
    return m, uh
