"""ParametricGeometry — mesh parameters and electrode / fibre placement (P4b)."""
import numpy as np
import pytest

from emgop.fem.geometry import ParametricGeometry

CYL = dict(r_bone=10, r_muscle=35, r_fat=38, r_skin=40, length=240.0)


def test_circle_is_ellipse_ratio_one():
    g = ParametricGeometry(**CYL)
    assert g.shape == "circle"
    assert g.a_skin == g.b_skin == 40


def test_ellipse_axes():
    g = ParametricGeometry(**CYL, ellipse_ratio=1.4)
    assert g.shape == "ellipse"
    assert g.a_skin == pytest.approx(56.0)   # 40 * 1.4
    assert g.b_skin == 40


def test_electrode_on_skin_circle():
    g = ParametricGeometry(**CYL)
    assert np.allclose(g.electrode_on_skin(0.0, 120.0), [40, 0, 120])
    assert np.allclose(g.electrode_on_skin(90.0, 5.0), [0, 40, 5], atol=1e-9)


def test_electrode_on_skin_ellipse():
    g = ParametricGeometry(**CYL, ellipse_ratio=1.4)
    assert np.allclose(g.electrode_on_skin(0.0, 0.0), [56, 0, 0])
    assert np.allclose(g.electrode_on_skin(90.0, 0.0), [0, 40, 0], atol=1e-9)


def test_r_skin_at():
    g = ParametricGeometry(**CYL, ellipse_ratio=1.4)
    assert g.r_skin_at(0.0) == pytest.approx(56.0)
    assert g.r_skin_at(90.0) == pytest.approx(40.0)


def test_fibre_xy_radial_is_distance_from_centre():
    g = ParametricGeometry(**CYL)
    x, y = g.fibre_xy_radial(30.0, 0.0)
    assert x == pytest.approx(30.0) and abs(y) < 1e-12
    x, y = g.fibre_xy_radial(30.0, 90.0)
    assert abs(x) < 1e-9 and y == pytest.approx(30.0)


def test_fibre_xy_below_skin_depth():
    """A fibre `depth` below the skin along θ=0 sits at x = a_skin − depth."""
    g = ParametricGeometry(**CYL, ellipse_ratio=1.4)
    x, y = g.fibre_xy_below_skin(10.0, 0.0)
    assert x == pytest.approx(56.0 - 10.0) and abs(y) < 1e-9


def test_fibre_points_centred_and_clipped():
    g = ParametricGeometry(**CYL)
    pts, zg = g.fibre_points(5.0, 0.0, 120.0, nz=64, dz=1.0)
    assert pts.shape == (64, 3)
    assert zg[0] == pytest.approx(-(64 - 1) / 2.0)   # centred window
    assert abs(zg.mean()) < 1e-9
    assert np.all(pts[:, 0] == 5.0) and np.all(pts[:, 1] == 0.0)
    assert pts[:, 2].min() >= 0.5 and pts[:, 2].max() <= 240.0 - 0.5   # z clip


def test_fibre_points_z_clip_at_ends():
    g = ParametricGeometry(**CYL)
    # a window centred at z=2 would run negative without the clip
    pts, _ = g.fibre_points(5.0, 0.0, 2.0, nz=64, dz=1.0)
    assert pts[:, 2].min() >= 0.5


def test_mesh_params_circle():
    p = ParametricGeometry(**CYL).mesh_params()
    assert p["shape"] == "circle"
    assert p["radius_cort_bone"] == 10 and p["radius_muscle"] == 35
    assert p["radius_fat"] == 38 and p["radius_skin"] == 40
    assert p["radius_canc_bone"] == 9.0


def test_mesh_params_ellipse():
    p = ParametricGeometry(**CYL, ellipse_ratio=1.4).mesh_params()
    assert p["shape"] == "ellipse" and p["ellipse_ratio"] == 1.4
    assert p["a_skin"] == pytest.approx(56.0) and p["b_skin"] == 40
    assert p["a_muscle"] == pytest.approx(49.0) and p["b_muscle"] == 35
