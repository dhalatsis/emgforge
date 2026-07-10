"""TissueTable + the single-source anisotropy constant (P1, P4a)."""
import numpy as np
import pytest

from emgop.fem.conductivity import TissueTable


def test_analytical_table():
    t = TissueTable.analytical()
    assert np.array_equal(t["Muscle"], np.diag([0.10, 0.10, 0.50]))
    assert t["Fat"] == 0.04 and t["Skin"] == 1.00
    assert t["Cortical Bone"] == 0.02 and t["Cancellous Bone"] == 0.02


def test_mri_analytical_table():
    t = TissueTable.mri_analytical()
    assert np.array_equal(t["muscle"], np.diag([0.10, 0.10, 0.50]))
    assert t["fat_skin"] == 0.04 and t["skin"] == 1.00


def test_emgop_table_is_production_default():
    t = TissueTable.emgop()
    assert np.array_equal(t["Muscle"], np.diag([0.2455, 0.2455, 1.2275]))
    assert t["Fat"] == 0.0379 and t["Skin"] == pytest.approx(4.55e-4)


def test_emgop_table_is_a_copy_not_the_module_dict():
    """Mutating a table must not corrupt the module defaults."""
    import emgop.fem.constants as C
    t = TissueTable.emgop()
    t["Muscle"] = np.zeros((3, 3))
    assert np.array_equal(C.CONDUCTIVITY["Muscle"], np.diag([0.2455, 0.2455, 1.2275]))


def test_muscle_fibre_derives_from_the_single_ratio():
    from emgop.tissue import ANISOTROPY_RATIO, SIGMA_MUSCLE_CROSS
    assert TissueTable.emgop()["Muscle"][2, 2] == pytest.approx(ANISOTROPY_RATIO * SIGMA_MUSCLE_CROSS)
    assert TissueTable.analytical()["Muscle"][2, 2] == pytest.approx(ANISOTROPY_RATIO * 0.10)


def test_single_source_shared_across_modules():
    """emgop.tissue is the one definition; fem.constants and mri read it."""
    import emgop.tissue as T
    import mri.core.fem_solver as fs
    from emgop.fem.constants import ANISOTROPY_RATIO as fem_ratio
    assert T.ANISOTROPY_RATIO == fs.ANISOTROPY_RATIO == fem_ratio == 5


def test_tissue_table_is_a_dict():
    """It must be a drop-in for the conductivity= constructor argument."""
    assert isinstance(TissueTable.analytical(), dict)
