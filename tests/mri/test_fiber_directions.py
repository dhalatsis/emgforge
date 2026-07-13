"""Per-muscle fibre-direction anisotropy — the pieces that align σ with each
muscle's fibres. Pure numpy/scipy (no dolfinx, no MRI data), so it runs on a
pip-only install.

Covers the two load-bearing operations:
  * ``rotate_conductivity`` — puts the high-σ axis on the fibre, preserving the
    5:1 transversely-isotropic eigenvalues.
  * ``_estimate_pca`` — recovers a muscle's fibre axis from its voxel cloud.
"""
import numpy as np

from emgforge.mri.core.fiber_directions import (
    MuscleFiberModel, MuscleInfo, rotate_conductivity, SIGMA_MUSCLE_Z,
    SIGMA_MUSCLE_CROSS, SIGMA_MUSCLE_FIBER,
)


def _angle_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))


def test_rotate_conductivity_puts_high_sigma_on_the_fibre():
    d = np.array([0.4, -0.2, 1.0])
    d /= np.linalg.norm(d)
    S = rotate_conductivity(SIGMA_MUSCLE_Z, d)

    assert np.allclose(S, S.T)                       # symmetric tensor
    w, V = np.linalg.eigh(S)
    # eigenvalues unchanged: transversely isotropic (σ⊥, σ⊥, σ∥)
    assert np.allclose(sorted(w), [SIGMA_MUSCLE_CROSS, SIGMA_MUSCLE_CROSS, SIGMA_MUSCLE_FIBER])
    # the largest-σ eigenvector points along the fibre
    assert _angle_deg(V[:, int(np.argmax(w))], d) < 1e-3


def test_rotate_conductivity_identity_when_fibre_is_z():
    S = rotate_conductivity(SIGMA_MUSCLE_Z, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(S, SIGMA_MUSCLE_Z)


def test_pca_recovers_a_synthetic_fibre_axis():
    rng = np.random.RandomState(0)
    axis = np.array([0.3, 0.1, 1.0])
    axis /= np.linalg.norm(axis)
    t = np.linspace(-60.0, 60.0, 800)[:, None]
    coords = t * axis + rng.normal(scale=2.0, size=(800, 3))   # elongated cloud

    fm = MuscleFiberModel()                           # no NIfTI needed for PCA
    mi = MuscleInfo(label=7, tissue_type="muscle")
    fm._estimate_pca(mi, coords)

    assert _angle_deg(mi.fiber_direction, axis) < 3.0
    assert mi.fiber_direction[2] > 0                  # canonical +z sign


def test_muscle_tensor_is_five_to_one_transversely_isotropic():
    mi = MuscleInfo(label=7, tissue_type="muscle")
    mi.fiber_direction = np.array([0.0, 0.3, 1.0])
    mi.fiber_direction /= np.linalg.norm(mi.fiber_direction)
    w = np.linalg.eigvalsh(mi.conductivity_tensor)
    assert np.isclose(w.max() / w.min(), SIGMA_MUSCLE_FIBER / SIGMA_MUSCLE_CROSS)
