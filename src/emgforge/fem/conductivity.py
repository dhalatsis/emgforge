"""Named per-tissue conductivity tables.

A ``TissueTable`` maps a tissue name to its conductivity — a scalar (isotropic,
S/m) or a 3×3 array (anisotropic). It is a drop-in for the ``conductivity=``
argument of ``FEMModel`` / ``MRIFEMModel``, replacing the hand-written dicts the
sanity wrappers previously inlined (and, before that, monkeypatched).

The muscle anisotropy in every table derives from the single source of truth,
``emgforge.tissue`` (dolfinx-free).
"""
from __future__ import annotations

import numpy as np

from emgforge.tissue import ANISOTROPY_RATIO, SIGMA_MUSCLE_CROSS

# Analytical muscle tensor (σ_cross = 0.10, ratio 5 → σ_fibre = 0.50). This is the
# cylinder/ellipse "apples-to-apples with emgforge.analytical" value, distinct from the
# production emgforge cross-fibre σ (0.2455) — see 02_fem.md §14.4 decision 3.
_ANALYTICAL_MUSCLE_CROSS = 0.10


def _analytical_muscle() -> np.ndarray:
    c = _ANALYTICAL_MUSCLE_CROSS
    return np.diag([c, c, ANISOTROPY_RATIO * c])


def pennation_rotation_matrix(alpha_deg: float, axis: str = "x") -> np.ndarray:
    """Rotation that tilts the fibre direction (nominally +z) by ``alpha_deg`` (pennation).

    The SAME matrix must rotate both the muscle σ tensor AND the fibre query paths, or the
    conductivity anisotropy and the fibre geometry disagree. Import it in both places rather
    than re-deriving the convention. ``axis`` is the in-plane axis the fibre tilts about:
    'x' tilts z→(0,sinα,cosα), 'y' tilts z→(sinα,0,cosα).
    """
    a = np.radians(alpha_deg)
    ca, sa = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    if axis == "y":
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def rotate_muscle_tensor(sigma: np.ndarray, alpha_deg: float, axis: str = "x") -> np.ndarray:
    """σ' = R σ Rᵀ — rotate an anisotropic muscle tensor by the pennation angle.

    Eigenvalues (the physical conductivities) are invariant; only the fibre axis tilts. At
    α=0 this is the identity; at α=90 it swaps the fibre axis into the in-plane direction.
    """
    R = pennation_rotation_matrix(alpha_deg, axis)
    return R @ np.asarray(sigma) @ R.T


class TissueTable(dict):
    """{tissue_name: σ} — σ scalar (iso) or 3×3 (anisotropic). Drop-in for
    ``FEMModel(..., conductivity=…)`` and ``MRIFEMModel(..., conductivity=…)``."""

    @classmethod
    def emgforge(cls) -> "TissueTable":
        """The production ``emgforge.fem`` defaults (muscle σ_cross = 0.2455 S/m)."""
        from emgforge.fem.constants import CONDUCTIVITY

        return cls({k: (v.copy() if hasattr(v, "copy") else v)
                    for k, v in CONDUCTIVITY.items()})

    @classmethod
    def analytical(cls) -> "TissueTable":
        """Cylinder/ellipse analytical match: single bone, σ_muscle = diag(.1,.1,.5)."""
        return cls({
            "Muscle": _analytical_muscle(),
            "Fat": 0.04,
            "Skin": 1.00,
            "Cortical Bone": 0.02,
            "Cancellous Bone": 0.02,  # single-bone analytical
        })

    @classmethod
    def analytical_pennated(cls, alpha_deg: float, axis: str = "x") -> "TissueTable":
        """analytical() with the muscle tensor tilted by pennation ``alpha_deg``.

        This is the cross-pennation knob: the field φ changes because the anisotropy axis
        rotates. The fibre query paths must be rotated by the same angle (see
        ``pennation_rotation_matrix``) at scoring time.
        """
        t = cls.analytical()
        t["Muscle"] = rotate_muscle_tensor(_analytical_muscle(), alpha_deg, axis)
        return t

    @classmethod
    def mri_analytical(cls) -> "TissueTable":
        """MRI-forearm tissue names, analytical conductivities matching the cyl/ellipse tiers."""
        return cls({
            "muscle": _analytical_muscle(),
            "fat_skin": 0.04,
            "skin": 1.00,
            "bone": 0.02,       # radius/ulna — matches the cyl tier's Cortical Bone
        })
