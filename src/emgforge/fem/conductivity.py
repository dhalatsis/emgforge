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
    def mri_analytical(cls) -> "TissueTable":
        """MRI-forearm tissue names, analytical conductivities matching the cyl/ellipse tiers."""
        return cls({
            "muscle": _analytical_muscle(),
            "fat_skin": 0.04,
            "skin": 1.00,
            "bone": 0.02,       # radius/ulna — matches the cyl tier's Cortical Bone
        })
