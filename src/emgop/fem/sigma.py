"""Injectable conductivity-field (σ-map) builders.

A σ-builder maps a mesh + its per-cell tissue markers to a DG0 tensor `Function`
— the per-cell anisotropic conductivity field the weak form multiplies against
∇u. This is the one genuinely backend-specific part of the reciprocity solve
(see `_refactor/10_leadfield.md` §3), so the future `LeadField` engine receives a
builder rather than containing one.

`LayeredSigma` here is the cylinder/ellipse builder (lifted from
`FEMModel.build_conductivity_map`). The MRI-anatomy builder (`AnatomySigma`, with
per-muscle fibre rotation) lives in `mri.core` — landing in L3 — so the heavy
`[mri]` dependencies stay out of `emgop.fem`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from dolfinx import fem

from .constants import GROUP_NAMES


@dataclass
class LayeredSigma:
    """σ-map for a layered (cylinder/ellipse) mesh.

    Muscle cells get the (anisotropic) tensor from ``tissue_table``; every other
    tissue gets an isotropic ``σ·I``. ``group_names`` maps physical tag → material.
    """
    tissue_table: dict
    group_names: dict = field(default_factory=lambda: GROUP_NAMES)

    def __call__(self, mesh, cell_markers) -> "fem.Function":
        dim = mesh.topology.dim
        V_tensor = fem.functionspace(mesh, ("DG", 0, (dim, dim)))
        sigma = fem.Function(V_tensor)
        material_map = {v: k for k, v in self.group_names.items()}
        with sigma.vector.localForm() as loc:
            for cell_index, marker in enumerate(cell_markers.values):
                material = material_map[int(marker)]
                if material == "Muscle":
                    loc.setValuesBlocked([cell_index], self.tissue_table[material].flatten())
                else:
                    loc.setValuesBlocked(
                        [cell_index],
                        (self.tissue_table[material] * np.eye(dim)).flatten(),
                    )
        return sigma
