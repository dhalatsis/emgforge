"""Shared tissue-conductivity constants — the single source of truth.

Deliberately dolfinx-free and dependency-light, so every consumer can import it
cheaply: the FEM solver (`emgforge.fem.constants`), the analytical pointcloud path
(`neural_field.pointcloud.analytical`, which must stay FEM-free), and the MRI anatomy
solver (`emgforge.mri.core.fem_solver`). Before this module those three each defined their
own `ANISOTROPY_RATIO = 5` — three copies that could silently drift.

Values are S/m. Muscle is anisotropic: the fibre-direction (z) conductivity is
`ANISOTROPY_RATIO` times the cross-fibre value.
"""
from __future__ import annotations

# Muscle fibre-to-cross conductivity ratio (σ_zz / σ_xx).
ANISOTROPY_RATIO = 5

# Muscle cross-fibre conductivity (S/m). Fibre-direction σ = ratio × this.
SIGMA_MUSCLE_CROSS = 0.2455
