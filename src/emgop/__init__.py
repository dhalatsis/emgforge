"""
Lightweight EMG forward operator package scaffold.

This package currently provides sampling and meshing helpers extracted
from the legacy `generate_mesh_dataset.py` script. Additional domains
can be added under subpackages (geometry, fem, voxel, etc.) as the
refactor progresses.
"""

__all__ = [
    "sampling",
    "meshing",
]

