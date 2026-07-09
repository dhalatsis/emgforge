"""Spatial (time-domain) MUAP refactor.

Revives the 2024/2025 spatial line-source method (``SFAP = (CSD @ φ)``) as a
clean, FFT-free alternative to the Fourier ``pare``/``radon`` pipeline.

>>> from muap_generator.spatial_refactor import compute_sfap_spatial, SpatialConfig
"""

from muap_generator.spatial_refactor.spatial_sfap import (  # noqa: F401
    Fibre,
    SpatialConfig,
    build_csd_matrix,
    compute_muap_spatial,
    compute_sfap_spatial,
    rosenfalck_dvm_dz,
    rosenfalck_d2vm_dz2,
)

__all__ = [
    "SpatialConfig",
    "Fibre",
    "compute_sfap_spatial",
    "compute_muap_spatial",
    "build_csd_matrix",
    "rosenfalck_dvm_dz",
    "rosenfalck_d2vm_dz2",
]
