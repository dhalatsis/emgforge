"""Synthesis engines: φ(z) → SFAP.

Two engines share one input contract. The volume conductor enters only through
φ(z); neither engine cares whether it came from the analytical cylinder or from
an FEM solve.

``engines.fourier``
    2-D frequency-domain synthesis (Farina 2004). Finite-fibre effects via the
    ``pare`` operator in (kt, kz), then a Radon section back to time.
    **Window-centred** output.

``engines.spatial``
    FFT-free time-domain line-source integral, ``SFAP = (CSD @ φ)``.
    **Physical-time native** — the NMJ fires at t = 0. Single-fibre only; the
    multi-fibre sum over a ``FibreBed`` lives in ``emgforge.synthesis.field_to_muap``.

>>> from emgforge.synthesis.engines.spatial import compute_sfap_spatial, SpatialConfig
"""

from emgforge.synthesis.engines.spatial import (  # noqa: F401
    SpatialConfig,
    build_csd_matrix,
    compute_sfap_spatial,
    rosenfalck_vm,
)

__all__ = [
    "SpatialConfig",
    "compute_sfap_spatial",
    "build_csd_matrix",
    "rosenfalck_vm",
]
