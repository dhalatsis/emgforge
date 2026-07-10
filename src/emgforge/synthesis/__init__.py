"""emgforge.synthesis — SFAP/MUAP synthesis from a reciprocal lead field φ(z).

Two engines share one input contract, ``φ(z) → waveform``:

1. **Fourier** (``emgforge.synthesis.fourier``) — 2-D frequency-domain synthesis with
   fibre-end modelling via the ``pare`` operator in (kt, kz). Window-centred output.
2. **Spatial** (``emgforge.synthesis.spatial_refactor``) — FFT-free time-domain
   line-source integral ``SFAP = (CSD @ φ)``. Physical-time native.

The volume conductor enters only through φ(z); the engines do not care whether it
came from the analytical cylinder or from an FEM solve.

Quick start
-----------
>>> from emgforge.synthesis import generate_muap_from_phi, get_optimal_config
>>> result = generate_muap_from_phi(phi_matrix, dz_mm=2.5, config=get_optimal_config())
>>> result.t_ms, result.muap
"""

from emgforge.synthesis.api import (
    MUAPConfig,
    MUAPResult,
    generate_muap_from_phi,
    get_adaptive_config,
    get_mri_config,
    get_optimal_config,
    get_truncated_input_config,
)
from emgforge.synthesis.adaptive_w import choose_w as adaptive_choose_w  # noqa: F401
from emgforge.synthesis.conventions import (  # noqa: F401
    Conventions,
    FARINA_DEFAULT,
    FEM_NEURODEC,
)

__all__ = [
    "MUAPConfig",
    "MUAPResult",
    "generate_muap_from_phi",
    "get_optimal_config",
    "get_adaptive_config",
    "get_mri_config",
    "get_truncated_input_config",
    "adaptive_choose_w",
    "Conventions",
    "FARINA_DEFAULT",
    "FEM_NEURODEC",
]
