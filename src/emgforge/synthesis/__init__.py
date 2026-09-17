"""emgforge.synthesis — SFAP/MUAP synthesis from a reciprocal lead field φ(z).

Two engines share one input contract, ``φ(z) → waveform``:

1. **Fourier** (``emgforge.synthesis.engines.fourier``) — 2-D frequency-domain
   synthesis with fibre-end modelling via the ``pare`` operator in (kt, kz).
   Window-centred output.
2. **Spatial** (``emgforge.synthesis.engines.spatial``) — FFT-free time-domain
   line-source integral ``SFAP = (CSD @ φ)``. Physical-time native.

The volume conductor enters only through φ(z); the engines do not care whether it
came from the analytical cylinder or from an FEM solve.

The **production route** is the spatial engine with :func:`production_config` —
direct line-source synthesis, validated against the closed-form line-source oracle
and the Farina (2004) cylinder (``synthesis/DIRECT_LINE_SOURCE.md``). It is the
default of ``field_to_muap(config=None)``. The Fourier engine is kept for comparison
only and warns once per process when selected.

Quick start
-----------
>>> from emgforge.synthesis import FibreBed, field_to_muap, production_config
>>> bed = FibreBed.uniform(50, dz_mm=0.977, len1_mm=60, len2_mm=60, v=4.0)
>>> result = field_to_muap(phi_matrix, bed)                 # == production_config()
>>> result.t_ms, result.muap                                # physical time, t=0 at the NMJ
"""

from emgforge.synthesis.api import (
    FOURIER_ROUTE_WARNING,
    MUAPConfig,
    MUAPResult,
    field_to_muap,
    generate_muap_from_phi,
    get_adaptive_config,
    get_mri_config,
    get_optimal_config,
    get_truncated_input_config,
)
from emgforge.synthesis.config import SynthesisConfig  # noqa: F401
from emgforge.synthesis.fibres import Fibre, FibreBed, NonUniformDz  # noqa: F401
from emgforge.synthesis.engines.spatial import SpatialConfig, production_config  # noqa: F401
from emgforge.synthesis.adaptive_w import choose_w as adaptive_choose_w  # noqa: F401
from emgforge.synthesis.conventions import (  # noqa: F401
    Conventions,
    FARINA_DEFAULT,
    FEM_NEURODEC,
)
from emgforge.synthesis.metrics import waveform_features  # noqa: F401

__all__ = [
    "production_config",
    "FOURIER_ROUTE_WARNING",
    "MUAPConfig",
    "MUAPResult",
    "SynthesisConfig",
    "SpatialConfig",
    "Fibre",
    "FibreBed",
    "NonUniformDz",
    "field_to_muap",
    "generate_muap_from_phi",
    "get_optimal_config",
    "get_adaptive_config",
    "get_mri_config",
    "get_truncated_input_config",
    "adaptive_choose_w",
    "Conventions",
    "FARINA_DEFAULT",
    "FEM_NEURODEC",
    "waveform_features",
]
