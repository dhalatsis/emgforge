"""
muap_generator - MUAP generation from reciprocal field data.

Two pipelines are provided:

1. **Fourier** (recommended) - 2D frequency-domain synthesis with proper
   fiber-end modelling via the `pare` function in (kt, kz).
2. **Numerical** (experimental) - Direct time-domain convolution with
   Tukey windowed fiber ends.

Quick start
-----------
>>> from muap_generator import generate_muap_from_npz, MUAPConfig
>>> result = generate_muap_from_npz("sample.npz", n_fibers=50)
>>> result.t_ms, result.muap

Or from a phi matrix directly:
>>> from muap_generator import generate_muap_from_phi
>>> result = generate_muap_from_phi(phi_matrix, dz_mm=2.5)

Low-level access:
>>> from muap_generator.fourier import SFAPParams, compute_sfap_from_phi_z
>>> from muap_generator.numerical import NumericalConfig, compute_sfap_numerical
"""

from muap_generator.api import (
    MUAPConfig,
    MUAPResult,
    generate_muap_from_phi,
    generate_muap_from_npz,
    generate_muaps_batch,
    get_optimal_config,
    get_fast_config,
    get_high_quality_config,
    get_adaptive_config,
    get_mri_config,
    get_truncated_input_config,
)
from muap_generator.adaptive_w import choose_w as adaptive_choose_w  # noqa: F401
from muap_generator.conventions import (  # noqa: F401
    Conventions,
    FARINA_DEFAULT,
    FEM_NEURODEC,
)

__all__ = [
    "MUAPConfig",
    "MUAPResult",
    "generate_muap_from_phi",
    "generate_muap_from_npz",
    "generate_muaps_batch",
    "get_optimal_config",
    "get_fast_config",
    "get_high_quality_config",
    "get_adaptive_config",
    "get_mri_config",
    "get_truncated_input_config",
    "adaptive_choose_w",
    "Conventions",
    "FARINA_DEFAULT",
    "FEM_NEURODEC",
]
