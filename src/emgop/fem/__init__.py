# Version information for the FEM module
__version__ = "2.0.0"
__version_info__ = {
    "version": "2.0.0",
    "features": ["native_point_source", "electrode_return_modes", "ground_placement"],
    "backwards_compatible": True,
}

from .constants import ANISOTROPY_RATIO, CONDUCTIVITY, GROUP_NAMES
from .solver import ConstrainedLinearProblem, FEMModel
from .sanity import (
    choose_source_point_near_surface,
    load_manifest,
    load_meta,
    make_probe_points_from_meta,
    plot_slices,
    run_one,
    run_manifest_entry,
)
from .electrode_configs import (
    ElectrodeType,
    DetectionMode,
    ElectrodeConfig,
    sample_electrode_area,
    compute_electrode_centers,
    compute_ground_position,
    evaluate_detection,
    ElectrodeFEMSolver,
)
from .point_source import NativePointSource
from .rotation import rotate_conductivity_tensor, rotate_point_in_cylinder

__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    # Constants
    "ANISOTROPY_RATIO",
    "CONDUCTIVITY",
    "GROUP_NAMES",
    "ConstrainedLinearProblem",
    "FEMModel",
    "choose_source_point_near_surface",
    "load_manifest",
    "load_meta",
    "make_probe_points_from_meta",
    "plot_slices",
    "run_one",
    "run_manifest_entry",
    # Electrode configurations
    "ElectrodeType",
    "DetectionMode",
    "ElectrodeConfig",
    "sample_electrode_area",
    "compute_electrode_centers",
    "compute_ground_position",
    "evaluate_detection",
    "ElectrodeFEMSolver",
    # Native point source
    "NativePointSource",
    # Rotation for pennation
    "rotate_conductivity_tensor",
    "rotate_point_in_cylinder",
]

