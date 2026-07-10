import numpy as np

from emgop.tissue import ANISOTROPY_RATIO, SIGMA_MUSCLE_CROSS

GROUP_NAMES = {
    "Cancellous Bone": 1,
    "Cortical Bone": 2,
    "Muscle": 3,
    "Fat": 4,
    "Skin": 5,
    "Boundary": 6,
}

CONDUCTIVITY = {
    "Cancellous Bone": 0.075,
    "Cortical Bone": 0.02,
    "Fat": 0.0379,
    "Skin": 4.55e-4,
    "Muscle": np.diag([SIGMA_MUSCLE_CROSS, SIGMA_MUSCLE_CROSS,
                       ANISOTROPY_RATIO * SIGMA_MUSCLE_CROSS]),
}

__all__ = ["GROUP_NAMES", "ANISOTROPY_RATIO", "CONDUCTIVITY"]

