import numpy as np

GROUP_NAMES = {
    "Cancellous Bone": 1,
    "Cortical Bone": 2,
    "Muscle": 3,
    "Fat": 4,
    "Skin": 5,
    "Boundary": 6,
}

ANISOTROPY_RATIO = 5
CONDUCTIVITY = {
    "Cancellous Bone": 0.075,
    "Cortical Bone": 0.02,
    "Fat": 0.0379,
    "Skin": 4.55e-4,
    "Muscle": np.diag([0.2455, 0.2455, ANISOTROPY_RATIO * 0.2455]),
}

__all__ = ["GROUP_NAMES", "ANISOTROPY_RATIO", "CONDUCTIVITY"]

