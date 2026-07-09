"""Analytical multilayer-cylindrical volume-conductor model (Farina 2004).

An object-oriented implementation of the analytical 4-layer cylindrical
volume conductor and its surface-EMG forward model, after:

    D. Farina, L. Mesin, S. Martina, R. Merletti, "A surface EMG generation
    model with multilayer cylindrical description of the volume conductor",
    IEEE Trans. Biomed. Eng., 51(3), 415-426, 2004.

This is the analytical *reference* model — the ground truth the FEM and
synthesis engines are validated against. Build a case as
``CylindricalVolumeConductor`` + ``MotorUnit`` + ``DetectionSystem``, then drive
it with ``SignalGenerator`` to produce SFAP/MUAP waveforms:

>>> from emgop.analytical import (
...     CylindricalVolumeConductor, MotorUnit, DetectionSystem, SignalGenerator)
>>> sg = SignalGenerator(vc, mu, det, v=4.0, fsamp=4096.0, w=256)
>>> t_ms, sig, positions = sg.generate_muap()
"""

from .volume_conductor import CylindricalVolumeConductor
from .motor_unit import MotorUnit
from .detection_system import DetectionSystem
from .signal_generator import SignalGenerator
from .visualizer import Visualizer

__all__ = [
    "CylindricalVolumeConductor",
    "MotorUnit",
    "DetectionSystem",
    "SignalGenerator",
    "Visualizer",
]
