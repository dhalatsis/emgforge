"""Coordinate-network architectures for the learned volume conductor.

Ported from the closed Exp 1-4 work (neural-forward-emg/training/neural_field/models).
MLP + random Fourier features was the best MUAP performer there; SIREN had the lowest
pointwise error but its high-frequency jaggedness wrecked MUAPs (SFAP is proportional
to phi'', so field roughness is amplified) -- keep both to re-test that on our benchmark.
"""
from .neural_field import MLP, SIREN, FourierFeatures

__all__ = ["MLP", "SIREN", "FourierFeatures"]
