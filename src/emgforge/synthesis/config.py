"""Synthesis config base — the type ``field_to_muap`` dispatches on.

``MUAPConfig`` (the Fourier engine, window-centred) and ``SpatialConfig`` (the spatial
engine, physical-time) both subclass this. The two still carry overlapping fields
(``v``, ``fsamp``, ``w`` …); consolidating those into the base is a later step. For now
this is a marker so ``field_to_muap(field, bed, config)`` can pick the engine from the
config's type rather than a string flag.
"""

from __future__ import annotations


class SynthesisConfig:
    """Base marker for the per-engine synthesis configs."""
