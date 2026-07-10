"""Tiny helper to centralize the choice of MUAP config for MRI scripts.

Use in any MRI script that builds a MUAPConfig:

    from mri.core._pipeline_config import build_muap_config
    cfg = build_muap_config(half_mm=half, auto=args.auto)

Legacy:  MUAPConfig(len1_mm=half, len2_mm=half, w=256)
Auto:    get_mri_config() with len1_mm/len2_mm filled — Butterworth
         smoothing + edge_taper="auto". The edge_taper is critical for
         MRI cases where the lead-field has edge/peak ≥ 0.3 (which is
         almost always — see progress_reports/adhoc/10).

Both branches preserve all other fields at their library defaults.
"""
from __future__ import annotations

from emgforge.synthesis.api import MUAPConfig, get_mri_config


def build_muap_config(half_mm: float, auto: bool = False, **overrides):
    """Return a MUAPConfig.

    Parameters
    ----------
    half_mm : float
        len1_mm and len2_mm (cylinder half-length on each side of NMJ).
    auto : bool
        If True, use ``get_mri_config()`` — Butterworth smoothing
        (c=0.03 o=2) + edge_taper="auto" (15 samples when edge/peak ≥
        0.3). This is the right choice for FEM-derived MRI lead fields.
    **overrides : any
        Forwarded as field overrides on the resulting config.
    """
    if auto:
        cfg = get_mri_config()
        cfg.len1_mm = float(half_mm)
        cfg.len2_mm = float(half_mm)
    else:
        cfg = MUAPConfig(len1_mm=float(half_mm), len2_mm=float(half_mm), w=256)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
