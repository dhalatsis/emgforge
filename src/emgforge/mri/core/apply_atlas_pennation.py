"""Apply atlas-based pennation to a MuscleFiberModel.

Without DT-MRI we can't measure per-subject pennation. The Lieber atlas
(forearm_muscle_atlas.json) provides a typical-value prior for 26
forearm muscles. We can:

  1. Set ALL muscles to the median forearm pennation (~5.8°).
  2. Provide a label→atlas-key mapping (when a user has identified
     specific labels) and pull per-muscle pennation from the atlas.
  3. Use a heuristic based on muscle shape (elongated, parallel-fibred
     muscles like brachioradialis get lower pennation; compact muscles
     like supinator get higher).

Example (PD_PROPELLER template01 — the canonical case)::

    from emgforge.mri.core.apply_atlas_pennation import (
        apply_default_pennation, apply_atlas_pennation,
        load_template01_label_map,
    )
    apply_default_pennation(fiber_model, value_deg=5.8)

    # Canonical label → atlas-key mapping for the PD_PROPELLER MRI used by
    # this repo. Cross-verified against mri-auto-seg's
    # nnUNet_data/.../dataset.json (label_remap.json). See
    # mri/data/template01_label_to_muscle.json for the full table.
    label_to_atlas = load_template01_label_map()
    apply_atlas_pennation(fiber_model, label_to_atlas)

Manually-mapped use::

    apply_atlas_pennation(fiber_model, label_to_muscle_name={
        7: "extensor_digitorum_communis_middle",  # autoseg name: ED
        12: "flexor_digitorum_profundus_middle",  # autoseg name: FDP
    })
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ATLAS_PATH = _DATA_DIR / "forearm_muscle_atlas.json"
TEMPLATE01_LABEL_MAP_PATH = _DATA_DIR / "template01_label_to_muscle.json"


def load_atlas() -> dict:
    with ATLAS_PATH.open() as f:
        return json.load(f)


def load_template01_label_map() -> dict[int, str]:
    """Canonical segmentation-label → Lieber-atlas-key mapping for the
    PD_PROPELLER template01 MRI used by this repo.

    Cross-verified against mri-auto-seg/nnUNet_data/.../dataset.json. Returns
    a ``{label_int: atlas_key_str}`` dict, suitable for
    ``apply_atlas_pennation``. Only entries with a non-null ``atlas_key`` in
    the JSON are returned (some labels have no Lieber-atlas analogue —
    e.g. ANC, EDM, PQ — and fall through to the fallback).

    See also
    --------
    mri/data/template01_label_to_muscle.json   the underlying source.
    """
    with TEMPLATE01_LABEL_MAP_PATH.open() as f:
        m = json.load(f)
    out: dict[int, str] = {}
    for lbl_str, info in m["muscles"].items():
        key = info.get("atlas_key")
        if key:
            out[int(lbl_str)] = key
    return out


def load_template01_label_names() -> dict[int, dict]:
    """Full per-label metadata for template01 (abbrev, name, atlas_key,
    compartment, notes). Returns ``{label_int: info_dict}``. Useful when you
    want anatomical names for display rather than atlas-key lookup.
    """
    with TEMPLATE01_LABEL_MAP_PATH.open() as f:
        m = json.load(f)
    return {int(k): v for k, v in m["muscles"].items()}


def apply_default_pennation(fiber_model, value_deg: float = 5.8):
    """Set ``pennation_deg`` on every muscle to a single default.

    Default 5.8° = median forearm pennation per Lieber atlas.
    """
    for m in fiber_model.muscles.values():
        if m.tissue_type == "muscle":
            m.pennation_deg = float(value_deg)


def apply_atlas_pennation(fiber_model, label_to_muscle_name: dict,
                            fallback_deg: float = 5.8):
    """Look up pennation per muscle from the Lieber atlas.

    Parameters
    ----------
    label_to_muscle_name : dict[int, str]
        Mapping from segmentation label → atlas key (see
        forearm_muscle_atlas.json for available keys).
    fallback_deg : float
        Pennation used for muscles whose label isn't mapped.
    """
    atlas = load_atlas()["muscles"]
    for label, m in fiber_model.muscles.items():
        if m.tissue_type != "muscle":
            continue
        name = label_to_muscle_name.get(label)
        if name is None or name not in atlas:
            m.pennation_deg = float(fallback_deg)
            continue
        m.pennation_deg = float(atlas[name]["pennation_deg"])


def apply_heuristic_pennation(fiber_model, low_deg: float = 3.0,
                                high_deg: float = 9.0):
    """Assign pennation per muscle from a simple shape heuristic.

    More elongated muscles (high pca_elongation) get LOWER pennation
    (parallel-fibred); compact muscles get HIGHER pennation. Maps
    linearly between ``low_deg`` and ``high_deg`` across the muscle
    population.
    """
    muscles = [m for m in fiber_model.muscles.values()
                if m.tissue_type == "muscle"]
    if not muscles:
        return
    elongs = np.array([m.pca_elongation for m in muscles])
    # Higher elongation → LOWER pennation (parallel)
    # Lower elongation → HIGHER pennation (compact pennate)
    if elongs.std() < 1e-6:
        for m in muscles:
            m.pennation_deg = float(0.5 * (low_deg + high_deg))
        return
    e_min, e_max = elongs.min(), elongs.max()
    for m in muscles:
        norm = (m.pca_elongation - e_min) / max(e_max - e_min, 1e-6)
        # norm=0 (least elongated) → high pennation
        # norm=1 (most elongated) → low pennation
        m.pennation_deg = float(high_deg + (low_deg - high_deg) * norm)


__all__ = [
    "load_atlas",
    "load_template01_label_map",
    "load_template01_label_names",
    "apply_default_pennation",
    "apply_atlas_pennation",
    "apply_heuristic_pennation",
]
