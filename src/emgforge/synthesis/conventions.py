"""Pipeline sign/timing conventions for the Farina MUAP engine.

Historically the six conventions (C1–C6) plus an overall output polarity were
hardcoded across ``fourier.py``, the inline engine in ``api.py``, and the
``physical_sfap`` wrapper — and re-implemented by hand in every
``datasets/pm`` script. This module centralises them into ONE explicit,
frozen object with named presets, so the engine's sign/timing behaviour is
data rather than scattered code.

See ``emgforge.synthesis/refactor/KNOWLEDGE.md`` §2 for the full convention table
and ``reports/progress_reports/2026-06-12_pm_convention_sweep.md`` for the
FEM-path validation (posz=0 + polarity=−1 → r=+0.737 vs Neurodec).

Convention map
--------------
========  ==========================================================  ===========================
field     meaning                                                     where it acts
========  ==========================================================  ===========================
iap_flip  C1: ``V2 = -np.flip(V2)`` on the Rosenfalck IAP spectrum    build_spe2_iap_spectrum
swap_ends C2: swap L1(proximal) ↔ L2(distal) in the pare operator     fiber_field_contribution
output_   C5: ``np.flip`` the radon section                           section_from_field_spectrum
 flip
center_   C6: return ``t`` with NMJ-fire at t=0 (i.e. ``t − T/2``)     engine t_ms return
 time
polarity  overall output sign (+1 / −1)                               section_from_field_spectrum
========  ==========================================================  ===========================

Not represented as fields (they are geometric *inputs*, not global toggles):
  * C3 ``posz`` — per-fibre axial NMJ offset. The *value* is the caller's
    decision; the validated FEM convention is posz=0 (the NMJ is already
    encoded in the FEM φ array indexing). Documented on ``FEM_NEURODEC``.
  * C4 ``z_det`` — detector axial position (electrode geometry).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Conventions:
    """The engine-level sign/timing knobs. Defaults == frozen Farina behaviour."""

    iap_flip: bool = True          # C1
    swap_ends: bool = False        # C2
    output_flip: bool = True       # C5
    center_time: bool = False      # C6
    polarity: int = 1              # overall output sign


# The validated Farina / analytical behaviour — byte-for-byte identical to
# ``MUAPConfig()`` prior to the refactor. This is the regression-frozen default.
FARINA_DEFAULT = Conventions()

# The validated FEM-vs-Neurodec convention (PM convention sweep → r=+0.740,
# zero time shift, trough at +18.6 ms). Two differences vs the Farina default:
#   * output_flip=False — the sweep's hand-rolled engine applied a *double*
#     np.flip (which cancels to no net flip); reproduced here as C5 off. With
#     the default single flip the MUAP is time-reversed (trough at −19 ms).
#   * polarity=-1 — negate the output (negative-dominant trough, matching
#     Neurodec).
# Callers on the FEM path additionally pass posz=0 (the NMJ is already encoded
# in the FEM φ array indexing — see the module docstring). Verified through the
# production engine api._compute_muap_core, not just the hand-rolled sweep.
FEM_NEURODEC = Conventions(output_flip=False, polarity=-1)
