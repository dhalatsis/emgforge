"""The Rosenfalck intracellular action potential — one definition, both engines.

The spatial engine samples the IAP itself (``rosenfalck_vm``) and differentiates
it numerically; the Fourier engine (``engines.fourier.build_spe2_iap_spectrum``)
builds the analytic derivative ``dVm/dz`` on its own z-grid. The two therefore use
different *expressions* — the function vs its derivative — but they must share the
one physical scale, the Rosenfalck amplitude ``A``. Keeping ``A`` here means a change
to the IAP magnitude lands in both engines at once, instead of one silently diverging.

Note the sibling ``analytical.signal_generator`` carries its own ``96`` (in **mV**, not
V): it is the deliberately-independent reference implementation, so it is *not* wired to
this constant — coupling the reference to the thing it validates would defeat its purpose.
"""

from __future__ import annotations

import numpy as np

# Rosenfalck (1969) amplitude A = 96, given in mV; scaled to V for SI consistency.
# Audit 2026-06-10 — the mV→V scaling fixes a ~10³× MUAP-amplitude mismatch vs the
# Neurodec ground truth. Both synthesis engines read the IAP scale from here.
ROSENFALCK_AMPLITUDE_V = 96e-3


def rosenfalck_vm(z_mm: np.ndarray) -> np.ndarray:
    """Rosenfalck intracellular action potential (depolarisation above baseline):
    Vm(z) = A·z³·e^{−z} [A in V], zero for z < 0 (wave not yet arrived).
    Onset is smooth (V, V', V'' all → 0 as z→0⁺), so no spurious wavefront source.
    """
    z = np.asarray(z_mm, dtype=float)
    return np.where(z >= 0.0, ROSENFALCK_AMPLITUDE_V * z**3 * np.exp(-z), 0.0)


__all__ = ["ROSENFALCK_AMPLITUDE_V", "rosenfalck_vm"]
