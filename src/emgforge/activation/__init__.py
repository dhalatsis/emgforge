"""emgforge.activation — neural drive → motor-unit spike trains.

A clean-room, phenomenological reimplementation of NeuroMotion's motoneuron-pool tier
(Fuglevand / iEMG-simulator parametrisation) — recruitment + onion-skin rate coding +
ISI-renewal spiking. No membrane biophysics, no vendored code. The spike trains drive
the FEM-derived MUAPs from ``emgforge.synthesis`` to make interference EMG.

>>> from emgforge.activation import MotoneuronPool, drive
>>> E = drive.trapezoid(peak=0.4, rise_s=2, hold_s=2, fall_s=2, fs=2048)
>>> E = drive.add_common_drive(E, sigma=0.02, fs=2048)
>>> pool = MotoneuronPool(n_mu=100, fs=2048)
>>> spikes = pool.spike_trains(E, seed=0)      # list of per-MU spike sample indices
"""

from emgforge.activation import drive  # noqa: F401
from emgforge.activation.emg import compound_emg, rms_envelope
from emgforge.activation.pool import DEFAULTS, ISI_CV, MotoneuronPool

__all__ = ["MotoneuronPool", "DEFAULTS", "ISI_CV", "drive", "compound_emg", "rms_envelope"]
