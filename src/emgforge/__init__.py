"""emgforge — synthetic EMG forward modelling.

A volume conductor produces a reciprocal lead field φ(z) along each muscle fibre;
a synthesis engine turns φ(z) into an SFAP, and the sum over a motor unit's fibres
into a MUAP.

Subpackages
-----------
``emgforge.synthesis``
    φ(z) → SFAP → MUAP. Two engines (Fourier, spatial) behind one input contract.
    Pure NumPy/SciPy — no FEM stack required.

The FEM volume conductor (``emgop``) and the MRI anatomy pipeline (``mri``) are
still separate top-level import roots; folding them in here is tracked as a
follow-up.
"""

__version__ = "0.1.0"
