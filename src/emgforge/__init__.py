"""emgforge — synthetic EMG forward modelling.

A volume conductor produces a reciprocal lead field φ(z) along each muscle fibre;
a synthesis engine turns φ(z) into an SFAP, and the sum over a motor unit's fibres
into a MUAP.

Subpackages
-----------
``emgforge.synthesis``
    φ(z) → SFAP → MUAP. Two engines (Fourier, spatial) behind one input contract.
    Pure NumPy/SciPy — no FEM stack required.
``emgforge.fem``
    The FEM volume conductor and the shared ``LeadField`` reciprocity solve.
``emgforge.mri``
    MRI-forearm anatomy → FEM lead field (pulls the ``[mri]`` extra).
``emgforge.{meshing, sampling, voxel, pointcloud, analytical}``
    The dataset spine — mesh building, fibre sampling, and φ sampling.

Importing ``emgforge`` itself is dependency-light; the FEM subpackages pull the
dolfinx stack only when imported. ``emgforge.Simulator`` / ``emgforge.Recording``
(the neural pool + MUAPs → EMG facade, NumPy only) are resolved lazily on first use.
"""

__version__ = "0.1.0"

__all__ = ["Simulator", "Recording", "__version__"]


def __getattr__(name):
    if name in ("Simulator", "Recording"):
        from emgforge import simulator
        return getattr(simulator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
