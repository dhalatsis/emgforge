"""neural_field — volume-conductor solution datasets for neural-field / PINN training.

Samples the ``emgforge`` FEM (volume conductor) solution onto voxel grids and point
clouds, producing ``(position → φ, σ, source, tissue)`` training sets for learned
representations of the lead field. This is the ML/dataset sibling of the forward-model
package: it depends **one-way** on ``emgforge`` (the volume conductor) — ``emgforge``
never imports ``neural_field``.

Subpackages
-----------
``neural_field.voxel``
    Geometry + source → voxel grids.
``neural_field.pointcloud``
    FEM solutions → point-cloud / PINN datasets (φ, conductivity, source, tissue labels).

The generation pipeline (scripts) consumes ``emgforge`` mesh + FEM outputs; see
``scripts/neural_field/``.
"""

__version__ = "0.1.0"
