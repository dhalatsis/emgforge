"""
Point cloud generation module for neural network training data.

This module provides tools for generating point cloud training data
from FEM solutions, suitable for:
- Neural field training: u(x, y, z | electrode_position)
- PINN training: Learning the PDE solution with physics constraints

Sampling Strategies:
- uniform: Uniform random sampling in the domain
- near_electrode: Concentrated sampling near electrode(s)
- stratified_tissue: Equal sampling from each tissue layer
- surface_biased: Higher density near skin surface
- mixed: Combination of strategies

Example usage:
    from emgop.pointcloud import generate_pointcloud_sample, sample_points

    # Generate training sample from FEM solve
    sample = generate_pointcloud_sample(
        model=fem_model,
        uh=solution,
        meta=metadata,
        source_position=electrode_pos,
        n_interior_points=10000,
        n_boundary_points=2000,  # For PINN
        sampling_strategy="near_electrode",
    )

    # Sample contains:
    # - points: (N, 3) query coordinates
    # - u: (N,) solution values
    # - sigma: (N, 6) conductivity tensors
    # - electrode_position: (3,) electrode coords
    # - boundary_points: (M, 3) for PINN
    # - boundary_normals: (M, 3) for PINN
"""

from .sampling import (
    SamplingStrategy,
    sample_points,
    sample_points_uniform_cylinder,
    sample_points_near_electrode,
    sample_points_stratified_tissue,
    sample_points_surface_biased,
    sample_boundary_points,
)

from .evaluate import (
    evaluate_fem_solution,
    evaluate_conductivity_at_points,
    compute_tissue_labels,
    compute_source_field,
    evaluate_all_fields,
)

from .generate import (
    generate_pointcloud_sample,
    save_pointcloud_sample,
    load_pointcloud_sample,
)

from .analytical import (
    compute_conductivity_analytical,
)


__all__ = [
    # Sampling
    "SamplingStrategy",
    "sample_points",
    "sample_points_uniform_cylinder",
    "sample_points_near_electrode",
    "sample_points_stratified_tissue",
    "sample_points_surface_biased",
    "sample_boundary_points",
    # Evaluation
    "evaluate_fem_solution",
    "evaluate_conductivity_at_points",
    "compute_tissue_labels",
    "compute_source_field",
    "evaluate_all_fields",
    # Generation
    "generate_pointcloud_sample",
    "save_pointcloud_sample",
    "load_pointcloud_sample",
    # Analytical
    "compute_conductivity_analytical",
]
