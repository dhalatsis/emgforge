# Pennation Angle in the FEM Pipeline

This document describes how muscle fiber orientation (pennation angle) is modeled and implemented in the FEM solver.

## Physical Background

### What is Pennation Angle?

In skeletal muscle, fibers are not always aligned parallel to the muscle's longitudinal axis. The **pennation angle** (θ) is the angle between the muscle fiber direction and the longitudinal axis of the muscle.

```
Longitudinal axis (Z)
        ↑
        │      ╱ Muscle fiber
        │    ╱
        │  ╱ θ (pennation angle)
        │╱
        └──────→ Radial direction
```

- **θ = 0°**: Fibers aligned with Z-axis (fusiform muscles like biceps)
- **θ = 15-30°**: Typical pennate muscles (gastrocnemius, deltoid)
- **θ > 30°**: Highly pennate muscles

### Why Does It Matter for EMG?

Muscle tissue is **electrically anisotropic** - current flows more easily along the fiber direction than across it. The conductivity ratio is typically 5:1 (longitudinal:transverse).

When fibers are pennated, this anisotropy direction rotates, affecting:
- How current spreads from motor units
- The shape and amplitude of detected EMG signals
- Volume conductor effects in the forward model

## Mathematical Model

### Default Conductivity Tensor (θ = 0°)

With fibers aligned along Z, the muscle conductivity tensor is diagonal:

```
        ┌ σ⊥   0    0  ┐
Σ₀  =   │  0  σ⊥   0  │
        └  0   0   σ∥ ┘

where:
  σ⊥ = 0.2455 S/m  (transverse conductivity)
  σ∥ = 1.2275 S/m  (longitudinal conductivity, 5× higher)
```

### Rotation for Pennation

To model pennated fibers, we rotate the conductivity tensor around an axis perpendicular to both the fiber direction and the radial direction.

For a cylindrical muscle geometry, at each point (x, y, z):

1. **Compute the radial vector** in the XY plane:
   ```
   v_radial = [x, y, 0] / |[x, y, 0]|
   ```

2. **Compute the rotation axis** (tangent to cylinder surface):
   ```
   k = [0, 0, 1]  (Z-axis)
   v_rotation = -cross(v_radial, k)
   ```

   This gives a vector perpendicular to both the radial direction and Z, pointing tangentially around the cylinder.

3. **Apply Rodrigues' rotation formula** to get rotation matrix R:
   ```
   R = I + sin(θ)K + (1 - cos(θ))K²

   where K is the skew-symmetric matrix:
        ┌  0   -k₃   k₂ ┐
   K =  │  k₃   0   -k₁ │
        └ -k₂   k₁   0  ┘
   ```

4. **Rotate the conductivity tensor**:
   ```
   Σ_rotated = R · Σ₀ · Rᵀ
   ```

### Geometric Interpretation

The rotation tilts the principal axis of anisotropy (originally Z) toward the radial direction:

```
    Z                      Z
    ↑                      ↑    ╱ Rotated principal axis
    │                      │  ╱
    │   θ = 0°             │╱    θ = 30°
    │                      │
    └──→ radial            └──→ radial
```

This means current preferentially flows along the tilted fiber direction, not purely along Z.

## Implementation

### File: `src/emgop/fem/rotation.py`

```python
def rotate_conductivity_tensor(
    centroid: np.ndarray,
    tensor: np.ndarray,
    theta_deg: float,
) -> np.ndarray:
    """
    Rotate a conductivity tensor by theta degrees around an axis
    perpendicular to the radial direction.

    Parameters
    ----------
    centroid : (3,) array
        Cell centroid (x, y, z)
    tensor : (3, 3) array
        Conductivity tensor to rotate
    theta_deg : float
        Pennation angle in degrees

    Returns
    -------
    (3, 3) array
        Rotated conductivity tensor
    """
```

### File: `src/emgop/fem/solver.py`

The `FEMModel.apply_pinnation()` method applies the rotation to all muscle cells:

```python
def apply_pinnation(self, theta: float):
    """
    Apply pennation angle rotation to muscle conductivity tensors.

    Parameters
    ----------
    theta : float
        Pennation angle in degrees
    """
    for cell_index, marker in enumerate(self.cell_markers.values):
        if material_map[marker] == "Muscle":
            # Get cell centroid
            centroid = compute_cell_centroid(cell_index)

            # Rotate tensor based on position
            rotated = rotate_conductivity_tensor(
                centroid,
                CONDUCTIVITY["Muscle"],
                theta
            )

            # Store in DG0 tensor function
            self.sigma_anisotropic[cell_index] = rotated
```

### Key Design Choices

1. **Position-dependent rotation**: The rotation axis varies with position around the cylinder, so fibers always tilt "outward" from the radial direction.

2. **Cell-by-cell application**: Each mesh cell gets its own rotated tensor based on its centroid location.

3. **Fresh model per angle**: Since `apply_pinnation()` modifies the conductivity in-place, create a new `FEMModel` for each pennation angle to avoid accumulating rotations.

## Usage

### Command Line

```bash
# Single solve with pennation angle
python scripts/01_solve_fem.py \
    --mesh meshes/sample.msh \
    --meta metadata/sample.json \
    --out_dir fem_output \
    --pennation_angle 30.0

# Generate pennation sweep dataset
python scripts/legacy/05_generate_pennation_dataset.py \
    --mesh meshes/sample.msh \
    --meta metadata/sample.json \
    --out_dir datasets/pennation \
    --n_angles 20 \
    --angle_min 0 --angle_max 45
```

### Python API

```python
from emgop.fem import FEMModel

# Create model
model = FEMModel(mesh_path, build_conductivity_map=True)

# Apply pennation (must be done BEFORE solving)
model.apply_pinnation(theta=30.0)  # 30 degrees

# Solve
uh = model.solve_for_point(source_point)
```

## Verification

The effect of pennation can be verified by comparing solutions at different angles:

```python
# Solutions at 0° and 30° should differ
u_0 = solve_with_pennation(0.0)
u_30 = solve_with_pennation(30.0)

# Difference is small but measurable
diff = np.abs(u_0 - u_30)
print(f"Max difference: {diff.max()}")  # ~1e-5 for typical cases
```

The difference is subtle because:
1. The conductivity change is moderate (rotation, not magnitude change)
2. The effect is strongest in the muscle layer
3. Boundary conditions dominate near the surface

See `sanity_checks/dataset_verification/verify_datasets.ipynb` for visualization of the pennation effect.

## References

1. Stegeman, D.F., et al. "Modeling motor unit action potentials." *IEEE Engineering in Medicine and Biology Magazine*, 1994.
2. Lowery, M.M., et al. "A multiple-layer finite-element model of the surface EMG signal." *IEEE Transactions on Biomedical Engineering*, 2002.
3. Mesin, L., et al. "Simulation of surface EMG signals for a multilayer volume conductor." *IEEE Transactions on Biomedical Engineering*, 2006.
