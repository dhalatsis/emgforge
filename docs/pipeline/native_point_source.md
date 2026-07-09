# Native Point Source Implementation

This document describes the implementation of `NativePointSource`, a pure FEniCSx/dolfinx implementation that replicates `scifem.PointSource` behavior without external dependencies.

## Table of Contents

1. [Background](#background)
2. [Mathematical Foundation](#mathematical-foundation)
3. [How scifem.PointSource Works](#how-scifempointsource-works)
4. [NativePointSource Implementation](#nativepointsource-implementation)
5. [Code Walkthrough](#code-walkthrough)
6. [Usage](#usage)
7. [Verification](#verification)
8. [Performance](#performance)

---

## Background

### The Problem

In FEM (Finite Element Method), we often need to apply point sources to the right-hand side (RHS) vector. A point source represents a Dirac delta function δ(x - x₀) at location x₀. In the weak formulation:

```
∫ f(x) v(x) dx  where f(x) = δ(x - x₀)
```

This integral evaluates to `v(x₀)` - the test function evaluated at the point.

### Why Native Implementation?

- **Reduced dependencies**: No need for the `scifem` package
- **Version compatibility**: Works with any dolfinx version
- **Educational**: Understanding the underlying mechanics
- **Customization**: Easy to modify for special cases

---

## Mathematical Foundation

### Weak Form with Point Source

Consider the conductivity equation:

```
∇·(σ∇u) = f    in Ω
```

The weak form is:

```
∫_Ω σ∇u·∇v dx = ∫_Ω f·v dx    ∀v ∈ V
```

For a point source f(x) = γ·δ(x - x₀), the RHS becomes:

```
∫_Ω γ·δ(x - x₀)·v dx = γ·v(x₀)
```

### Finite Element Discretization

In the finite element space, any function v is represented as:

```
v(x) = Σᵢ vᵢ·φᵢ(x)
```

where φᵢ are basis functions and vᵢ are DoF (degrees of freedom) values.

The RHS vector b has entries:

```
bᵢ = ∫_Ω f·φᵢ dx = γ·φᵢ(x₀)
```

So applying a point source means **adding γ·φᵢ(x₀) to each DoF i** where φᵢ(x₀) ≠ 0.

### Which Basis Functions are Non-Zero?

For standard Lagrange elements, basis functions have **local support** - they're only non-zero within cells containing their associated DoF. Therefore:

1. Find the cell containing point x₀
2. Only basis functions associated with that cell's DoFs are non-zero at x₀
3. Evaluate those basis functions at x₀
4. Add contributions to the corresponding DoFs

---

## How scifem.PointSource Works

Based on analysis of the [scifem source code](https://github.com/scientificcomputing/scifem), here's the algorithm:

### Step 1: Locate Points in Cells

```python
# Build bounding box tree for collision detection
tree = geometry.bb_tree(mesh, tdim)

# Find candidate cells that might contain each point
cell_candidates = geometry.compute_collisions_points(tree, points)

# Verify actual collisions (point inside cell)
cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)
```

### Step 2: Transform to Reference Coordinates

Each cell in FEniCSx has a **reference element** (e.g., unit tetrahedron). To evaluate basis functions, we need the point's coordinates in reference space:

```python
# Get coordinate map
cmap = mesh.geometry.cmap

# Get cell's geometric coordinates
cell_coords = mesh.geometry.x[mesh.geometry.dofmap[cell]]

# Pull back: physical → reference coordinates
ref_point = cmap.pull_back(physical_point, cell_coords)
```

### Step 3: Evaluate Basis Functions

Basis functions are defined on the reference element. Use basix to tabulate:

```python
# Get the basix element
basix_element = V.element.basix_element

# Tabulate basis functions at reference points
# Returns shape: (n_derivatives, n_points, n_basis_functions, value_size)
tab = basix_element.tabulate(0, ref_points)  # 0 = no derivatives

# Extract values (derivative order 0)
basis_values = tab[0]  # Shape: (n_points, n_basis, value_size)
```

### Step 4: Apply to Vector

```python
for i, cell in enumerate(cells):
    cell_dofs = V.dofmap.cell_dofs(cell)
    for j, dof in enumerate(cell_dofs):
        b[dof] += magnitude * basis_values[i, j]
```

---

## NativePointSource Implementation

### Class Overview

```python
class NativePointSource:
    """
    Point source implementation using pure FEniCSx/dolfinx.

    Parameters
    ----------
    V : FunctionSpace
        The function space
    points : np.ndarray
        Point coordinates, shape (n, 3) or (3,)
    magnitude : float
        Magnitude of each point source
    """

    def __init__(self, V, points, magnitude=1.0):
        self.V = V
        self.points = np.atleast_2d(points)
        self.magnitude = magnitude
        self.mesh = V.mesh

        # Precompute cell contributions
        self._cells, self._basis_values, self._valid_indices = \
            self._compute_cell_contributions()

    def apply_to_vector(self, b):
        """Add point source contributions to vector b."""
        ...
```

### Key Implementation Details

#### 1. Point Location

```python
def _compute_cell_contributions(self):
    mesh = self.mesh
    tdim = mesh.topology.dim

    # Step 1: Build bounding box tree
    tree = geometry.bb_tree(mesh, tdim)

    # Step 2: Find candidate cells (fast, approximate)
    cell_candidates = geometry.compute_collisions_points(tree, self.points)

    # Step 3: Verify actual collisions (exact)
    cells = geometry.compute_colliding_cells(mesh, cell_candidates, self.points)

    # Get first colliding cell for each point
    point_cells = []
    valid_indices = []
    for i in range(len(self.points)):
        cell_list = cells.links(i)
        if len(cell_list) > 0:
            point_cells.append(cell_list[0])
            valid_indices.append(i)

    # Handle points outside mesh
    if len(valid_indices) != len(self.points):
        missing = len(self.points) - len(valid_indices)
        print(f"Warning: {missing} point(s) outside mesh domain")
```

#### 2. Coordinate Transformation

```python
    # Get coordinate map for transformations
    cmap = mesh.geometry.cmap
    x_dofs = mesh.geometry.dofmap
    x_coords = mesh.geometry.x

    ref_points = []
    for pt_idx, cell in zip(valid_indices, point_cells):
        # Get cell's geometric node coordinates
        cell_dofs = x_dofs[cell]
        cell_coords = x_coords[cell_dofs]

        # Transform physical → reference coordinates
        physical_pt = self.points[pt_idx].reshape(1, -1)
        ref_pt = cmap.pull_back(physical_pt, cell_coords)
        ref_points.append(ref_pt[0])

    ref_points = np.array(ref_points)
```

#### 3. Basis Function Evaluation

```python
def _tabulate_basis(self, ref_points):
    element = self.V.element

    # Tabulate using basix
    # Output: (n_derivatives, n_points, n_basis, value_size)
    tab = element.basix_element.tabulate(0, ref_points)

    # Get values at derivative order 0
    basis_vals = tab[0]  # (n_points, n_basis, value_size)

    # Squeeze value_size dimension for scalar elements
    if basis_vals.ndim == 3 and basis_vals.shape[-1] == 1:
        basis_vals = basis_vals.squeeze(-1)

    return basis_vals  # Shape: (n_points, n_basis)
```

#### 4. Vector Application

```python
def apply_to_vector(self, b):
    if len(self._cells) == 0:
        return

    # Get array to modify
    if hasattr(b, 'x'):
        b_array = b.x.array  # dolfinx.fem.Function
    elif hasattr(b, 'vector'):
        b_array = b.vector.array
    else:
        b_array = b.array

    dofmap = self.V.dofmap

    # Add contributions
    for i, cell in enumerate(self._cells):
        cell_dofs = dofmap.cell_dofs(cell)
        for j, dof in enumerate(cell_dofs):
            b_array[dof] += self.magnitude * self._basis_values[i, j]
```

---

## Code Walkthrough

### Complete Flow Diagram

```
Input: V (FunctionSpace), points (N×3 array), magnitude (float)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. POINT LOCATION                                          │
│     ┌──────────────────┐                                    │
│     │ bb_tree(mesh)    │ Build bounding box tree            │
│     └────────┬─────────┘                                    │
│              ▼                                              │
│     ┌──────────────────────────┐                            │
│     │ compute_collisions_points │ Fast candidate search     │
│     └────────┬─────────────────┘                            │
│              ▼                                              │
│     ┌──────────────────────────┐                            │
│     │ compute_colliding_cells  │ Exact containment test     │
│     └────────┬─────────────────┘                            │
│              ▼                                              │
│     cells[] = [cell_id for each valid point]                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. COORDINATE TRANSFORMATION                               │
│     For each (point, cell):                                 │
│     ┌──────────────────────────┐                            │
│     │ cell_coords = x[dofmap]  │ Get cell geometry          │
│     └────────┬─────────────────┘                            │
│              ▼                                              │
│     ┌──────────────────────────┐                            │
│     │ cmap.pull_back(pt, coords)│ Physical → Reference      │
│     └────────┬─────────────────┘                            │
│              ▼                                              │
│     ref_points[] = [ξ, η, ζ for each point]                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. BASIS FUNCTION EVALUATION                               │
│     ┌──────────────────────────────┐                        │
│     │ basix_element.tabulate(0, ξ) │ Evaluate φᵢ(ξ)         │
│     └────────┬─────────────────────┘                        │
│              ▼                                              │
│     basis_values[n_points, n_basis] = φᵢ(ξⱼ)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. APPLY TO VECTOR (called separately)                     │
│     For each point i in cell c:                             │
│       For each local DoF j in cell c:                       │
│         global_dof = dofmap.cell_dofs(c)[j]                 │
│         b[global_dof] += magnitude × basis_values[i, j]     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    Output: Modified b vector
```

### Example: Single Point in Tetrahedron

Consider a point x₀ = (10, 0, 120) in a P1 (linear) tetrahedral mesh:

1. **Find cell**: Cell 42 contains the point
2. **Get cell geometry**: 4 vertices at corners
3. **Pull back**: x₀ → ξ₀ = (0.3, 0.2, 0.4) in reference tet
4. **Tabulate**: For P1, basis functions are:
   - φ₀(ξ) = 1 - ξ - η - ζ = 0.1
   - φ₁(ξ) = ξ = 0.3
   - φ₂(ξ) = η = 0.2
   - φ₃(ξ) = ζ = 0.4
5. **Apply**: If magnitude = 1.0:
   - b[dof₀] += 1.0 × 0.1 = 0.1
   - b[dof₁] += 1.0 × 0.3 = 0.3
   - b[dof₂] += 1.0 × 0.2 = 0.2
   - b[dof₃] += 1.0 × 0.4 = 0.4

Note: The basis values sum to 1.0 (partition of unity property).

---

## Usage

### Basic Usage

```python
from emgop.fem import NativePointSource
from dolfinx import fem

# Single point
point = np.array([10.0, 0.0, 120.0])
ps = NativePointSource(V, point, magnitude=1.0)

b = fem.Function(V)
ps.apply_to_vector(b)
```

### Multiple Points (Distributed Electrode)

```python
# Sample 256 points in a sphere (electrode surface)
center = np.array([35.0, 0.0, 120.0])
radius = 3.0
points = sample_points_in_sphere(center, radius, n=256)

# Distribute magnitude across points
magnitude = 1.0 / len(points)

ps = NativePointSource(V, points, magnitude=magnitude)
ps.apply_to_vector(b)
```

### Bipolar Configuration (Source + Ground)

```python
# Source electrode (+1 total current)
source_ps = NativePointSource(V, source_points, magnitude=+1.0/n_source)
source_ps.apply_to_vector(b)

# Ground electrode (-1 total current)
ground_ps = NativePointSource(V, ground_points, magnitude=-1.0/n_ground)
ground_ps.apply_to_vector(b)

# Now b sums to ~0 (current balance)
```

### With ElectrodeFEMSolver

```python
from emgop.fem.electrode_configs import ElectrodeFEMSolver

# Use native implementation
solver = ElectrodeFEMSolver(
    mesh_path,
    return_mode="localized",
    ground_position=ground_pos,
    use_native_point_source=True,  # Enable native implementation
)

uh = solver.solve(source_point)
```

---

## Verification

### Test Results

The native implementation was verified against scifem.PointSource:

| Test | Relative Difference | Status |
|------|---------------------|--------|
| Single point | 1.01×10⁻¹⁶ | PASS |
| 256 distributed points | 9.25×10⁻¹⁷ | PASS |
| Bipolar (±1) | 8.41×10⁻¹⁷ | PASS |
| Full FEM solver (RHS) | 7.95×10⁻¹⁷ | PASS |
| Full FEM solver (solution) | 3.17×10⁻¹⁵ | PASS |

All differences are at **machine precision** (~10⁻¹⁶), confirming identical behavior.

### Field Comparisons

2D slices, fiber potentials, and surface plots show:
- XY slice max difference: 6.66×10⁻¹⁶
- XZ slice max difference: 7.77×10⁻¹⁶
- Skin surface max difference: 7.77×10⁻¹⁶

### Running Verification Tests

```bash
cd /path/to/emgforge
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scifem
export PYTHONPATH=src

# Run numerical tests
python tests/fem_residual_debug/test_native_point_source.py

# Generate comparison plots
python sanity_checks/scripts/native_vs_scifem_field_plots.py
```

---

## Performance

### Timing Comparison

| N Points | Native (ms) | Scifem (ms) | Speedup |
|----------|-------------|-------------|---------|
| 1 | 1687 | 1676 | 0.99× |
| 10 | 1788 | 1808 | 1.01× |
| 50 | 1731 | 1669 | 0.96× |
| 100 | 1694 | 1775 | 1.05× |
| 256 | 1759 | 1643 | 0.93× |
| 512 | 1624 | 1617 | 1.00× |

Performance is essentially identical. The large absolute times (~1.7s) are dominated by:
1. Bounding box tree construction
2. Collision detection
3. Coordinate transformation

The actual basis evaluation and vector modification are negligible.

### Memory Usage

Both implementations have similar memory footprints:
- Store cell indices: O(n_points)
- Store basis values: O(n_points × n_basis_per_cell)
- For P1 elements with 256 points: ~256 × 4 × 8 bytes = 8 KB

---

## References

1. [FEniCSx Documentation](https://docs.fenicsproject.org/)
2. [scifem GitHub](https://github.com/scientificcomputing/scifem)
3. [Basix Documentation](https://docs.fenicsproject.org/basix/main/)
4. [FEniCS Discourse: PointSource in dolfinx](https://fenicsproject.discourse.group/t/pointsource-in-dolfinx/8337)

---

## File Locations

- **Implementation**: `src/emgop/fem/point_source.py`
- **Integration**: `src/emgop/fem/electrode_configs.py` (ElectrodeFEMSolver)
- **Tests**: `tests/fem_residual_debug/test_native_point_source.py`
- **Sanity checks**: `sanity_checks/scripts/native_point_source_sanity.py`
- **Field visualization**: `sanity_checks/scripts/native_vs_scifem_field_plots.py`
- **Notebooks**: `sanity_checks/notebooks/native_*.ipynb`
