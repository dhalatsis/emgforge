# FEM pipeline (forward model)

This document describes the **forward FEM problem** we solve on generated limb-like meshes, how the source is modeled, and how the implementation in this repo is structured.

## Goal

Given a 3D domain \(\Omega\) representing layered biological tissue (bone/muscle/fat/skin), and a current injection/source located near the surface, solve for the electric potential \(u(x)\) inside the domain under a **pure Neumann** formulation that is compatible with charge conservation and the constant-nullspace of Neumann Poisson-type problems.

This solution is later used as the **training target** \(u\) for voxel/operator learning, while the voxelized conductivity tensor field \(\Sigma(x)\) and a voxelized source channel \(s(x)\) are used as **model inputs**.

## PDE being solved

We solve (in weak form) the conductivity equation:

\[
\nabla \cdot (\Sigma(x)\nabla u(x)) = f(x) \quad \text{in }\Omega
\]

with **Neumann boundary condition** (constant here):

\[
(\Sigma\nabla u)\cdot n = g \quad \text{on }\partial\Omega
\]

In the current code, \(g\) is constant and defaults to 0.

### Critical constraint: Neumann compatibility / solvability

Pure Neumann problems have a constant nullspace: if \(u\) is a solution, so is \(u+c\). A solvable Neumann problem must satisfy:

\[
\int_\Omega f(x)\,dx + \int_{\partial\Omega} g\,ds = 0
\]

We enforce this in two ways:

1) **Construct the forcing to be mean-free** (zero integral) by adding a uniform “return” term (sink).

2) **Solve with PETSc constant nullspace**, and explicitly remove the nullspace component from the RHS.

## Tissue model and conductivity tensor

Meshes are generated with Gmsh physical volume groups (subdomains):

- `Cancellous Bone` (id=1)
- `Cortical Bone` (id=2)
- `Muscle` (id=3)
- `Fat` (id=4)
- `Skin` (id=5)

The FEM reads these markers from the `.msh` file and builds a piecewise-constant tensor field \(\Sigma(x)\) using a DG0 tensor function.

Defined in `src/emgop/fem/constants.py`:

- **Isotropic tissues** use scalar conductivity \(\sigma\) and convert to tensor \(\sigma I\).
- **Muscle** is anisotropic (diagonal tensor):
  - \(\sigma_{xx}=\sigma_{yy}=0.2455\)
  - \(\sigma_{zz} = 5\times 0.2455\) (anisotropy ratio = 5)

### Pennation angle (fiber orientation)

By default, muscle fibers are aligned with the Z-axis. The **pennation angle** allows rotating the anisotropy direction to simulate muscles where fibers are oriented at an angle to the longitudinal axis.

```bash
# Apply 30° pennation angle
python scripts/01_solve_fem.py --mesh mesh.msh --meta meta.json --out_dir out/ --pennation_angle 30.0
```

Implementation:
- `FEMModel.apply_pinnation(theta)` rotates the conductivity tensor at each muscle cell
- Uses Rodrigues' rotation formula around an axis tangent to the cylinder surface
- Must be called **before** solving (modifies `sigma_anisotropic` in-place)

See `docs/pennation_angle.md` for detailed mathematical description.

## Source modeling (what is \(f(x)\)?)

The implementation supports two main “source modes” for the volumetric forcing term \(f(x)\).

### A) Gaussian volumetric source (default)

This is the default CLI mode (`--source_mode gaussian`). We construct:

- A 3D Gaussian blob centered at a chosen source point \(x_s\):

\[
f_\text{blob}(x) = \mathcal{N}(x; x_s, \sigma^2)
\]

- Then subtract its mean so that \(\int_\Omega f(x)\,dx = 0\):

\[
f(x) = f_\text{blob}(x) - \frac{1}{|\Omega|}\int_\Omega f_\text{blob}(x)\,dx
\]

Implementation details:
- Implemented in `FEMModel.assign_source_to_point()`.
- Gaussian std dev is controlled by `source_sigma` (mesh units).
- The subtraction constant is computed via assembly (integral / volume), then the function is re-interpolated as blob-minus-constant.

### B) “Point” source / electrode (discrete point injection + uniform sink)

Enabled with `--source_mode point`.

In this mode, the volumetric forcing term is:
- a **uniform sink** of total \(-1\): \(f_\text{sink}(x) = -1/|\Omega|\)
- plus a discrete injected source of total \(+1\) applied to the RHS vector using `scifem.PointSource`.

Implementation details:
- `assign_source_to_point()` sets `source_function = Constant(-1/volume)`.
- `ConstrainedLinearProblem.add_point_source(points)` applies a point source to the assembled RHS:
  - If you pass \(N\) points, each has magnitude \(1/N\), so total injection is +1.

The code supports two point injection styles:
- **Single-point electrode**: if `point_electrode=True`, pass exactly one point.
- **Distributed electrode** (current default when point mode is used in `sanity.py`): sample `sampled_electrode_points` random points in a sphere of radius `electrode_radius` around the source location and distribute the +1 across them.

## Enforcing mean-free solution / handling the nullspace

Because of the constant nullspace under pure Neumann conditions, the linear system is singular without constraints. We handle this explicitly:

- Assemble matrix \(A\) from bilinear form \(a(u,v)\).
- Assemble RHS vector \(b\) from linear form \(L(v)\).
- Set PETSc nullspace to constant:
  - `nullspace = PETSc.NullSpace().create(constant=True)`
  - `A.setNullSpace(nullspace)`
- Remove nullspace component from RHS:
  - `nullspace.remove(b)`
- Solve `A u = b` with PETSc KSP.

This logic is in `src/emgop/fem/solver.py` in `ConstrainedLinearProblem`.

## Weak form used in code

Spaces (built in `FEMModel.build_model()`):
- \(V_u\): `CG1` scalar space for \(u\)
- \(V_\Sigma\): `DG0` tensor space for \(\Sigma\)
- \(V_f\): `CGk` space for Gaussian source (degree controlled by `source_degree`, default 1)

Forms (in `FEMModel.solve_for_point()`):

- Trial/test: `u = TrialFunction(V_scalar)`, `v = TestFunction(V_scalar)`
- Bilinear form:
  - if conductivity map built:
    - \(a(u,v)=\int_\Omega (\Sigma \nabla u)\cdot\nabla v\,dx\)
  - else (debug/ablations):
    - \(a(u,v)=\int_\Omega \nabla u\cdot\nabla v\,dx\)
- Linear form:
  - \(L(v)=\int_\Omega f\,v\,dx + \int_{\partial\Omega} g\,v\,ds\)

## Source placement (how do we choose \(x_s\)?)

The sanity runner provides a few convenient source placement options in `src/emgop/fem/sanity.py`:

### 1) “Near-surface at mid-height” (default)

`choose_source_point_near_surface(r_skin, length, ...)` places:
- \(z = 0.5L\)
- \(r = r_\text{skin} - \varepsilon\)
- \(\theta\) from `--angle` (radians)

The inward offset \(\varepsilon\) is controlled by:
- `--eps_mode skin_frac`: \(\varepsilon = eps\_value \cdot thickness\_skin\) (with a minimum of 0.2 units)
- `--eps_mode abs`: \(\varepsilon = eps\_value\) (mesh units)
- `--eps_mode r_frac`: \(\varepsilon = eps\_value \cdot r_\text{skin}\)

### 2) Normalized cylindrical coordinates (recommended for reproducibility)

`--source_cyl r_norm,theta_deg,z_norm` uses `source_point_from_cyl_normalized()`:
- \(r = r_\text{norm} \cdot r_\text{skin}\)
- \(\theta = theta\_deg\) (degrees)
- \(z = z_\text{norm}\cdot L\)

Optional clamping:
- `--source_layer skin|muscle`: clamp radial position into a band.
  - muscle band is between `radius_cort_bone` and `radius_muscle`
- `--source_margin`: stay away from layer interfaces by this amount (mesh units)

## Outputs written by the FEM sanity runner

The primary runner is `scripts/01_solve_fem.py`, which calls `emgop.fem.sanity.run_one()`.

For a single run, it writes to `--out_dir`:

- `u.npy`: the solved potential vector (DoF ordering of `CG1` on the mesh)
- `sigma.npy`: conductivity tensor function coefficients (DG0 tensor values per cell)
- `source.npy`: the volumetric source function coefficients (Gaussian or constant sink) on `V_pol`
- `probes_points.npy`: fixed sanity probe points (world coords)
- `probes_values.npy`: evaluated `u` at probe points
- `summary.json`: a compact record of run configuration and sanity stats
- `plots/` (optional): slice images for qualitative inspection (recommended with MPI size 1)

### `summary.json` fields

Written in `run_one()`:
- `mesh`, `meta`
- `mpi_size`
- `source_point` (world coords) **(critical for later voxelization)**
- `source_mode` (`gaussian` or `point`)
- `source_sigma` (if Gaussian)
- `fem_source` (path to `source.npy`)
- `source_cyl` (if used), `source_layer`, `source_margin`
- `r_skin`, `length`
- `mean_u` (domain mean of \(u\))
- `probe` stats (min/max/absmax, finite check)

## CLI usage reference

### Single mesh

```bash
PYTHONPATH=src mpirun -n 1 python scripts/01_solve_fem.py \\\n+  --mesh /path/to/sample_000000.msh \\\n+  --meta /path/to/sample_000000.json \\\n+  --out_dir /path/to/fem/sample_000000 \\\n+  --plots \\\n+  --source_mode gaussian \\\n+  --source_sigma 0.1\n+```

### Manifest mode

```bash
PYTHONPATH=src mpirun -n 1 python scripts/01_solve_fem.py \\\n+  --manifest /path/to/manifest.csv \\\n+  --index 0 \\\n+  --out_root /path/to/fem \\\n+  --source_mode gaussian \\\n+  --source_sigma 0.1\n+```

### Source placement via normalized cylindrical coordinates

```bash
PYTHONPATH=src mpirun -n 1 python scripts/01_solve_fem.py \\\n+  --mesh /path/to/sample_000000.msh \\\n+  --meta /path/to/sample_000000.json \\\n+  --out_dir /path/to/fem/sample_000000 \\\n+  --source_mode gaussian \\\n+  --source_sigma 0.1 \\\n+  --source_cyl 0.95,30,0.50 \\\n+  --source_layer skin \\\n+  --source_margin 0.5\n+```

## Implementation map (files to read)

- `src/emgop/fem/constants.py`: conductivity definitions and physical group ids
- `src/emgop/fem/solver.py`: FEMModel, weak form, nullspace solve, point source application
- `src/emgop/fem/sanity.py`: source placement helpers, probe evaluation, plotting, outputs
- `scripts/01_solve_fem.py`: CLI wrapper\n+
